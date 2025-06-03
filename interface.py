import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Fix protobuf compatibility issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from tqdm import tqdm
import concurrent.futures
import glob

import boto3
import zipfile

import requests
from botocore.exceptions import NoCredentialsError, ClientError

# load_dotenv()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,
    chunk_overlap=512,
    separators=["\n\n", "\n", ".", " ", ""]
)

st.set_page_config(page_title="PDF Document Chatbot", page_icon="🤖")
st.title("📄 PDF Document Chatbot")

@st.cache_resource
def initialize_vector_store():
    """Initialize vector store - local or download from S3"""
    local_path = os.path.join(os.path.dirname(__file__), "vector_store")
    
    # Check if running locally (vector store already exists)
    if os.path.exists(local_path):
        # Verify the vector store is actually valid by checking for key files
        chroma_db_path = os.path.join(local_path, "chroma.sqlite3")
        if os.path.exists(chroma_db_path):
            st.info("📁 Using local vector store")
            return local_path
        else:
            st.warning("⚠️ Local vector store exists but appears incomplete, re-downloading...")
    
    # Running on cloud - download from S3
    st.info("🔄 Loading vector store from S3... This may take 2-3 minutes on first load.")
    
    try:
        # Debug: Check if secrets are available
        try:
            access_key = st.secrets["AWS_ACCESS_KEY_ID"]
            secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
            region = st.secrets.get("AWS_REGION", "us-east-1")
            bucket_name = st.secrets["S3_BUCKET_NAME"]
            st.write(f"Debug: Using bucket '{bucket_name}' in region '{region}'")
        except KeyError as e:
            st.error(f"❌ Missing secret: {e}")
            st.stop()
        
        # Initialize S3 client with secrets
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        object_key = "vector_store.zip"
        local_zip_path = "vector_store.zip"
        
        # Test S3 connection first
        try:
            st.write("Debug: Testing S3 connection...")
            s3_client.head_bucket(Bucket=bucket_name)
            st.write("✅ S3 bucket accessible")
            
            # Check if file exists
            response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
            total_size = response['ContentLength']
            st.write(f"✅ File found: {total_size / (1024*1024):.1f}MB")
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            st.error(f"❌ S3 connection test failed ({error_code}): {e}")
            st.stop()
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"📥 Downloading {total_size / (1024*1024):.1f}MB from S3...")
        
        # Download with simpler approach (no callback for now)
        try:
            st.write("Debug: Starting download...")
            s3_client.download_file(bucket_name, object_key, local_zip_path)
            st.write("✅ Download completed")
            
        except Exception as download_error:
            st.error(f"❌ Download failed: {download_error}")
            # Try to get more specific error info
            if hasattr(download_error, 'response'):
                st.error(f"Response: {download_error.response}")
            st.stop()
        
        # Extract
        status_text.text("📦 Extracting vector store...")
        progress_bar.progress(0.9)
        
        try:
            # Remove existing vector_store directory if it exists but is incomplete
            if os.path.exists("vector_store"):
                import shutil
                shutil.rmtree("vector_store")
            
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            st.write("✅ Extraction completed")
            
            # Verify extraction was successful
            extracted_files = os.listdir('vector_store') if os.path.exists('vector_store') else []
            st.write(f"Extracted files: {extracted_files}")
            
            # Check for critical files
            chroma_db_path = os.path.join("vector_store", "chroma.sqlite3")
            if not os.path.exists(chroma_db_path):
                st.error("❌ Critical vector store files missing after extraction")
                st.stop()
                
        except Exception as extract_error:
            st.error(f"❌ Extraction failed: {extract_error}")
            st.stop()
        
        # Cleanup zip file
        try:
            os.remove(local_zip_path)
        except:
            pass  # Don't fail if cleanup fails
        
        # Clear progress indicators
        progress_bar.progress(1.0)
        status_text.text("✅ Vector store loaded successfully!")
        
        return "vector_store"
        
    except Exception as e:
        st.error(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        # Show more debug info
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()


def verify_vector_store(persist_directory):
    """Verify that the vector store is properly initialized and accessible"""
    try:
        if not os.path.exists(persist_directory):
            st.error(f"❌ Vector store directory does not exist: {persist_directory}")
            return False
            
        # Check for essential Chroma files
        chroma_db_path = os.path.join(persist_directory, "chroma.sqlite3")
        if not os.path.exists(chroma_db_path):
            st.error(f"❌ Chroma database file missing: {chroma_db_path}")
            return False
            
        # List all files in the directory for debugging
        all_files = []
        for root, dirs, files in os.walk(persist_directory):
            for file in files:
                all_files.append(os.path.relpath(os.path.join(root, file), persist_directory))
        # st.write(f"Debug: All files in vector store: {all_files}")
        
        # Try to load the vector store to ensure it's accessible
        try:
            test_vectordb = Chroma(
                persist_directory=persist_directory, 
                embedding_function=embedding_model
            )
            
            # Try a simple operation to verify it works
            collection_count = test_vectordb._collection.count()
            # st.write(f"Debug: Vector store contains {collection_count} documents")
            
            if collection_count == 0:
                st.warning("⚠️ Vector store is not accessible due to some error contact your administrator for further assistance.")
                # For now, let's continue even with 0 documents
                return True  # Changed from False to True to allow empty databases
            
            return True
            
        except Exception as chroma_error:
            st.error(f"❌ Chroma initialization failed: {chroma_error}")
            
            # Try alternative initialization methods
            try:
                st.write("Debug: Trying alternative Chroma initialization...")
                
                # Method 1: Try without specifying collection name
                test_vectordb2 = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_model
                )
                collection_count2 = len(test_vectordb2.get()['ids'])
                st.write(f"Debug: Alternative method found {collection_count2} documents")
                return collection_count2 >= 0  # Accept even 0 documents
                
            except Exception as alt_error:
                st.error(f"❌ Alternative Chroma initialization also failed: {alt_error}")
                return False
        
    except Exception as e:
        st.error(f"❌ Vector store verification failed with unexpected error: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False


# Modified vector store loading with improved verification
persist_directory = initialize_vector_store()

# Verify the vector store before proceeding
if not verify_vector_store(persist_directory):
    st.error("❌ Vector store verification failed completely.")
    
    # Provide user options
    st.write("**Possible solutions:**")
    st.write("1. Check if your S3 vector store contains the correct files")
    st.write("2. Verify that the vector store was created with the same embedding model")
    st.write("3. Try clearing the Streamlit cache and reloading")
    
    if st.button("🔄 Clear Cache and Retry"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("🔧 Continue Anyway (for testing)"):
        st.warning("⚠️ Continuing with potentially broken vector store...")
    else:
        st.stop()

# Initialize vector database
try:
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    st.success("✅ Vector database initialized successfully!")
except Exception as e:
    st.error(f"❌ Failed to initialize vector database: {e}")
    st.stop()

# LLM setup
try:
    llm = ChatOpenAI(model="gpt-4o")
except Exception as e:
    st.error(f"❌ Failed to initialize OpenAI model: {e}")
    st.stop()

# Chat prompt template
chat_prompt_template = """
Role: You are a helpful assistant specialized in answering questions based on the given documents. Be precise and clear in your responses.

Context:
{context}

Question: {question}

Instructions:
- Answer in clear, simple language based on the provided context
- Always cite the specific document(s) and page number(s) you used to answer the question
- Format your citations like: "According to [Document Name] (Page X)..." or "As mentioned in [Document Name] (Page X)..."
- If information comes from multiple sources, cite all relevant sources
- Use the conversation history to provide contextual responses when relevant
- if not enough context is provided tell the user that but still try to answer the question based on the context provided
"""

# Create prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", chat_prompt_template),
    ("human", "{question}"),
])

# Format retrieved documents into a single string
def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    
    formatted_docs = []
    seen_content = set()
    
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'Unknown page')
        content_hash = hash(doc.page_content[:100])
        
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            doc_info = f"Source: {os.path.basename(source)} (Page {page + 1})\nContent: {doc.page_content}"
            formatted_docs.append(doc_info)
    
    return "\n\n---\n\n".join(formatted_docs)

# Retriever function with error handling
def get_retriever(k=20):
    try:
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 2,
                "lambda_mult": 0.7
            }
        )
    except Exception as e:
        st.error(f"❌ Error creating retriever: {e}")
        # Fallback to basic retriever
        return vectordb.as_retriever(search_kwargs={"k": k})

# Chat history store
if 'message_histories' not in st.session_state:
    st.session_state.message_histories = {}

def get_message_history(session_id):
    """Get or create message history for a session"""
    if session_id not in st.session_state.message_histories:
        st.session_state.message_histories[session_id] = ChatMessageHistory()
    return st.session_state.message_histories[session_id]

# Create the chain
chain = (
    {
        "context": RunnableLambda(lambda inputs: format_docs(get_retriever().invoke(inputs["question"]))),
        "question": itemgetter("question"),
    }
    | chat_prompt
    | llm
    | StrOutputParser()
)

# Chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Utility functions for PDF processing (keep your existing functions)
def process_single_pdf(pdf_file):
    """Process a single PDF file and return the text chunks"""
    try:
        print(f"Processing: {os.path.basename(pdf_file)}")
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        return texts, None
    except Exception as e:
        return None, f"Error processing {pdf_file}: {e}"

def load_and_index_pdf(folder_path, max_workers=4, batch_size=50, force_reindex=False):
    """Load and index all PDF files from a folder and its subfolders with parallel processing"""
    
    if not force_reindex:
        return

    pdf_files = glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Starting parallel processing...")
    
    all_texts = []
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, pdf_file): pdf_file 
                         for pdf_file in pdf_files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_pdf), 
                          total=len(pdf_files), desc="Processing PDFs"):
            texts, error = future.result()
            if texts:
                all_texts.extend(texts)
            if error:
                errors.append(error)
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
    
    if all_texts:
        print(f"\nIndexing {len(all_texts)} text chunks in batches...")
        
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Adding to vector store"):
            batch = all_texts[i:i + batch_size]
            vectordb.add_documents(batch)
        
        vectordb.persist()
        print(f"Indexing completed! Total chunks: {len(all_texts)}")
    else:
        print("No documents were successfully processed.")

# STREAMLIT UI

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session"

def submit_question():
    """Submit function with better error handling"""
    question = st.session_state.user_input.strip()
    if question:
        # Add user message to Streamlit UI
        st.session_state.messages.append({"role": "user", "content": question})
        
        try:
            # Check if vector store has documents before querying
            doc_count = vectordb._collection.count() if hasattr(vectordb, '_collection') else 0
            
            if doc_count == 0:
                response = "I don't have any documents loaded in my vector store yet. Please ensure that the vector store contains indexed documents before asking questions."
            else:
                response = chain_with_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
            
            # Add bot response to Streamlit UI
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}\n\nThis might be due to an empty vector store or connection issues."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear input
        st.session_state.user_input = ""

# Chat container
chat_container = st.container()

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your question:", 
        key="user_input", 
        placeholder="Type your question here...",
        height=None,
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button(label="Send", on_click=submit_question)

# Display chat messages
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

# # Sidebar with debug info
# with st.sidebar:
#     st.header("Debug Information")
    
#     try:
#         doc_count = vectordb._collection.count() if hasattr(vectordb, '_collection') else "Unknown"
#         st.write(f"📊 Documents in vector store: {doc_count}")
#     except:
#         st.write("📊 Documents in vector store: Unable to count")
    
#     st.write(f"📁 Vector store path: {persist_directory}")
    
#     if os.path.exists(persist_directory):
#         files = os.listdir(persist_directory)
#         st.write(f"📄 Files in vector store: {files}")
    
#     if st.button("🔄 Refresh Vector Store Info"):
#         st.rerun()

# Clear chat button
if st.button("Clear chat"):
    st.session_state.messages = []
    if st.session_state.session_id in st.session_state.message_histories:
        del st.session_state.message_histories[st.session_state.session_id]
    st.rerun()