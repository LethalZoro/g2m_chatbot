import os

# Fix protobuf compatibility issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ANONYMIZED_TELEMETRY"] = "False"


import streamlit as st
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
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

@st.cache_resource
def initialize_vector_store():
    """Initialize vector store - local or download from S3"""
    local_path = os.path.join(os.path.dirname(__file__), "vector_store")
    
    # Check if running locally (vector store already exists)
    if os.path.exists(local_path):
        st.info("📁 Using local vector store")
        return local_path
    
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
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            st.write("✅ Extraction completed")
        except Exception as extract_error:
            st.error(f"❌ Extraction failed: {extract_error}")
            st.stop()
        
        # Cleanup
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

# Replace your current vector store initialization with:


# Setup embeddings and text splitter with optimized settings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Initialize or load vectorstore with optimized batch size
persist_directory = initialize_vector_store()

# persist_directory = r"D:\Coding\Job\Salik Labs\g2m_AI\vector_store"

if not os.path.exists(persist_directory):
    # Fallback for when vector store doesn't exist
    st.error("Vector store not found. Please ensure vector_store folder is in your repository.")
    st.stop()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


# if os.path.exists(persist_directory):
#     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
# else:
#     vectordb = Chroma(embedding_function=embedding_model, persist_directory=persist_directory)

# LLM setup
llm = ChatOpenAI(model="gpt-4o")

# FIXED: Simple prompt template without MessagesPlaceholder in the chain
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
"""

# FIXED: Create prompt template that RunnableWithMessageHistory can work with
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", chat_prompt_template),
    ("human", "{question}"),
])

# Format retrieved documents into a single string
def format_docs(docs):
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

# Retriever function
def get_retriever(k=20):
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 2,
            "lambda_mult": 0.7
        }
    )

# Chat history store - FIXED: Use proper session management
if 'message_histories' not in st.session_state:
    st.session_state.message_histories = {}

def get_message_history(session_id):
    """Get or create message history for a session"""
    if session_id not in st.session_state.message_histories:
        st.session_state.message_histories[session_id] = ChatMessageHistory()
    # print("History",st.session_state.message_histories[session_id])
    return st.session_state.message_histories[session_id]

# FIXED: Simplified chain without redundant history handling
chain = (
    {
        "context": RunnableLambda(lambda inputs: format_docs(get_retriever().invoke(inputs["question"]))),
        "question": itemgetter("question"),
    }
    | chat_prompt
    | llm
    | StrOutputParser()
)

# Chain with history - FIXED: Proper configuration for RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="question",
    history_messages_key="chat_history",  # Changed from "history" to avoid conflict
)

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

def chat(session_id, k=20):
    """Example interactive chat function for non-Streamlit usage"""
    print("Starting chat session. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]: 
            print("Exiting chat session.")
            break
        try:
            response = chain_with_history.invoke(
                {"question": user_input}, 
                config={"configurable": {"session_id": session_id}}
            )
            print(f"Bot: {response}")
        except Exception as e:
            print("Error:", e)

# STREAMLIT UI
st.set_page_config(page_title="PDF Document Chatbot", page_icon="🤖")
st.title("📄 PDF Document Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session"

def submit_question():
    """FIXED: Simplified submit function that lets LangChain handle history"""
    question = st.session_state.user_input.strip()
    if question:
        # Add user message to Streamlit UI
        st.session_state.messages.append({"role": "user", "content": question})
        
        try:
            # FIXED: Let LangChain handle the history automatically
            response = chain_with_history.invoke(
                {"question": question},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            
            # Add bot response to Streamlit UI
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Error: {e}"
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

# JavaScript for Enter key submission
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const textArea = document.querySelector('textarea[data-testid="stTextArea"]');
    if (textArea) {
        textArea.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.altKey) {
                event.preventDefault();
                const form = textArea.closest('form');
                if (form) {
                    form.requestSubmit();
                }
            }
        });
    }
});
</script>
""", unsafe_allow_html=True)

# Display chat messages
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

# Clear chat button
if st.button("Clear chat"):
    st.session_state.messages = []
    if st.session_state.session_id in st.session_state.message_histories:
        del st.session_state.message_histories[st.session_state.session_id]
    st.rerun()