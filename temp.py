from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
import os
from tqdm import tqdm
import concurrent.futures
import glob


load_dotenv()
# Setup embeddings and text splitter with optimized settings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Increased chunk size for better table coverage
    chunk_overlap=400,  # Increased overlap to prevent content loss
    separators=["\n\n", "\n", ".", " ", ""]  # Added period separator for better structure preservation
)

# Initialize or load vectorstore with optimized batch size
persist_directory = r"D:\Coding\Job\Salik Labs\g2m_AI\vector_store"
if os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
else:
    vectordb = Chroma(embedding_function=embedding_model, persist_directory=persist_directory)

# LLM setup
llm = ChatOpenAI(model="gpt-4o")

# Prompt template for answering questions
chat_prompt_template = """
Role: You are a helpful assistant specialized in answering questions based on the given documents. Be precise and clear in your responses.

Context:
{context}

Conversation history:
{history}

Question: {question}

Instructions:
- Answer in clear, simple language based on the provided context
- Always cite the specific document(s) and page number(s) you used to answer the question
- Format your citations like: "According to [Document Name] (Page X)..." or "As mentioned in [Document Name] (Page X)..."
- If information comes from multiple sources, cite all relevant sources
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", chat_prompt_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# Format retrieved documents into a single string
def format_docs(docs):
    formatted_docs = []
    seen_content = set()  # To avoid duplicate content
    
    for doc in docs:
        # Extract source information from metadata
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'Unknown page')
        
        # Create a hash of the content to check for duplicates
        content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
        
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            # Format document with source info
            doc_info = f"Source: {os.path.basename(source)} (Page {page + 1})\nContent: {doc.page_content}"
            formatted_docs.append(doc_info)
    
    return "\n\n---\n\n".join(formatted_docs)

# Retriever function (search top k relevant docs)
def get_retriever(k=20):  # Increased from 15 to get more relevant chunks
    return vectordb.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diverse results
        search_kwargs={
            "k": k,
            "fetch_k": k * 2,  # Fetch more candidates for MMR
            "lambda_mult": 0.7  # Balance between relevance and diversity
        }
    )

# Define chain with message history for conversation continuity
chain = (
    {
        "context": RunnableLambda(lambda inputs: format_docs(get_retriever().invoke(inputs["question"]))),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | chat_prompt
    | llm
    | StrOutputParser()
)

# Chat history store
message_histories = {}

def get_message_history(session_id):
    if session_id not in message_histories:
        message_histories[session_id] = ChatMessageHistory()
        message_histories[session_id].add_ai_message("Hello! How can I assist you today?")
    return message_histories[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="question",
    history_messages_key="history",
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

    # Get all PDF files recursively from the folder
    pdf_files = glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Starting parallel processing...")
    
    all_texts = []
    errors = []
    
    # Process PDFs in parallel with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {executor.submit(process_single_pdf, pdf_file): pdf_file 
                         for pdf_file in pdf_files}
        
        # Process results with progress bar
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
        
        # Add documents in batches for better performance
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Adding to vector store"):
            batch = all_texts[i:i + batch_size]
            vectordb.add_documents(batch)
        
        vectordb.persist()
        print(f"Indexing completed! Total chunks: {len(all_texts)}")
    else:
        print("No documents were successfully processed.")


# Example interactive chat function
def chat(session_id, k=20):  # Increased k for more context
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

if __name__ == "__main__":
    # Example usage:
    # 1. Load your PDF document(s) once:
    load_and_index_pdf(r"D:\Coding\Job\Salik Labs\g2m_AI\docs", max_workers=8, batch_size=1000,force_reindex=False)
    # 2. Start chat session with a unique ID:
    chat(session_id="default_session")
