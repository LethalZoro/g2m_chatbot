import os
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def convert_chroma_to_faiss():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load existing Chroma vector store
    chroma_path = r"D:\Coding\Job\Salik Labs\g2m_AI\vector_store"
    print(f"Loading Chroma vector store from: {chroma_path}")
    chroma_vectordb = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)
    
    # Get all documents using the correct method
    print("Loading documents from Chroma...")
    
    try:
        # Method 1: Get all documents with their embeddings
        collection = chroma_vectordb._collection
        result = collection.get(include=['documents', 'metadatas'])
        
        print(f"Found {len(result['documents'])} documents")
        
        # Convert to Document objects
        documents = []
        for i, (doc_text, metadata) in enumerate(zip(result['documents'], result['metadatas'])):
            if metadata is None:
                metadata = {}
            documents.append(Document(page_content=doc_text, metadata=metadata))
            
            if i % 100 == 0:
                print(f"Processed {i} documents...")
        
        print(f"Converted {len(documents)} documents to Document objects")
        
        # Create FAISS vector store from documents
        print("Creating FAISS vector store...")
        
        if len(documents) == 0:
            print("❌ No documents found in Chroma vector store!")
            return
            
        # Create FAISS in batches to avoid memory issues
        batch_size = 1000
        faiss_vectordb = None
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
            
            if faiss_vectordb is None:
                # Create the first batch
                faiss_vectordb = FAISS.from_documents(batch, embedding_model)
            else:
                # Add subsequent batches
                batch_vectordb = FAISS.from_documents(batch, embedding_model)
                faiss_vectordb.merge_from(batch_vectordb)
        
        # Save FAISS vector store
        output_path = r"D:\Coding\Job\Salik Labs\g2m_AI\vector_store_faiss"
        print(f"Saving FAISS vector store to: {output_path}")
        faiss_vectordb.save_local(output_path)
        
        print("✅ Conversion completed!")
        print(f"FAISS vector store saved to: {output_path}")
        print("\nNext steps:")
        print("1. Zip the vector_store_faiss folder")
        print("2. Upload to S3 as vector_store.zip")
        print("3. Update your interface.py to use FAISS")
        
        # Test the converted vector store
        print("\n🧪 Testing the converted FAISS vector store...")
        test_query = "test query"
        results = faiss_vectordb.similarity_search(test_query, k=3)
        print(f"Test search returned {len(results)} results")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        
        # Alternative method: Try using similarity search to get all docs
        print("\n🔄 Trying alternative method...")
        try:
            # Get a sample to understand the structure
            sample_docs = chroma_vectordb.similarity_search("", k=10)
            print(f"Sample docs type: {type(sample_docs)}")
            if sample_docs:
                print(f"First doc type: {type(sample_docs[0])}")
                print(f"First doc content preview: {sample_docs[0].page_content[:100]}...")
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")

if __name__ == "__main__":
    convert_chroma_to_faiss()