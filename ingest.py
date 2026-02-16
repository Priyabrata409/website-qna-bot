
import os
import time


from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def ingest_url(url):
    """
    Scrapes a URL, chunks the text, and stores embeddings in Pinecone.
    """
    print(f"Loading content from {url}...")
    loader = WebBaseLoader(url)
    docs = loader.load()

    print(f"Splitting {len(docs)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} splits.")

    print("Initializing Pinecone and Embeddings...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env")

    # Initialize Pinecone Client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, if not create it (Serverless spec for AWS us-east-1 is a safe default for free tier)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating it...")
        try:
            pc.create_index(
                name=index_name,
                dimension=1536, # OpenAI text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
            print(f"Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Indexing documents to Pinecone...")
    docsearch = PineconeVectorStore.from_documents(
        splits, 
        embeddings, 
        index_name=index_name
    )
    print("Ingestion complete!")
    return docsearch

if __name__ == "__main__":
    # Test with a URL
    test_url = "https://python.langchain.com/docs/get_started/introduction"
    # ingest_url(test_url)
    print("Run this script by importing ingest_url or uncommenting the test call.")
