import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load API keys from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Local embeddings model (same one used for upload)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone index
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# Ask questions in a loop
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    results = vector_store.similarity_search(query, k=1)
    if results:
        print(f"\nAnswer: {results[0].page_content}")
    else:
        print("\nNo relevant answer found.")
