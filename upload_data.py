import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create HuggingFace embeddings object (free local model)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Read FAQ file
with open("faqs.txt", "r", encoding="utf-8") as f:
    faqs = f.read().strip().split("\n\n")

# Create vector store connected to Pinecone
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# Upload FAQ data
for faq in faqs:
    vector_store.add_texts([faq])

print("✅ FAQ data uploaded to Pinecone using free HuggingFace embeddings!")
