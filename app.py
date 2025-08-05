import os
import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# Streamlit UI
st.set_page_config(page_title="Structured Q/A Bot", page_icon="🤖")
st.title("🤖 Structured Q/A Bot")
st.write("Ask me anything from the stored FAQ data.")

# Session state for conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Input box + Ask button (works with Enter too)
with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("Your question:")
    submit = st.form_submit_button("Ask")

# Process the query
if submit and user_query.strip():
    results = vector_store.similarity_search_with_score(user_query, k=1)

    if results:
        best_match, score = results[0]  # unpack document & similarity score

        # For cosine similarity: higher = more similar (1.0 = perfect match)
        if score >= 0.80:  # Only accept results with >=80% similarity
            raw_text = best_match.page_content
            # Extract only answer
            if "A:" in raw_text:
                answer = raw_text.split("A:", 1)[1].strip()
            else:
                answer = raw_text
        else:
            answer = "Sorry, I couldn't find an answer."
    else:
        answer = "Sorry, I couldn't find an answer."

    # Store conversation
    st.session_state["messages"].append(("You", user_query))
    st.session_state["messages"].append(("Bot", answer))

# Show conversation history
for role, msg in st.session_state["messages"]:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")
