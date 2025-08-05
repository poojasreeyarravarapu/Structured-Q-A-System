# Structured Q/A System
An AI-powered FAQ Question Answering Bot that retrieves accurate answers from your stored knowledge base using HuggingFace embeddings and Pinecone vector search.
Built with Python + Streamlit for a clean, interactive web interface.

**Domain Specific Chatbot**

>> Answers user queries quickly and accurately.
>> Uses LangChain + HuggingFace embeddings for semantic understanding.
>> Uses Pinecone for vector search to retrieve relevant context from your knowledge base.

🚀 **Features**
- Semantic Search → Understands meaning, not just keywords
- HuggingFace Embeddings → No OpenAI API required
- Pinecone Vector DB → Fast and scalable retrieval
- Instant Response → No model training needed
- Streamlit UI → Simple, browser-based chatbot interface

🛠 **Tech Stack**
* Python  
* Streamlit (UI)
* LangChain (integration & retrieval logic)
* HuggingFace Sentence Transformers (embeddings)
* Pinecone (vector database)
* dotenv (environment variables management)

**Key Components & Flow**
**Step-by-step architecture:**

1. User Query → The user asks a question.

2. Embedding Generation (LangChain + HuggingFace Sentence Transformers):

    The query is converted into a vector embedding (numerical representation of meaning).

    Example: "How do I reset my password?" → [0.021, -0.114, 0.98, ...]

3. Vector Database Search (Pinecone):

    This vector is compared to stored document embeddings.

    Finds most relevant documents (e.g., internal knowledge base, FAQs).

4. Response Retrieval (LangChain)

    The most similar document is returned as the answer.

    No LLM text generation is used — the answer comes directly from your stored knowledge base.

5. Final Response to User:

    The bot replies instantly.

    If the query isn’t covered in the knowledge base, it can respond with a default “No answer found” or suggest escalation.



**SETUP INSTRUCTIONS**

1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/Structured-Q-A-System.git
cd Structured-Q-A-System
```

2️⃣ Create & activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Set up environment variables like in the .env.example file in the repository


📤 Upload your FAQ data to Pinecone
Place your FAQ content in faqs.txt in the format:
```bash
Q: How do I reset my password?
A: Go to the login page, click "Forgot Password", and follow the instructions.

Q: How can I update my email?
A: Navigate to account settings and click "Update Email".
```
Then run:
```bash
python upload_data.py
```

💬 Run the chatbot
```bash
streamlit run app.py
```
