# RAG Chatbot

**🔗 Live Demo: [https://rag-chatbot-six-silk.vercel.app](https://rag-chatbot-six-silk.vercel.app)**

A simple **Retrieval-Augmented Generation (RAG)** chatbot.  
Upload a document → Ask questions → Get answers grounded in your document.

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It solves the problem of LLMs not knowing about *your* private documents.

Instead of relying on the LLM's training data, RAG:
1. **Splits** your document into small chunks
2. **Embeds** each chunk as a vector (a list of numbers representing meaning)
3. **Stores** all vectors in a vector store (FAISS)
4. When you ask a question, **finds the most similar chunks** using cosine similarity
5. **Sends** those chunks + your question to the LLM as context
6. The LLM **answers based only on the retrieved context** — not from memory

This means the LLM can only answer from your document, not hallucinate from training data.

---

## RAG Flow (step by step)

```
User uploads a file
       │
       ▼
Read text from file (PDF / DOCX / TXT)
       │
       ▼
Split into chunks (RecursiveCharacterTextSplitter)
  chunk_size=1000, chunk_overlap=200
       │
       ▼
Embed each chunk (HuggingFace: sentence-transformers/all-MiniLM-L6-v2)
  → converts text to a 384-dimensional vector
  → free, runs locally, no API key needed
       │
       ▼
Store vectors in FAISS (in-memory vector store)
       │
       ▼
User asks a question
       │
       ▼
Embed the question (same embedding model)
       │
       ▼
Similarity search in FAISS → top 4 most relevant chunks
       │
       ▼
Build prompt: context (4 chunks) + question
       │
       ▼
Send to Groq LLM (llama-3.3-70b-versatile)
       │
       ▼
Return answer to user
```

---

## Tech Stack

| Component | Library | Why |
|---|---|---|
| Text splitting | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` splits smartly by paragraphs/sentences |
| Embeddings | `langchain-huggingface` + `sentence-transformers` | Free, local, no API key needed |
| Vector store | `faiss-cpu` via `langchain-community` | Fast in-memory similarity search |
| LLM | `langchain-groq` (llama-3.3-70b-versatile) | Fast, free tier available |
| API | `FastAPI` | Simple Python REST API |
| Frontend | Vanilla HTML/CSS/JS | No framework needed |
| Deployment | Vercel | Frontend + Python serverless |

---

## Project Structure

```
├── api/
│   └── index.py     ← FastAPI Python serverless function
├── frontend/
│   └── index.html   ← Static HTML
├── rag.py           ← RAG pipeline (imported by api/index.py)
├── requirements.txt
├── vercel.json
└── README.md


---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start the server
uvicorn api.index:app --reload

# Open the UI
# Go to: http://localhost:8000
# (index.html talks to /api which is localhost:8000/api)
```

---