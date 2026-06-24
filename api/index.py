"""
FastAPI server — HTTP wrapper around the RAG pipeline.
All the actual RAG logic lives in rag.py.
"""

import os
import sys

# Vercel serverless has a read-only filesystem.
# HuggingFace model cache must go to /tmp (the only writable dir on Vercel).
os.environ.setdefault("HF_HOME", "/tmp/huggingface")

# Allow importing rag.py from the project root (needed on Vercel)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import build_vector_store, ask_question

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory vector store — lives for the lifetime of this server instance
vector_store = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receive uploaded file and build the RAG vector store."""
    global vector_store

    file_bytes = await file.read()

    try:
        vector_store, chunk_count, char_count = build_vector_store(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": f"File '{file.filename}' uploaded and indexed successfully.",
        "chunks": chunk_count,
        "characters": char_count,
    }


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question against the uploaded document."""
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a file first.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, sources = ask_question(vector_store, request.question.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=answer, sources=sources)
