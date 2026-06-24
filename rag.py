"""
RAG Pipeline — Retrieval-Augmented Generation
----------------------------------------------
Flow:
  1. User uploads a file (PDF / DOCX / TXT)
  2. Read the file text
  3. Split into chunks  (RecursiveCharacterTextSplitter)
  4. Create embeddings  (HuggingFace sentence-transformers — free, no API key)
  5. Store in FAISS vector store (in-memory)
  6. User asks a question
  7. Retrieve top-4 similar chunks from FAISS (similarity search)
  8. Build prompt: context chunks + question
  9. Send to Groq LLM (llama-3.3-70b-versatile)
  10. Return the answer
"""

import os
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

load_dotenv()


# ── Step 4: Embeddings model ───────────────────────────────────────────────────
# HuggingFace runs locally — no API key needed.
# Downloads ~90MB model on first run, then cached.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ── Step 9: Groq LLM ──────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)


# ── Step 8: Prompt template ────────────────────────────────────────────────────
# We tell the LLM: answer ONLY from the context, not from your training data.
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided document context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
    input_variables=["context", "question"],
)


# ── Step 1-2: Read text from uploaded file ────────────────────────────────────
def read_file_text(filename: str, file_bytes: bytes) -> str:
    """Extract plain text from PDF, DOCX, or TXT files."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    elif ext == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        doc = Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")


# ── Step 3-5: Build the vector store from a document ─────────────────────────
def build_vector_store(filename: str, file_bytes: bytes) -> tuple:
    """
    Takes a file, extracts text, splits into chunks,
    embeds them, and stores in FAISS.
    Returns the FAISS vector_store + stats (chunk count, character count).
    """
    # Step 2: Read text
    text = read_file_text(filename, file_bytes)

    if not text.strip():
        raise ValueError("Could not extract any text from the file.")

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    # Step 4+5: Embed chunks and store in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, len(chunks), len(text)


# ── Step 6-10: Answer a question using the vector store ──────────────────────
def ask_question(vector_store, question: str) -> tuple:
    """
    Takes the FAISS vector store and a question.
    Retrieves top-4 relevant chunks, builds a prompt, calls Groq LLM.
    Returns the answer and the retrieved source chunks.
    """
    # Step 7: Retriever — similarity search, top 4 chunks
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 7: Retrieve the 4 most relevant chunks
    retrieved_docs = retriever.invoke(question)

    # Step 8: Concatenate chunks into one context string
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Step 8: Build the final prompt (context + question)
    final_prompt = prompt.invoke({"context": context_text, "question": question})

    # Step 9: Send to Groq LLM and get the answer
    answer = llm.invoke(final_prompt)

    # Step 10: Return answer + source chunk previews
    sources = [doc.page_content[:200] + "..." for doc in retrieved_docs]

    return answer.content, sources
