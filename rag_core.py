"""Core RAG + voice helpers for MediBot."""
import io
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

VECTOR_DIR = Path("vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # must match the model used to build your index
LLM_MODEL = "openai/gpt-oss-20b"      # use the SAME model name that works in connect_memory_with_llm.py
WHISPER_MODEL = "whisper-large-v3"    # Groq speech-to-text model
BATCH_SIZE = 64

# label shown in the app -> (language name for the LLM, language code for voice)
LANGUAGES = {
    "English": ("English", "en"),
    "हिन्दी (Hindi)": ("Hindi", "hi"),
    "मराठी (Marathi)": ("Marathi", "mr"),
}

PROMPT = """You are a helpful medical information assistant.
Answer the question using ONLY the context below.
If the context only partly answers it, share the relevant parts and say what is missing.
Say "I don't know based on the provided documents." only if the context has nothing relevant.
Do not invent facts.
Do not start with "Yes" or "No" unless the question is a yes/no question.
Write the whole answer in {language}.

Context:
{context}

Question: {question}

Answer:"""


# ---------- models ----------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------- books (one FAISS index per folder inside vectorstore/) ----------
def list_books():
    if not VECTOR_DIR.exists():
        return []
    return sorted(p.name for p in VECTOR_DIR.iterdir() if (p / "index.faiss").exists())


def slugify(filename):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", Path(filename).stem).strip("_").lower()
    return slug or "book"


def build_book_index(pdf_path, display_name, embeddings, progress=None):
    """Read a PDF, chunk it, embed it and save it as vectorstore/<name>/."""
    docs = PyPDFLoader(pdf_path).load()
    if not docs or not any(d.page_content.strip() for d in docs):
        raise ValueError(
            "No text found in this PDF. It may be a scanned (image-only) PDF."
        )
    for d in docs:
        d.metadata["source"] = display_name

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    db = None
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)
        if progress:
            progress(min(i + BATCH_SIZE, total) / total)

    path = VECTOR_DIR / slugify(display_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(path))
    return path.name, len(docs), total


def load_books(names, embeddings):
    """Load one book, or merge several books into one searchable index."""
    db = FAISS.load_local(
        str(VECTOR_DIR / names[0]), embeddings, allow_dangerous_deserialization=True
    )
    for name in names[1:]:
        other = FAISS.load_local(
            str(VECTOR_DIR / name), embeddings, allow_dangerous_deserialization=True
        )
        db.merge_from(other)
    return db


# ---------- question answering ----------
def to_english(question, llm):
    prompt = (
        "Translate the following medical question into English. "
        "Output only the translation, nothing else.\n\n" + question
    )
    return llm.invoke(prompt).content.strip()


def answer_question(question, db, llm, language="English", k=6):
    # The books are in English, so search with an English version of the question.
    search_query = question if language == "English" else to_english(question, llm)
    docs = db.max_marginal_relevance_search(search_query, k=k, fetch_k=20)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = PROMPT.format(context=context, question=question, language=language)
    answer = llm.invoke(prompt).content.strip()

    sources = [
        {
            "book": d.metadata.get("source", "book"),
            "page": int(d.metadata.get("page", 0)) + 1,   # PDF page number
            "text": d.page_content,
        }
        for d in docs
    ]
    return answer, sources, search_query


# ---------- voice ----------
def transcribe(audio_bytes, language_code="en"):
    """Speech -> text using Groq Whisper."""
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    result = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model=WHISPER_MODEL,
        language=language_code,
        temperature=0,
    )
    return result.text.strip()


def text_to_speech(text, language_code="en"):
    """Text -> mp3 bytes using gTTS (needs internet)."""
    from gtts import gTTS

    clean = re.sub(r"[*_#`>|]+", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()[:1500]
    buffer = io.BytesIO()
    gTTS(text=clean, lang=language_code).write_to_fp(buffer)
    return buffer.getvalue()