from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def create_chunks(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(extracted_data)


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    print("Step 1: loading PDFs...")
    documents = load_pdf_files(DATA_PATH)
    print("Length of PDF pages:", len(documents))

    print("Step 2: creating chunks...")
    text_chunks = create_chunks(documents)
    print("Length of text chunks:", len(text_chunks))

    print("Step 3: loading embedding model...")
    embedding_model = get_embedding_model()

    print("Step 4: building FAISS index (this can take several minutes)...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("Vector store saved at:", DB_FAISS_PATH)