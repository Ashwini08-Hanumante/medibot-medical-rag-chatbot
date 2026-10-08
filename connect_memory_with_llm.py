import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

PROMPT_TEMPLATE = """You are a helpful medical information assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )


def answer_question(question, db, llm, k=3):
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    ).format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content, docs


if __name__ == "__main__":
    db = load_vectorstore()
    llm = load_llm()

    while True:
        q = input("\nAsk a medical question (or 'exit'): ")
        if q.lower() == "exit":
            break
        answer, sources = answer_question(q, db, llm)
        print("\nANSWER:\n", answer)
        print("\nSOURCES:")
        for s in sources:
            print(f"- {s.metadata.get('source')} (page {s.metadata.get('page')})")