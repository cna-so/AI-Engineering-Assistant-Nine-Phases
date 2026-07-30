# chat_rag.py
# Run after ingest: uv run python phases/02_rag/ingest_docs.py
# Then: uv run python phases/02_rag/rag_chat.py

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_PATH = Path(__file__).resolve().parent / ".qdrant"
COLLECTION_NAME = "phase2_docs"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

llm = ChatOpenRouter(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    """You are a documentation assistant.
Answer only from the retrieved context.
If the answer is not in the context, say you don't know.

Question:
{question}

Context:
{context}
"""
)


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def main():
    if "OPENROUTER_API_KEY" not in os.environ:
        raise ValueError("Set OPENROUTER_API_KEY first (e.g. in .env).")

    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"No Qdrant store at {QDRANT_PATH}. Run: uv run python phases/02_rag/ingest_docs.py"
        )
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    client = QdrantClient(path=str(QDRANT_PATH))
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    chain = (
        {
            "question": lambda x: x,
            "context": lambda x: format_docs(retriever.invoke(x)),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Phase 2 RAG Chat")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
