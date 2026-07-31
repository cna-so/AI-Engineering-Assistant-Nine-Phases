# Ingest local markdown docs into a local Qdrant store.
# Run: uv run python phases/02_rag/ingest_docs.py
# Needs: OPENROUTER_API_KEY

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

DOCS_DIR = Path(__file__).resolve().parent / "docs"
QDRANT_PATH = Path(__file__).resolve().parent / ".qdrant"
COLLECTION = "phase6_docs"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
VECTOR_SIZE = 1536  # text-embedding-3-small


def load_markdown_docs(directory: Path) -> list[Document]:
    docs = []
    for path in sorted(directory.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "type": "markdown",
                    "filename": path.name,
                },
            )
        )
    return docs


def main():
    if "OPENROUTER_API_KEY" not in os.environ:
        raise ValueError("Set OPENROUTER_API_KEY first.")

    # Must be an Embeddings object — a model name string is not enough.
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    raw_docs = load_markdown_docs(DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    # Local on-disk Qdrant (no Docker / remote server needed)
    client = QdrantClient(path=str(QDRANT_PATH))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(raw_docs)} docs into '{COLLECTION}'")
    print(f"Store path: {QDRANT_PATH}")


if __name__ == "__main__":
    main()
