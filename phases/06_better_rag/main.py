import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Config
# -----------------------------
DOCS_DIR = Path(__file__).resolve().parent / "docs"
QDRANT_PATH = Path(__file__).resolve().parent / ".qdrant"
COLLECTION = "phase6_docs"

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "google/gemini-2.5-flash-lite"


# -----------------------------
# Models
# -----------------------------
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

llm = ChatOpenRouter(
    model=CHAT_MODEL,
    temperature=0,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Local Qdrant allows only one client per path — reuse a single instance.
_qdrant_client: QdrantClient | None = None
_vectorstore: QdrantVectorStore | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_PATH.exists():
            raise FileNotFoundError(
                f"No Qdrant store at {QDRANT_PATH}. "
                "Run: uv run python phases/06_better_rag/ingest_docs.py"
            )
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
    return _qdrant_client


def get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=COLLECTION,
            embedding=embeddings,
        )
    return _vectorstore


def close_qdrant_client() -> None:
    """Release the local Qdrant lock before interpreter shutdown."""
    global _qdrant_client, _vectorstore
    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None
    _vectorstore = None


# -----------------------------
# Load docs
# -----------------------------
def load_markdown_docs(directory: Path) -> List[Document]:
    docs = []
    for path in directory.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        source_parts = str(path).replace("\\", "/").split("/")

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "category": source_parts[1] if len(source_parts) > 1 else "general",
                    "extension": "md",
                },
            )
        )
    return docs


# -----------------------------
# Indexing with parent/child docs
# -----------------------------
def build_parent_retriever() -> ParentDocumentRetriever:
    raw_docs = load_markdown_docs(DOCS_DIR)
    vectorstore = get_vectorstore()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(raw_docs)
    return retriever


# -----------------------------
# Base retriever with metadata filtering idea
# -----------------------------
def build_filtered_vector_retriever(category: str | None = None):
    vectorstore = get_vectorstore()

    search_kwargs = {"k": 8}

    if category:
        search_kwargs["filter"] = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="category",
                    match=qdrant_models.MatchValue(value=category),
                )
            ]
        )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


# -----------------------------
# Multi-query retriever
# -----------------------------
def build_multi_query_retriever(base_retriever):
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )


# -----------------------------
# Contextual compression
# -----------------------------
def build_compression_retriever(base_retriever):
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )


# -----------------------------
# Reranker
# -----------------------------
def build_rerank_retriever(base_retriever):
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base"
    )
    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=4,
    )
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker,
    )


# -----------------------------
# Retrieval pipeline
# -----------------------------
def build_better_rag_retriever(category: str | None = None):
    parent_retriever = build_parent_retriever()

    filtered_retriever = build_filtered_vector_retriever(category=category)

    multi_query_retriever = build_multi_query_retriever(filtered_retriever)

    compressed_retriever = build_compression_retriever(multi_query_retriever)

    reranked_retriever = build_rerank_retriever(compressed_retriever)

    return {
        "parent_retriever": parent_retriever,
        "filtered_retriever": filtered_retriever,
        "multi_query_retriever": multi_query_retriever,
        "compressed_retriever": compressed_retriever,
        "reranked_retriever": reranked_retriever,
    }


# -----------------------------
# Format docs
# -----------------------------
def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(
            f"[{i}] source={doc.metadata.get('source')} category={doc.metadata.get('category')}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


# -----------------------------
# Ask question
# -----------------------------
def ask(question: str, category: str | None = None):
    retrievers = build_better_rag_retriever(category=category)

    docs = retrievers["reranked_retriever"].invoke(question)

    context = format_docs(docs)

    prompt = f"""You are a grounded documentation assistant.
Answer only using the retrieved context.
If the answer is not in the context, say you don't know.

Question:
{question}

Context:
{context}
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    try:
        answer = ask("What is LangGraph?", category=None)
        print(answer)
    finally:
        close_qdrant_client()
