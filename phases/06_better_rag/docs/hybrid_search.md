# Hybrid Search Notes

Hybrid search combines keyword (BM25-style) matching with vector similarity.
Metadata filtering restricts retrieval by fields such as source, language, or doc type.
Reranking re-scores an initial candidate set so the LLM sees the best context first.
