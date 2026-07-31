# OpenAI API (overview for RAG projects)

OpenAI provides chat completions, embeddings, and tool-calling APIs used by
many RAG and agent systems.

## Embeddings

Embeddings convert text into vectors for similarity search.
Common pattern: embed documents once, embed queries at request time,
then retrieve nearest neighbors from a vector store.

## Tool calling

Models can return structured tool calls. The application executes the tool
and feeds results back into the conversation.

## Notes for OpenRouter users

OpenRouter exposes many models (including OpenAI-compatible ones) behind a
single API key and base URL.
