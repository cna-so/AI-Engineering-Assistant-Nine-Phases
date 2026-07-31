# Anthropic API (overview for RAG projects)

Anthropic provides Claude models used for chat, tool use, and long-context
reasoning.

## Strengths in agent workflows

- Strong instruction following
- Reliable tool-use patterns
- Large context windows useful for document-heavy answers

## Typical use in this project

Claude (via OpenRouter) is the chat model for the assistant.
Embeddings may come from a separate OpenAI-compatible embedding model.
