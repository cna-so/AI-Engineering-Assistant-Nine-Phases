# LangChain

LangChain is a framework for developing applications powered by language models.
It provides abstractions for prompts, models, tools, retrievers, and chains.

## Building blocks

- **Chat models**: wrappers around LLM providers
- **Prompt templates**: reusable prompt structures
- **Tools**: functions the model can call
- **Retrievers**: interfaces that fetch relevant documents for a query
- **Document loaders & splitters**: ingest and chunk source material

## Relationship to LangGraph

LangChain focuses on components and composition.
LangGraph focuses on durable, stateful agent workflows built on those components.
