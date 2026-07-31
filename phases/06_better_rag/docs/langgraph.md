# LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs.
It extends LangChain with graph-based workflows.

## Core concepts

- **State**: shared data structure passed between nodes
- **Nodes**: functions that read state and return updates
- **Edges**: connections between nodes (including conditional routing)
- **START / END**: special markers for graph entry and exit
- **Checkpointers**: persist state across turns (e.g. MemorySaver, SqliteSaver)

## Typical flow

START → agent node → (optional tool node) → generate answer → END

## When to use LangGraph

Use LangGraph when you need explicit control over agent loops, branching,
human-in-the-loop interrupts, or multi-agent collaboration.
