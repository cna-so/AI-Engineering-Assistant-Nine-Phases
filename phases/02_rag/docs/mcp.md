# Model Context Protocol (MCP)

MCP is an open protocol that standardizes how applications provide context
and tools to large language models.

## Goals

- Connect LLMs to data sources and tools through a common interface
- Allow hosts (clients) to discover and call tools exposed by MCP servers
- Separate tool implementation from the model provider

## Components

- **Host / Client**: the AI application that consumes MCP servers
- **Server**: exposes resources, prompts, and tools
- **Transport**: how client and server communicate (stdio, HTTP, etc.)

## Why it matters for agents

Instead of hard-coding every integration, an agent can discover MCP tools at
runtime and call them uniformly.
