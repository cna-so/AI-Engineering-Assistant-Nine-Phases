# AI Engineering Assistant — Nine Phases

Progressive LangGraph + OpenRouter project. Each phase is a **full runnable script**.
To advance: copy the previous `main.py` into the next folder and grow it, or run that phase directly.

## Setup

```bash
uv sync
# put keys in .env
# OPENROUTER_API_KEY=...
# GITHUB_TOKEN=...          # optional
# TAVILY_API_KEY=...        # optional (web search)
# ENABLE_REPO_CLONE=1       # optional (phase 8 local clone)
```

Run a phase:

```bash
uv run python phases/01_basic_chat/main.py
# or root (phase 1):
uv run python main.py
```

## Phases

| Phase | Path | Goal | Env |
|-------|------|------|-----|
| 1 Basic Chat | `phases/01_basic_chat/` | Chat models, streaming, message history | `OPENROUTER_API_KEY` |
| 2 RAG | `phases/02_rag/` | Load → chunk → embed → Qdrant → retrieve → answer | `OPENROUTER_API_KEY` |
| 3 Tool Calling | `phases/03_tool_calling/` | `search_docs`, `github_search`, `web_search` | + optional `GITHUB_TOKEN`, `TAVILY_API_KEY` |
| 4 LangGraph | `phases/04_langgraph/` | Explicit planner → tool? → answer graph | same as 3 |
| 5 Multiple Tools | `phases/05_multiple_tools/` | Calculator, arXiv, optional python/youtube | same as 3 |
| 6 Better RAG | `phases/06_better_rag/` | Multi-query, hybrid, parent docs, compression, rerank | `OPENROUTER_API_KEY` |
| 7 Memory | `phases/07_memory/` | Session state + summarization | `OPENROUTER_API_KEY` |
| 8 GitHub Tool | `phases/08_github_tool/` | Repo Q&A via API (+ optional clone) | + optional `GITHUB_TOKEN`, `ENABLE_REPO_CLONE` |
| 9 Evaluation | `phases/09_evaluation/` | Relevance, faithfulness, latency, retrieval quality | `OPENROUTER_API_KEY` |

## Copy workflow

1. Finish / understand phase N
2. `cp phases/0N_*/main.py phases/0N+1_*/main.py` (or start from the already-written next file)
3. Run the next phase and extend

Sample docs for RAG live in `phases/02_rag/docs/` (copied into phase 6).
Eval gold set: `phases/09_evaluation/datasets/gold_questions.json`.
