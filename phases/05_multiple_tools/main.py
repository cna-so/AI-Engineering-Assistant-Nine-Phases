import os
import math
import json
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper

from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# 1) Custom tools
# -----------------------------
@tool
def search_docs(query: str) -> str:
    """Search internal documentation and indexed project docs."""
    return f"[docs] Relevant documentation for: {query}"


@tool
def github_search(query: str) -> str:
    """Search GitHub repositories, code, or issues."""
    return f"[github] Relevant GitHub results for: {query}"


@tool
def web_search(query: str) -> str:
    """Search the public web for recent or general information."""
    return f"[web] Public web results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic mathematical expression safely."""
    allowed = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


@tool
def python_exec(code: str) -> str:
    """Execute short sandboxed Python code for small computations."""
    local_vars = {}
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        return json.dumps(local_vars, default=str)
    except Exception as e:
        return f"Python execution error: {str(e)}"


@tool
def youtube_transcript(video_id: str) -> str:
    """Fetch YouTube transcript by video ID."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join(snippet.text for snippet in list(transcript)[:80])
        return text[:4000]
    except Exception as e:
        return f"YouTube transcript error: {str(e)}"


# -----------------------------
# 2) ArXiv tool
# -----------------------------
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=4000,
    )
)


# -----------------------------
# 3) All tools
# -----------------------------
tools = [
    search_docs,
    github_search,
    web_search,
    calculator,
    python_exec,
    youtube_transcript,
    arxiv_tool,
]


# -----------------------------
# 4) State
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# -----------------------------
# 5) Model
# -----------------------------
llm = ChatOpenRouter(
    model="anthropic/claude-sonnet-4.5",
    temperature=0,
    api_key=os.environ["OPENROUTER_API_KEY"],
)

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a capable AI research and engineering assistant.

Available tools:
- search_docs(query): internal or indexed documentation
- github_search(query): repositories, code, issues, implementation details
- web_search(query): general or fresh public web information
- calculator(expression): arithmetic and math expressions
- python_exec(code): short sandboxed Python snippets for computation
- arxiv: search academic papers on arXiv
- youtube_transcript(video_id): fetch transcript of a YouTube video

Tool selection guidance:
- Use search_docs for framework or internal documentation questions.
- Use github_search for repo/code structure or implementation questions.
- Use web_search for public/fresh information.
- Use calculator for direct numeric calculations.
- Use python_exec for slightly more complex computations or transformations.
- Use arxiv for research paper discovery.
- Use youtube_transcript when the question is about a specific YouTube video.
- If no tool is needed, answer directly.
- Prefer the smallest sufficient tool.
"""


# -----------------------------
# 6) Planner node
# -----------------------------
def planner(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# -----------------------------
# 7) Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("planner", planner)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)

builder.add_edge("tools", "planner")

graph = builder.compile()


# -----------------------------
# 8) Chat loop
# -----------------------------
def main():
    print("Phase 5 - Multi-Tool LangGraph Agent")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = graph.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })

        print("\nAssistant:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()