import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1) Define tools
# -----------------------------
@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"[docs] Results for: {query}"


@tool
def github_search(query: str) -> str:
    """Search GitHub repositories or code."""
    return f"[github] Results for: {query}"


@tool
def web_search(query: str) -> str:
    """Search the public web."""
    return f"[web] Results for: {query}"


tools = [search_docs, github_search, web_search]


# -----------------------------
# 2) Define graph state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# -----------------------------
# 3) Create model
# -----------------------------
llm = ChatOpenRouter(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
    api_key=os.environ["OPENROUTER_API_KEY"],
)

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful AI assistant.

You have access to these tools:
- search_docs(query): use for internal/project docs
- github_search(query): use for repo/code questions
- web_search(query): use for internet/fresh info

Decide whether you need a tool before answering.
If a tool is needed, call it.
If not, answer directly.
"""


# -----------------------------
# 4) Planner node
# -----------------------------
def planner(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# -----------------------------
# 5) Build graph
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
# 6) Run chat loop
# -----------------------------
def main():
    print("Phase 4 - LangGraph Tool Calling Agent")
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