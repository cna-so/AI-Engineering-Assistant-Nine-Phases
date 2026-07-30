# === Phase 1: Basic Chat ===
# Learn: chat models, prompt templates, streaming, message history
#
# User → LLM → Answer (no tools)
#
# Run: uv run python phases/01_basic_chat/main.py

import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and briefly."

llm = ChatOpenRouter(
    model="anthropic/claude-sonnet-4-5",
    temperature=0,
)


def chatbot(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=MemorySaver())


def chat():
    config = {"configurable": {"thread_id": str(uuid4())}}
    print("Phase 1 — Basic Chat (LangGraph + OpenRouter)")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        print("Assistant: ", end="", flush=True)

        for msg_chunk, _metadata in graph.stream(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
            stream_mode="messages",
        ):
            if getattr(msg_chunk, "content", None):
                print(msg_chunk.content, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    if "OPENROUTER_API_KEY" not in os.environ:
        raise ValueError("Set OPENROUTER_API_KEY first.")
    chat()
