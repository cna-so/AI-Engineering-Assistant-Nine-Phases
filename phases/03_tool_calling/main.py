import os
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

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
tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOpenRouter(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
    api_key=os.environ["OPENROUTER_API_KEY"],
)

llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are a helpful AI assistant.

You have access to these tools:
- search_docs(query): use for internal/project docs
- github_search(query): use for repo/code questions
- web_search(query): use for internet/fresh info

Decide whether you need a tool before answering.
"""

def run_agent(user_input: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return ai_msg.content

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            result = tools_by_name[tool_name].invoke(tool_args)

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_id,
                )
            )

if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        answer = run_agent(user_input)
        print(f"\nAssistant: {answer}\n")