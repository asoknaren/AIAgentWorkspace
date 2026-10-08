"""Introduction to LangChain create_agent, tools, and tool-call middleware.

Set OPENAI_API_KEY in the project-root .env file before running. LangSmith
tracing is enabled automatically when LANGSMITH_API_KEY is present.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.tools import tool


load_dotenv(override=True)

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "langchain-create-agent-intro")


@tool
def get_weather(city: str) -> str:
    """Return a sample weather report for a city."""
    reports = {
        "London": "Rainy, 14 degrees C",
        "Rome": "Sunny, 27 degrees C",
        "Tokyo": "Cloudy, 19 degrees C",
    }
    return reports.get(city.title(), "Clear, 20 degrees C")


@tool
def get_population(city: str) -> str:
    """Return the sample population of a city."""
    populations = {
        "London": "8.9 million",
        "Rome": "2.8 million",
        "Tokyo": "about 14 million",
    }
    return populations.get(city.title(), "unknown")


@wrap_tool_call
def log_tool_call(request, handler):
    """Observe a tool call, then let the normal handler execute it."""
    tool_call = request.tool_call
    print(f"[wrap_tool_call] {tool_call['name']}({tool_call['args']})")
    return handler(request)


def main() -> None:
    """Create an agent, inspect its graph, and run a tool-using request."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the project-root .env file.")

    tools = [get_weather, get_population]

    # create_agent supplies the model/tool orchestration loop. The model chooses
    # whether to answer directly or request one or more tools.
    agent = create_agent(
        model="openai:gpt-4o",
        tools=tools,
        system_prompt=(
            "You are a concise travel assistant. Use the available tools for "
            "weather and population questions, then summarize their results."
        ),
        middleware=[log_tool_call],
    )

    # create_agent returns a compiled LangGraph graph. Its nodes and edges can
    # be inspected just like a graph assembled directly with StateGraph.
    print("=== LangGraph underneath create_agent ===")
    print(agent.get_graph().draw_mermaid())

    print("\n=== Agent run ===")
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Compare the weather and population of London and Rome.",
                )
            ]
        }
    )
    print("\nFinal answer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
