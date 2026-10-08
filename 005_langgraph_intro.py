"""A small, runnable introduction to LangGraph state and orchestration.

Set OPENAI_API_KEY in the project .env file before running. To trace runs in
LangSmith, also set LANGSMITH_API_KEY and LANGSMITH_TRACING=true.
"""

import os
import operator
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv(override=True)

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "langgraph-intro")


class GraphState(TypedDict):
    """Shared data passed between nodes while the graph runs."""

    # add_messages merges new messages into the existing conversation.
    messages: Annotated[list[AnyMessage], add_messages]
    # operator.add accumulates each node's list of execution notes.
    node_history: Annotated[list[str], operator.add]


@tool
def lookup_course_fact(
    topic: Literal["state", "reducer", "node", "condition", "loop"],
) -> str:
    """Look up a short definition of a LangGraph concept from this lesson."""
    facts = {
        "state": "State is the shared data that flows between graph nodes.",
        "reducer": "A reducer combines a node's update with the current state value.",
        "node": "A node is a Python function or runnable that reads and updates state.",
        "condition": "A conditional edge chooses the next node based on current state.",
        "loop": "A loop is made by routing an edge back to an earlier node.",
    }
    return facts[topic]


def configure_langsmith() -> None:
    """Enable tracing when a LangSmith API key is configured."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", "langgraph-intro")
        print("LangSmith tracing enabled for project:", os.environ["LANGSMITH_PROJECT"])
    else:
        print("LangSmith tracing is off; set LANGSMITH_API_KEY to enable it.")


def build_graph():
    """Build a model and tool graph with a conditional tool-call loop."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [lookup_course_fact]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: GraphState) -> dict:
        """Ask the model for a response or a tool call."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "node_history": ["chatbot"]}

    tool_runner = ToolNode(tools)

    def tools_node(state: GraphState) -> dict:
        """Execute requested tools and record that this node ran."""
        updates = tool_runner.invoke(state)
        return {**updates, "node_history": ["tools"]}

    builder = StateGraph(GraphState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "chatbot")
    # This condition routes to tools when the model asks for them, otherwise END.
    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {"tools": "tools", END: END},
    )
    # Returning to chatbot after tools creates the tool-call loop.
    builder.add_edge("tools", "chatbot")

    # A checkpointer is required for state to persist across calls sharing a thread_id.
    return builder.compile(checkpointer=MemorySaver())


def main() -> None:
    """Run one request that requires the model to use the concept lookup tool."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the project-root .env file.")

    configure_langsmith()
    graph = build_graph()

    result = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    "Use the lesson lookup tool to explain state, reducers, nodes, "
                    "conditional routing, and loops in simple terms.",
                )
            ],
            "node_history": [],
        },
        config={"configurable": {"thread_id": "langgraph-intro-demo"}},
    )

    print("\nAssistant:")
    print(result["messages"][-1].content)
    print("\nNodes executed:", " -> ".join(result["node_history"]))


if __name__ == "__main__":
    main()