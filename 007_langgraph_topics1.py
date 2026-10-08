"""Demonstrate LangGraph supersteps and in-memory checkpoints.

This example uses ordinary Python nodes rather than an LLM, so the graph
execution mechanics are easy to see and require no API key.
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class LessonState(TypedDict):
    """Values passed between nodes and stored in each checkpoint."""

    request: str
    research_a: str
    research_b: str
    summary: str
    events: Annotated[list[str], operator.add]


def prepare_node(state: LessonState) -> dict:
    """Start the run and record the request being processed."""
    print("  node: prepare")
    return {"events": [f"Prepared: {state['request']}"]}


def research_a_node(state: LessonState) -> dict:
    """One branch of work in the parallel research superstep."""
    print("  node: research_a")
    return {
        "research_a": f"First perspective on {state['request']}",
        "events": ["Research branch A completed"],
    }


def research_b_node(state: LessonState) -> dict:
    """Another branch that can run in the same superstep as research_a."""
    print("  node: research_b")
    return {
        "research_b": f"Second perspective on {state['request']}",
        "events": ["Research branch B completed"],
    }


def summarize_node(state: LessonState) -> dict:
    """Join the branch results after both parallel nodes finish."""
    print("  node: summarize")
    summary = f"{state['research_a']}; {state['research_b']}"
    return {"summary": summary, "events": ["Summary created"]}


def build_graph():
    """Build a graph whose two research nodes execute in parallel."""
    builder = StateGraph(LessonState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("research_a", research_a_node)
    builder.add_node("research_b", research_b_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "prepare")
    # Both nodes are scheduled after prepare and form one parallel superstep.
    builder.add_edge("prepare", "research_a")
    builder.add_edge("prepare", "research_b")
    # summarize runs only after both parallel branches complete.
    builder.add_edge(["research_a", "research_b"], "summarize")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=MemorySaver())


def print_checkpoint_history(graph, config: dict) -> None:
    """Print saved state snapshots from newest to oldest."""
    history = list(graph.get_state_history(config))
    print(f"Saved checkpoints: {len(history)}")

    for snapshot in reversed(history):
        step = snapshot.metadata.get("step", "?")
        source = snapshot.metadata.get("source", "?")
        next_nodes = ", ".join(snapshot.next) or "(complete)"
        print(
            f"  step {step}: source={source}, next={next_nodes}, "
            f"events={len(snapshot.values.get('events', []))}"
        )


def main() -> None:
    """Run twice on one thread to show both supersteps and memory."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "supersteps-demo"}}

    print("=== First invocation ===")
    first_result = graph.invoke(
        {
            "request": "how LangGraph coordinates work",
            "research_a": "",
            "research_b": "",
            "summary": "",
            "events": [],
        },
        config=config,
    )
    print("Summary:", first_result["summary"])
    print("Accumulated events:", first_result["events"])

    print("\n=== Checkpoints from the first invocation ===")
    print_checkpoint_history(graph, config)

    print("\n=== Second invocation on the same thread ===")
    second_result = graph.invoke(
        {"request": "how checkpointing supports recovery"},
        config=config,
    )
    print("Summary:", second_result["summary"])
    print("Events retained across calls:", second_result["events"])

    print("\n=== Checkpoints after the second invocation ===")
    print_checkpoint_history(graph, config)


if __name__ == "__main__":
    main()
