"""Explain LangGraph memory, checkpointers, SQLite, and time travel.

The MemorySaver example runs using the standard LangGraph dependency. The
SQLite example also requires the optional langgraph-checkpoint-sqlite package.
"""

from pathlib import Path
from typing import Annotated
import operator

from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class ConversationState(TypedDict):
    """Conversation data saved and restored by a checkpointer."""

    message: str
    memories: Annotated[list[str], operator.add]
    reply: str


def remember_node(state: ConversationState) -> dict:
    """Add the new user message to the accumulating memory field."""
    print("  node: remember")
    return {"memories": [state["message"]]}


def reply_node(state: ConversationState) -> dict:
    """Create a deterministic reply using all information in state."""
    print("  node: reply")
    remembered = "; ".join(state["memories"])
    return {
        "reply": f"I remember {len(state['memories'])} message(s): {remembered}"
    }


def build_graph(checkpointer):
    """Compile a small two-node graph with the supplied saver."""
    builder = StateGraph(ConversationState)
    builder.add_node("remember", remember_node)
    builder.add_node("reply", reply_node)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=checkpointer)


def show_history(graph, config: dict) -> list:
    """Display saved snapshots in execution order and return newest first."""
    history = list(graph.get_state_history(config))
    print(f"Saved snapshots: {len(history)}")
    for snapshot in reversed(history):
        step = snapshot.metadata.get("step", "?")
        next_nodes = ", ".join(snapshot.next) or "(complete)"
        memories = snapshot.values.get("memories", [])
        print(f"  step {step}: next={next_nodes}; memories={memories}")
    return history


def demo_memory_and_replay() -> None:
    """Show thread memory, checkpoint history, and replay from a checkpoint."""
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "memory-demo"}}

    print("=== MemorySaver: same thread, multiple invocations ===")
    first = graph.invoke(
        {"message": "My favorite color is blue.", "memories": [], "reply": ""},
        config=config,
    )
    print("Reply:", first["reply"])

    second = graph.invoke({"message": "I also like hiking."}, config=config)
    print("Reply:", second["reply"])
    print("\nA checkpointer saves state snapshots at graph steps:")
    history = show_history(graph, config)

    # Find the snapshot after remember ran but before reply was scheduled to finish.
    replay_point = next(
        snapshot
        for snapshot in history
        if snapshot.next == ("reply",)
        and snapshot.values.get("message") == "I also like hiking."
    )
    replay_config = {
        "configurable": {
            "thread_id": "memory-demo",
            "checkpoint_id": replay_point.config["configurable"]["checkpoint_id"],
        }
    }
    print("\n=== Time travel / replay ===")
    print("Selected an earlier snapshot where the next node was:", replay_point.next)
    replayed = graph.invoke(None, config=replay_config)
    print("Replay resumed from that saved state:", replayed["reply"])
    print("The original latest conversation state remains inspectable:")
    current = graph.get_state(config)
    print(" ", current.values["reply"])


def demo_sqlite_persistence(db_path: Path) -> None:
    """Persist a thread in SQLite and reopen it using a new saver instance."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as error:
        raise RuntimeError(
            "SQLite checkpointing needs the optional dependency. Install it with "
            "`uv add langgraph-checkpoint-sqlite`."
        ) from error

    config = {"configurable": {"thread_id": "sqlite-demo"}}
    print("\n=== SqliteSaver: checkpoint data stored in a database file ===")

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_graph(saver)
        result = graph.invoke(
            {"message": "Remember that I prefer window seats.", "memories": [], "reply": ""},
            config=config,
        )
        print("First run:", result["reply"])

    # Reopening the database creates a new saver but restores the same thread.
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_graph(saver)
        result = graph.invoke({"message": "What seat do I prefer?"}, config=config)
        print("After reopening SQLite:", result["reply"])

    print("SQLite database:", db_path)


def main() -> None:
    """Run the in-memory and disk-backed checkpoint demonstrations."""
    demo_memory_and_replay()
    demo_sqlite_persistence(Path("langgraph_topics2.sqlite"))


if __name__ == "__main__":
    main()
