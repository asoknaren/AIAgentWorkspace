# LangGraph Topics: State, Supersteps, Memory, and Replay

This guide explains the concepts demonstrated in `007_langgraph_topics1.py` and `008_langgraph_topics2.py`. The examples use deterministic Python functions, so they illustrate graph behavior without needing an LLM or API key.

## What is a graph?

A LangGraph application is a workflow described as a directed graph. **Nodes** perform work, and **edges** determine which node or nodes run next. The graph carries shared **state** between nodes.

The standard construction pattern is:

1. Define the state schema.
2. Create a `StateGraph` builder for that schema.
3. Add nodes, which are Python functions or other runnables.
4. Connect nodes with edges.
5. Compile the builder into an executable graph.
6. Call `invoke` with an initial state and optional configuration.

`START` marks where execution begins and `END` marks a completed run. They are graph control markers, not application nodes.


## How `create_agent` relates to `StateGraph`

`create_agent` and `StateGraph` are not competing frameworks. `create_agent` returns a compiled LangGraph graph with a prebuilt model-and-tools loop. In effect, it connects a model node, a tools node, and routing that sends tool results back to the model. You can inspect that graph with `agent.get_graph()`.

### Use `create_agent` for a standard agent loop

Choose `create_agent` when the main workflow is:

1. Receive a user request.
2. Let the model decide whether to call one or more tools.
3. Run the requested tools.
4. Give tool results back to the model for its final response.

You provide the model, tools, and system prompt; the agent manages the model/tool-call loop. This is a good fit for assistants, support agents, and research helpers where the model can decide which tools to use.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_population],
    system_prompt="You are a concise travel assistant. Use tools when useful.",
)

result = agent.invoke({"messages": [("user", "Compare Rome and London.")]})
print(result["messages"][-1].content)
```

### Use `StateGraph` for custom orchestration

Choose `StateGraph` when your application, rather than the model's usual tool loop, must explicitly control what happens next. Examples include:

- Route requests to different specialist nodes.
- Run independent nodes in parallel and join their results.
- Add custom conditions, loops, or stop rules.
- Insert human review or approval steps.
- Coordinate multiple model calls or non-LLM services in a fixed sequence.
- Specify the state fields and exactly how each node updates them.

With `StateGraph`, you define the state, nodes, and edges yourself, then compile the graph. It takes more code but provides direct control over workflow behavior.

```python
from langgraph.graph import END, START, StateGraph

builder = StateGraph(MyState)
builder.add_node("classify", classify_request)
builder.add_node("billing", handle_billing)
builder.add_node("support", handle_support)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", choose_route)
builder.add_edge("billing", END)
builder.add_edge("support", END)

graph = builder.compile()
result = graph.invoke({"request": "I was charged twice"})
```

### Quick decision rule

- **The model chooses tools in a conventional assistant loop:** start with `create_agent`.
- **Your application chooses the next step through custom logic:** use `StateGraph`.
- **Not sure yet:** start with `create_agent`, then switch to a custom graph if you need workflow control the standard agent loop does not provide. A custom graph can also include an agent as one of its nodes.


## State and node updates

State is the shared data flowing through the graph. In the examples it is described with a `TypedDict`:

```python
class LessonState(TypedDict):
    request: str
    research_a: str
    research_b: str
    summary: str
    events: Annotated[list[str], operator.add]
```

A node receives the current state and returns a dictionary containing only the fields it wants to update. For example, a research node can return `{"research_a": "..."}` without rebuilding the entire state. LangGraph merges that update into the shared state according to each field's reducer.

## Reducers: replace or combine

A **reducer** defines how an update is combined with the existing value for one state field. If a field has no reducer, its value is normally replaced by the update. This is useful for fields such as `summary` or `reply`, where the newest result should win.

For accumulating values, annotate the field with a reducer. The topics examples use `operator.add` for lists:

```python
events: Annotated[list[str], operator.add]
```

If a node returns `{"events": ["Research complete"]}`, that list is appended to the existing `events` list rather than replacing it. Reducers matter especially when multiple nodes update the same field in one superstep: LangGraph combines those updates using that field's reducer.

## Nodes, edges, and supersteps

A **superstep** is a graph execution round. All nodes scheduled for the same round run before the graph advances to the next round. Nodes connected from the same predecessor can run in parallel; a downstream join waits for all of its prerequisite nodes to finish.

The first topics example builds this sequence:

```text
START -> prepare -> (research_a and research_b) -> summarize -> END
```

This produces these execution rounds:

1. `prepare` runs.
2. `research_a` and `research_b` run in the same superstep.
3. `summarize` runs after both research branches finish.

An edge from a single node to another is a direct transition. An edge from one node to several nodes schedules branches. A join edge such as `builder.add_edge(["research_a", "research_b"], "summarize")` waits for both branches before scheduling the summary node.

## Compiling and invoking a graph

`builder.compile()` validates and turns the builder into an executable graph. When a checkpointer is passed to `compile`, the graph also saves state snapshots as it runs:

```python
graph = builder.compile(checkpointer=MemorySaver())
result = graph.invoke(initial_state, config=config)
```

`invoke` runs synchronously from the supplied state through scheduled nodes and edges until the graph reaches `END`, then returns the resulting state. The optional `config` is where checkpoint settings such as `thread_id` are provided.

## Memory and thread IDs

There are two related meanings of “memory” in these examples:

- **State within a run:** values are passed from node to node. Reducers can accumulate updates during that run.
- **Remembering across invocations:** a checkpointer saves graph state, and a later call using the same `thread_id` can continue from that thread's saved state.

A `thread_id` identifies one conversation or workflow instance:

```python
config = {"configurable": {"thread_id": "conversation-1"}}
graph.invoke(first_input, config=config)
result = graph.invoke(next_input, config=config)
```

Use a different thread ID for an independent conversation. Without a checkpointer, graph state does not persist automatically between separate invocations.

## Checkpointers and `MemorySaver`

A **checkpointer** stores snapshots of graph state, typically after supersteps. A snapshot includes the state values and execution information such as which nodes are next. Checkpoints support resuming, inspecting history, and replaying from an earlier point.

`MemorySaver` stores checkpoints in process memory:

```python
graph = builder.compile(checkpointer=MemorySaver())
```

It is convenient for demos and short-lived sessions. Its data is not durable: it is lost when the process exits, and it is not shared with another process.

## SQLite persistence

`SqliteSaver` stores checkpoints in a SQLite database file. It supports the same checkpointer behavior as `MemorySaver`, but its saved data remains available after the Python process closes, as long as the database file is kept. The project needs the `langgraph-checkpoint-sqlite` package for this saver.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("memory.db") as saver:
    graph = builder.compile(checkpointer=saver)
    result = graph.invoke(input_state, config=config)
```

Opening the same database later and invoking with the same `thread_id` restores that thread's saved state. Keep the saver open while using the graph, and close it when finished. SQLite is a local persistence option; a production deployment with multiple workers may need a shared database/checkpointer suited to that deployment.

## Inspecting checkpoint history

`graph.get_state(config)` returns the latest saved snapshot for a thread. `graph.get_state_history(config)` returns the saved snapshots, newest first. Each snapshot exposes values, metadata, the next scheduled nodes, and a configuration containing its checkpoint ID.

The topics examples reverse the history when printing it so that the output reads chronologically. The exact number and metadata of snapshots depend on the graph run; the important information is the saved state and where execution can continue.

## Time travel and replay

**Time travel** means selecting an earlier checkpoint rather than using the latest state. To replay from it, pass its `checkpoint_id` along with the same `thread_id`. Calling `invoke(None, config=that_checkpoint_config)` resumes from the selected snapshot's saved execution point. In the topics example, the selected snapshot has `reply` as the next node, so replay continues by running `reply`.

Replay does not erase the original later checkpoint history. The replay produces a continuation from the selected point, which can be inspected separately. This is useful for understanding what happened, recovering from an intermediate state, or exploring an alternate continuation. Be thoughtful when replaying workflows with external side effects, such as sending a payment or notification: those effects may happen again if the replay executes the corresponding node.

