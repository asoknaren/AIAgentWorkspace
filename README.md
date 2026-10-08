# LangChain and MCP example

This example connects a LangChain agent to a local MCP server that provides an
order-status lookup tool.

1. Put a valid `OPENAI_API_KEY` in the project-root `.env` file (copy
   `.env.example` if needed).
2. Run the example from the project directory:

   ```powershell
   uv run python .\402_langchain_mcp_example.py
   ```

The MCP server in `401_local_mcp_server.py` exposes `get_order_status` using
the MCP Python SDK. The client configures that server with the `stdio`
transport; it starts the server as a subprocess using the current Python
interpreter. `MultiServerMCPClient.get_tools()` discovers the server's tools
and adapts them to LangChain tools. Those tools are passed to `create_agent`,
so the model can choose to call the MCP tool while answering the user's
request.

The order records are sample data. For a real integration, replace them with
your data source and secure any required credentials outside source control.

## LangGraph Studio for development and testing

LangGraph Studio is a visual UI that connects to a local development server
(`langgraph dev`). Use it to run a graph, watch each node execute, inspect
state, and replay from earlier steps without writing test scripts.

### Setup

1. Dependencies: `langgraph-cli[inmem]` and `colorama` (Windows console
   colors) are installed with `uv sync`. Python 3.11 or later is required.
2. Put `LANGSMITH_API_KEY` (needed to connect Studio) and, for the `intro`
   graph, `OPENAI_API_KEY` in the project-root `.env` file. Set
   `LANGSMITH_TRACING=false` if you do not want runs traced to LangSmith.
3. [langgraph.json](./langgraph.json) lists the graphs Studio can load:

   | Graph ID | File | Input fields |
   | --- | --- | --- |
   | `intro` | `006_langgraph_intro.py` | `messages`, `node_history` |
   | `researcher` | `007_langgraph_topics1.py` | `request` (plus `research_a`, `research_b`, `summary`, `events`) |
   | `time_travel` | `008_langgraph_topics2.py` | `message` |

### Run

```powershell
uv run langgraph dev
```

Open the Studio URL printed in the terminal
(`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`). The API
is at `http://127.0.0.1:2024` (docs at `/docs`). The server needs no Docker,
keeps state in memory, and hot-reloads when you edit the graph files. In
Safari, use `uv run langgraph dev --tunnel` because it blocks `localhost`
connections to Studio.

### Using Studio to test

- Pick a graph, enter the input fields from the table above, and submit.
- Watch nodes execute. Parallel nodes (`research_a` and `research_b`) run in
  one step, so they appear together before `summarize`.
- Open a step to see its state. For finished runs, check the thread's output
  for the final values, such as `summary` and `events`.
- Re-run a thread from any earlier step to test a change without starting over.
- Use separate threads for independent test cases.

### Adding a graph to Studio

Export a compiled graph from a module-level variable named `graph`, built
**without** a custom checkpointer (the server provides persistence), then add
it to `langgraph.json`:

```python
graph = build_graph()
```

```json
{ "graphs": { "my_graph": "./my_file.py:graph" } }
```

Keep a checkpointer argument on `build_graph()` so scripts that run the graph
directly can still pass `MemorySaver()` or `SqliteSaver`.

### Troubleshooting

- `Could not find graph 'graph'`: the file does not export a top-level
  `graph`, or the name in `langgraph.json` does not match.
- `ConsoleRenderer ... requires the colorama package`: run `uv add colorama`.
- Runs seem stuck: only one background worker runs at a time, so a stuck run
  blocks later ones. Cancel it or restart `langgraph dev`. Also check that no
  second server is already using port 2024.