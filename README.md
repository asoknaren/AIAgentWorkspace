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