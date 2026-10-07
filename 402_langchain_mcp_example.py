import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
    raise RuntimeError("Set a valid OPENAI_API_KEY in .env or your environment.")


async def main() -> None:
    server_path = Path(__file__).with_name("401_local_mcp_server.py")
    client = MultiServerMCPClient({
        "orders": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_path)],
        }
    })

    tools = await client.get_tools()
    print("MCP tools available to the agent:")
    for tool in tools:
        print(f" - {tool.name}")

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o", temperature=0),
        tools=tools,
        system_prompt="Answer order and product questions using the available MCP tools.",
    )
    user_requests = [
        "Can you check order A100?",
        "What items are in order A100 and what was the total?",
        "Which orders does alice have?",
        "Is the Wireless Mouse (MS-02) in stock, and how much does it cost?",
        "Please cancel order A200.",
        "Please cancel order A100.",
        "What's the status of order Z999?",
    ]

    for request in user_requests:
        print(f"\nUser: {request}")
        result = await agent.ainvoke({"messages": [("user", request)]})
        print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
