import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
    raise RuntimeError("Set a valid OPENAI_API_KEY in .env or your environment.")


class PrintLLMContext(BaseCallbackHandler):
    def __init__(self) -> None:
        self.call_number = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **_kwargs: Any,
    ) -> None:
        del _kwargs
        self.call_number += 1
        model_name = serialized.get("kwargs", {}).get("model_name", "chat model")
        print(f"\nLLM call #{self.call_number} ({model_name})")
        for prompt_number, prompt_messages in enumerate(messages, start=1):
            print(f"Prompt {prompt_number}:")
            for message in prompt_messages:
                print(message.pretty_repr())
        print("-" * 100)


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
        model=ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            callbacks=[PrintLLMContext()],
        ),
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
        print("-" * 100) 
        input("Press Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
