import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
    raise RuntimeError("Set a valid OPENAI_API_KEY in .env or your environment.")


async def main() -> None:
    client = MultiServerMCPClient({
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest", "--isolated"],
        }
    })

    browser_tools = await client.get_tools()
    print(f"Loaded {len(browser_tools)} browser tools:")
    for tool in browser_tools:
        print(" -", tool.name)

    browser_agent = create_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=browser_tools,
        system_prompt="You are a web research assistant. Use the browser tools to complete the task, then report clearly.",
    )

    result = await browser_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "Go to https://news.ycombinator.com/ and tell me the titles of the top three stories on the front page.",
        }]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())