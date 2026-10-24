import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
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

    # One session for the whole run: get_tools() alone opens a new session per call, which resets the browser.
    async with client.session("playwright") as session:
        browser_tools = await load_mcp_tools(session)
        print(f"Loaded {len(browser_tools)} browser tools:")
        for tool in browser_tools:
            print(" -", tool.name)

        browser_agent = create_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=browser_tools,
            system_prompt=(
                "You are a study assistant. Call browser_navigate, then read the page with browser_evaluate "
                "using exactly this function: () => document.querySelector('#mw-content-text').innerText.slice(0, 6000). "
                "Do NOT call browser_snapshot: full pages exceed the model's context limit. "
                "Base your answer only on the extracted text."
            ),
        )

        result = await browser_agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": (
                    "I have an exam on machine learning. Open https://en.wikipedia.org/wiki/Machine_learning, "
                    "read the introduction, and give me: (1) a 2-sentence plain-English summary, "
                    "(2) 5 flashcards as 'Q: ... / A: ...', and (3) 3 related topics worth studying next."
                ),
            }]
        })
    for msg in result["messages"]:
        if msg.type == "tool":  # shows the real browser error, if any
            print(f"[tool:{msg.name}] {str(msg.content)[:300]}")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())