import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv(override=True)


@tool
def get_share_price(symbol: str) -> float:
	"""Return the current sample share price for a ticker symbol."""
	fake_prices = {"AAPL": 241.5, "GOOG": 168.2, "AMZN": 198.0}
	return fake_prices.get(symbol.upper(), 0.0)


@tool
def get_weather(city: str) -> str:
	"""Return a fictional weather report for a city."""
	reports = {
		"seattle": "Cloudy, 16 C",
		"paris": "Sunny, 22 C",
		"tokyo": "Light rain, 19 C",
	}
	return reports.get(city.lower(), f"No sample forecast for {city}")


class Company(BaseModel):
	name: str = Field(description="The company name")
	ticker: str = Field(description="The stock ticker symbol")
	founded_year: int = Field(description="The year the company was founded")


def run_examples() -> None:
	"""Run the LangChain building-block examples from lab 1."""
	if not os.getenv("OPENAI_API_KEY"):
		raise RuntimeError("Set OPENAI_API_KEY in the project-root .env file.")

	llm = ChatOpenAI(model="gpt-4o")

	print("=== First model call ===")
	reply = llm.invoke("In one sentence, what does an autonomous AI agent do?")
	print(reply.content)

	print("\n=== Streaming ===")
	for chunk in llm.stream("Write a two-line poem about autonomous agents."):
		print(chunk.content, end="", flush=True)
	print()

	print("\n=== Messages ===")
	messages = [
		SystemMessage("You are a terse assistant. Answer in exactly five words."),
		HumanMessage("What is the capital of France?"),
	]
	print(llm.invoke(messages).content)

	print("\n=== Tool metadata and direct invocation ===")
	print("name:", get_share_price.name)
	print("description:", get_share_price.description)
	print("args:", get_share_price.args)
	print("sample price:", get_share_price.invoke({"symbol": "AAPL"}))

	print("\n=== Manual tool loop with two tools ===")
	tools = {get_share_price.name: get_share_price, get_weather.name: get_weather}
	llm_with_tools = llm.bind_tools(list(tools.values()))
	conversation = [
		HumanMessage(
			"Use the tools to find Amazon's share price and the sample weather in Seattle."
		)
	]
	ai_message = llm_with_tools.invoke(conversation)
	conversation.append(ai_message)

	for call in ai_message.tool_calls:
		result = tools[call["name"]].invoke(call["args"])
		conversation.append(
			ToolMessage(content=str(result), tool_call_id=call["id"])
		)

	final = llm_with_tools.invoke(conversation)
	print(final.content)

	print("\n=== Structured output ===")
	structured_llm = llm.with_structured_output(Company)
	company = structured_llm.invoke(
		"Give structured details about Amazon the technology company."
	)
	print(company)
	print("ticker:", company.ticker)

if __name__ == "__main__":
	run_examples()
