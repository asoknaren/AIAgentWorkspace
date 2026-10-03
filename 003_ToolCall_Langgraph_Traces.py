import os
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "false"

# 1. Define custom tools using Python type hints and docstrings
# LangChain inspects the docstring and type hints to generate the JSON Schema.
@tool
def calculate_travel_time(distance_miles: float, speed_mph: float) -> str:
    """Calculate the estimated travel time given distance in miles and speed in miles per hour."""
    if speed_mph <= 0:
        return "Error: Speed must be greater than zero."
    hours = distance_miles / speed_mph
    minutes = int((hours % 1) * 60)
    return f"{int(hours)} hours and {minutes} minutes"

@tool
def get_flight_status(flight_number: str) -> str:
    """Check the real-time departure and arrival status of a specific flight number."""
    # Simulated API call
    mock_db = {
        "AA104": "On Time. Departs at 18:30 from Terminal 8.",
        "UA82": "Delayed by 45 minutes due to maintenance."
    }
    return mock_db.get(flight_number.upper(), f"No record found for {flight_number}")

# 2. Collect tools and initialize the LLM
tools = [calculate_travel_time, get_flight_status]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Create the agent
agent = create_agent(model=llm, tools=tools)

# 4. Invoke the agent
user_prompt = "How long will it take to drive 320 miles at 65 mph, and what's the status of flight AA104?"
for event in agent.stream(
    {"messages": [("user", user_prompt)]},
    stream_mode="debug",
):
    pprint(event)