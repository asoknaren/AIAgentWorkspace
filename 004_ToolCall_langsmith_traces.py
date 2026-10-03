import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key or langsmith_api_key == "your_langsmith_api_key_here":
    raise RuntimeError("Set a valid LANGSMITH_API_KEY in .env to upload traces.")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "agent-class")


@tool
def calculate_travel_time(distance_miles: float, speed_mph: float) -> str:
    """Calculate the estimated travel time given distance and speed."""
    if speed_mph <= 0:
        return "Error: Speed must be greater than zero."
    hours = distance_miles / speed_mph
    minutes = int((hours % 1) * 60)
    return f"{int(hours)} hours and {minutes} minutes"


@tool
def get_flight_status(flight_number: str) -> str:
    """Check the status of a specific flight number."""
    mock_db = {
        "AA104": "On Time. Departs at 18:30 from Terminal 8.",
        "UA82": "Delayed by 45 minutes due to maintenance.",
    }
    return mock_db.get(flight_number.upper(), f"No record found for {flight_number}")


tools = [calculate_travel_time, get_flight_status]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(model=llm, tools=tools)

user_prompt = "How long will it take to drive 320 miles at 65 mph, and what's the status of flight AA104?"
response = agent.invoke({"messages": [("user", user_prompt)]})
print(response["messages"][-1].content)
