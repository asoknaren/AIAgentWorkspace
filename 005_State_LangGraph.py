import os
from typing import Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 1. Initialize the LLM
# Replace or ensure OPENAI_API_KEY is set in your environment
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 2. Define the State
# This represents the data that travels between nodes in the graph.
class SupportState(TypedDict):
    ticket: str
    category: Literal["technical", "billing", "general"]
    response: str


# 3. Define the Structured Output Schema for Classification
class CategoryClassification(BaseModel):
    category: Literal["technical", "billing", "general"] = Field(
        description="The category that best matches the user's issue: 'technical', 'billing', or 'general'."
    )


# 4. Define the Nodes

def classify_intent_node(state: SupportState) -> dict:
    """Classifies the user's inquiry into one of three support categories."""
    classifier = llm.with_structured_output(CategoryClassification)
    
    prompt = (
        "Classify the following user support ticket into one of three categories: "
        "'technical', 'billing', or 'general'.\n\n"
        f"Ticket: {state['ticket']}"
    )
    result = classifier.invoke(prompt)
    
    # Return updates to the state
    return {"category": result.category}


def technical_support_node(state: SupportState) -> dict:
    """Handles IT hardware, software bugs, VPN, and system errors."""
    prompt = (
        "You are an IT Support Specialist. Provide concise, step-by-step "
        f"troubleshooting instructions for this issue:\n{state['ticket']}"
    )
    res = llm.invoke(prompt)
    return {"response": f"[IT Support]:\n{res.content}"}


def billing_support_node(state: SupportState) -> dict:
    """Handles payments, refunds, invoices, and subscription queries."""
    prompt = (
        "You are an Accounts & Billing Specialist. Address the customer's "
        f"billing or refund inquiry politely and clearly:\n{state['ticket']}"
    )
    res = llm.invoke(prompt)
    return {"response": f"[Billing Department]:\n{res.content}"}


def general_support_node(state: SupportState) -> dict:
    """Handles general company queries, hours of operation, and feedback."""
    prompt = (
        "You are a Customer Service Representative. Answer the customer's general inquiry:\n"
        f"{state['ticket']}"
    )
    res = llm.invoke(prompt)
    return {"response": f"[General Inquiries]:\n{res.content}"}


# 5. Define the Routing Logic
def route_ticket(state: SupportState) -> str:
    """Reads the category from state and returns the name of the next node."""
    category = state["category"]
    if category == "technical":
        return "technical_support"
    elif category == "billing":
        return "billing_support"
    return "general_support"


# 6. Build the Graph
builder = StateGraph(SupportState)

# Add Nodes
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("technical_support", technical_support_node)
builder.add_node("billing_support", billing_support_node)
builder.add_node("general_support", general_support_node)

# Add Edges
builder.add_edge(START, "classify_intent")

# Add Conditional Branching from the classifier
builder.add_conditional_edges(
    "classify_intent",
    route_ticket,
    {
        "technical_support": "technical_support",
        "billing_support": "billing_support",
        "general_support": "general_support",
    },
)

# Connect specialist nodes to END
builder.add_edge("technical_support", END)
builder.add_edge("billing_support", END)
builder.add_edge("general_support", END)

# Compile into an executable runnable
app = builder.compile()


# 7. Test the Graph
if __name__ == "__main__":
    test_tickets = [
        "I was charged twice on my credit card for this month's plan!",
        "My laptop screen is flickering and my VPN disconnects every 5 minutes.",
        "What are your business hours on weekends?",
    ]

    for ticket in test_tickets:
        print(f"\nIncoming Ticket: \"{ticket}\"")
        result = app.invoke({"ticket": ticket})
        print(f"Assigned Category: {result['category']}")
        print(f"Department Response:\n{result['response']}")
        print("-" * 50)