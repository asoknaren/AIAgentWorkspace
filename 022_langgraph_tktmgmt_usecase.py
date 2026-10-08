import os
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# --- 1. Schemas & State ---

class TriageAnalysis(BaseModel):
    ticket_type: str = Field(description="'defect', 'enhancement', or 'support'")
    sentiment_score: float = Field(description="Score between -1.0 (furious) and 1.0 (delighted)")
    is_angry: bool = Field(description="True if customer displays overt frustration or urgency")
    suggested_department: str = Field(description="Target department: 'Core_Dev', 'Product_Mgmt', 'Support_Tier2'")

class SimilarDefect(BaseModel):
    crm_number: str
    release: str
    similarity_score: float
    workaround_or_status: str

class TicketProcessingState(TypedDict):
    crm_number: str
    agency_id: str
    release_number: Optional[str]
    ticket_text: str
    communication_history: List[str]
    
    # Analysis outputs
    ticket_type: str
    sentiment_score: float
    is_angry: bool
    assigned_department: str
    similar_defects: List[dict]
    recommended_action: str


# --- 2. Node Implementations ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def triage_and_sentiment_node(state: TicketProcessingState) -> dict:
    """Classifies category, assesses customer sentiment, and determines initial department."""
    structured_llm = llm.with_structured_output(TriageAnalysis)
    
    prompt = f"""
    Analyze CRM Ticket #{state['crm_number']} for Agency {state['agency_id']}.
    Communication History:
    {chr(10).join(state['communication_history'])}
    
    Latest Message:
    {state['ticket_text']}
    """
    
    analysis = structured_llm.invoke(prompt)
    
    return {
        "ticket_type": analysis.ticket_type,
        "sentiment_score": analysis.sentiment_score,
        "is_angry": analysis.is_angry,
        "assigned_department": analysis.suggested_department
    }

def defect_matching_node(state: TicketProcessingState) -> dict:
    """Searches vector index for matching open/closed defects in the same release."""
    release = state.get("release_number", "Unknown")
    
    # Query vector store filtering by release
    # (Mocked tool call: search_release_defects(query=state['ticket_text'], release=release))
    found_defects = [
        {
            "crm_number": "CRM-8821",
            "release": release,
            "similarity_score": 0.89,
            "workaround_or_status": "Known issue in UI grid rendering; fixed in hotfix patch 2.4.1"
        }
    ]
    
    return {
        "similar_defects": found_defects,
        "recommended_action": f"Potential duplicate of {found_defects[0]['crm_number']}. Apply hotfix patch."
    }

def escalation_router_node(state: TicketProcessingState) -> dict:
    """Escalates angry customer tickets to high-priority queues."""
    return {
        "assigned_department": "Customer_Success_Escalations",
        "recommended_action": "ALERT: Customer sentiment critical. Notify Account Executive."
    }

def standard_routing_node(state: TicketProcessingState) -> dict:
    """Assigns standard priority and updates ticket status in SQL Server."""
    return {
        "recommended_action": f"Routed to {state['assigned_department']}. Ready for agent pick-up."
    }


# --- 3. Conditional Edge Logic ---

def route_decision(state: TicketProcessingState) -> str:
    # Priority 1: Angry customer immediately routes to escalation path
    if state["is_angry"]:
        return "escalate"
    # Priority 2: Defects go to similarity matching
    if state["ticket_type"] == "defect":
        return "match_defects"
    # Priority 3: Standard routing
    return "standard_route"


# --- 4. Graph Construction ---

builder = StateGraph(TicketProcessingState)

builder.add_node("triage", triage_and_sentiment_node)
builder.add_node("match_defects", defect_matching_node)
builder.add_node("escalate", escalation_router_node)
builder.add_node("standard_route", standard_routing_node)

builder.add_edge(START, "triage")

builder.add_conditional_edges(
    "triage",
    route_decision,
    {
        "escalate": "escalate",
        "match_defects": "match_defects",
        "standard_route": "standard_route"
    }
)

builder.add_edge("match_defects", END)
builder.add_edge("escalate", END)
builder.add_edge("standard_route", END)

crm_agent = builder.compile()


if __name__ == "__main__":
    sample_tickets = [
        {
            "crm_number": "CRM-1042",
            "agency_id": "AGENCY-17",
            "release_number": "2.4.0",
            "communication_history": ["The issue started after our latest upgrade."],
            "ticket_text": "The customer grid sometimes renders blank rows when we scroll. Can you investigate?",
        },
        {
            "crm_number": "CRM-1043",
            "agency_id": "AGENCY-08",
            "release_number": "2.4.0",
            "communication_history": [
                "We reported this yesterday and have not heard back.",
                "Our entire team is blocked from processing applications.",
            ],
            "ticket_text": "This is urgent. The application crashes every time we submit a case, and we need help now.",
        },
        {
            "crm_number": "CRM-1044",
            "agency_id": "AGENCY-22",
            "release_number": "2.5.0",
            "communication_history": [],
            "ticket_text": "Could you add an option to export the monthly dashboard as a spreadsheet?",
        },
        {
            "crm_number": "CRM-1045",
            "agency_id": "AGENCY-31",
            "release_number": "2.4.1",
            "communication_history": ["Our new staff need access to the reporting area."],
            "ticket_text": "How do I grant a colleague permission to view reports?",
        },
    ]

    for ticket in sample_tickets:
        result = crm_agent.invoke(ticket)
        print(f"\nTicket: {ticket['crm_number']} | {ticket['ticket_text']}")
        print(f"Type: {result['ticket_type']}")
        print(f"Sentiment: {result['sentiment_score']:.2f} | Angry: {result['is_angry']}")
        print(f"Department: {result['assigned_department']}")
        print(f"Recommended action: {result['recommended_action']}")
        if result["similar_defects"]:
            print(f"Similar defects: {result['similar_defects']}")