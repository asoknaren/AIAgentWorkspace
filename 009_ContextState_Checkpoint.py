"""Show how LangChain tools read run context and update agent state.

Context is supplied by the application for each invocation; state is mutable
conversation data saved by the checkpointer. Set OPENAI_API_KEY in the project-
root .env file before running. The in-memory checkpointer keeps this demo's cart
for the lifetime of the Python process.
"""

import operator
import os
from dataclasses import dataclass
from typing import Annotated

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


load_dotenv(override=True)


@dataclass(frozen=True)
class ShopperContext:
    """Per-invocation information supplied by the application."""

    customer_name: str
    loyalty_tier: str
    discount_percent: int


class ShoppingState(AgentState):
    """Mutable conversation data retained by the agent's checkpointer."""

    cart_items: Annotated[list[str], operator.add]


@tool
def add_item_to_cart(
    item: str,
    runtime: ToolRuntime[ShopperContext, ShoppingState],
) -> Command:
    """Add one item to the shopper's conversation cart."""
    return Command(
        update={
            "cart_items": [item],
            "messages": [
                ToolMessage(
                    content=f"Added {item} to the cart.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def get_cart(runtime: ToolRuntime[ShopperContext, ShoppingState]) -> str:
    """Read the cart currently stored in the agent's state."""
    items = runtime.state.get("cart_items", [])
    if not items:
        return "The cart is empty."
    return "Cart items: " + ", ".join(items)


@tool
def get_loyalty_benefit(runtime: ToolRuntime[ShopperContext, ShoppingState]) -> str:
    """Read the current shopper's per-invocation loyalty context."""
    shopper = runtime.context
    return (
        f"{shopper.customer_name} is a {shopper.loyalty_tier} member "
        f"with a {shopper.discount_percent}% discount."
    )


def main() -> None:
    """Run two turns to contrast per-run context with checkpointed state."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the project-root .env file.")

    agent = create_agent(
        model="openai:gpt-4o",
        tools=[add_item_to_cart, get_cart, get_loyalty_benefit],
        system_prompt=(
            "You are a shopping assistant. Use add_item_to_cart whenever the "
            "shopper asks to add items. Use get_cart for cart questions and "
            "get_loyalty_benefit for membership or discount questions. Do not "
            "invent cart contents or loyalty benefits."
        ),
        state_schema=ShoppingState,
        context_schema=ShopperContext,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "context-state-demo"}}

    context = ShopperContext(
        customer_name="Alex",
        loyalty_tier="Gold",
        discount_percent=10,
    )

    print("=== Turn 1: a tool updates state ===")
    first_turn = agent.invoke(
        {
            "messages": [("user", "Add a notebook and a pen to my cart.")],
        },
        config=config,
        context=context,
    )
    print(first_turn["messages"][-1].content)
    print("State cart:", first_turn.get("cart_items", []))

    print("\n=== Turn 2: tools read state and context ===")
    second_turn = agent.invoke(
        {
            "messages": [
                ("user", "What's in my cart, and what is my member discount?")
            ]
        },
        config=config,
        # Context is supplied again; it is not stored in the checkpointed state.
        context=context,
    )
    print(second_turn["messages"][-1].content)
    print("State cart:", second_turn.get("cart_items", []))

    print("\n=== Turn 3: adding another item to the cart ===")
    third_turn = agent.invoke(
        {
            # Don't resend cart_items: the checkpointer already holds them, and
            # the operator.add reducer would append the input a second time.
            "messages": [("user", "Add a ruler to my cart.")],
        },
        config=config,
        context=context,
    )
    print(third_turn["messages"][-1].content)
    print("State cart:", third_turn.get("cart_items", []))

    print("\n=== Turn 4: checking the final cart and loyalty benefit ===")
    fourth_turn = agent.invoke(
        {
            "messages": [
                ("user", "What's in my cart now, and what is my member discount?")
            ]
        },
        config=config,
        context=context,
    )
    print(fourth_turn["messages"][-1].content)
    print("State cart:", fourth_turn.get("cart_items", []))


if __name__ == "__main__":
    main()