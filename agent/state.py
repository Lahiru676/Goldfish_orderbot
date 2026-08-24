"""The shape of data that flows through the LangGraph graph for one customer.

One AgentState exists per WhatsApp phone number. It's loaded from storage
before a graph run and saved back after - see conversation_store.py.
"""
from typing import TypedDict


class CartItem(TypedDict):
    product_id: str
    name: str
    price: float
    quantity: int


class DeliveryAddress(TypedDict):
    city: str
    street: str
    building_number: str


class AgentState(TypedDict):
    
    phone: str
    customer_name: str | None

    
    customer_language: str | None

    
    customer_phone: str | None


    messages: list[dict]

    
    intent: str | None

    
    welcomed: bool

    
    flow_stage: str

    
    turn_start_stage: str

    
    cart: list[CartItem]

    
    cart_at_turn_start: list[CartItem]

    
    delivery_method: str | None

    
    delivery_address: DeliveryAddress | None

    
    shipping_method: dict | None

    
    supabase_customer_id: str | None

    
    order_just_created: dict | None

    
    registration_stage: str | None

    
    human_takeover: bool

    
    reply: str | None

    
    grounding_context: str | None

    
    guardrail_retries: int

    
    guardrail_feedback: str | None

    
    is_first_message: bool

    
    browsing_category: str | None

    # Tracks the current position while navigating a multi-level product tree.
    # None = show category-level roots. When set, show children of this product ID.
    # Supports arbitrary depth (brand → group → size → processing variant).
    browsing_parent_id: str | None

    # Set when the bot showed one specific product and is waiting for the
    # customer to give a quantity. If the next message is a bare number we
    # skip the full catalog re-search and add this product directly.
    # Stores {id, name, price, unit_type}. Cleared after use or on cart reset.
    pending_product: dict | None


def new_state(phone: str, customer_name: str | None = None) -> AgentState:
    """Creates a fresh, empty conversation state for a phone number we haven't seen before."""
    return AgentState(
        phone=phone,
        customer_name=customer_name,
        customer_language=None,
        customer_phone=None,
        messages=[],
        intent=None,
        welcomed=False,
        flow_stage="browsing",
        turn_start_stage="browsing",
        cart=[],
        cart_at_turn_start=[],
        delivery_method=None,
        delivery_address=None,
        shipping_method=None,
        supabase_customer_id=None,
        order_just_created=None,
        registration_stage=None,
        human_takeover=False,
        reply=None,
        grounding_context=None,
        guardrail_retries=0,
        guardrail_feedback=None,
        is_first_message=False,
        browsing_category=None,
        browsing_parent_id=None,
        pending_product=None,
    )
