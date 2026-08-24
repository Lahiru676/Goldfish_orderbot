"""Builds the LangGraph graph that decides how to answer each WhatsApp message.

    START -> router -> {order_taking | product_qa | delivery_info |
                         order_support | general_info} -> guardrail -> END
                                                              |
                                          (rejected once) loops back to the
                                          same domain node for one retry

    START -> router -> escalate -> END   (fixed reply, no guardrail needed)

See the conversation that designed this for why: a single LLM router can
misclassify overlapping intents (e.g. "where's my order and I'm annoyed"),
so every domain node's draft reply is double-checked by the guardrail
before it's allowed to reach the customer.
"""
import logging

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.state import AgentState

logger = logging.getLogger(__name__)

DOMAIN_NODES = ["order_taking", "product_qa", "delivery_info", "order_support", "general_info"]


def _route_after_router(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "escalate":
        return "escalate"
    if intent in DOMAIN_NODES:
        return intent
    logger.warning("Unexpected intent %r after router, defaulting to general_info", intent)
    return "general_info"


def _route_after_guardrail(state: AgentState) -> str:
    if state.get("human_takeover"):
        return END
    if state.get("guardrail_retries", 0) > 0:
        return state["intent"]
    return END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", nodes.router_node)
    graph.add_node("order_taking", nodes.order_taking_node)
    graph.add_node("product_qa", nodes.product_qa_node)
    graph.add_node("delivery_info", nodes.delivery_info_node)
    graph.add_node("order_support", nodes.order_support_node)
    graph.add_node("general_info", nodes.general_info_node)
    graph.add_node("escalate", nodes.escalate_node)
    graph.add_node("guardrail", nodes.guardrail_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _route_after_router, DOMAIN_NODES + ["escalate"])

    for node_name in DOMAIN_NODES:
        graph.add_edge(node_name, "guardrail")

    graph.add_conditional_edges("guardrail", _route_after_guardrail, DOMAIN_NODES + [END])
    graph.add_edge("escalate", END)

    logger.info("Agent graph built: router -> %s (+escalate) -> guardrail -> END", DOMAIN_NODES)
    return graph.compile()


# Built once when this module is first imported, then reused for every turn.
compiled_graph = build_graph()
