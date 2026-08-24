"""Real pickup/delivery options, read from the store's `shipping_methods`
table (discovered by inspecting the real schema - there's no public anon
access to this table, so it always uses the service key).
"""
import logging

from agent.repositories import supabase_client

logger = logging.getLogger(__name__)

# The store's name for their free in-store pickup option (vs. a priced
# delivery zone). Matched by exact name since that's how it's stored.
PICKUP_METHOD_NAME = "איסוף מהחנות"


def list_delivery_zones() -> list[dict]:
    """All active delivery zones (pickup excluded), each with a city/price."""
    methods = supabase_client.rest_get(
        "shipping_methods",
        {"select": "id,name,price,city", "active": "eq.true", "order": "sort_order"},
        use_service_key=True,
    )
    return [m for m in methods if m["name"] != PICKUP_METHOD_NAME]


def find_delivery_zone_for_city(city: str) -> dict | None:
    """Best-effort match of a customer-typed city to a known delivery zone."""
    zones = list_delivery_zones()
    for zone in zones:
        if zone.get("city") and zone["city"] in city:
            logger.info("Matched delivery zone %r for city=%r", zone["name"], city)
            return zone
    logger.warning("No delivery zone matched city=%r - falling back to unpriced delivery", city)
    return None
