"""Order storage.

Writes real orders into the store's actual Supabase `orders`/`order_items`
tables when SUPABASE_SERVICE_KEY is configured. If it's not (e.g. local
testing), falls back to a local data/pending_orders.json file so nothing is
lost and the rest of the agent doesn't have to care where orders live.

Real schema facts learned by testing against the live database (not
guessed - see the conversation that built this):
- orders.user_id is a foreign key to Supabase Auth's users table, not just
  `profiles` - see agent/repositories/customers.py for how we resolve that.
- phone numbers must be in local Israeli format ("0524215758"), not
  WhatsApp's international format - see agent/phone.py.
- status/payment_type/payment_status are free-text columns with check
  constraints - valid values were found by inspecting real rows:
  status: ראשוני (new) / לטיפול (in progress) / נלקח (picked up)
  payment_type: מזומן (cash) / אשראי (card) / צק (check) / אחר (הערות)
  payment_status: pending / paid / authorized / payment_failed
- pickup orders still need non-null city/street/building_number (NOT NULL
  columns) - real pickup orders use "עמדה" / "--" / "--" as placeholders.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

# Israel Standard Time is UTC+2; Daylight Saving (March–October) is UTC+3.
# We store in UTC and derive Israel local time only for logging.
_IST = timezone(timedelta(hours=2))   # base offset — close enough for logs

from agent import config
from agent.repositories import customers as customers_repo
from agent.repositories import supabase_client

logger = logging.getLogger(__name__)

_PENDING_ORDERS_PATH = os.path.join(config.DATA_DIR, "pending_orders.json")

# Placeholder address fields real pickup orders use in this store's system
# (pickup has no real delivery address, but the columns are NOT NULL).
PICKUP_PLACEHOLDER = {"city": "עמדה", "street": "--", "building_number": "--"}

DEFAULT_PAYMENT_TYPE = "מזומן"  # cash - WhatsApp orders don't take payment online today

# "לטיפול" ("in progress") rather than "ראשוני" ("initial"/draft) -
# deliberate choice: every other channel's order sits in "ראשוני" until a
# staff member manually clicks to advance it (there's no automatic
# trigger for this anywhere in the system, confirmed by checking real
# order data). The store owner explicitly asked for WhatsApp orders to be
# fully active immediately instead of waiting in that manual review
# queue, accepting the tradeoff that no staff member reviews a WhatsApp
# order before it's treated as confirmed.
NEW_ORDER_STATUS = "לטיפול"


def _load_all() -> list[dict]:
    if not os.path.exists(_PENDING_ORDERS_PATH):
        return []
    with open(_PENDING_ORDERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_all(orders: list[dict]) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(_PENDING_ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)


def create_order(
    phone: str,
    customer_name: str | None,
    items: list[dict],
    delivery_method: str,
    delivery_address: dict | None = None,
    shipping_method: dict | None = None,
) -> dict:
    """Places a real order (or a local fallback one). delivery_method is
    "pickup" or "delivery". shipping_method is the matched shipping_methods
    row (id/name/price) for a delivery order, or None for pickup/unmatched.

    `phone` is the contact phone the customer explicitly gave for THIS
    order (see nodes._stage_collecting_contact) - the caller must NOT pass
    the WhatsApp number here, even though it's available; the WhatsApp
    account isn't necessarily who the order is for.
    """
    items_total = round(sum(item["price"] * item["quantity"] for item in items), 2)
    shipping_cost = shipping_method["price"] if shipping_method else 0
    total = round(items_total + shipping_cost, 2)

    if not config.SUPABASE_SERVICE_KEY:
        return _create_local_order(phone, customer_name, items, delivery_method, delivery_address, total)

    return _create_supabase_order(
        phone, customer_name, items, delivery_method, delivery_address, shipping_method, total, shipping_cost,
    )


def _create_supabase_order(
    phone: str,
    customer_name: str | None,
    items: list[dict],
    delivery_method: str,
    delivery_address: dict | None,
    shipping_method: dict | None,
    total: float,
    shipping_cost: float,
) -> dict:
    user_id = customers_repo.find_or_create_customer(phone, customer_name)

    # Stamp exact server time at the moment the order is placed.
    # Sent explicitly in the POST so the DB row always has our timestamp
    # rather than relying on Supabase's default (which can vary by trigger order).
    now_utc = datetime.now(timezone.utc)
    now_israel = now_utc.astimezone(_IST)
    logger.info(
        "Order timestamp: %s (Israel: %s)",
        now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        now_israel.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if delivery_method == "pickup" or not delivery_address:
        address_fields = dict(PICKUP_PLACEHOLDER)
    else:
        address_fields = {
            "city": delivery_address["city"],
            "street": delivery_address["street"],
            "building_number": delivery_address["building_number"],
        }

    from agent.phone import to_orders_table_phone

    order_row = supabase_client.rest_post("orders", {
        "user_id": user_id,
        "customer_name": customer_name or "WhatsApp customer",
        "phone_number_1": to_orders_table_phone(phone),
        "payment_type": DEFAULT_PAYMENT_TYPE,
        "payment_status": "pending",
        "status": NEW_ORDER_STATUS,
        "source": "pos",
        "general_comment": "הזמנה מווצאפ",
        "installments": 1,
        "hidden_from_admin": False,
        "total_amount": total,
        "shipping_method": shipping_method["name"] if shipping_method else "pickup",
        "shipping_cost": shipping_cost,
        "shipping_method_id": shipping_method["id"] if shipping_method else None,
        "created_at": now_utc.isoformat(),
        **address_fields,
    })

    if not order_row:
        logger.error("Supabase order creation failed for phone=%s - falling back to local storage", phone)
        return _create_local_order(phone, customer_name, items, delivery_method, delivery_address, total)

    order_id = order_row["id"]

    # A DB trigger sets hidden_from_admin=True on every new WhatsApp order. The
    # trigger fires BEFORE INSERT so PostgREST's RETURNING clause already shows
    # True, but we always PATCH regardless — belt-and-suspenders to cover any
    # timing variation (AFTER trigger on some Supabase versions).
    fixed = supabase_client.rest_patch(
        "orders",
        {"id": f"eq.{order_id}"},
        {"hidden_from_admin": False},
    )
    if fixed:
        logger.info("Forced hidden_from_admin=False for order %s", order_id)
    else:
        logger.error("Failed to unhide order %s — it will not appear in POS/website", order_id)

    for item in items:
        supabase_client.rest_post("order_items", {
            "order_id": order_id,
            "product_id": item.get("product_id"),
            "product_name": item["name"],
            "price": item["price"],
            "quantity": item["quantity"],
            "created_at": now_utc.isoformat(),
        })

    logger.info(
        "Created REAL Supabase order %s for phone=%s, total=%.2f, at=%s",
        order_id, phone, total, now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return {
        "id": order_id,
        "phone": phone,
        "customer_name": customer_name,
        "items": items,
        "total": total,
        "status": NEW_ORDER_STATUS,
        "created_at": now_utc.isoformat(),
    }


def _create_local_order(
    phone: str, customer_name: str | None, items: list[dict], delivery_method: str,
    delivery_address: dict | None, total: float,
) -> dict:
    order = {
        "id": str(uuid.uuid4()),
        "phone": phone,
        "customer_name": customer_name,
        "items": items,
        "delivery_method": delivery_method,
        "delivery_address": delivery_address,
        "total": total,
        "status": "pending_sync",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    orders = _load_all()
    orders.append(order)
    _save_all(orders)
    logger.warning(
        "Order %s recorded LOCALLY ONLY (no SUPABASE_SERVICE_KEY configured) - phone=%s items=%d total=%.2f",
        order["id"], phone, len(items), total,
    )
    return order


def get_orders_for_phone(phone: str) -> list[dict]:
    if not config.SUPABASE_SERVICE_KEY:
        return [o for o in _load_all() if o["phone"] == phone]

    from agent.phone import to_orders_table_phone

    rows = supabase_client.rest_get(
        "orders",
        {
            "select": "id,status,total_amount,created_at,shipping_method",
            "phone_number_1": f"eq.{to_orders_table_phone(phone)}",
            "order": "created_at.desc",
            "limit": "5",
        },
        use_service_key=True,
    )
    return [
        {"id": r["id"], "phone": phone, "status": r["status"], "total": r["total_amount"],
         "created_at": r["created_at"], "items": [], "shipping_method": r["shipping_method"]}
        for r in rows
    ]


def get_latest_order_for_phone(phone: str) -> dict | None:
    orders = get_orders_for_phone(phone)
    if not orders:
        return None
    latest = max(orders, key=lambda o: o["created_at"])

    if config.SUPABASE_SERVICE_KEY and not latest["items"]:
        rows = supabase_client.rest_get(
            "order_items",
            {"select": "product_name,quantity,price,product_id", "order_id": f"eq.{latest['id']}"},
            use_service_key=True,
        )
        latest["items"] = [
            {"name": r["product_name"], "quantity": r["quantity"], "price": r["price"],
             "product_id": r.get("product_id")}
            for r in rows
        ]

    return latest


def get_all_orders_for_phone(phone: str) -> list[dict]:
    """Returns last 5 orders with their items populated — used for order history display."""
    orders = get_orders_for_phone(phone)
    if not orders or not config.SUPABASE_SERVICE_KEY:
        return orders
    for order in orders:
        if not order["items"]:
            rows = supabase_client.rest_get(
                "order_items",
                {"select": "product_name,quantity,price,product_id", "order_id": f"eq.{order['id']}"},
                use_service_key=True,
            )
            order["items"] = [
                {"name": r["product_name"], "quantity": r["quantity"], "price": r["price"],
                 "product_id": r.get("product_id")}
                for r in rows
            ]
    return orders


def get_reorder_cart_items(phone: str) -> list[dict]:
    """Fetches the most recent completed order's items in cart-compatible format.
    Returns an empty list if no prior orders exist."""
    latest = get_latest_order_for_phone(phone)
    if not latest or not latest.get("items"):
        return []
    return [
        {
            "product_id": i.get("product_id") or "",
            "name": i["name"],
            "price": i.get("price") or 0,
            "quantity": i.get("quantity") or 1,
            "unit_type": "unit",   # unit_type not stored on order_items — default to unit
        }
        for i in latest["items"]
    ]
