"""The actual LangGraph nodes: the router that decides intent, one node per
business domain (orders, product questions, delivery, order support,
general info), an escalation node, and a guardrail that checks every draft
reply before it's allowed to go out.

Every node takes the current AgentState and returns a dict of the fields it
wants to update - that's how LangGraph nodes work.
"""
import logging

from agent import config, escalation, llm
from agent import phone as phone_utils
from agent.repositories import orders as orders_repo
from agent.repositories import products as products_repo
from agent.repositories import shipping as shipping_repo
from agent.state import AgentState

logger = logging.getLogger(__name__)

VALID_INTENTS = {
    "order_taking",
    "product_qa",
    "delivery_info",
    "order_support",
    "general_info",
    "escalate",
}

DELIVERY_INFO_TEXT = """
Business hours:
Sunday: closed
Monday-Tuesday: 10:00-17:00
Wednesday-Thursday: 10:00-20:00
Friday: 10:00-12:00
Closed on holidays.

Orders are accepted daily until 15:00 to guarantee availability - recommend ordering 2 days ahead.
Serving Beitar Illit, Beit Shemesh and surrounding areas.
Address: Rabbi Akiva 23, Beitar Illit. Phone: 02-581-9955.
""".strip()

GENERAL_INFO_TEXT = """
Gold Fish (גולד פיש) - boutique fishmonger & butcher in Beitar Illit.
Address: Rabbi Akiva 23, Beitar Illit. Phone: 02-581-9955.
Kosher certifications used across the catalog include: עד"ח (Eidah Chareidit), חב"ד שייקביץ, קרסיק חב"ד, רובין.
The certification depends on the specific product - check that product's own hechsher field.
""".strip()

# Used when the LLM/guardrail pipeline genuinely fails (e.g. guardrail
# exhausted its retries, or a node got no reply back) - this must NOT be
# English-only, since most customers write Hebrew (see feedback memory:
# Hebrew-first testing) and a wrong-language fallback is exactly the kind
# of language-mixing customers have complained about. Kept as a plain
# Hebrew string rather than an LLM call, since this IS the fallback for
# when things already went wrong - it must not depend on another LLM call
# that could also fail.
SAFE_FALLBACK_REPLY = "סליחה, לא הבנתי לגמרי. אפשר לכתוב את זה במילים אחרות? אני רוצה לוודא שאני נותן לך את המידע הנכון."

# Customers sometimes switch languages mid-conversation (e.g. start in
# Hebrew, then ask "give me this in English"). A bare "reply in their
# language" instruction is ambiguous once the chat history itself has
# mixed languages in it - the LLM can default to whichever language
# dominates the history instead of the one actually being spoken right
# now. Always be explicit about WHICH message's language to match.
LANGUAGE_INSTRUCTION = (
    "Reply in the SAME language as the customer's MOST RECENT message - check the actual words of "
    "their last message, not which language dominates the conversation so far. Even a short reply "
    "(\"yes\", \"pickup\", a name) can signal they've switched languages - if it's recognizably a "
    "different language than before, switch to it and KEEP using it for this reply and from now on, "
    "until they switch again. Do not drift back to an earlier language on your own."
)


def language_instruction(lang: str | None = None) -> str:
    """Returns an explicit language directive for an LLM prompt.

    When the customer's language has been detected and stored in state, this
    gives a firm 'always reply in X' instruction — much stronger than asking
    the LLM to guess from the latest message, which caused language drift in
    production. Falls back to the per-message auto-detection constant when
    the language is not yet known (first message of a brand new conversation).
    """
    if lang:
        return (
            f"Always reply in {lang}. "
            f"This customer's language is {lang} — never switch to a different language, "
            f"even if they write a word or two in another language."
        )
    return LANGUAGE_INSTRUCTION


# --- small helpers shared by several nodes ---

def recent_context(state: AgentState) -> list[dict]:
    """The last few chat turns, used as context for every LLM call.

    Always prepends the current cart as a pinned system message so the LLM
    knows what's been ordered even when conversation history exceeds the
    context window (48+ messages in production caused the LLM to lose track).

    If this turn's previous attempt was rejected by the guardrail
    (guardrail_feedback set), appends a note about WHY so the retried node
    targets that specific problem instead of repeating the same mistake.
    """
    cart = state.get("cart") or []
    prefix = []
    if cart:
        prefix = [{
            "role": "system",
            "content": (
                f"[CURRENT CART — {len(cart)} item(s), pinned for reference]:\n"
                f"{_format_cart(cart)}\n"
                f"Total so far: {sum((i.get('price') or 0) * (i.get('quantity') or 0) for i in cart):.2f} ILS\n"
                "[END CART]"
            ),
        }]

    messages = state["messages"][-config.MAX_CONTEXT_MESSAGES:]
    feedback = state.get("guardrail_feedback")
    if feedback:
        messages = messages + [{
            "role": "system",
            "content": (
                f"Your previous draft reply was rejected by quality control: {feedback}. "
                "Fix that specific problem in your new reply - pay special attention if it "
                "was about replying in the wrong language."
            ),
        }]
    return prefix + messages


def last_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message["role"] == "user":
            return message["content"]
    return ""


def extract_query_terms(context_messages: list[dict], category_names: list[str]) -> dict:
    """Asks the LLM what product/category the customer means, so we can
    search the real product table.

    Shows the LLM the REAL catalog (every actual product name + its
    category) instead of asking it to blindly translate/guess a Hebrew
    search term - that guessing is what caused real mismatches in
    production (e.g. a generic translated term for "fresh fish"
    coincidentally matched unrelated products in other categories,
    because no real fresh-fish item actually contains the literal words
    "fish"/"fresh" in its name). Picking a name straight from the real
    list means the later DB search is matching against itself, not a guess.

    Takes recent chat history, not just the latest message in isolation -
    a bare follow-up like "1" or "yes" has no product name in it at all,
    but read together with the assistant's own previous question ("how
    much carp roe would you like?") it's obvious what's being ordered. Only
    looking at the latest message used to make the bot lose track of the
    product entirely on these short answers.
    """
    catalog = products_repo.list_all_product_names()
    catalog_text = "\n".join(f"{p['category']}: {p['name']}" for p in catalog) or "(catalog unavailable)"

    system = (
        "Below is the REAL catalog of this kosher fish/meat shop (category: product name), in Hebrew - "
        "every line is an exact real product.\n\n"
        f"{catalog_text}\n\n"
        "Decide what the customer is asking about/ordering, to search this catalog:\n"
        "- If they mean a SPECIFIC product (even if they wrote in English or phrased it differently), "
        "copy that product's name EXACTLY as written above into search_term - verbatim, don't "
        "translate or paraphrase it yourself, just pick the matching real line.\n"
        "- If they're only naming a category in general (e.g. \"fresh fish\", \"meat\", \"do you have "
        "chicken\") rather than a specific product, leave search_term EMPTY and only set category.\n"
        "- If truly unclear, leave both empty.\n"
        "Use the whole conversation below for context - e.g. if the assistant just asked about a "
        'specific product and the customer\'s latest reply is just a quantity ("1", "2 קילו") or '
        "confirmation, that reply is about THAT product, not something unclear.\n\n"
        f"Known categories: {', '.join(category_names)}.\n"
        'Reply JSON: {"search_term": "<EXACT real product name copied from the catalog above, or '
        'empty string>", "category": "<one of the known categories above, or empty string>"}'
    )
    return llm.chat_json([{"role": "system", "content": system}] + context_messages)


# --- router ---

ROUTER_SYSTEM_PROMPT = """You are routing an incoming WhatsApp message for Gold Fish, a kosher fish/meat shop, to the right handler.

Classify the customer's LATEST message into exactly one of these intents:
- "order_taking": wants to order something, add/remove items, or confirm an order.
- "product_qa": asking about products, prices, availability, or kosher certification.
- "delivery_info": asking about delivery areas, cost, timing, or opening hours.
- "order_support": asking about an existing order's status, complaints, order history,
    OR wants to re-order / repeat a previous order (e.g. "same as last time",
    "add my previous order", "תוסיף לי מה שהזמנתי", "מה הזמנתי בפעם הקודמת",
    "show me my orders", "הזמנות קודמות", "הזמן שוב").
- "general_info": general business questions (location, contact, etc.) not covered above.
- "escalate": explicitly asks for a human/manager, or is abusive/very upset.

Reply with JSON: {"intent": "<one of the labels above>"}
"""

# Stages of the structured order flow where the customer is expected to
# answer a specific question we just asked (pickup/delivery? address? name?
# confirm?). While one of these is active, the router must NOT reclassify
# their reply into a different domain - e.g. an address like "Rabbi Akiva
# 10, Beitar Illit" looks like a delivery_info question on its own, but
# here it's actually the answer to "what's your delivery address?" and
# must stay in order_taking or the whole flow breaks.
ACTIVE_CHECKOUT_STAGES = {"choosing_delivery", "collecting_address", "collecting_name", "confirming_summary", "choosing_parent"}

# If the customer explicitly asks for a human even mid-checkout, still let
# that through instead of forcing order_taking.
ESCALATE_KEYWORDS = ("נציג", "מנהל", "בן אדם", "human", "manager", "agent")

# Phrases that explicitly signal "cancel what I had and start a completely
# new order from scratch". Matched against the lowercased user message.
# Deliberately specific to avoid false positives (e.g. "בטל" alone could
# mean "remove this one item", not "reset everything").
_NEW_ORDER_PHRASES = (
    # Hebrew
    "הזמנה חדשה", "התחל מחדש", "להתחיל מחדש", "להזמין מחדש",
    "בטל הכל", "לבטל הכל", "לבטל את ההזמנה",
    # English
    "new order", "start over", "start again", "start fresh", "restart",
)


def router_node(state: AgentState) -> dict:
    stage = state.get("flow_stage") or "browsing"
    user_text = last_user_text(state)

    if stage in ACTIVE_CHECKOUT_STAGES and not any(kw in user_text for kw in ESCALATE_KEYWORDS):
        logger.info("Router: phone=%s mid-checkout (stage=%s) - forcing order_taking", state["phone"], stage)
        return {"intent": "order_taking"}

    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}] + recent_context(state)
    result = llm.chat_json(messages)
    intent = result.get("intent")
    if intent not in VALID_INTENTS:
        logger.warning("Router returned unknown intent %r, defaulting to general_info", intent)
        intent = "general_info"
    logger.info("Router: phone=%s intent=%s", state["phone"], intent)
    return {"intent": intent}


# --- domain nodes ---

def product_qa_node(state: AgentState) -> dict:
    categories = products_repo.list_categories()
    category_names = [c["category_name"] for c in categories]
    terms = extract_query_terms(recent_context(state), category_names)
    search_term = terms.get("search_term") or None
    category = terms.get("category") or None

    if not search_term and not category:
        # The customer asked something general ("what do you have?") rather
        # than about a specific item - there's no single product to search
        # for, so show the category list instead of (wrongly) saying
        # nothing in the catalog matches.
        logger.info("product_qa: no specific item/category extracted, showing category overview")
        catalog_context = "Categories available in the catalog:\n" + "\n".join(
            f"- {c['category_name']}" for c in categories
        )
    else:
        results = products_repo.search_products(query=search_term, category=category)
        if results:
            catalog_context = "\n".join(
                f"- {p['name']}: {p['price']} ILS per {p.get('unit_type') or 'unit'}"
                f"{' (currently out of stock)' if not p.get('in_stock', True) else ''}"
                f" - hechsher: {p.get('hechsher') or 'n/a'}"
                for p in results
            )
        else:
            catalog_context = "(no matching products found in the catalog)"

    system = (
        "You are a friendly WhatsApp assistant for Gold Fish, a kosher fish/meat shop in Beitar Illit. "
        "Answer using ONLY the info below - never invent a price, product, or kosher certification. "
        "If you're given a category list rather than specific products, mention the categories and ask "
        "which one they're interested in (don't claim nothing is available). "
        f"{language_instruction(state.get('customer_language'))} Keep it short - this is WhatsApp.\n\n"
        f"{catalog_context}"
    )
    reply = llm.chat([{"role": "system", "content": system}] + recent_context(state))
    return {"reply": reply, "grounding_context": catalog_context}


def delivery_info_node(state: AgentState) -> dict:
    zones = shipping_repo.list_delivery_zones()
    if zones:
        zones_text = "Delivery areas:\n" + "\n".join(
            f"- {z['name']} ({z['city']}): {int(z['price'])} ILS" for z in zones
        )
    else:
        zones_text = "Delivery areas: contact the shop for details."

    context = f"{DELIVERY_INFO_TEXT}\n\n{zones_text}"
    system = (
        "You are a WhatsApp assistant for Gold Fish. Answer the customer's question about "
        "delivery, hours, or ordering cutoffs using ONLY the info below - don't invent anything. "
        f"{language_instruction(state.get('customer_language'))} Keep it short.\n\n"
        f"{context}"
    )
    reply = llm.chat([{"role": "system", "content": system}] + recent_context(state))
    return {"reply": reply, "grounding_context": context}


def general_info_node(state: AgentState) -> dict:
    system = (
        "You are a WhatsApp assistant for Gold Fish. Answer the customer's general question about "
        "the business using ONLY the info below. If you don't know, offer to check with the team. "
        "Help the customer add order items to their cart."
        "Tell i would happy to help them with their order."
        "Reply with emojis and a friendly tone. Don't invent anything. "
        "message of the conversation, so repeating it here would look like the bot is re-introducing "
        "itself on every message. Just answer the question directly.\n"
        f"{language_instruction(state.get('customer_language'))} Keep it short.\n\n"
        f"{GENERAL_INFO_TEXT}"
    )
    reply = llm.chat([{"role": "system", "content": system}] + recent_context(state))
    return {"reply": reply, "grounding_context": GENERAL_INFO_TEXT}


def order_support_node(state: AgentState) -> dict:
    """Handles order status, complaints, order history, and re-order requests.

    Sub-intent classification decides which DB query to run before calling the LLM:
      current_cart  → customer has live items not yet placed; describe what's in it
      latest_order  → status/tracking question about the most recent placed order
      order_history → customer wants a list of past orders
      reorder       → customer wants to repeat a previous order; populate cart from DB
      complaint     → something went wrong; escalate if needed
    """
    phone = state.get("customer_phone") or state["phone"]
    cart  = state.get("cart") or []
    lang  = state.get("customer_language")

    # ── RC-08: classify sub-intent ─────────────────────────────────────────────
    sub_system = (
        "Classify the customer's latest message for a kosher fish/meat shop into ONE of:\n"
        '- "current_cart"  : asking what is currently in their cart / what they are ordering right now\n'
        '- "latest_order"  : asking about the status/delivery of a specific past order\n'
        '- "order_history" : wants a list of their previous orders\n'
        '- "reorder"       : wants to repeat / re-add a previous order to the cart\n'
        '- "complaint"     : reporting a problem, wrong item, late delivery, etc.\n'
        'Reply JSON: {"sub_intent": "<one of the five labels>"}'
    )
    sub_result = llm.chat_json(
        [{"role": "system", "content": sub_system}] + recent_context(state)
    )
    sub = sub_result.get("sub_intent", "latest_order")
    logger.info("order_support sub_intent=%s for phone=%s", sub, state["phone"])

    # ── Build order_context for the LLM reply ─────────────────────────────────

    extra_state_update = {}   # additional state fields to return (e.g. updated cart)

    if sub == "current_cart":
        # RC-03: always describe live cart first
        if cart:
            cart_total = sum((i.get("price") or 0) * (i.get("quantity") or 0) for i in cart)
            order_context = (
                f"Active cart (NOT yet placed):\n{_format_cart(cart)}\n"
                f"Cart total: {cart_total:.2f} ILS.\n"
                "This cart has not been confirmed yet — it will be placed at checkout."
            )
        else:
            order_context = "The customer's cart is currently empty."

    elif sub == "order_history":
        # RC-06: fetch all recent orders with items
        all_orders = orders_repo.get_all_orders_for_phone(phone)
        if all_orders:
            lines = []
            for idx, o in enumerate(all_orders, 1):
                items_txt = ", ".join(f"{i['name']} x{i['quantity']}" for i in o.get("items") or [])
                placed = o.get("created_at", "")[:16].replace("T", " ")
                lines.append(
                    f"{idx}. Placed: {placed} | Status: {o['status']} | "
                    f"Total: {o['total']} ILS | Items: {items_txt or 'unknown'}"
                )
            order_context = "Customer's order history (most recent first):\n" + "\n".join(lines)
        else:
            order_context = "No previous orders found for this customer."

    elif sub == "reorder":
        # RC-07: fetch previous order items and merge into current cart
        reorder_items = orders_repo.get_reorder_cart_items(phone)
        if reorder_items:
            # Merge: keep existing cart items, add re-ordered items on top
            merged = list(cart) + reorder_items
            extra_state_update["cart"] = merged
            extra_state_update["flow_stage"] = "browsing"
            items_txt = _format_cart(reorder_items)
            cart_total = sum((i.get("price") or 0) * (i.get("quantity") or 0) for i in merged)
            order_context = (
                f"Re-order requested. These items from the customer's last order were added to cart:\n"
                f"{items_txt}\n"
                f"Updated cart total: {cart_total:.2f} ILS.\n"
                "Ask the customer to confirm or modify before checkout."
            )
        else:
            order_context = "No previous order found to re-order from."

    elif sub == "complaint":
        # Complaints: always fetch latest placed order as context
        latest = orders_repo.get_latest_order_for_phone(phone)
        if latest:
            items_txt = ", ".join(f"{i['name']} x{i['quantity']}" for i in latest.get("items") or [])
            order_context = (
                f"Most recent order: id={latest['id']}, status={latest['status']}, "
                f"placed_at={latest.get('created_at','')[:16]}, total={latest['total']} ILS, "
                f"items: {items_txt or 'unknown'}."
            )
        elif cart:
            order_context = f"Customer has active cart (not yet placed):\n{_format_cart(cart)}"
        else:
            order_context = "No order or cart found for this customer."

    else:
        # latest_order (default): RC-03 — live cart first, then DB
        if cart:
            cart_total = sum((i.get("price") or 0) * (i.get("quantity") or 0) for i in cart)
            order_context = (
                f"The customer has an active cart in this session (NOT yet placed as an order):\n"
                f"{_format_cart(cart)}\n"
                f"Cart total: {cart_total:.2f} ILS.\n"
                "This cart has not been confirmed yet."
            )
        else:
            latest = orders_repo.get_latest_order_for_phone(phone)
            if latest:
                items_txt = ", ".join(f"{i['name']} x{i['quantity']}" for i in latest.get("items") or [])
                order_context = (
                    f"Most recent order: id={latest['id']}, status={latest['status']}, "
                    f"placed_at={latest.get('created_at','')[:16]}, total={latest['total']} ILS, "
                    f"items: {items_txt or 'unknown'}.\n"
                    "Note: status 'pending_sync' means the order was recorded by the WhatsApp bot "
                    "but hasn't been confirmed in the shop's system yet."
                )
            else:
                order_context = "No orders found for this phone number."

    # ── LLM reply ─────────────────────────────────────────────────────────────
    system = (
        "You are a WhatsApp assistant for Gold Fish (kosher fish/meat shop). "
        "Use ONLY the order information provided below — do not invent details. "
        "Don't promise refunds or compensation yourself; if you can't resolve it, say a team member will follow up and set needs_human=true. "
        f"{language_instruction(lang)} Keep replies short and friendly.\n\n"
        f"ORDER INFO:\n{order_context}\n\n"
        'Reply JSON: {"reply": "<your reply>", "needs_human": true/false, '
        '"reason": "<only if needs_human=true>"}'
    )
    result = llm.chat_json([{"role": "system", "content": system}] + recent_context(state))
    reply  = result.get("reply") or SAFE_FALLBACK_REPLY

    if result.get("needs_human"):
        escalation.notify_staff(state["phone"], state["customer_name"], result.get("reason", "order support"))

    return {"reply": reply, "grounding_context": order_context, **extra_state_update}


def _is_new_order_request(state: AgentState) -> bool:
    """True when the customer explicitly asks to start a fresh order from scratch.

    Only fires when there is something to reset (a non-empty cart or an
    active checkout stage). 'New order' on an empty cart in browsing is
    just a normal order start and should reach _stage_browsing as usual.
    """
    stage = state.get("turn_start_stage") or state.get("flow_stage") or "browsing"
    has_active_order = bool(state.get("cart")) or stage != "browsing"
    if not has_active_order:
        return False
    text = last_user_text(state).lower()
    return any(phrase in text for phrase in _NEW_ORDER_PHRASES)


def _handle_new_order_request(state: AgentState) -> dict:
    """Resets the active order and replies with a fresh-start acknowledgment."""
    categories_data = products_repo.list_categories()
    categories_text = "\n".join(f"- {c['category_name']}" for c in categories_data)

    system = (
        "The customer explicitly asked to start a brand-new order, cancelling whatever was in "
        "their cart. Acknowledge this with one short friendly line (e.g. 'Okay! Let's start "
        "fresh.'), then immediately ask what they'd like to order and list the available "
        "categories so they can pick one. Keep it brief — this is WhatsApp.\n"
        f"{language_instruction(state.get('customer_language'))}\n\n"
        f"Available categories:\n{categories_text}"
    )
    try:
        reply = llm.chat([{"role": "system", "content": system}] + recent_context(state))
    except Exception as exc:
        logger.error("_handle_new_order_request LLM call failed: %s", exc)
        reply = "בסדר! נתחיל מחדש 😊 מה תרצה להזמין?"

    logger.info("order_taking: phone=%s requested new order — resetting cart and flow", state["phone"])
    return {
        "reply": reply,
        "cart": [],
        "flow_stage": "browsing",
        "delivery_method": None,
        "delivery_address": None,
        "shipping_method": None,
        "browsing_category": None,
        "browsing_parent_id": None,
        "grounding_context": (
            f"Customer requested a fresh start — cart cleared, flow reset to browsing.\n"
            f"Available categories:\n{categories_text}"
        ),
    }


def order_taking_node(state: AgentState) -> dict:
    """Dispatches to whichever stage of the structured order flow this
    customer is currently in:

        browsing -> choosing_delivery -> [collecting_address] ->
        [collecting_contact] -> confirming_summary -> (order placed) -> browsing

    [collecting_address] is skipped for pickup orders.
    [collecting_contact] (name + phone, asked explicitly - never taken from
    the WhatsApp account itself) is skipped only if we already collected
    both earlier THIS conversation.
    """
    # If an order was already placed earlier THIS turn (e.g. the guardrail
    # rejected our first attempt at phrasing the confirmation and sent us
    # back here), do NOT decide anything again or place a second order -
    # just repeat the confirmation we already generated. The order is a
    # real, already-happened side effect; it must never be silently dropped.
    if state.get("order_just_created"):
        order = state["order_just_created"]
        return {
            "reply": order["confirmation_message"],
            "grounding_context": (
                f"An order was already placed earlier this turn - items={order['items']}, "
                f"total={order['total']} ILS. Just repeat that confirmation, nothing else to decide."
            ),
        }

    # If the customer explicitly asks to start a new order from scratch,
    # reset the cart and flow before doing anything else — regardless of
    # which stage they were in.
    if _is_new_order_request(state):
        return _handle_new_order_request(state)

    # Use the snapshot taken at the start of this turn, not the live
    # flow_stage - see the comment on turn_start_stage in state.py.
    stage = state.get("turn_start_stage") or "browsing"
    logger.info("order_taking: phone=%s stage=%s", state["phone"], stage)

    if stage == "choosing_delivery":
        return _stage_choosing_delivery(state)
    if stage == "collecting_address":
        return _stage_collecting_address(state)
    if stage == "collecting_contact":
        return _stage_collecting_contact(state)
    if stage == "confirming_summary":
        return _stage_confirming_summary(state)
    if stage == "choosing_parent":
        return _stage_choosing_parent(state)
    return _stage_browsing(state)


def _stage_browsing(state: AgentState) -> dict:
    """Customer is picking products and building their cart."""

    # Fast path: if we were waiting for quantity on a specific product and the
    # customer just replied with a bare number, skip the full catalog search and
    # add it directly. This prevents the re-search from picking the wrong product
    # when conversation history is long.
    pending = state.get("pending_product")
    if pending:
        raw = last_user_text(state).strip().replace(",", ".")
        try:
            qty = float(raw)
        except ValueError:
            qty = 0
        if qty > 0 and raw.replace(".", "", 1).replace("-", "", 1).lstrip("-").replace(".", "").isdigit():
            qty = int(qty) if qty == int(qty) else qty
            unit = pending.get("unit_type") or "unit"
            cart_item = {
                "product_id": pending["id"],
                "name": pending["name"],
                "price": pending["price"],
                "quantity": qty,
                "unit_type": unit,
            }
            cart = list(state.get("cart_at_turn_start") or state.get("cart") or []) + [cart_item]
            confirm_system = (
                f"Customer added {qty} {unit} of {pending['name']} (₪{pending['price']}/{unit}) to their cart. "
                f"Confirm briefly and ask if they'd like anything else or to confirm the order. "
                f"{language_instruction(state.get('customer_language'))} One short natural line."
            )
            reply = llm.chat([{"role": "system", "content": confirm_system}] + recent_context(state))
            logger.info(
                "browsing: pending_product shortcut — added %r x%s for phone=%s",
                pending["name"], qty, state["phone"],
            )
            return {
                "reply": reply,
                "cart": cart,
                "pending_product": None,
                "browsing_category": None,
                "browsing_parent_id": None,
                "is_first_message": False,
                "grounding_context": (
                    f"Added {qty} {unit} of {pending['name']} (₪{pending['price']}/{unit}) to cart. "
                    f"Cart now: {_format_cart(cart)}"
                ),
            }

    categories_data = products_repo.list_categories()
    categories = [c["category_name"] for c in categories_data]
    terms = extract_query_terms(recent_context(state), categories)
    search_term = terms.get("search_term") or None
    category = terms.get("category") or None

    # Greeting note injected into the LLM prompt when this is the first
    # turn of the session - the node owns the greeting so runner.py doesn't
    # prepend a separate welcome that would make two openers back-to-back.
    is_first = state.get("is_first_message")
    name = state.get("customer_name") or ""
    name_part = f" for {name}" if name else ""
    if is_first and not search_term and not category:
        # Pure greeting ("Hi") or vague intent ("I want to order") — respond naturally
        # without robotically listing categories or using order-bot phrasing.
        first_msg_note = (
            f"IMPORTANT: This is the customer's first message. Greet them warmly{name_part} "
            "and ask how you can help today. Do NOT list categories or push toward ordering "
            "unless they specifically mentioned a product. Keep it natural — just a friendly hello.\n"
        )
    elif is_first:
        first_msg_note = (
            f"IMPORTANT: This is the customer's first message. Start with a short warm "
            f"greeting{name_part}, then immediately continue with the rest of your response "
            "— write it as ONE natural message, not two separate ones.\n"
        )
    else:
        first_msg_note = ""
    # Shared extra fields to clear transient flags from the returned state.
    first_msg_clear = {"is_first_message": False} if is_first else {}

    # --- Category-only browsing: show parent products (kashrut / fish type) ---
    # When the customer names a category but not a specific product, fetch
    # parent products first. These are the second-level groupings (e.g.
    # "לנדאו" / "שייקביץ" for poultry, or "לברק" / "מושט" for fresh fish).
    # Only if no parents exist do we fall through to showing all products.
    if category and not search_term:
        parent_products = products_repo.get_parent_products(category)
        if parent_products:
            parents_text = "\n".join(
                f"- {p['name']}{' (' + p['hechsher'] + ')' if p.get('hechsher') else ''}"
                for p in parent_products
            )
            system_parents = (
                f"{first_msg_note}"
                f"You are a WhatsApp assistant for Gold Fish, a kosher fish/meat shop. "
                f"The customer wants to browse {category}. "
                f"Show them these available options and ask which they'd prefer:\n{parents_text}\n"
                f"Keep it brief — this is WhatsApp. {language_instruction(state.get('customer_language'))}"
            )
            reply = llm.chat([{"role": "system", "content": system_parents}] + recent_context(state))
            grounding_context = f"Options shown for {category}:\n{parents_text}"
            logger.info("order_taking: phone=%s category=%r has %d parent products, entering choosing_parent",
                        state["phone"], category, len(parent_products))
            return {
                **first_msg_clear,
                "reply": reply,
                "flow_stage": "choosing_parent",
                "browsing_category": category,
                "browsing_parent_id": None,
                "grounding_context": grounding_context,
            }
    # --- End category-only path ---

    if not search_term and not category:
        # Nothing specific named yet (e.g. "I want to order something") -
        # proactively show the menu instead of just asking an open-ended
        # "what would you like?" and waiting for the customer to ask for
        # it separately.
        candidates = []
        candidates_text = "Customer hasn't named a specific item yet - categories available:\n" + "\n".join(
            f"- {c['category_name']}" for c in categories_data
        )
    else:
        raw_candidates = products_repo.search_products(query=search_term, category=category)
        # Separate navigational group products (have children) from orderable leaf products.
        # Groups cannot be added to the cart directly — they require further selection.
        if raw_candidates:
            group_ids = products_repo.check_has_children([p["id"] for p in raw_candidates])
            candidates = [p for p in raw_candidates if p["id"] not in group_ids]
            group_candidates = [p for p in raw_candidates if p["id"] in group_ids]
        else:
            group_ids, candidates, group_candidates = set(), [], []

        # If ONLY groups were found (customer named a navigational group directly, e.g.
        # "Bistritzky cut-up"), navigate into the best match rather than trying to add
        # it to cart.
        if group_candidates and not candidates:
            if len(group_candidates) == 1:
                # Single clear match — navigate directly into that group
                nav_id = group_candidates[0]["id"]
            else:
                # Multiple groups — go to the common parent so the customer sees all siblings
                parent_ids = [g.get("parent_product_id") for g in group_candidates if g.get("parent_product_id")]
                nav_id = parent_ids[0] if parent_ids else group_candidates[0]["id"]
            nav_category = group_candidates[0].get("category") or category or ""
            logger.info(
                "browsing: phone=%s search matched only groups, navigating to id=%r",
                state["phone"], nav_id,
            )
            nav_system = (
                f"{first_msg_note}"
                f"You are a WhatsApp assistant for Gold Fish. The customer asked for something specific "
                f"— you're about to show their options. "
                + ("Write a very short reply that embeds the greeting and leads naturally into showing options."
                   if first_msg_note else "Write one short bridging sentence in the right tone.")
                + f" {language_instruction(state.get('customer_language'))}"
            )
            nav_reply = llm.chat([{"role": "system", "content": nav_system}] + recent_context(state))
            return {
                **first_msg_clear,
                "reply": nav_reply,
                "flow_stage": "choosing_parent",
                "browsing_parent_id": nav_id,
                "browsing_category": nav_category,
                "grounding_context": f"Navigating product hierarchy, target group id={nav_id}",
            }

        # Show leaf candidates to the cart LLM; note any groups as non-orderable sub-categories
        groups_note = (
            "Note: the following are sub-categories (NOT directly orderable — need further selection): "
            + ", ".join(f'"{g["name"]}"' for g in group_candidates)
            + ". Do NOT add these to the cart.\n\n"
            if group_candidates else ""
        )
        candidates_text = groups_note + ("\n".join(
            f'- id="{p["id"]}" name="{p["name"]}" price={p["price"]} ILS/{p.get("unit_type") or "unit"}'
            for p in candidates
        ) or "(no matching products found)")

    # Build from the frozen turn-start snapshot, not the live state["cart"] -
    # see the comment on cart_at_turn_start in state.py for why: a
    # guardrail-rejected retry re-runs this same function, and the live
    # cart would already include the PREVIOUS (unapproved) attempt's
    # speculative additions, silently duplicating items.
    # NOTE: don't use `state.get(...) or state["cart"]` here - an empty
    # list is a perfectly valid, meaningful snapshot (cart started empty
    # this turn) but Python treats it as falsy, which would silently fall
    # through to the live (wrong) cart and reintroduce the exact
    # duplication bug this snapshot exists to prevent.
    cart_baseline = state["cart_at_turn_start"]
    cart_text = _format_cart(cart_baseline)

    system = (
        f"{first_msg_note}"
        "You are taking a WhatsApp order for Gold Fish, a kosher fish/meat shop. Decide what to do "
        "with the customer's cart based on their latest message.\n\n"
        f"Current cart:\n{cart_text}\n\n"
        f"Products matching their latest message:\n{candidates_text}\n\n"
        "Rules:\n"
        "- Only add products from the matching-products list above, using their exact id/name/price. "
        "Never invent a product or price.\n"
        "- If their request is ambiguous (multiple matches, or no quantity given), ask a clarifying "
        'question instead of guessing - use action "clarify".\n'
        "- If the products list above says the customer hasn't named anything specific yet, use action "
        '"clarify" and reply by listing those categories (briefly, e.g. as a short list) and asking which '
        "one interests them - this is their first hint they want to order, so show them what's available "
        "instead of just asking an open-ended question.\n"
        '- If they confirm they want to place the order (e.g. "that\'s all", "confirm"), use action '
        '"ready_to_checkout" - but ONLY if the cart above is not empty. If the cart is empty, there is '
        "nothing to confirm: tell them their cart is empty and ask what they'd like to order instead "
        '(use action "clarify"). If using "ready_to_checkout", your reply should briefly acknowledge '
        "they're done, THEN ask whether they want pickup or delivery, in the SAME reply.\n"
        '- Otherwise use action "update_cart" and list items to add (with quantity) and/or remove.\n\n'
        'Reply JSON: {"action": "update_cart"|"clarify"|"ready_to_checkout", '
        '"add_items": [{"product_id": "...", "name": "...", "price": 0, "quantity": 1}], '
        f'"remove_product_ids": ["..."], "reply": "<your reply to the customer — use human-readable product names, NEVER raw product IDs or UUIDs. {language_instruction(state.get("customer_language"))}>"}}'
    )
    result = llm.chat_json([{"role": "system", "content": system}] + recent_context(state))

    # Don't just trust the prompt's "never invent a product" instruction -
    # the LLM did exactly that in production (added an item using the
    # product's NAME as its product_id, instead of leaving it alone, when
    # a short reply like "1" matched no real candidate). Drop anything
    # that isn't a real candidate id before it ever reaches the cart.
    real_ids = {p["id"] for p in candidates}
    product_map = {p["id"]: p for p in candidates}
    add_items = result.get("add_items") or []
    valid_add_items, rejected_add_items = [], []
    for item in add_items:
        pid = item.get("product_id")
        if pid in real_ids:
            # Always overwrite price and unit_type from the real catalog record —
            # the LLM sometimes copies the placeholder "price": 0 from the prompt
            # example, or echoes "price": null when the product has no price set.
            real = product_map[pid]
            item["price"] = real.get("price") or 0
            item["unit_type"] = real.get("unit_type") or "unit"
            valid_add_items.append(item)
        else:
            rejected_add_items.append(item)
    if rejected_add_items:
        logger.warning(
            "order_taking: dropped %d hallucinated add_items (not in real candidates) for phone=%s: %r",
            len(rejected_add_items), state["phone"], rejected_add_items,
        )

    cart = list(cart_baseline) + valid_add_items
    remove_ids = set(result.get("remove_product_ids") or [])
    if remove_ids:
        cart = [item for item in cart if item.get("product_id") not in remove_ids]

    reply = result.get("reply") or "איזה מוצר ובאיזו כמות תרצה?"
    # Cart AFTER this turn's add/remove, not the pre-turn snapshot - the
    # guardrail checks "I added X to your cart" claims against this, and it
    # can't verify an addition actually happened if the facts only show the
    # cart from before the addition.
    grounding_context = (
        f"Matching products (names are in Hebrew — translating them to the customer's language is valid):\n"
        f"{candidates_text}\n\nCart after this turn's update:\n{_format_cart(cart)}"
    )

    # If exactly one leaf product was shown but not yet added (LLM is asking
    # for quantity), remember it so the next bare-number reply skips re-search.
    if len(candidates) == 1 and not valid_add_items:
        p = candidates[0]
        new_pending = {"id": p["id"], "name": p["name"], "price": p["price"], "unit_type": p.get("unit_type") or "unit"}
        logger.info("browsing: set pending_product=%r for phone=%s", p["name"], state["phone"])
    else:
        new_pending = None

    if result.get("action") == "ready_to_checkout" and cart:
        # The system prompt above already told the LLM to ask pickup-or-
        # delivery as part of this SAME reply (instead of a separate
        # follow-up llm.chat call) - a second, independently-prompted call
        # was actually losing track of the customer's current language
        # (e.g. they switched to English, but a fresh call with a shorter
        # system prompt and no first-hand view of the decision drifted
        # back to Hebrew). Reusing `reply` keeps the language consistent.
        return {
            **first_msg_clear,
            "reply": reply,
            "cart": cart,
            "flow_stage": "choosing_delivery",
            "browsing_category": None,
            "browsing_parent_id": None,
            "pending_product": None,
            "grounding_context": grounding_context,
        }

    return {
        **first_msg_clear,
        "reply": reply,
        "cart": cart,
        "browsing_category": None,
        "browsing_parent_id": None,
        "pending_product": new_pending,
        "grounding_context": grounding_context,
    }


def _stage_choosing_parent(state: AgentState) -> dict:
    """Navigate a multi-level product hierarchy until the customer picks an orderable leaf.

    Uses browsing_parent_id to track the current navigation position:
      - None  → show category-level roots (brand selection, depth 0)
      - set   → show children of that specific product (depth 1, 2, 3 …)

    The Bistritzky poultry tree has 4 levels:
      brand root → whole/cut-up group → size variant group → processing leaf
    This function stays in 'choosing_parent' until a leaf (orderable product) is reached.
    """
    # Guard: if the cart was already modified this turn, the customer's selection was
    # processed. Handle guardrail retries by regenerating the confirmation instead of
    # re-navigating the hierarchy (which would use stale/cleared state and hallucinate).
    cart_this_turn = list(state.get("cart") or [])
    cart_at_start = list(state.get("cart_at_turn_start") or [])
    if len(cart_this_turn) > len(cart_at_start):
        last_added = cart_this_turn[-1]
        confirm_system = (
            f"The customer added {last_added['quantity']}x {last_added['name']} "
            f"(₪{last_added['price']}) to their cart. "
            f"Confirm this briefly and ask if they want anything else or to go to checkout. "
            f"{language_instruction(state.get('customer_language'))} Keep it very brief."
        )
        reply = llm.chat([{"role": "system", "content": confirm_system}] + recent_context(state))
        logger.info(
            "choosing_parent: cart already modified this turn (guardrail retry?), repeating confirmation for phone=%s",
            state["phone"],
        )
        return {
            "reply": reply,
            "flow_stage": "browsing",
            "browsing_category": None,
            "browsing_parent_id": None,
            "grounding_context": (
                f"Item already added this turn: {last_added['quantity']}x "
                f"{last_added['name']} at ₪{last_added['price']}."
            ),
        }

    parent_id = state.get("browsing_parent_id")
    category = state.get("browsing_category") or ""

    # Fetch the products at the current navigation level
    if parent_id:
        current_products = products_repo.get_child_products(parent_id)
        parent_record = products_repo.get_product_by_id(parent_id)
        context_name = (parent_record or {}).get("name") or category
    else:
        current_products = products_repo.get_parent_products(category)
        context_name = category

    if not current_products:
        logger.warning(
            "choosing_parent: no products for parent_id=%r category=%r, falling back to browsing",
            parent_id, category,
        )
        fallback = _stage_browsing(state)
        fallback.setdefault("flow_stage", "browsing")
        fallback["browsing_category"] = None
        fallback["browsing_parent_id"] = None
        return fallback

    # Classify: groups (have children → navigation only) vs. leaves (orderable → add to cart)
    group_ids = products_repo.check_has_children([p["id"] for p in current_products])
    leaf_products = [p for p in current_products if p["id"] not in group_ids]
    group_products = [p for p in current_products if p["id"] in group_ids]

    # Build a clear options list for the LLM identification step
    options_lines = []
    for p in group_products:
        options_lines.append(
            f'- id="{p["id"]}" name="{p["name"]}" [sub-category — NOT orderable directly, has further options]'
        )
    for p in leaf_products:
        options_lines.append(
            f'- id="{p["id"]}" name="{p["name"]}" price={p["price"]} ILS/{p.get("unit_type") or "unit"} [orderable]'
        )
    options_text = "\n".join(options_lines)

    # Ask the LLM to identify what the customer chose from the current level
    identify_system = (
        f"The customer is browsing: {context_name}\n\n"
        f"Available options:\n{options_text}\n\n"
        "Based on their latest message and the full conversation context, identify what they picked.\n"
        "- 'option 1', 'the first one', 'the second', a partial name, or a matching keyword all count.\n"
        "- If they picked an [orderable] item, also return their quantity (0 if not mentioned).\n"
        "- If genuinely unclear, set unclear to true.\n"
        'Reply JSON: {"chosen_id": "<exact id from list above, or empty>", '
        '"chosen_name": "<name of chosen item, or empty>", "quantity": 0, "unclear": false}'
    )
    id_result = llm.chat_json([{"role": "system", "content": identify_system}] + recent_context(state))
    chosen_id = (id_result.get("chosen_id") or "").strip()
    chosen_name_hint = (id_result.get("chosen_name") or "").strip()

    # Validate that the LLM returned a real product ID — LLMs sometimes mangle UUIDs slightly.
    # Fall back to name matching before giving up and asking the customer to pick again.
    real_ids_set = {p["id"] for p in current_products}
    if chosen_id and chosen_id not in real_ids_set:
        name_match = None
        if chosen_name_hint:
            name_match = next(
                (p for p in current_products
                 if p["name"].lower() == chosen_name_hint.lower()
                 or chosen_name_hint.lower() in p["name"].lower()),
                None,
            )
        if name_match:
            logger.warning(
                "choosing_parent: LLM returned invalid id=%r, matched by name=%r -> %r",
                chosen_id, chosen_name_hint, name_match["id"],
            )
            chosen_id = name_match["id"]
        else:
            logger.warning(
                "choosing_parent: LLM returned invalid id=%r (no name fallback) — treating as unclear",
                chosen_id,
            )
            chosen_id = ""

    if not chosen_id or id_result.get("unclear"):
        # Still unclear — show the options and ask the customer to pick
        display_system = (
            f"You are a WhatsApp assistant for Gold Fish. Show the customer their options under "
            f"{context_name} and ask them to choose. Keep it brief — this is WhatsApp.\n"
            f"Sub-categories (tap to see more options): "
            + (", ".join(p["name"] for p in group_products) or "none")
            + "\nOrderable products: "
            + (", ".join(f"{p['name']} ₪{p['price']}/{p.get('unit_type') or 'unit'}" for p in leaf_products) or "none")
            + f"\n{language_instruction(state.get('customer_language'))}"
        )
        reply = llm.chat([{"role": "system", "content": display_system}] + recent_context(state))
        return {
            "reply": reply,
            "grounding_context": f"Showing options for {context_name}:\n{options_text}",
        }

    chosen_product = next((p for p in current_products if p["id"] == chosen_id), None)
    chosen_name = (chosen_product or {}).get("name") or chosen_id

    if chosen_id in group_ids:
        # Customer picked a sub-category — navigate one level deeper
        logger.info(
            "choosing_parent: phone=%s chose group %r (%s), navigating deeper",
            state["phone"], chosen_id, chosen_name,
        )
        # Immediately fetch and display the children of the chosen group
        children = products_repo.get_child_products(chosen_id)
        child_group_ids = products_repo.check_has_children([c["id"] for c in children]) if children else set()
        child_leaves = [c for c in children if c["id"] not in child_group_ids]
        child_groups = [c for c in children if c["id"] in child_group_ids]

        child_lines = []
        for c in child_groups:
            child_lines.append(f"- {c['name']} (has more options)")
        for c in child_leaves:
            child_lines.append(f"- {c['name']} — ₪{c['price']}/{c.get('unit_type') or 'unit'}")
        child_options_text = "\n".join(child_lines) or "(no further options found)"

        show_system = (
            f"You are a WhatsApp assistant for Gold Fish. The customer chose {chosen_name}. "
            f"Show them what's available and ask them to pick one. Keep it concise.\n"
            f"{language_instruction(state.get('customer_language'))}\n\n"
            f"Options under {chosen_name}:\n{child_options_text}"
        )
        reply = llm.chat([{"role": "system", "content": show_system}] + recent_context(state))

        return {
            "reply": reply,
            "flow_stage": "choosing_parent",
            "browsing_parent_id": chosen_id,
            "grounding_context": f"Navigated into {chosen_name}. Options:\n{child_options_text}",
        }

    # Customer picked an orderable leaf product
    price = (chosen_product or {}).get("price") or 0
    unit = (chosen_product or {}).get("unit_type") or "unit"
    quantity = id_result.get("quantity") or 0

    logger.info(
        "choosing_parent: phone=%s chose leaf %r (%s), qty=%s",
        state["phone"], chosen_id, chosen_name, quantity,
    )

    if quantity and int(quantity) > 0:
        # Quantity known — add to cart and return to browsing
        cart_item = {
            "product_id": chosen_id,
            "name": chosen_name,
            "price": price,
            "quantity": int(quantity),
            "unit_type": unit,
        }
        cart = list(state.get("cart_at_turn_start") or state.get("cart") or []) + [cart_item]

        confirm_system = (
            f"The customer just added {int(quantity)}x {chosen_name} (₪{price}/{unit}) to their cart. "
            f"Confirm this warmly and ask if they'd like anything else or want to go to checkout. "
            f"{language_instruction(state.get('customer_language'))} Keep it brief."
        )
        reply = llm.chat([{"role": "system", "content": confirm_system}] + recent_context(state))

        return {
            "reply": reply,
            "cart": cart,
            "flow_stage": "browsing",
            "browsing_category": None,
            "browsing_parent_id": None,
            "grounding_context": f"Added {int(quantity)}x {chosen_name} to cart. Price: ₪{price}/{unit}.",
        }
    else:
        # Leaf chosen but no quantity — ask how many and return to browsing.
        # The next turn's _stage_browsing handles the quantity via extract_query_terms,
        # which reads the conversation context to know which product is being quantified.
        ask_system = (
            f"The customer wants {chosen_name} (₪{price}/{unit}). "
            f"Ask how many they'd like. One short line only. "
            f"{language_instruction(state.get('customer_language'))}"
        )
        reply = llm.chat([{"role": "system", "content": ask_system}] + recent_context(state))

        return {
            "reply": reply,
            "flow_stage": "browsing",
            "browsing_category": None,
            "browsing_parent_id": None,
            "pending_product": {"id": chosen_id, "name": chosen_name, "price": price, "unit_type": unit},
            "grounding_context": (
                f"Customer chose {chosen_name} (₪{price}/{unit}) from the product hierarchy. "
                f"Asking for quantity. The pending_product shortcut will handle their reply."
            ),
        }


# Shared by every "answer this one pending question" stage below
# (choosing_delivery / collecting_address / collecting_name /
# confirming_summary). Real customers often say something else entirely
# mid-checkout - try to add one more item, ask about a past order, ask an
# unrelated question. Without this, the stage handler just re-asks its own
# question and ignores what they actually said, which the guardrail then
# (correctly) rejects for not addressing their message - this caused a real
# stuck-conversation pattern in production. wants_more_items lets a stage
# hand control back to browsing instead of getting stuck mid-checkout.
def off_topic_instruction(lang: str | None = None) -> str:
    """Instruction appended to mid-checkout stage prompts so the LLM handles
    off-topic messages (add item, change order, etc.) without getting stuck.
    Takes the stored customer language so the response stays in the right language.
    """
    return (
        "If their latest message does NOT actually answer this pending question (e.g. they ask something "
        "else, or want to add/change products), set wants_more_items to true ONLY if they're clearly "
        "trying to add or change items - otherwise leave it false. Either way, write a reply that FIRST "
        "briefly acknowledges what they actually said - without inventing any facts you don't have (e.g. "
        "if they ask about a past order, just say you'll help with that right after) - THEN, unless "
        "wants_more_items is true, still ask this pending question again so we can finish the current "
        f"order. If wants_more_items is true, just acknowledge you'll let them add more items first. "
        f"{language_instruction(lang)}"
    )


def _has_contact_details(state: AgentState) -> bool:
    """Whether we've already explicitly collected a name AND phone number
    for this order THIS conversation - never satisfied by WhatsApp's own
    account info, only by _stage_collecting_contact actually asking."""
    return bool(state.get("customer_name")) and bool(state.get("customer_phone"))


def _stage_choosing_delivery(state: AgentState) -> dict:
    """Customer is answering "pickup or delivery?"."""
    system = (
        'The customer was just asked whether they want "pickup" or "delivery" for their current '
        'order. Decide what their latest message means: "pickup", "delivery", or "unclear".\n'
        f"{off_topic_instruction(state.get('customer_language'))}\n"
        'Reply JSON: {"choice": "pickup"|"delivery"|"unclear", "wants_more_items": true/false, '
        '"reply": "<see instructions above>"}'
    )
    result = llm.chat_json([{"role": "system", "content": system}] + recent_context(state))
    choice = result.get("choice")
    reply = result.get("reply") or "איסוף עצמי או משלוח?"
    grounding_context = "The customer was asked to choose pickup or delivery."

    if result.get("wants_more_items"):
        logger.info("order_taking: phone=%s wants to add more items mid-checkout - back to browsing", state["phone"])
        browsing_result = _stage_browsing(state)
        browsing_result.setdefault("flow_stage", "browsing")
        return browsing_result

    if choice not in ("pickup", "delivery"):
        return {"reply": reply, "grounding_context": grounding_context}

    if choice == "pickup":
        next_stage = "confirming_summary" if _has_contact_details(state) else "collecting_contact"
        return {
            "reply": reply, "delivery_method": "pickup", "flow_stage": next_stage,
            "grounding_context": grounding_context,
        }

    # For delivery: skip address collection if we already have it from registration
    if state.get("delivery_address"):
        next_stage = "confirming_summary" if _has_contact_details(state) else "collecting_contact"
        return {"reply": reply, "delivery_method": "delivery", "flow_stage": next_stage,
                "grounding_context": grounding_context}

    return {"reply": reply, "delivery_method": "delivery", "flow_stage": "collecting_address",
            "grounding_context": grounding_context}


def _stage_collecting_address(state: AgentState) -> dict:
    """Customer is giving their delivery address (city/street/building)."""
    system = (
        "Extract a delivery address (city, street, building number) from the customer's message(s), "
        "for their current order. Israeli address - the catalog/business is in Beitar Illit, customers "
        "may be there or nearby cities (Beit Shemesh, Jerusalem, Tzur Hadassah, etc).\n"
        f"{off_topic_instruction(state.get('customer_language'))}\n"
        'Reply JSON: {"city": "...", "street": "...", "building_number": "...", "complete": true/false, '
        '"wants_more_items": true/false, "reply": "<if complete, a short confirmation; if not complete, '
        "ask for exactly what's missing or follow the instructions above>\"}"
    )
    result = llm.chat_json([{"role": "system", "content": system}] + recent_context(state))
    reply = result.get("reply") or "מה כתובת המשלוח? (עיר, רחוב, מספר בית)"

    if result.get("wants_more_items"):
        logger.info("order_taking: phone=%s wants to add more items mid-checkout - back to browsing", state["phone"])
        browsing_result = _stage_browsing(state)
        browsing_result.setdefault("flow_stage", "browsing")
        return browsing_result

    if not result.get("complete"):
        return {"reply": reply, "grounding_context": "Collecting a delivery address - not complete yet."}

    address = {
        "city": result.get("city", ""),
        "street": result.get("street", ""),
        "building_number": result.get("building_number", ""),
    }
    zone = shipping_repo.find_delivery_zone_for_city(address["city"])
    next_stage = "confirming_summary" if _has_contact_details(state) else "collecting_contact"
    grounding_context = (
        f"Delivery address collected: {address}. "
        + (f"Matched delivery zone: {zone['name']}, cost {zone['price']} ILS." if zone
           else "No delivery zone matched this city - cost will be confirmed by staff.")
    )
    return {
        "reply": reply, "delivery_address": address, "shipping_method": zone,
        "flow_stage": next_stage, "grounding_context": grounding_context,
    }


def _stage_collecting_contact(state: AgentState) -> dict:
    """Customer is giving their name AND a contact phone number for this
    order. Always asked explicitly here, even though we technically know
    the WhatsApp account's number and profile name - neither is reliable
    enough to put on a real order (a shared family phone, ordering for
    someone else, a WhatsApp nickname instead of a real name, etc), so the
    customer is asked directly and THESE values (not anything from the
    webhook) are what get saved to Supabase."""
    system = (
        "The customer was just asked for their full name and a contact phone number, to put on their "
        "current order - we deliberately do NOT use their WhatsApp account's own name/number for this, "
        "they need to actually tell us. Extract whichever of the two their latest message(s) give - "
        "they might give both together or one at a time. A valid phone number looks like a real "
        "Israeli phone number (e.g. 0524215758, 052-421-5758, +972524215758) - if what they wrote "
        "doesn't look like one, leave phone empty so we ask again.\n"
        f"{off_topic_instruction(state.get('customer_language'))}\n"
        'Reply JSON: {"name": "<name, or empty string if not given in this message>", '
        '"phone": "<phone number, or empty string if not given or not a real-looking number>", '
        '"wants_more_items": true/false, "reply": "<see instructions above - if name and/or phone is '
        'still missing after this message, ask for exactly what\'s still missing>"}'
    )
    result = llm.chat_json([{"role": "system", "content": system}] + recent_context(state))
    reply = result.get("reply") or "מה השם המלא ומספר הטלפון להזמנה?"

    if result.get("wants_more_items"):
        logger.info("order_taking: phone=%s wants to add more items mid-checkout - back to browsing", state["phone"])
        browsing_result = _stage_browsing(state)
        browsing_result.setdefault("flow_stage", "browsing")
        return browsing_result

    name = result.get("name") or state.get("customer_name") or ""
    raw_phone = result.get("phone") or ""
    # Don't just trust the LLM's "this looks like a phone number" judgment -
    # double check server-side before ever saving it (same reasoning as the
    # add_items id check in _stage_browsing).
    phone = raw_phone if phone_utils.looks_like_phone_number(raw_phone) else ""
    phone = phone or state.get("customer_phone") or ""

    if not name or not phone:
        return {
            "reply": reply, "customer_name": name or None, "customer_phone": phone or None,
            "grounding_context": "Collecting the customer's name and contact phone number for the order - not complete yet.",
        }

    return {
        "reply": reply, "customer_name": name, "customer_phone": phone, "flow_stage": "confirming_summary",
        "grounding_context": f"Customer's name for the order: {name}. Contact phone for the order: {phone}.",
    }


def _stage_confirming_summary(state: AgentState) -> dict:
    """Show the full order summary and ask for final confirmation."""
    summary_text = _format_order_summary(state)

    zones = shipping_repo.list_delivery_zones()
    if zones:
        zones_text = "Delivery areas:\n" + "\n".join(
            f"- {z['name']} ({z['city']}): {int(z['price'])} ILS" for z in zones
        )
    else:
        zones_text = "Delivery areas: contact the shop for details."

    system = (
        'The customer was just shown an order summary and asked to confirm it. Decide whether their '
        'latest message confirms it (e.g. "yes", "confirm", "that\'s right").\n'
        f"{off_topic_instruction(state.get('customer_language'))}\n"
        f"If they ask about delivery locations or areas, answer using this info: {zones_text}\n"
        'Reply JSON: {"confirmed": true/false, "wants_more_items": true/false, "reply": "<if not '
        "confirmed and not wanting to add items, show them the summary below and ask them to confirm; "
        'don\'t invent anything not in it>"}\n\nOrder summary:\n' + summary_text
    )
    result = llm.chat_json([{"role": "system", "content": system}] + recent_context(state))
    reply = result.get("reply") or summary_text

    if result.get("wants_more_items"):
        logger.info("order_taking: phone=%s wants to add more items mid-checkout - back to browsing", state["phone"])
        browsing_result = _stage_browsing(state)
        browsing_result.setdefault("flow_stage", "browsing")
        return browsing_result

    if not result.get("confirmed"):
        return {"reply": reply, "grounding_context": f"Order summary shown, awaiting confirmation:\n{summary_text}\n\n{zones_text}"}

    # Use the explicitly-collected contact phone for the order/Supabase
    # record, NOT state["phone"] (the WhatsApp number) - see
    # _stage_collecting_contact for why. _has_contact_details() already
    # guaranteed this is set before the flow could ever reach this stage.
    order = orders_repo.create_order(
        state["customer_phone"], state["customer_name"], state["cart"],
        state["delivery_method"], state.get("delivery_address"), state.get("shipping_method"),
    )
    items_text = ", ".join(f"{i['name']} x{i['quantity']}" for i in order["items"])

    # Generate the confirmation message once, in the customer's own
    # language, and store it - every later attempt this turn (if the
    # guardrail rejects the phrasing) reuses this exact text instead of
    # asking the LLM to re-decide anything about an order that already
    # happened.
    confirmation_system = (
        "An order was just successfully placed for a customer of Gold Fish, a kosher fish/meat shop, "
        f"via WhatsApp. Items: {items_text}. Total: {order['total']} ILS. Write a short, friendly "
        "confirmation message for the customer, stating the total and that the team will confirm "
        f"availability and delivery details shortly. {language_instruction(state.get('customer_language'))}"
    )
    order["confirmation_message"] = llm.chat(
        [{"role": "system", "content": confirmation_system}] + recent_context(state)
    )

    return {
        "reply": order["confirmation_message"],
        "cart": [],
        "delivery_method": None,
        "delivery_address": None,
        "shipping_method": None,
        "flow_stage": "browsing",
        "order_just_created": order,
        "grounding_context": f"An order WAS just successfully created: items={items_text}, total={order['total']} ILS.",
    }


def _format_cart(cart: list[dict]) -> str:
    lines = []
    for item in cart:
        unit = item.get("unit_type") or "unit"
        price = item.get("price") or 0
        qty = item.get("quantity") or 0
        lines.append(f"- {item.get('name', '?')} x{qty} {unit} ({price} ILS/{unit})")
    return "\n".join(lines) or "(empty)"


def _format_order_summary(state: AgentState) -> str:
    items_total = sum((item.get("price") or 0) * (item.get("quantity") or 0) for item in state["cart"])
    shipping = state.get("shipping_method")
    shipping_cost = shipping["price"] if shipping else 0

    lines = [_format_cart(state["cart"]), f"Items total: {items_total:.2f} ILS"]

    if state.get("delivery_method") == "pickup":
        lines.append("Method: pickup (free)")
    elif state.get("delivery_method") == "delivery":
        address = state.get("delivery_address") or {}
        address_text = f"{address.get('street', '')} {address.get('building_number', '')}, {address.get('city', '')}"
        if shipping:
            lines.append(f"Method: delivery to {address_text} - {shipping_cost} ILS ({shipping['name']})")
        else:
            lines.append(f"Method: delivery to {address_text} - cost to be confirmed by staff")

    lines.append(f"Total: {items_total + shipping_cost:.2f} ILS")
    lines.append(f"Name: {state.get('customer_name') or 'unknown'}")
    lines.append(f"Phone: {state.get('customer_phone') or 'unknown'}")
    return "\n".join(lines)


def escalate_node(state: AgentState) -> dict:
    user_text = last_user_text(state)
    escalation.notify_staff(state["phone"], state["customer_name"], f"Customer asked for a human: {user_text!r}")
    system = (
        "The customer asked to speak with a human support agent. Tell them warmly that a support agent "
        "will join the conversation shortly and they should wait. Do NOT say you'll keep helping as the "
        f"bot — a real human IS coming. {language_instruction(state.get('customer_language'))} "
        "Keep it to 1-2 sentences."
    )
    reply = llm.chat([{"role": "system", "content": system}] + recent_context(state))
    # human_takeover=True pauses the bot for this conversation from this
    # point forward. The support dashboard will handle further replies.
    return {"reply": reply, "human_takeover": True}


# --- guardrail ---

BANNED_PHRASES = ["full refund", "free delivery", "% off", "discount code"]


def guardrail_node(state: AgentState) -> dict:
    """Checks the draft reply before it's allowed to be sent. Cheap
    deterministic checks first, then an LLM judge for quality/grounding.
    Allows exactly one retry before falling back to a safe reply + escalation.
    """
    reply = (state.get("reply") or "").strip()

    failure_reason = None
    if not reply:
        failure_reason = "empty reply"
    elif len(reply) > 1500:
        failure_reason = "reply too long for WhatsApp"
    elif any(phrase in reply.lower() for phrase in BANNED_PHRASES):
        failure_reason = "reply contains an unapproved promise (refund/discount/etc.)"

    if failure_reason is None:
        # Kept deliberately simple for this MVP demo phase - an earlier,
        # more elaborate multi-criteria version (addressing the question +
        # fact-grounding + strict language-matching, all in one judgment)
        # was rejecting too many genuinely fine replies (e.g. translating a
        # Hebrew product name into English), burning retries and
        # escalating ordinary orders over nothing. Just catch outright
        # fabrication - the failure mode that actually matters (inventing a
        # price/product/policy that isn't real) - and let everything else
        # through.
        facts = state.get("grounding_context") or "(this node had no specific facts to ground its reply in)"
        system = (
            "You are a final sanity check on a draft WhatsApp reply from a shop's AI assistant, before "
            "it's sent to a customer. You're given the FACTS the assistant actually had available, and "
            "its DRAFT REPLY.\n\n"
            f"FACTS the assistant had:\n{facts}\n\n"
            f"DRAFT REPLY:\n{reply}\n\n"
            "Only reject the reply if it states something as fact that flatly contradicts or isn't "
            "supported at all by the FACTS above - e.g. a price, product, or policy that's just made "
            "up. Translating/rephrasing a real product name into another language is fine, not "
            "fabrication. Don't reject for tone, phrasing, language choice, or not being a complete "
            "answer - approve anything that isn't a clear factual fabrication.\n\n"
            'Reply JSON: {"approved": true/false, "reason": "short reason"}'
        )
        verdict = llm.chat_json([{"role": "system", "content": system}])
        # If the judge call itself failed to return parseable JSON, default to approving -
        # we don't want a broken judge to block every single reply.
        if verdict.get("approved", True):
            logger.info("Guardrail: approved reply for phone=%s", state["phone"])
            # Reset explicitly - if this approval followed an earlier
            # rejection THIS turn, guardrail_retries is still 1 here, and
            # leaving it that way would persist into next turn's state and
            # make _route_after_guardrail (graph.py) wrongly treat every
            # future reply as a retry-in-progress.
            return {"guardrail_retries": 0, "guardrail_feedback": None}
        failure_reason = verdict.get("reason", "guardrail rejected the reply")

    logger.warning("Guardrail: rejected draft reply for phone=%s - %s", state["phone"], failure_reason)

    if state.get("order_just_created"):
        # A real order was already placed this turn - that's an
        # irreversible fact. No amount of guardrail unhappiness with the
        # PHRASING is allowed to hide it from the customer behind a generic
        # "let me get a team member" message, so we keep the confirmation
        # text we already generated and don't escalate over wording alone.
        logger.warning(
            "Guardrail rejected the order confirmation's phrasing, but an order was placed - "
            "keeping the confirmation message anyway for phone=%s", state["phone"],
        )
        return {
            "reply": state["order_just_created"]["confirmation_message"],
            "guardrail_retries": 0, "guardrail_feedback": None,
        }

    if state.get("guardrail_retries", 0) >= 1:
        # Already retried once - during this MVP demo phase there's no
        # staff actively monitoring escalations, so going silent here would
        # just permanently freeze the chat for no benefit. Log it (in case
        # someone IS watching) but keep the bot answering customers itself
        # instead of human_takeover blocking all future replies.
        escalation.notify_staff(
            state["phone"], state["customer_name"], f"Guardrail repeatedly rejected a draft reply: {failure_reason}"
        )
        return {"reply": SAFE_FALLBACK_REPLY, "guardrail_retries": 0, "guardrail_feedback": None}

    # Feed the rejection reason into the retried node's call (see
    # recent_context) so it targets this specific problem instead of
    # blindly re-running the same prompt with the same odds of repeating
    # the same mistake.
    return {"guardrail_retries": state.get("guardrail_retries", 0) + 1, "guardrail_feedback": failure_reason}
