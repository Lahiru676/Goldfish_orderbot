"""The single function api.py calls to run one customer message through the
agent: load their conversation state, run the graph, save the state back.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
import logging

from agent.config import LANGFUSE_ENABLED, MAX_CONTEXT_MESSAGES
from agent.conversation_store import conversation_store, get_thread_lock
from agent.graph import compiled_graph
from agent.nodes import language_instruction
from agent import llm
from agent.phone import to_local_il_format, looks_like_phone_number
from agent.repositories import customers as customers_repo
from agent.repositories import shipping as shipping_repo

logger = logging.getLogger(__name__)


def _message(role: str, content: str) -> dict:
    """Build a stored chat message with a reliable UTC timestamp."""
    return {"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()}

def _build_welcome_message(user_text: str, returning_name: str | None = None, lang: str | None = None) -> str:
    """Generates a welcome message — personalized if this is a returning customer."""
    lang_instr = language_instruction(lang)
    if returning_name:
        system = (
            f"You are the WhatsApp assistant for Gold Fish (גולד פיש), a kosher "
            f"fish and meat shop in Beitar Illit, Israel. This is a RETURNING customer "
            f"named {returning_name}. Greet them warmly by name and offer to help with "
            f"placing an order, checking an existing order, or any other question. "
            f"2-3 short lines, friendly. {lang_instr}"
        )
        fallback = (
            f"שלום {returning_name}, שמחים לראות אותך שוב!\n"
            "איך אפשר לעזור היום? אפשר לבצע הזמנה חדשה, לבדוק הזמנה קיימת, או כל שאלה אחרת."
        )
    else:
        system = (
            "You are the WhatsApp assistant for Gold Fish (גולד פיש), a kosher "
            "fish and meat shop in Beitar Illit, Israel. This is the customer's "
            "very first message, so write a short welcome: briefly introduce the "
            "shop (kosher fishmonger/butcher) and mention you can help with "
            "browsing products, placing an order, checking an existing order, or "
            f"any other question. Keep it to 2-3 short sentences/lines, friendly "
            f"and not overly formal. {lang_instr}"
        )
        fallback = (
            "שלום וברוכים הבאים לגולד פיש!\n"
            "אני העוזר הדיגיטלי של החנות - אפשר לשאול אותי על מוצרים ומחירים, "
            "לבצע הזמנה, לבדוק סטטוס הזמנה קיימת, או כל שאלה אחרת.\n"
            "איך אפשר לעזור?"
        )
    try:
        return llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ])
    except Exception as exc:
        logger.error("Failed to generate welcome message, falling back to Hebrew default: %s", exc)
        return fallback

def _reg_ask(user_text: str, instruction: str, fallback: str, lang: str | None = None) -> str:
    """Generates a short registration question in the customer's language."""
    try:
        return llm.chat([
            {"role": "system", "content":
             f"You are registering a new customer for Gold Fish (WhatsApp). {instruction}. "
             f"One short friendly line. {language_instruction(lang)}"},
            {"role": "user", "content": user_text},
        ])
    except Exception:
        return fallback


def _detect_language(text: str) -> str | None:
    """Detects the language of the customer's message.
    Returns a plain English name like 'Hebrew', 'English', 'Arabic', 'Russian', etc.
    Returns None if detection fails — the caller will fall back to auto-detection per message.
    """
    try:
        result = llm.chat_json([
            {"role": "system", "content":
             "What language is this text written in? "
             'Reply JSON: {"language": "<language name in English, e.g. Hebrew, English, Arabic, Russian, Sinhala>"}'},
            {"role": "user", "content": text[:300]},
        ])
        lang = (result.get("language") or "").strip()
        if lang:
            logger.info("Detected language: %r from text: %r", lang, text[:80])
            return lang
        return None
    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
        return None


def _extract_name_from_message(text: str) -> str | None:
    try:
        result = llm.chat_json([
            {"role": "system", "content":
             "Extract the person's full name from the message (Hebrew or English). "
             'Reply JSON: {"name": "<name, or empty string if not found>"}'},
            {"role": "user", "content": text},
        ])
        name = (result.get("name") or "").strip()
        return name if len(name) >= 2 else None
    except Exception as exc:
        logger.error("Name extraction failed: %s", exc)
        return None


def _extract_phone_from_message(text: str) -> str | None:
    try:
        result = llm.chat_json([
            {"role": "system", "content":
             "Extract a phone number from the message (Israeli format like 0524215758 or international). "
             'Reply JSON: {"phone": "<digits only, or empty string if not found>"}'},
            {"role": "user", "content": text},
        ])
        digits = "".join(ch for ch in (result.get("phone") or "") if ch.isdigit())
        return digits if looks_like_phone_number(digits) else None
    except Exception as exc:
        logger.error("Phone extraction failed: %s", exc)
        return None


def _extract_address_from_message(text: str) -> dict:
    try:
        return llm.chat_json([
            {"role": "system", "content":
             "Extract a delivery address from the message (Israeli address). "
             'Reply JSON: {"city": "...", "street": "...", "building_number": "...", "complete": true/false}. '
             "Set complete=true only when you found city, street AND building number."},
            {"role": "user", "content": text},
        ])
    except Exception as exc:
        logger.error("Address extraction failed: %s", exc)
        return {"complete": False}


if LANGFUSE_ENABLED:
    try:
        from langfuse import get_client, observe, propagate_attributes
    except Exception as exc:
        logger.warning("Langfuse enabled but unavailable in runner (%s). Continuing without tracing.", exc)

        def observe(*_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        def propagate_attributes(**_kwargs):
            return nullcontext()

        def get_client():
            return None
else:
    def observe(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def propagate_attributes(**_kwargs):
        return nullcontext()

    def get_client():
        return None

# We keep more raw history in storage than we feed to the LLM each turn
# (recent_context() in nodes.py trims that separately) - this just stops
# the saved conversation document from growing forever.
_MAX_STORED_MESSAGES = MAX_CONTEXT_MESSAGES * 4


@observe(name="whatsapp-agent-turn", as_type="agent")
async def run_agent_turn(phone: str, customer_name: str | None, user_text: str) -> str | None:
    """Runs one WhatsApp message through the agent graph.

    Returns the reply to send back, or None if the bot should stay silent
    (a human has taken over this conversation).
    """
    # @observe captures all of this function's args as the trace input by
    # default - that would mean the phone number shows up twice (once here,
    # once correctly as user_id below) plus customer_name. Restrict it to
    # just the actual message, which is what's useful to read in the UI.
    langfuse_client = get_client()
    if langfuse_client:
        langfuse_client.update_current_span(input=user_text)

    # Wrap the WHOLE turn (not just the graph call) so even a turn that
    # exits early because a human has taken over still gets tagged with
    # user_id/session_id - otherwise those conversations are untaggable and
    # impossible to filter for in the Langfuse UI.
    with propagate_attributes(
        user_id=phone,
        session_id=phone,
        trace_name="whatsapp-agent-turn",
        metadata={"channel": "whatsapp", "customer_name": customer_name or "unknown"},
        tags=["whatsapp", "support-bot"],
    ):
        lock = get_thread_lock(phone)
        async with lock:
            state = conversation_store.load(phone)

            # Detect the customer's language from their first message and
            # store it in state. All LLM prompts this turn (and every
            # future turn) use an explicit "always reply in X" directive
            # instead of the vaguer "match their last message" fallback,
            # which caused language drift in production.
            if not state.get("customer_language"):
                detected = _detect_language(user_text)
                if detected:
                    state["customer_language"] = detected
                    logger.info("Language set to %r for phone=%s", detected, phone)

            # Every message: verify the customer has a real Supabase profile.
            #
            # Case 1 — customer_phone not set yet: look them up by WhatsApp
            #   number. If found in Supabase, pre-fill name+phone so the
            #   welcome is personalized and checkout skips contact collection.
            #
            # Case 2 — customer_phone already set (from a previous turn):
            #   verify the profile actually exists in Supabase. If it doesn't,
            #   reset registration state so the bot re-registers them. This
            #   catches broken states where MongoDB says "registered" but
            #   Supabase has no real profile (e.g. after deleting a test user).
            if not state.get("customer_phone") and not state.get("registration_stage"):
                profile = customers_repo.find_customer_by_whatsapp_phone(phone)
                if profile:
                    state["customer_name"] = profile.get("customer_name") or state.get("customer_name")
                    state["customer_phone"] = profile.get("phone_number_1") or phone
                    logger.info(
                        "Identified returning customer name=%r phone=%s for WhatsApp=%s",
                        state["customer_name"], state["customer_phone"], phone,
                    )
            elif state.get("customer_phone") and not state.get("registration_stage"):
                # Try both the stored contact phone and the actual WhatsApp number
                # (for non-Israeli numbers, bot email lookup uses whichever was
                # used at registration — usually these are the same, but try both)
                profile = (
                    customers_repo.find_customer_by_whatsapp_phone(state["customer_phone"]) or
                    customers_repo.find_customer_by_whatsapp_phone(phone)
                )
                if not profile:
                    logger.warning(
                        "customer_phone=%s is set in MongoDB but no Supabase profile found — "
                        "resetting registration for WhatsApp=%s",
                        state["customer_phone"], phone,
                    )
                    state["customer_phone"] = None
                    state["customer_name"] = None
                    state["registration_stage"] = None
                    state["welcomed"] = False

            state["messages"].append(_message("user", user_text))

            if state["human_takeover"]:
                logger.info("Phone=%s is under human control - logging message only, not replying", phone)
                conversation_store.save(state)
                return None

            # ── Registration gate ──────────────────────────────────────────
            # Collects name → phone → address+city for new customers before
            # letting them into the normal chat. Known customers (customer_phone
            # already set in state) skip this entirely.
            #
            # Step 1 (stage=None, no customer_phone): welcome + ask name
            # Step 2 (stage="name"):  extract name  → ask phone
            # Step 3 (stage="phone"): extract phone → ask address+city
            # Step 4 (stage="address"): extract address → create Supabase profile → done

            reg_stage = state.get("registration_stage")

            lang = state.get("customer_language")

            # If the customer asks to cancel / restart while stuck in any
            # registration step, wipe the registration state and start over
            # from step 1 (ask for name again). Without this they are
            # completely trapped — the registration gate runs before the
            # LangGraph graph, so the "new order" reset in order_taking_node
            # never fires while registration is active.
            _REG_RESET_PHRASES = (
                "הזמנה חדשה", "התחל מחדש", "להתחיל מחדש", "ביטול", "לבטל",
                "new order", "cancel", "start over", "start again", "start fresh",
            )
            if reg_stage and any(p in user_text.lower() for p in _REG_RESET_PHRASES):
                logger.info("Registration reset requested at stage=%r for WhatsApp=%s", reg_stage, phone)
                state["registration_stage"] = "name"
                state["customer_name"] = None
                state["customer_phone"] = None
                reply = _reg_ask(
                    user_text,
                    "The customer wants to start fresh. Acknowledge with one short line, then ask for their full name.",
                    "בסדר! נתחיל מחדש 😊 מה שמך המלא?",
                    lang=lang,
                )
                state["messages"].append(_message("assistant", reply))
                conversation_store.save(state)
                return reply

            if reg_stage == "name":
                name = _extract_name_from_message(user_text)
                if not name:
                    reply = "סליחה, לא הצלחתי להבין. מה שמך המלא?"
                    state["messages"].append(_message("assistant", reply))
                    conversation_store.save(state)
                    return reply
                state["customer_name"] = name
                state["registration_stage"] = "phone"
                reply = _reg_ask(user_text,
                                 f"Ask for their phone number (one line, address them as {name})",
                                 f"תודה {name}! מה מספר הטלפון שלך?",
                                 lang=lang)
                logger.info("Registration: got name=%r, asking for phone, WhatsApp=%s", name, phone)
                state["messages"].append(_message("assistant", reply))
                conversation_store.save(state)
                return reply

            elif reg_stage == "phone":
                contact_phone = _extract_phone_from_message(user_text)
                if not contact_phone:
                    reply = _reg_ask(user_text,
                                     "Phone number was unclear — ask again for a valid phone number",
                                     "לא הצלחתי לזהות מספר. אנא הכנס מספר טלפון תקין (לדוגמה: 0524215758)",
                                     lang=lang)
                    state["messages"].append(_message("assistant", reply))
                    conversation_store.save(state)
                    return reply
                state["customer_phone"] = contact_phone
                state["registration_stage"] = "address"
                reply = _reg_ask(user_text,
                                 "Ask for their full delivery address: city, street name and building number (one friendly line)",
                                 "מה כתובת המשלוח שלך? (עיר, רחוב ומספר בית)",
                                 lang=lang)
                logger.info("Registration: got phone=%r, asking for address, WhatsApp=%s", contact_phone, phone)
                state["messages"].append(_message("assistant", reply))
                conversation_store.save(state)
                return reply

            elif reg_stage == "address":
                parsed = _extract_address_from_message(user_text)
                if not parsed.get("complete"):
                    # Tell the LLM exactly which fields are still missing so it
                    # asks for just those, instead of repeating the whole thing.
                    missing = []
                    if not parsed.get("city"):
                        missing.append("city (עיר)")
                    if not parsed.get("street"):
                        missing.append("street (רחוב)")
                    if not parsed.get("building_number"):
                        missing.append("building number (מספר בית)")
                    missing_str = ", ".join(missing) if missing else "city, street and building number"
                    instruction = (
                        f"Address was incomplete — still missing: {missing_str}. "
                        "Ask only for the missing part(s), briefly. "
                        "If the customer seems to be trying to order something instead of giving an address, "
                        "gently remind them we need their address first to complete registration."
                    )
                    reply = _reg_ask(user_text, instruction,
                                     "לא קיבלתי כתובת מלאה. אנא ציין עיר, רחוב ומספר בית.",
                                     lang=lang)
                    state["messages"].append(_message("assistant", reply))
                    conversation_store.save(state)
                    return reply

                delivery_address = {
                    "city": parsed["city"],
                    "street": parsed["street"],
                    "building_number": parsed["building_number"],
                }
                state["delivery_address"] = delivery_address
                zone = shipping_repo.find_delivery_zone_for_city(parsed["city"])
                if zone:
                    state["shipping_method"] = zone
                state["registration_stage"] = None

                # Create the Supabase profile (auth user + profiles row)
                customers_repo.find_or_create_customer(state["customer_phone"], state["customer_name"])
                logger.info(
                    "Registration complete: name=%r phone=%r address=%r WhatsApp=%s",
                    state["customer_name"], state["customer_phone"], delivery_address, phone,
                )

                name = state["customer_name"]
                city = parsed["city"]
                try:
                    greet_system = (
                        f"You are the WhatsApp assistant for Gold Fish (גולד פיש), a kosher fish/meat shop. "
                        f"A new customer just finished registering: name={name}, city={city}. "
                        f"Welcome them by name, confirm we have their address in {city}, and ask how you "
                        f"can help today. 2-3 short friendly lines. {language_instruction(lang)}"
                    )
                    reply = llm.chat([
                        {"role": "system", "content": greet_system},
                        {"role": "user", "content": user_text},
                    ])
                except Exception as exc:
                    logger.error("Registration completion greeting failed: %s", exc)
                    reply = f"מעולה {name}! ההרשמה הושלמה. שמרנו את הכתובת שלך ב{city}. איך אפשר לעזור היום?"

                state["messages"].append(_message("assistant", reply))
                conversation_store.save(state)
                return reply

            elif not state.get("customer_phone"):
                # First message from an unrecognized number — start registration
                state["registration_stage"] = "name"
                welcome = _build_welcome_message(user_text, returning_name=None, lang=lang)
                ask_name = _reg_ask(user_text,
                                    "Ask for their full name to register them",
                                    "מה שמך המלא?",
                                    lang=lang)
                reply = f"{welcome}\n\n{ask_name}"
                state["welcomed"] = True
                logger.info("Registration started: asked for name, WhatsApp=%s", phone)
                state["messages"].append(_message("assistant", reply))
                conversation_store.save(state)
                return reply
            # ── End registration gate ──────────────────────────────────────

            is_first_message = not state.get("welcomed")

            # Snapshot the stage and cart BEFORE the graph runs - see the
            # comments on turn_start_stage/cart_at_turn_start in state.py
            # for why order_taking_node must build from these instead of
            # the live flow_stage/cart.
            state["turn_start_stage"] = state.get("flow_stage") or "browsing"
            state["cart_at_turn_start"] = list(state.get("cart") or [])

            # order_just_created is only meant to survive WITHIN one turn -
            # so a guardrail-rejected retry still repeats the same real
            # order's confirmation instead of losing track of it (see
            # order_taking_node/guardrail_node). It must NOT survive into a
            # later, unrelated turn: if last turn placed an order, this
            # turn's message has nothing to do with it, and without this
            # reset the bot would keep repeating that old order's
            # confirmation forever, ignoring everything the customer says
            # after it (this actually happened in production - see logs).
            state["order_just_created"] = None

            # Same transient-within-a-turn reasoning as order_just_created
            # above, applied to the guardrail's last rejection reason - it
            # must not leak into a later, unrelated turn and make the bot
            # think it's still correcting an old mistake.
            state["guardrail_feedback"] = None

            # Tell _stage_browsing (and any other node that cares) whether
            # this is the first message — so it can include a greeting in
            # its own reply instead of having runner.py prepend a separate
            # welcome that caused two redundant opening messages.
            state["is_first_message"] = is_first_message

            logger.info("Running agent graph for phone=%s, message=%r", phone, user_text[:200])
            new_state = compiled_graph.invoke(state)

            # Hebrew default, not English - most customers write Hebrew, and
            # this only fires if every node in the graph somehow returned no
            # reply at all (see nodes.SAFE_FALLBACK_REPLY for the same
            # reasoning, used by the more common failure paths).
            reply = new_state.get("reply") or "סליחה, אפשר לכתוב את זה במילים אחרות?"
            if is_first_message:
                if new_state.get("intent") == "order_taking":
                    # _stage_browsing already embedded the greeting in its
                    # reply - no separate prepend needed.
                    logger.info("First-message greeting embedded by order_taking node, phone=%s", phone)
                else:
                    # Non-order nodes don't include a greeting (they're
                    # told not to), so prepend a standalone welcome here.
                    returning_name = new_state.get("customer_name") or state.get("customer_name")
                    welcome_message = _build_welcome_message(
                        user_text, returning_name=returning_name,
                        lang=new_state.get("customer_language") or state.get("customer_language"),
                    )
                    reply = f"{welcome_message}\n\n{reply}"
                    logger.info("Prepended welcome for first-message non-order intent, phone=%s", phone)
                new_state["welcomed"] = True

            new_state["messages"].append(_message("assistant", reply))
            new_state["messages"] = new_state["messages"][-_MAX_STORED_MESSAGES:]

            # Stays inside the lock - saving after releasing it would let a
            # second message for the same customer load stale state in the
            # gap between releasing the lock and this write landing.
            conversation_store.save(new_state)
            logger.info(
                "Agent turn complete for phone=%s intent=%s human_takeover=%s reply=%r",
                phone, new_state.get("intent"), new_state.get("human_takeover"), reply[:200],
            )
            return reply
