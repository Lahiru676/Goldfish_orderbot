"""Loads/saves each customer's conversation state (their chat history, cart,
and human_takeover flag) and makes sure two messages from the SAME customer
are never processed at the same time.

Storage: MongoDB if MONGODB_URI is set, otherwise a local JSON file
(data/conversations.json) so the agent still works for local testing.
"""
import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

from agent import config
from agent.state import AgentState, new_state

logger = logging.getLogger(__name__)

_LOCAL_STORE_PATH = os.path.join(config.DATA_DIR, "conversations.json")

# One asyncio.Lock per phone number, created the first time we see that
# number. Two messages from the same customer arriving close together will
# wait for each other here; messages from different customers never block
# each other since they use different locks.
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def get_thread_lock(phone: str) -> asyncio.Lock:
    return _locks[phone]


class ConversationStore:
    def __init__(self):
        self._collection = None
        if config.MONGODB_URI:
            try:
                from pymongo import MongoClient

                client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
                self._collection = client[config.MONGODB_DB_NAME]["conversations"]
                logger.info("ConversationStore: connected to MongoDB database %r", config.MONGODB_DB_NAME)
            except Exception as e:
                logger.error("ConversationStore: could not connect to MongoDB (%s) - using local file instead", e)
                self._collection = None

        if self._collection is None:
            logger.warning(
                "ConversationStore: MONGODB_URI not set/unreachable - using local file %s", _LOCAL_STORE_PATH
            )

    def load(self, phone: str) -> AgentState:
        if self._collection is not None:
            doc = self._collection.find_one({"_id": phone})
            if doc:
                doc.pop("_id", None)
                logger.info("Loaded conversation for %s from MongoDB (%d messages so far)", phone, len(doc.get("messages", [])))
                return doc  # type: ignore[return-value]
        else:
            all_states = self._load_local_file()
            if phone in all_states:
                logger.info("Loaded conversation for %s from local file", phone)
                return all_states[phone]

        logger.info("No existing conversation for %s - starting fresh", phone)
        return new_state(phone)

    def save(self, state: AgentState) -> None:
        phone = state["phone"]
        if self._collection is not None:
            self._collection.replace_one({"_id": phone}, dict(state), upsert=True)
            return

        all_states = self._load_local_file()
        all_states[phone] = state
        self._save_local_file(all_states)

    def _load_local_file(self) -> dict:
        if not os.path.exists(_LOCAL_STORE_PATH):
            return {}
        with open(_LOCAL_STORE_PATH, encoding="utf-8") as f:
            return json.load(f)

    def _save_local_file(self, all_states: dict) -> None:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(_LOCAL_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(all_states, f, ensure_ascii=False, indent=2)


conversation_store = ConversationStore()


# ── Support dashboard helpers ─────────────────────────────────────────────────

def list_all_conversations() -> list[dict]:
    """Returns a summary of every conversation for the support dashboard sidebar."""
    if conversation_store._collection is not None:
        docs = list(conversation_store._collection.find(
            {},
            {
                "customer_name": 1,
                "human_takeover": 1,
                "registration_stage": 1,
                "messages": {"$slice": -1},  # only the last message for preview
            }
        ))
        result = []
        for doc in docs:
            phone = str(doc.get("_id", ""))
            msgs = doc.get("messages") or []
            last = msgs[-1] if msgs else None
            result.append({
                "phone": phone,
                "customer_name": doc.get("customer_name"),
                "human_takeover": bool(doc.get("human_takeover")),
                "registration_active": bool(doc.get("registration_stage")),
                "last_message_text": (last.get("content") or "")[:80] if last else "",
                "last_message_role": last.get("role") if last else None,
                "last_ts": last.get("ts") if last else None,
            })
        result.sort(key=lambda x: x.get("last_ts") or "", reverse=True)
        return result
    return []


def get_conversation_messages(phone: str) -> list[dict]:
    """Returns the full message history for a conversation."""
    if conversation_store._collection is not None:
        doc = conversation_store._collection.find_one({"_id": phone}, {"messages": 1, "_id": 0})
        return (doc.get("messages") or []) if doc else []
    return (conversation_store.load(phone).get("messages") or [])


def set_human_takeover(phone: str, active: bool) -> None:
    """Toggles human_takeover for a conversation directly in the store."""
    if conversation_store._collection is not None:
        conversation_store._collection.update_one(
            {"_id": phone},
            {"$set": {"human_takeover": active}},
        )
    else:
        state = conversation_store.load(phone)
        state["human_takeover"] = active
        conversation_store.save(state)


def append_user_message(phone: str, text: str) -> None:
    """Appends a customer message directly to the conversation (used for media)."""
    msg = {"role": "user", "content": text, "ts": datetime.now(timezone.utc).isoformat()}
    if conversation_store._collection is not None:
        conversation_store._collection.update_one({"_id": phone}, {"$push": {"messages": msg}})
    else:
        state = conversation_store.load(phone)
        state["messages"].append(msg)
        conversation_store.save(state)


def append_support_message(phone: str, text: str) -> None:
    """Saves a support-agent message into the conversation history.

    Stored as role='assistant' with from_support=True so the bot treats it
    as an assistant turn (no LLM confusion) but the dashboard can style it
    differently from bot replies.
    """
    msg = {"role": "assistant", "content": text, "from_support": True,
           "ts": datetime.now(timezone.utc).isoformat()}
    if conversation_store._collection is not None:
        conversation_store._collection.update_one(
            {"_id": phone},
            {"$push": {"messages": msg}},
        )
    else:
        state = conversation_store.load(phone)
        state["messages"].append(msg)
        conversation_store.save(state)
