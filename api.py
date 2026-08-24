from collections import deque
from datetime import datetime, timezone
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import base64
import json
import logging
import os
import requests
import threading

# WhatsApp Flow encryption (RSA + AES-GCM)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

load_dotenv()

# Must run BEFORE importing agent.* - some of those modules (e.g.
# ConversationStore) log during import (when they connect to MongoDB), and
# without a handler configured yet, those log lines would be silently lost.
_log_format = "%(asctime)s [%(name)s] %(levelname)s  %(message)s"

# Write to file (persistent, survives restarts)
_file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "webhook.log"))
_file_handler.setFormatter(logging.Formatter(_log_format))

# Also write to stdout so `docker compose logs app -f` shows live traces
_stdout_handler = logging.StreamHandler()
_stdout_handler.setFormatter(logging.Formatter(_log_format))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stdout_handler])

from agent.runner import run_agent_turn
from agent.repositories import products as products_repo
from agent.repositories import shipping as shipping_repo
from agent.support import manager as _support_manager, router as _support_router


app = FastAPI()
app.include_router(_support_router)

# Secrets and config pulled from .env
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")  # used once by Meta to verify your webhook URL
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")  # lets us call the WhatsApp API
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID") or os.getenv("PHONE_NUMBER_ID")  # your business number's ID
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "vXX.X")  # which version of Meta's API to call

# The URL we POST to whenever we want to send a WhatsApp message out.
SEND_MESSAGE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"

PROCESSED_MESSAGE_IDS_MAX = 2000
processed_message_ids = deque(maxlen=PROCESSED_MESSAGE_IDS_MAX)
processed_message_ids_set = set()
processed_message_ids_lock = threading.Lock()


def remember_message_id(message_id: str) -> bool:
    """Returns True if the message_id is new, False if already processed."""
    with processed_message_ids_lock:
        if message_id in processed_message_ids_set:
            return False
        if len(processed_message_ids) == processed_message_ids.maxlen:
            oldest = processed_message_ids.popleft()
            processed_message_ids_set.discard(oldest)
        processed_message_ids.append(message_id)
        processed_message_ids_set.add(message_id)
        return True


@app.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    """One-time check Meta does when you first set up the webhook URL.

    Meta sends a GET request with a token; if it matches our VERIFY_TOKEN,
    we echo back their "challenge" value to prove we control this URL.
    """
    params = dict(request.query_params)

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)

    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook/whatsapp")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """Called by WhatsApp every time a message (or status update) happens.

    We pull the actual text message out of Meta's nested JSON payload and
    hand it off to the LangGraph agent in the background.
    """
    data = await request.json()
    logging.info("Incoming webhook: %s", data)

    try:
        # Meta wraps the real content several layers deep: entry -> changes -> value.
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return {"status": "ignored"}

        message = value["messages"][0]
        from_number = message["from"]  # the sender's WhatsApp number
        incoming_message_id = message.get("id")

        # WhatsApp's profile name for this contact, if Meta included it - used
        # so the agent can address the customer by name.
        contacts = value.get("contacts") or []
        customer_name = contacts[0].get("profile", {}).get("name") if contacts else None

        if incoming_message_id and not remember_message_id(incoming_message_id):
            logging.info("Duplicate webhook ignored for message_id=%s", incoming_message_id)
            return {"status": "duplicate"}

        if incoming_message_id:
            mark_message_as_read_and_show_typing(incoming_message_id)

        if message.get("type") == "text":
            user_text = message["text"]["body"]
            background_tasks.add_task(process_and_reply, from_number, customer_name, user_text)
        elif message.get("type") in ("image", "document", "audio", "video", "sticker"):
            media_type = message["type"]
            media_obj = message.get(media_type) or {}
            background_tasks.add_task(
                process_media_message, from_number, customer_name,
                media_type, media_obj.get("id"), media_obj.get("filename"), media_obj.get("caption") or "",
            )

    except Exception as e:
        logging.error("Error: %s", str(e))

    return {"status": "ok"}


def send_whatsapp_message(to_number: str, message_text: str):
    """Sends a text message to a WhatsApp number via Meta's Graph API."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {
            "body": message_text
        }
    }

    response = requests.post(SEND_MESSAGE_URL, headers=headers, json=payload, timeout=20)
    logging.info("Send response: %s %s", response.status_code, response.text)


def mark_message_as_read_and_show_typing(message_id: str):
    """Marks an incoming message as read and asks WhatsApp to show typing indicator."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {
            "type": "text",
        },
    }
    response = requests.post(SEND_MESSAGE_URL, headers=headers, json=payload, timeout=20)
    logging.info("Read/typing response: %s %s", response.status_code, response.text)
    if response.ok:
        return

    # Fallback for accounts/API versions that don't support typing_indicator yet.
    fallback_payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    fallback_response = requests.post(
        SEND_MESSAGE_URL, headers=headers, json=fallback_payload, timeout=20
    )
    logging.info(
        "Read-only fallback response: %s %s",
        fallback_response.status_code,
        fallback_response.text,
    )


async def process_media_message(
    from_number: str, customer_name: str | None,
    media_type: str, media_id: str | None, filename: str | None, caption: str,
):
    """Handles an incoming media message (image / document / audio / video)."""
    from agent.conversation_store import conversation_store as _cs, append_user_message
    state = _cs.load(from_number)
    display = filename or media_type
    text = f"[{media_type}: {display}]" + (f" {caption}" if caption else "")

    if state.get("human_takeover"):
        append_user_message(from_number, text)
        await _support_manager.broadcast({
            "type": "message",
            "phone": from_number,
            "name": state.get("customer_name") or customer_name,
            "text": caption or "",
            "media_type": media_type,
            "media_id": media_id,
            "filename": filename,
            "direction": "inbound",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    else:
        # Bot can't process media — acknowledge and ask them to describe in text
        label = {"image": "תמונה", "document": "קובץ", "audio": "הודעה קולית", "video": "וידאו"}.get(media_type, "קובץ")
        append_user_message(from_number, text)
        send_whatsapp_message(from_number, f"קיבלתי את ה{label} 📎 אם יש שאלה ספציפית, אשמח לעזור בטקסט 😊")


async def process_and_reply(from_number: str, customer_name: str | None, user_text: str):
    """Runs the LangGraph agent for this message, outside the webhook request path."""
    from agent.conversation_store import conversation_store as _cs
    pre_state = _cs.load(from_number)
    was_human = bool(pre_state.get("human_takeover"))

    logging.info("Message from %s (%s): %s", from_number, customer_name, user_text)
    reply_text = await run_agent_turn(from_number, customer_name, user_text)

    if reply_text is None:
        # Human takeover is active — push the inbound message to the dashboard.
        await _support_manager.broadcast({
            "type": "message",
            "phone": from_number,
            "name": pre_state.get("customer_name") or customer_name,
            "text": user_text,
            "direction": "inbound",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        logging.info("Not replying to %s - conversation is under human control", from_number)
        return

    send_whatsapp_message(from_number, reply_text)

    # If this turn just triggered an escalation (human_takeover flipped to True),
    # broadcast an escalation alert to the support dashboard.
    if not was_human:
        post_state = _cs.load(from_number)
        if post_state.get("human_takeover"):
            await _support_manager.broadcast({
                "type": "escalation",
                "phone": from_number,
                "name": customer_name or post_state.get("customer_name"),
                "reason": "Customer requested human support",
            })


FLOW_PRIVATE_KEY_PATH = os.path.join(os.path.dirname(__file__), "keys", "private.pem")

def _load_private_key():
    with open(FLOW_PRIVATE_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _decrypt_flow_request(encrypted_body: dict) -> dict:
    """Decrypts a WhatsApp Flow request encrypted with our public key.

    Meta uses hybrid encryption:
    1. Generates a random AES-256 key and encrypts it with our RSA public key
    2. Encrypts the actual payload with that AES key (AES-GCM)

    We reverse both steps to get the plaintext JSON.
    """
    private_key = _load_private_key()

    encrypted_aes_key = base64.b64decode(encrypted_body["encrypted_aes_key"])
    encrypted_flow_data = base64.b64decode(encrypted_body["encrypted_flow_data"])
    initial_vector = base64.b64decode(encrypted_body["initial_vector"])

    # Decrypt the AES key using RSA-OAEP
    aes_key = private_key.decrypt(
        encrypted_aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Decrypt the flow payload using AES-GCM
    decrypted = AESGCM(aes_key).decrypt(initial_vector, encrypted_flow_data, None)
    return json.loads(decrypted.decode("utf-8"))


def _encrypt_flow_response(response_dict: dict, aes_key: bytes, initial_vector: bytes) -> str:
    """Encrypts our response back to Meta using the same AES key they sent us."""
    # Flip ALL bytes of IV as required by the WhatsApp Flows protocol
    flipped_iv = bytes(b ^ 0xFF for b in initial_vector)
    encrypted = AESGCM(aes_key).encrypt(flipped_iv, json.dumps(response_dict).encode("utf-8"), None)
    return base64.b64encode(encrypted).decode("utf-8")


@app.get("/flow")
async def flow_health_check():
    """Meta calls this to verify the endpoint is reachable before publishing a flow."""
    return {"status": "active"}


@app.post("/flow")
async def whatsapp_flow_endpoint(request: Request, background_tasks: BackgroundTasks):
    """WhatsApp Flow Data Exchange endpoint.

    Meta encrypts every request with our public key. We decrypt it, handle the
    action (ping / INIT / data_exchange), encrypt the response, and return it.
    """
    raw_body = await request.json()
    logging.info("Flow raw request keys: %s", list(raw_body.keys()))

    # ── ping comes in plaintext — not encrypted ───────────────────────
    if raw_body.get("action") == "ping":
        return {"data": {"status": "active"}}

    # ── decrypt all other requests ────────────────────────────────────
    try:
        private_key = _load_private_key()
        encrypted_aes_key = base64.b64decode(raw_body["encrypted_aes_key"])
        encrypted_flow_data = base64.b64decode(raw_body["encrypted_flow_data"])
        initial_vector = base64.b64decode(raw_body["initial_vector"])

        aes_key = private_key.decrypt(
            encrypted_aes_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        decrypted_bytes = AESGCM(aes_key).decrypt(initial_vector, encrypted_flow_data, None)
        body = json.loads(decrypted_bytes.decode("utf-8"))
    except Exception as exc:
        logging.error("Flow decryption failed: %s  raw=%s", exc, raw_body)
        # HTTP 421 tells the WhatsApp client to re-download the public key and retry
        from fastapi.responses import Response as FastAPIResponse
        return FastAPIResponse(status_code=421)

    logging.info("Flow decrypted body: %s", body)

    action = body.get("action")
    flow_token = body.get("flow_token", "")
    screen = body.get("screen", "")
    data = body.get("data", {})

    # ── ping can arrive encrypted — respond with encrypted status ─────
    if action == "ping":
        return PlainTextResponse(_encrypt_flow_response({"data": {"status": "active"}}, aes_key, initial_vector))

    # ── INIT: flow opened — return initial data for the first screen ──
    if action == "INIT":
        zones = shipping_repo.list_delivery_zones()
        cities = [{"id": z["id"], "title": z.get("city") or z["name"]} for z in zones if z.get("city")]
        response = {
            "screen": screen or "SCREEN_1",
            "data": {"cities": cities},
        }
        return PlainTextResponse(_encrypt_flow_response(response, aes_key, initial_vector))

    # ── data_exchange: customer submitted a screen ────────────────────
    if action == "data_exchange":
        wa_id = flow_token

        if flow_token.startswith("SUBMIT:"):
            wa_id = flow_token.removeprefix("SUBMIT:")
            background_tasks.add_task(handle_flow_submission, wa_id, screen, data)
            response = {
                "screen": "SUCCESS",
                "data": {
                    "extension_message_response": {
                        "params": {"flow_token": flow_token}
                    }
                },
            }
            return PlainTextResponse(_encrypt_flow_response(response, aes_key, initial_vector))

        zones = shipping_repo.list_delivery_zones()
        cities = [{"id": z["id"], "title": z.get("city") or z["name"]} for z in zones if z.get("city")]
        response = {
            "screen": screen,
            "data": {"cities": cities},
        }
        return PlainTextResponse(_encrypt_flow_response(response, aes_key, initial_vector))

    logging.warning("Flow: unhandled action=%s", action)
    return PlainTextResponse(_encrypt_flow_response({"screen": screen, "data": {}}, aes_key, initial_vector))


async def handle_flow_submission(wa_id: str, screen: str, data: dict):
    """Processes a completed flow submission — registration or checkout."""
    logging.info("Flow submission from wa_id=%s screen=%s data=%s", wa_id, screen, data)
    # Registration and checkout logic will be wired here once
    # the flows are built and published. For now just log.


if __name__ == "__main__":

    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
