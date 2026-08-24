"""Small shared helper for talking to Supabase's REST API (PostgREST) and
Auth Admin API. Used by every repository so they don't each reimplement
headers/error handling.

Two keys are used for two different things:
- SUPABASE_ANON_KEY: public, read-only, same access the storefront website
  has in the browser. Fine for reading products/categories.
- SUPABASE_SERVICE_KEY: full read/write access, bypasses Row Level
  Security. Needed for anything touching customers/orders. Treat it like a
  password - never log it, never send it anywhere except Supabase's API.
"""
import logging

import requests

from agent import config

logger = logging.getLogger(__name__)


def _headers(use_service_key: bool) -> dict:
    key = config.SUPABASE_SERVICE_KEY if use_service_key else config.SUPABASE_ANON_KEY
    return {"apikey": key or "", "Authorization": f"Bearer {key or ''}"}


def rest_get(table: str, params: dict, use_service_key: bool = False) -> list[dict]:
    url = f"{config.SUPABASE_URL}/rest/v1/{table}"
    try:
        response = requests.get(url, headers=_headers(use_service_key), params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error("Supabase GET %s failed: %s", table, e)
        return []


def rest_post(table: str, body: dict) -> dict | None:
    """Always uses the service key - every write needs to bypass RLS."""
    url = f"{config.SUPABASE_URL}/rest/v1/{table}"
    headers = {**_headers(use_service_key=True), "Content-Type": "application/json", "Prefer": "return=representation"}
    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        rows = response.json()
        return rows[0] if rows else None
    except requests.RequestException as e:
        logger.error("Supabase POST %s failed: %s | body=%s", table, e, body)
        return None


def rest_patch(table: str, match_params: dict, body: dict) -> dict | None:
    """Always uses the service key."""
    url = f"{config.SUPABASE_URL}/rest/v1/{table}"
    headers = {**_headers(use_service_key=True), "Content-Type": "application/json", "Prefer": "return=representation"}
    try:
        response = requests.patch(url, headers=headers, params=match_params, json=body, timeout=10)
        response.raise_for_status()
        rows = response.json()
        return rows[0] if rows else None
    except requests.RequestException as e:
        logger.error("Supabase PATCH %s failed: %s | body=%s", table, e, body)
        return None


def bot_email_for_phone(phone: str) -> str:
    """Generates a stable internal email from a phone number.

    Supabase Auth requires a user account for every order (orders.user_id
    is a FK to auth.users). Phone-based auth requires SMS setup which this
    project doesn't use, so we create users with a bot-only email instead.
    The email is never sent to the customer — it's just a unique key.
    """
    digits = "".join(ch for ch in phone if ch.isdigit())
    return f"wa.{digits}@bot.goldfish"


def find_auth_user_id_by_email(email: str) -> str | None:
    """Looks up an existing Supabase Auth user by email. Used to recover
    when auth_admin_create_user returns 422 (user already exists).

    The admin users list API doesn't support email filtering, so we fetch
    all users (paginated) and search locally. This is called rarely — only
    for non-Israeli numbers that can't store phone_number_1 in the DB.
    """
    url = f"{config.SUPABASE_URL}/auth/v1/admin/users"
    headers = _headers(use_service_key=True)
    page = 1
    while True:
        try:
            response = requests.get(url, headers=headers, params={"per_page": 1000, "page": page}, timeout=10)
            response.raise_for_status()
            data = response.json()
            users = data.get("users") or []
            for user in users:
                if user.get("email") == email:
                    logger.info("Found existing auth user %s for email=%s", user["id"], email)
                    return user["id"]
            if len(users) < 1000:
                break
            page += 1
        except requests.RequestException as e:
            logger.error("Supabase find_auth_user_by_email failed for email=%s: %s", email, e)
            return None
    return None


def auth_admin_create_user(phone: str, customer_name: str | None) -> str | None:
    """Creates a Supabase Auth user using a bot-generated internal email.
    Phone auth (SMS) is not required. This also auto-creates a matching
    `profiles` row via a Postgres trigger on the Supabase side.
    If the user already exists (422), looks up and returns their existing ID.
    Returns the user's id, or None on failure.
    """
    url = f"{config.SUPABASE_URL}/auth/v1/admin/users"
    headers = {**_headers(use_service_key=True), "Content-Type": "application/json"}
    email = bot_email_for_phone(phone)
    body = {
        "email": email,
        "email_confirm": True,
        "user_metadata": {"customer_name": customer_name, "whatsapp_phone": phone},
    }
    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)

        if response.status_code == 422:
            # User already exists with this email — find and return their ID
            logger.warning("Auth user already exists for email=%s, looking up existing id", email)
            return find_auth_user_id_by_email(email)

        response.raise_for_status()
        user_id = response.json().get("id")
        logger.info("Created Supabase auth user %s (email=%s) for phone=%s", user_id, email, phone)
        return user_id
    except requests.RequestException as e:
        logger.error("Supabase auth admin create_user failed for phone=%s: %s", phone, e)
        return None
