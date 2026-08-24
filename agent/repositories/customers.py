"""Resolves a WhatsApp customer to a real Supabase customer record.

Important, non-obvious fact learned by testing against the real database:
`orders.user_id` has a foreign key to Supabase's Auth `users` table, not
just to `profiles`. So placing a real order for a new customer means:
  1. Create a Supabase Auth user (this auto-creates a matching, mostly
     empty `profiles` row via a trigger on their side)
  2. Fill in that profile's customer_name and phone_number_1

If SUPABASE_SERVICE_KEY isn't configured, falls back to a local fake id so
the rest of the agent keeps working without a real database connection.
"""
import logging
import uuid
from agent import config
from agent.phone import to_international_il_format, to_local_il_format
from agent.repositories import supabase_client

logger = logging.getLogger(__name__)


def find_customer_by_whatsapp_phone(whatsapp_phone: str) -> dict | None:
    """Looks up a customer in Supabase profiles by their WhatsApp phone number.

    WhatsApp gives us international format ("972524215758"); the profiles
    table stores local Israeli format ("0524215758") - we convert before
    querying. Returns {id, customer_name, phone_number_1} or None.

    For non-Israeli numbers, phone_number_1 can't be stored (DB constraint),
    so we fall back to looking up the auth user by their bot-generated email
    (wa.<digits>@bot.goldfish) and then fetch the profile by that user's id.
    """
    if not config.SUPABASE_SERVICE_KEY:
        return None

    local_phone = to_local_il_format(whatsapp_phone)
    # Try local format first (Israeli numbers: "0524215758"), then the raw
    # WhatsApp number as fallback for non-Israeli numbers.
    for lookup_phone in dict.fromkeys([local_phone, whatsapp_phone]):
        rows = supabase_client.rest_get(
            "profiles",
            {"select": "id,customer_name,phone_number_1", "phone_number_1": f"eq.{lookup_phone}", "limit": "1"},
            use_service_key=True,
        )
        if rows:
            logger.info("Identified returning customer %s for WhatsApp phone=%s", rows[0]["id"], whatsapp_phone)
            return rows[0]

    # phone_number_1 lookup failed — the number may be non-Israeli (so
    # phone_number_1 is NULL in the DB due to the format constraint). Try
    # finding the auth user by their bot-generated email, then get the profile.
    bot_email = supabase_client.bot_email_for_phone(whatsapp_phone)
    user_id = supabase_client.find_auth_user_id_by_email(bot_email)
    if user_id:
        rows = supabase_client.rest_get(
            "profiles",
            {"select": "id,customer_name,phone_number_1", "id": f"eq.{user_id}", "limit": "1"},
            use_service_key=True,
        )
        if rows:
            logger.info(
                "Identified returning customer %s via bot email for WhatsApp phone=%s",
                user_id, whatsapp_phone,
            )
            return rows[0]
    return None


def find_or_create_customer(contact_phone: str, customer_name: str | None) -> str:
    """Returns a Supabase profiles.id (= auth user id) for this phone number,
    creating the customer if they don't exist yet.

    contact_phone is the number the customer explicitly gave us for this
    order (asked in chat) - NOT necessarily the WhatsApp number they're
    texting from. That's deliberate: the WhatsApp account isn't always the
    right contact (shared phone, ordering for someone else, etc), so the
    agent always asks and uses what the customer actually says here.
    """
    if not config.SUPABASE_SERVICE_KEY:
        fake_id = str(uuid.uuid4())
        logger.warning(
            "No SUPABASE_SERVICE_KEY configured - using a throwaway fake customer id (%s) for phone=%s",
            fake_id, contact_phone,
        )
        return fake_id

    local_phone = to_local_il_format(contact_phone)

    # Look up by phone_number_1 first (works for Israeli numbers stored in local format)
    existing = supabase_client.rest_get(
        "profiles",
        {"select": "id,customer_name", "phone_number_1": f"eq.{local_phone}", "limit": "1"},
        use_service_key=True,
    )
    if existing:
        logger.info("Found existing Supabase customer %s for phone=%s", existing[0]["id"], contact_phone)
        return existing[0]["id"]

    # Not found by phone - create a new auth user with a bot-generated email.
    # (Phone-based auth is not enabled; email auth always works.)
    user_id = supabase_client.auth_admin_create_user(contact_phone, customer_name)
    if not user_id:
        fake_id = str(uuid.uuid4())
        logger.error("Could not create a real Supabase customer for phone=%s - using fake id %s", contact_phone, fake_id)
        return fake_id

    # Patch the auto-created profiles row with name + phone.
    # phone_number_1 has an Israeli-format check constraint in the DB,
    # so only write it if the number is in local Israeli format (0 + digits).
    profile_update = {"customer_name": customer_name}
    if local_phone.startswith("0") and local_phone[1:].isdigit() and len(local_phone) in (9, 10):
        profile_update["phone_number_1"] = local_phone

    supabase_client.rest_patch(
        "profiles",
        {"id": f"eq.{user_id}"},
        profile_update,
    )
    logger.info("Created new Supabase customer %s for phone=%s", user_id, contact_phone)
    return user_id
