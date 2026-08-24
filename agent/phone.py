"""WhatsApp gives us phone numbers in international format with no leading
"+" (e.g. "972524215758"). The store's Supabase database has a check
constraint requiring Israeli LOCAL format instead (e.g. "0524215758") on
orders/profiles phone fields - discovered by testing, not documented
anywhere. This converts between the two.
"""
import hashlib
import logging

logger = logging.getLogger(__name__)

ISRAEL_COUNTRY_CODE = "972"


def looks_like_phone_number(text: str) -> bool:
    """Basic sanity check on a phone number the customer typed in chat
    (e.g. "0524215758", "052-421-5758", "+972524215758") - just enough to
    catch the LLM extracting something that obviously isn't a phone number
    at all, before we ever save it. Real format/constraint enforcement
    happens at the DB layer (see to_orders_table_phone).
    """
    digits = "".join(ch for ch in text if ch.isdigit())
    return 9 <= len(digits) <= 12


def to_local_il_format(whatsapp_phone: str) -> str:
    """Converts "972524215758" -> "0524215758". Leaves already-local or
    non-Israeli numbers unchanged (best effort - we only have Israeli
    customers today, but this shouldn't crash on anything else).
    """
    digits = "".join(ch for ch in whatsapp_phone if ch.isdigit())

    if digits.startswith(ISRAEL_COUNTRY_CODE):
        local = "0" + digits[len(ISRAEL_COUNTRY_CODE):]
        logger.info("phone: %s -> %s (local IL format)", whatsapp_phone, local)
        return local

    if digits.startswith("0") and len(digits) in (9, 10):
        # Already local format - this is the common case now that
        # customers type their own number in chat (see
        # to_international_il_format) rather than us always deriving it
        # from WhatsApp's international-format wa_id.
        return digits

    logger.warning("phone: %s doesn't look like an Israeli number, leaving as-is", whatsapp_phone)
    return digits


def to_international_il_format(local_phone: str) -> str:
    """Converts "0524215758" -> "972524215758" - the OPPOSITE direction of
    to_local_il_format. Needed because Supabase Auth's admin API requires
    international format for a user's `phone` field (no leading 0,
    matches what WhatsApp's wa_id already looks like), while the
    orders/profiles TABLES need the opposite, local format (see
    to_orders_table_phone). Customers naturally type their own number in
    local format when asked in chat, so this conversion has to happen
    before creating their Supabase Auth user.
    """
    digits = "".join(ch for ch in local_phone if ch.isdigit())

    if digits.startswith("0"):
        international = ISRAEL_COUNTRY_CODE + digits[1:]
        logger.info("phone: %s -> %s (international IL format)", local_phone, international)
        return international

    if digits.startswith(ISRAEL_COUNTRY_CODE):
        return digits

    logger.warning("phone: %s doesn't look like an Israeli number, leaving as-is", local_phone)
    return digits


def to_orders_table_phone(whatsapp_phone: str) -> str:
    """Like to_local_il_format(), but for values actually written to the
    orders/profiles tables - those have a DB check constraint requiring
    Israeli local format ("0" + 8-9 digits), which a non-Israeli number
    (e.g. a developer testing from their own foreign phone, before going
    live with real Israeli customers) can never satisfy. Real inserts
    using the raw foreign number get rejected outright by Postgres and
    silently fall back to local-only storage - this avoids that.

    For a non-Israeli number, deterministically maps it to a placeholder
    Israeli-shaped number instead (same real number -> same placeholder
    every time, so that test customer's profile/order history still
    resolves consistently). Never use this for actually messaging the
    customer - only for fields the DB itself validates.
    """
    local = to_local_il_format(whatsapp_phone)
    if local.startswith("0") and local[1:].isdigit() and len(local) in (9, 10):
        return local

    digest = hashlib.sha256(whatsapp_phone.encode()).hexdigest()
    placeholder = "05" + "".join(ch for ch in digest if ch.isdigit())[:8]
    logger.warning(
        "phone: %s isn't an Israeli number - using placeholder %s for the orders/profiles tables only",
        whatsapp_phone, placeholder,
    )
    return placeholder
