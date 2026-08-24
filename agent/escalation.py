"""Notifies a human member of staff that a conversation needs their attention.

Posts to a Slack incoming webhook if SLACK_WEBHOOK_URL is configured;
otherwise just logs loudly so escalations are never silently dropped during
local testing.
"""
import logging

import requests

from agent import config

logger = logging.getLogger(__name__)


def notify_staff(phone: str, customer_name: str | None, reason: str) -> None:
    who = customer_name or phone
    message = f"WhatsApp customer needs a human: {who} ({phone})\nReason: {reason}"

    if not config.SLACK_WEBHOOK_URL:
        logger.warning("ESCALATION (no SLACK_WEBHOOK_URL configured, logging only): %s", message)
        return

    try:
        response = requests.post(config.SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)
        response.raise_for_status()
        logger.info("Escalation notification sent to Slack for %s", phone)
    except requests.RequestException as e:
        logger.error("Failed to send Slack escalation notification: %s", e)
