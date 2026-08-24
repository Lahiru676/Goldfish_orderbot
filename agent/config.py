"""All environment-based settings for the WhatsApp agent, in one place.

Every other module imports settings from here instead of calling
os.getenv() directly, so there's a single place to see what the agent
needs to run.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- OpenAI (used for every LLM call the agent makes) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Langfuse (optional observability for LLM traces) ---
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() in ("1", "true", "yes", "on")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# --- Supabase (the store's main database) ---
# SUPABASE_URL / SUPABASE_ANON_KEY are the *public* read-only credentials the
# storefront website itself uses in the browser - safe to read products and
# categories with, but Row Level Security blocks orders/customers data.
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
# SUPABASE_SERVICE_KEY is NOT available yet (see docs/STORE_CONTEXT.md). Once
# the store owner provides it (or a dedicated RPC), order writes/reads can
# move from the local pending-orders file to the real `orders` table.
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- MongoDB (chat history / conversation state only) ---
# If unset, the agent falls back to a local JSON file so it still works
# without a Mongo server during local development.
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "goldfish_agent")

# --- Human escalation ---
# If unset, escalations are just logged instead of pinging anyone.
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# --- Misc ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MAX_CONTEXT_MESSAGES = 12  # how many recent chat turns we feed back to the LLM each call
