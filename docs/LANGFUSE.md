# Langfuse In This Project

This project can send traces for every WhatsApp agent turn and OpenAI call to Langfuse.

## What Is Traced

- `run_agent_turn(...)` in `agent/runner.py` is traced as an `agent` observation.
- Every OpenAI chat completion in `agent/llm.py` is traced via Langfuse's OpenAI wrapper.
- Trace attributes include:
  - `user_id`: WhatsApp phone number
  - `session_id`: WhatsApp phone number
  - `trace_name`: `whatsapp-agent-turn`
  - metadata: channel/customer name

## 1) Install SDK

Activate your venv and install Langfuse:

```bash
source venv/bin/activate
pip install langfuse
```

## 2) Add Environment Variables

Set these in `.env`:

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

US region example:

```env
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

## 3) Restart API

```bash
python api.py
```

Then send a WhatsApp message and open your Langfuse project UI to see traces.

## Troubleshooting

- If `LANGFUSE_ENABLED=true` but `langfuse` is not installed, the app falls back to normal OpenAI client and logs a warning.
- If traces don't appear:
  - verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
  - verify `LANGFUSE_BASE_URL` region
  - check server logs for Langfuse warnings/errors
