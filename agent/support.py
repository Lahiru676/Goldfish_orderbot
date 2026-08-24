"""
Support dashboard for the Gold Fish customer-support team.
Mounted at /support in api.py.

Requires SUPPORT_TOKEN env var (set in .env) sent as:
  - HTTP: Authorization: Bearer <token>
  - WebSocket / image src: ?token=<token> query param

Routes:
  GET  /support                                       → HTML dashboard
  GET  /support/api/conversations                     → sidebar conversation list
  GET  /support/api/conversations/{phone}             → full message history
  POST /support/api/conversations/{phone}/takeover    → toggle human_takeover
  POST /support/api/conversations/{phone}/reply       → send text as support agent
  POST /support/api/conversations/{phone}/media       → upload & send a file
  GET  /support/api/media/{media_id}                  → proxy WhatsApp media to browser
  WS   /support/ws?token=...                          → real-time event stream
"""
import logging
import os
from datetime import datetime, timezone

import requests
from fastapi import APIRouter, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agent import conversation_store as cs

logger = logging.getLogger(__name__)

SUPPORT_TOKEN = os.getenv("SUPPORT_TOKEN", "")

_WA_TOKEN      = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
_PHONE_ID      = os.getenv("WHATSAPP_PHONE_NUMBER_ID") or os.getenv("PHONE_NUMBER_ID", "")
_API_VER       = os.getenv("GRAPH_API_VERSION", "v25.0")
_SEND_URL      = f"https://graph.facebook.com/{_API_VER}/{_PHONE_ID}/messages"
_MEDIA_UPLOAD  = f"https://graph.facebook.com/{_API_VER}/{_PHONE_ID}/media"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wa_headers() -> dict:
    return {"Authorization": f"Bearer {_WA_TOKEN}", "Content-Type": "application/json"}


def _send_text(to: str, text: str) -> None:
    r = requests.post(_SEND_URL, headers=_wa_headers(),
                      json={"messaging_product": "whatsapp", "to": to,
                            "type": "text", "text": {"body": text}}, timeout=20)
    logger.info("Support text → %s: %s", to, r.status_code)


def _auth(authorization: str = "") -> None:
    if not SUPPORT_TOKEN:
        raise HTTPException(503, "SUPPORT_TOKEN not configured on server")
    if authorization != f"Bearer {SUPPORT_TOKEN}":
        raise HTTPException(401, "Invalid support token")


def _auth_token(token: str, authorization: str) -> None:
    if not SUPPORT_TOKEN:
        raise HTTPException(503, "SUPPORT_TOKEN not configured")
    if token != SUPPORT_TOKEN and authorization != f"Bearer {SUPPORT_TOKEN}":
        raise HTTPException(401, "Invalid support token")


# ── WebSocket manager ─────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._ws: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._ws.append(ws)
        logger.info("Support WS +1 (%d total)", len(self._ws))

    def disconnect(self, ws: WebSocket) -> None:
        self._ws = [c for c in self._ws if c is not ws]
        logger.info("Support WS -1 (%d remaining)", len(self._ws))

    async def broadcast(self, data: dict) -> None:
        dead = []
        for ws in list(self._ws):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
router  = APIRouter(prefix="/support")


# ── Pydantic bodies ───────────────────────────────────────────────────────────

class TakeoverBody(BaseModel):
    active: bool

class ReplyBody(BaseModel):
    text: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@router.get("/api/conversations")
async def list_conversations(authorization: str = Header(default="")):
    _auth(authorization)
    return cs.list_all_conversations()


@router.get("/api/conversations/{phone}")
async def get_conversation(phone: str, authorization: str = Header(default="")):
    _auth(authorization)
    return cs.get_conversation_messages(phone)


@router.post("/api/conversations/{phone}/takeover")
async def set_takeover(phone: str, body: TakeoverBody, authorization: str = Header(default="")):
    _auth(authorization)
    cs.set_human_takeover(phone, body.active)
    await manager.broadcast({"type": "takeover", "phone": phone, "active": body.active, "ts": _now()})
    logger.info("Dashboard: human_takeover=%s for %s", body.active, phone)
    return {"ok": True}


@router.post("/api/conversations/{phone}/reply")
async def send_reply(phone: str, body: ReplyBody, authorization: str = Header(default="")):
    _auth(authorization)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty message")
    _send_text(phone, text)
    cs.append_support_message(phone, text)
    await manager.broadcast({"type": "message", "phone": phone, "text": text,
                              "direction": "outbound", "from_support": True, "ts": _now()})
    return {"ok": True}


@router.post("/api/conversations/{phone}/media")
async def send_media(phone: str, file: UploadFile, authorization: str = Header(default="")):
    _auth(authorization)
    data = await file.read()
    ct = file.content_type or "application/octet-stream"

    # Upload to WhatsApp
    r = requests.post(
        _MEDIA_UPLOAD,
        headers={"Authorization": f"Bearer {_WA_TOKEN}"},
        files={"file": (file.filename, data, ct)},
        data={"messaging_product": "whatsapp", "type": ct},
        timeout=60,
    )
    if not r.ok:
        logger.error("WA media upload failed: %s %s", r.status_code, r.text[:300])
        raise HTTPException(502, "WhatsApp media upload failed")

    media_id = r.json().get("id")

    # Decide message type
    if ct.startswith("image/"):
        msg_type, payload = "image",    {"id": media_id}
    elif ct.startswith("audio/"):
        msg_type, payload = "audio",    {"id": media_id}
    elif ct.startswith("video/"):
        msg_type, payload = "video",    {"id": media_id}
    else:
        msg_type, payload = "document", {"id": media_id, "filename": file.filename or "file"}

    requests.post(_SEND_URL, headers=_wa_headers(),
                  json={"messaging_product": "whatsapp", "to": phone,
                        "type": msg_type, msg_type: payload}, timeout=30)

    label = file.filename or msg_type
    cs.append_support_message(phone, f"[{msg_type}: {label}]")
    await manager.broadcast({"type": "message", "phone": phone,
                              "text": "", "media_type": msg_type,
                              "media_id": None, "filename": file.filename,
                              "direction": "outbound", "from_support": True, "ts": _now()})
    return {"ok": True}


@router.get("/api/media/{media_id}")
async def proxy_media(media_id: str, token: str = "",
                      authorization: str = Header(default="")):
    _auth_token(token, authorization)
    h = {"Authorization": f"Bearer {_WA_TOKEN}"}
    r = requests.get(f"https://graph.facebook.com/{_API_VER}/{media_id}", headers=h, timeout=10)
    if not r.ok:
        raise HTTPException(404, "Media not found")
    url = r.json().get("url")
    r2  = requests.get(url, headers=h, timeout=30, stream=True)
    ct  = r2.headers.get("content-type", "application/octet-stream")
    return StreamingResponse(r2.iter_content(8192), media_type=ct)


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = ""):
    if not SUPPORT_TOKEN or token != SUPPORT_TOKEN:
        await ws.close(code=4001)
        return
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Dashboard HTML ─────────────────────────────────────────────────────────────
# Layout: customer messages always on the LEFT, bot/agent always on the RIGHT.
# Timestamps are shown on every message. Historical messages saved before
# timestamp support are clearly marked as unavailable rather than guessed.

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold Fish · Support</title>
<style>
:root {
  --sw: 320px;
  --sb-bg: #18191f;
  --sb-bd: #2a2b35;
  --sb-item: #1e1f27;
  --sb-sel: #252634;
  --sb-txt: #d4d4d8;
  --sb-dim: #71717a;
  --bg: #f0f2f5;
  --white: #ffffff;
  --bd: #e4e6ea;
  --txt: #111827;
  --dim: #6b7280;
  --accent: #2563eb;
  --accent-bg: #eff6ff;
  --green: #16a34a;
  --green-bg: #f0fdf4;
  --amber: #b45309;
  --amber-bg: #fffbeb;
  --red: #dc2626;
  /* Bubble colours */
  --bbl-customer: #ffffff;
  --bbl-customer-bd: #e4e6ea;
  --bbl-bot: #f1f5f9;
  --bbl-agent: #2563eb;
  --bbl-agent-txt: #ffffff;
  --r: 16px;
  --r-sm: 8px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:14px;background:var(--bg);color:var(--txt);
  height:100vh;display:flex;overflow:hidden;
  -webkit-font-smoothing:antialiased;
}

/* ── Login ──────────────────────────────────────────────────────── */
#login{
  position:fixed;inset:0;background:var(--bg);
  display:flex;align-items:center;justify-content:center;z-index:100;
}
.lcard{
  background:var(--white);border:1px solid var(--bd);border-radius:20px;
  padding:40px 36px;width:360px;
  box-shadow:0 8px 32px rgba(0,0,0,.08);
  display:flex;flex-direction:column;gap:14px;
}
.lcard .ico{font-size:44px;text-align:center}
.lcard h2{font-size:20px;font-weight:700;text-align:center}
.lcard p{font-size:13px;color:var(--dim);text-align:center}
.lcard input{
  padding:11px 14px;border:1.5px solid var(--bd);border-radius:10px;
  font-size:14px;outline:none;width:100%;transition:border-color .15s;
}
.lcard input:focus{border-color:var(--accent)}
.lcard .btn{
  width:100%;padding:12px;background:var(--accent);color:#fff;
  border:none;border-radius:10px;font-size:14px;font-weight:600;
  cursor:pointer;transition:background .15s;
}
.lcard .btn:hover{background:#1d4ed8}
.lerr{font-size:12px;color:var(--red);text-align:center;display:none}

/* ── App shell ──────────────────────────────────────────────────── */
#app{display:none;width:100%;height:100%}
.layout{display:flex;height:100%}

/* ── Sidebar ────────────────────────────────────────────────────── */
aside{
  width:var(--sw);background:var(--sb-bg);flex-shrink:0;
  border-right:1px solid var(--sb-bd);
  display:flex;flex-direction:column;
}
.sb-top{
  padding:16px 14px 12px;border-bottom:1px solid var(--sb-bd);
  display:flex;flex-direction:column;gap:10px;
}
.sb-brand{display:flex;align-items:center;gap:8px}
.sb-brand-name{font-size:15px;font-weight:700;color:#fff}
.sb-count{
  font-size:10px;font-weight:700;color:#bfdbfe;background:rgba(37,99,235,.22);
  padding:3px 7px;border-radius:20px;min-width:22px;text-align:center;
}
.ws-chip{
  margin-left:auto;display:flex;align-items:center;gap:5px;
  font-size:11px;color:var(--sb-dim);
  background:rgba(255,255,255,.06);padding:3px 9px;border-radius:20px;
}
.wsdot{width:6px;height:6px;border-radius:50%;background:#4b5563}
.wsdot.on{background:#22c55e}
.sb-search{
  padding:8px 12px;background:rgba(255,255,255,.07);
  border:1px solid var(--sb-bd);border-radius:var(--r-sm);
  color:var(--sb-txt);font-size:13px;outline:none;width:100%;
}
.sb-search::placeholder{color:var(--sb-dim)}
.sb-search:focus{border-color:#3b82f6}
.tabs{display:flex;gap:2px}
.tab{
  flex:1;padding:5px 4px;text-align:center;font-size:12px;font-weight:500;
  color:var(--sb-dim);border-radius:6px;cursor:pointer;transition:all .15s;
}
.tab.on{background:rgba(37,99,235,.22);color:#93c5fd}
.tab:hover:not(.on){background:rgba(255,255,255,.05);color:var(--sb-txt)}

.clist{flex:1;overflow-y:auto}
.clist::-webkit-scrollbar{width:3px}
.clist::-webkit-scrollbar-thumb{background:#2a2b35;border-radius:2px}
.list-empty{padding:40px 20px;text-align:center;color:var(--sb-dim);font-size:12px;line-height:1.6}

.ci{
  display:flex;align-items:flex-start;gap:10px;padding:11px 14px;
  cursor:pointer;border-bottom:1px solid rgba(255,255,255,.04);
  transition:background .1s;position:relative;
}
.ci:hover{background:var(--sb-item)}
.ci.sel{background:var(--sb-sel)}
.ci.sel::before{
  content:'';position:absolute;left:0;top:0;bottom:0;
  width:3px;background:var(--accent);border-radius:0 2px 2px 0;
}
.ci.hum::before{background:var(--amber)!important}
.ci.hum{border-left:3px solid var(--amber);padding-left:11px}

.av{
  width:36px;height:36px;border-radius:50%;font-size:13px;font-weight:700;
  color:#fff;display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.cb{flex:1;min-width:0}
.ctop{display:flex;justify-content:space-between;align-items:baseline;gap:4px}
.cname{font-size:13px;font-weight:600;color:var(--sb-txt);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ctime{font-size:11px;color:var(--sb-dim);white-space:nowrap;flex-shrink:0}
.ci:first-child .ctime{color:#93c5fd;font-weight:600}
.cprev{font-size:12px;color:var(--sb-dim);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
.ctags{display:flex;gap:4px;margin-top:5px}
.tag{font-size:10px;font-weight:600;padding:2px 7px;border-radius:20px}
.tag-bot  {background:rgba(37,99,235,.14);color:#93c5fd}
.tag-hum  {background:rgba(180,83,9,.2); color:#fcd34d}
.tag-reg  {background:rgba(22,163,74,.12);color:#86efac}

/* ── Main panel ─────────────────────────────────────────────────── */
main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.empty{
  flex:1;display:flex;align-items:center;justify-content:center;
  flex-direction:column;gap:10px;color:var(--dim);
}
.empty-ico{font-size:44px;opacity:.25}

/* Chat view */
#cv{flex:1;display:none;flex-direction:column;overflow:hidden}

/* Header */
.ch{
  background:var(--white);border-bottom:1px solid var(--bd);
  padding:12px 20px;display:flex;align-items:center;gap:12px;flex-shrink:0;
}
.backbtn{display:none;background:none;border:0;color:var(--dim);font-size:28px;line-height:1;cursor:pointer}
.ch-av{width:38px;height:38px;border-radius:50%;font-size:14px;font-weight:700;
  color:#fff;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ch-info{flex:1;min-width:0}
.ch-name{font-size:15px;font-weight:700}
.ch-sub{font-size:12px;color:var(--dim);margin-top:1px}
.tkbtn{
  padding:8px 18px;border-radius:20px;border:none;
  font-size:13px;font-weight:600;cursor:pointer;
  display:flex;align-items:center;gap:6px;
  transition:all .15s;white-space:nowrap;flex-shrink:0;
}
.tkbtn.bot{background:var(--green-bg);color:var(--green)}
.tkbtn.bot:hover{background:#dcfce7}
.tkbtn.hum{background:var(--amber-bg);color:var(--amber)}
.tkbtn.hum:hover{background:#fef3c7}
/* ── Message thread ─────────────────────────────────────────────── */
/*
  LAYOUT RULE:
    Customer messages  → LEFT  (.from-c)
    Bot messages       → RIGHT (.from-b)
    Agent messages     → RIGHT (.from-a)
*/
#msgs{
  flex:1;overflow-y:auto;
  padding:16px 24px 10px;
  display:flex;flex-direction:column;gap:0;
  background:var(--bg);
}
#msgs::-webkit-scrollbar{width:4px}
#msgs::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:2px}

/* Day separator */
.dsep{
  display:flex;align-items:center;gap:8px;
  font-size:11px;font-weight:600;color:var(--dim);
  text-transform:uppercase;letter-spacing:.5px;
  margin:14px 0 8px;
}
.dsep::before,.dsep::after{content:'';flex:1;height:1px;background:var(--bd)}

/* Message row — the outer container sets left/right alignment */
.mrow{
  display:flex;         /* makes the inner .mi float left or right */
  width:100%;
  margin:2px 0;
}
.mrow.from-c { justify-content:flex-start  }  /* customer  → LEFT  */
.mrow.from-b { justify-content:flex-end    }  /* bot       → RIGHT */
.mrow.from-a { justify-content:flex-end    }  /* agent     → RIGHT */

/* Inner wrapper keeps bubble + meta together */
.mi{
  display:flex;flex-direction:column;
  max-width:62%;
}
.mrow.from-c .mi { align-items:flex-start }
.mrow.from-b .mi { align-items:flex-end   }
.mrow.from-a .mi { align-items:flex-end   }

/* Sender label */
.mlbl{
  font-size:11px;font-weight:600;color:var(--dim);
  margin-bottom:3px;padding:0 4px;
}

/* Bubble */
.bbl{
  padding:9px 13px;border-radius:var(--r);
  font-size:14px;line-height:1.55;word-break:break-word;white-space:pre-wrap;
}

/* Customer bubble — white card, left-pointing tail */
.from-c .bbl{
  background:var(--bbl-customer);
  border:1px solid var(--bbl-customer-bd);
  border-top-left-radius:3px;
  box-shadow:0 1px 2px rgba(0,0,0,.06);
}

/* Bot bubble — light gray, right-pointing tail */
.from-b .bbl{
  background:var(--bbl-bot);
  border-top-right-radius:3px;
}

/* Agent bubble — indigo, right-pointing tail */
.from-a .bbl{
  background:var(--bbl-agent);
  color:var(--bbl-agent-txt);
  border-top-right-radius:3px;
}

/* Timestamp below bubble */
.mts{
  font-size:11px;color:var(--dim);
  margin-top:3px;padding:0 4px;
}
.mts.missing{font-style:italic;opacity:.72}
.mrow.from-c .mts{text-align:left}
.mrow.from-b .mts,
.mrow.from-a .mts{text-align:right}

/* Spacing between consecutive messages from the same sender */
.mrow+.mrow.from-c.same-c { margin-top:1px }
.mrow+.mrow.from-b.same-b { margin-top:1px }
.mrow+.mrow.from-a.same-a { margin-top:1px }

/* Group: hide label for consecutive same-sender messages */
.mrow.same-c .mlbl,
.mrow.same-b .mlbl,
.mrow.same-a .mlbl { display:none }

/* First of a group — give a bit more top margin */
.mrow:not(.same-c):not(.same-b):not(.same-a){ margin-top:10px }

/* Media in bubbles */
.bbl img{max-width:220px;max-height:200px;border-radius:8px;display:block;cursor:pointer;object-fit:cover}
.doc-link{
  display:flex;align-items:center;gap:8px;
  padding:8px 12px;background:rgba(0,0,0,.06);border-radius:8px;
  text-decoration:none;color:inherit;font-size:13px;font-weight:500;
}
.from-a .doc-link{background:rgba(255,255,255,.18);color:#fff}

/* ── Reply area ─────────────────────────────────────────────────── */
.rarea{
  background:var(--white);border-top:1px solid var(--bd);
  padding:10px 16px 14px;flex-shrink:0;
}
.fprev{
  display:none;align-items:center;gap:8px;
  padding:6px 10px;background:var(--accent-bg);
  border:1px solid #bfdbfe;border-radius:var(--r-sm);margin-bottom:8px;
  font-size:13px;
}
.fprev .fname{flex:1;color:var(--accent);font-weight:500;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fprev img.thumb{height:36px;width:36px;object-fit:cover;border-radius:4px;flex-shrink:0}
.fclear{background:none;border:none;cursor:pointer;color:var(--accent);font-size:16px;padding:0 2px}
.rrow{display:flex;align-items:flex-end;gap:8px}
.abtn{
  width:38px;height:38px;border-radius:50%;
  background:none;border:1.5px solid var(--bd);
  font-size:17px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;color:var(--dim);transition:all .15s;
}
.abtn:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.abtn:disabled{opacity:.3;cursor:not-allowed}
.rinput{
  flex:1;padding:9px 14px;
  background:var(--bg);border:1.5px solid var(--bd);border-radius:20px;
  font-size:14px;font-family:inherit;resize:none;
  height:40px;max-height:120px;outline:none;color:var(--txt);
  transition:border-color .15s;
}
.rinput:focus{border-color:var(--accent);background:#fff}
.rinput:disabled{opacity:.35;cursor:not-allowed}
.rinput::placeholder{color:var(--dim)}
.sbtn{
  width:38px;height:38px;border-radius:50%;
  background:var(--accent);border:none;color:#fff;font-size:16px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;transition:background .15s;
}
.sbtn:hover:not(:disabled){background:#1d4ed8}
.sbtn:disabled{background:#93c5fd;cursor:not-allowed}
.bnotice{text-align:center;font-size:12px;color:var(--dim);padding:8px 0}

/* ── Toast ──────────────────────────────────────────────────────── */
#toast{
  position:fixed;bottom:20px;right:20px;
  background:#1f2937;color:#f9fafb;
  border-radius:10px;padding:12px 16px;font-size:13px;font-weight:500;
  display:none;z-index:999;max-width:300px;line-height:1.4;
  box-shadow:0 4px 20px rgba(0,0,0,.25);
  animation:slideup .2s ease;
}
#toast.warn{background:#7f1d1d}
@keyframes slideup{from{transform:translateY(8px);opacity:0}to{transform:translateY(0);opacity:1}}

@media (max-width:760px){
  :root{--sw:100%}
  aside{width:100%}
  main{display:none;position:fixed;inset:0;z-index:10;background:var(--bg)}
  body.chat-open aside{display:none}
  body.chat-open main{display:flex}
  .ch{padding:10px 12px}
  .backbtn{display:block}
  #msgs{padding:12px}
  .mi{max-width:84%}
}
</style>
</head>
<body>

<!-- Login -->
<div id="login">
  <div class="lcard">
    <div class="ico">🐟</div>
    <h2>Gold Fish Support</h2>
    <p>Enter your support token to sign in</p>
    <input id="tok" type="password" placeholder="Support token">
    <button class="btn" onclick="doLogin()">Sign in</button>
    <div id="lerr" class="lerr">Incorrect token — please try again</div>
  </div>
</div>

<!-- App -->
<div id="app">
<div class="layout">

<!-- Sidebar -->
<aside>
  <div class="sb-top">
    <div class="sb-brand">
      <span style="font-size:20px">🐟</span>
      <span class="sb-brand-name">Support</span>
      <span class="sb-count" id="sbcount">0</span>
      <div class="ws-chip">
        <div class="wsdot" id="wsdot"></div>
        <span id="wslbl">–</span>
      </div>
    </div>
    <input class="sb-search" id="srch" placeholder="🔍  Search…" oninput="renderList()">
    <div class="tabs">
      <div class="tab on" data-t="all"   onclick="setTab(this)">All</div>
      <div class="tab"    data-t="human" onclick="setTab(this)">👤 Human</div>
      <div class="tab"    data-t="wait"  onclick="setTab(this)">🔴 Waiting</div>
    </div>
  </div>
  <div class="clist" id="clist"></div>
</aside>

<!-- Main -->
<main>
  <div class="empty" id="empty">
    <div class="empty-ico">💬</div>
    <p style="font-size:14px">Select a conversation</p>
  </div>

  <div id="cv">
    <!-- Header -->
    <div class="ch">
      <button class="backbtn" type="button" aria-label="Back to conversations" onclick="closeChat()">‹</button>
      <div class="ch-av" id="hav"></div>
      <div class="ch-info">
        <div class="ch-name" id="hname"></div>
        <div class="ch-sub"  id="hsub"></div>
      </div>
      <button class="tkbtn bot" id="tkbtn" onclick="toggleTakeover()">🤖 Take over</button>
    </div>

    <!-- Messages -->
    <div id="msgs"></div>

    <!-- Reply -->
    <div class="rarea">
      <div class="fprev" id="fprev">
        <span id="fpico">📄</span>
        <div class="fname" id="fname"></div>
        <button class="fclear" onclick="clearFile()">✕</button>
      </div>
      <div class="rrow">
        <button class="abtn" id="abtn" onclick="document.getElementById('fi').click()" disabled>📎</button>
        <input type="file" id="fi" accept="image/*,.pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.zip" style="display:none" onchange="onFile(this)">
        <textarea class="rinput" id="ri" rows="1" disabled placeholder="Bot is active — take over to reply"></textarea>
        <button class="sbtn" id="sb" onclick="sendMsg()" disabled>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
      <div class="bnotice" id="bn"></div>
    </div>
  </div>
</main>
</div>
</div>

<div id="toast"></div>

<script>
'use strict';

// ── State ──────────────────────────────────────────────────────────
let tok        = localStorage.getItem('gf_tok') || '';
let phone      = null;
let convs      = [];
let humanOn    = false;
let ws         = null;
let activeTab  = 'all';
let pendingFile= null;
let lastRole   = null;

// ── Bootstrap ──────────────────────────────────────────────────────
if (tok) boot();
document.getElementById('tok').addEventListener('keydown', e => { if (e.key==='Enter') doLogin(); });

async function doLogin() {
  tok = document.getElementById('tok').value.trim();
  boot();
}
async function boot() {
  const r = await api('/support/api/conversations');
  if (!r) { showLoginErr(); return; }
  localStorage.setItem('gf_tok', tok);
  document.getElementById('login').style.display = 'none';
  document.getElementById('app').style.display   = 'block';
  convs = await r.json();
  renderList();
  connectWs();
  setInterval(refreshList, 25000);
}
function showLoginErr() {
  document.getElementById('lerr').style.display = 'block';
  tok = ''; localStorage.removeItem('gf_tok');
}

// ── WebSocket ──────────────────────────────────────────────────────
function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/support/ws?token=${enc(tok)}`);
  ws.onopen    = () => setWsUi(true);
  ws.onclose   = () => { setWsUi(false); setTimeout(connectWs, 4000); };
  ws.onerror   = () => setWsUi(false);
  ws.onmessage = e  => onEvent(JSON.parse(e.data));
  setInterval(() => ws && ws.readyState===1 && ws.send('ping'), 25000);
}
function setWsUi(on) {
  document.getElementById('wsdot').className = 'wsdot' + (on ? ' on' : '');
  document.getElementById('wslbl').textContent = on ? 'Live' : 'Reconnecting…';
}
function onEvent(ev) {
  if (ev.type === 'message') {
    const c = convs.find(x => x.phone === ev.phone);
    if (c) {
      c.last_message_text = ev.text || (ev.filename ? `📎 ${ev.filename}` : `[${ev.media_type||'media'}]`);
      c.last_ts = ev.ts || null;
    }
    renderList();
    if (ev.phone === phone) appendMsg(ev);
    else if (ev.direction==='inbound' && c?.human_takeover)
      toast(`💬 New message from ${c.customer_name || ev.phone}`);
  }
  if (ev.type === 'takeover') {
    const c = convs.find(x => x.phone === ev.phone);
    if (c) c.human_takeover = ev.active;
    renderList();
    if (ev.phone === phone) { humanOn = ev.active; updateReplyUi(); }
  }
  if (ev.type === 'escalation') {
    toast(`🚨 ${ev.name||ev.phone} needs a human agent`, true);
    refreshList();
  }
}

// ── Sidebar ────────────────────────────────────────────────────────
async function refreshList() {
  const r = await api('/support/api/conversations');
  if (r) { convs = await r.json(); renderList(); }
}
function setTab(el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('on'));
  el.classList.add('on'); activeTab = el.dataset.t; renderList();
}

const AV_COLORS = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#6366f1'];
function avColor(s) {
  let h = 0;
  for (let i=0;i<(s||'').length;i++) h = (h*31 + s.charCodeAt(i)) & 0xffffff;
  return AV_COLORS[Math.abs(h) % AV_COLORS.length];
}
function initials(n) {
  const p = (n||'').trim().split(/\s+/);
  return p.length >= 2 ? (p[0][0]||'')+(p[1][0]||'') : (n||'').substring(0,2).toUpperCase() || '?';
}
function fmtSidebarTime(ts) {
  if (!ts) return '';
  const d = new Date(ts), now = new Date();
  if (d.toDateString() === now.toDateString())
    return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  const diff = (now - d) / 86400000;
  if (diff < 7) return d.toLocaleDateString([], {weekday:'short'});
  return d.toLocaleDateString([], {month:'short', day:'numeric'});
}

function renderList() {
  const q = document.getElementById('srch').value.toLowerCase();
  let list = convs.filter(c => {
    const name = (c.customer_name||'').toLowerCase();
    if (q && !name.includes(q) && !c.phone.includes(q)) return false;
    if (activeTab==='human' && !c.human_takeover) return false;
    if (activeTab==='wait'  && !c.human_takeover) return false;
    return true;
  });
  list.sort((a,b) => {
    const ta = a.last_ts ? new Date(a.last_ts).getTime() : 0;
    const tb = b.last_ts ? new Date(b.last_ts).getTime() : 0;
    return (tb - ta) || String(a.customer_name||a.phone).localeCompare(String(b.customer_name||b.phone));
  });
  document.getElementById('sbcount').textContent = list.length;
  const html = list.map(c => {
    const name  = c.customer_name || c.phone;
    const col   = avColor(name);
    const init  = initials(name);
    const prev  = c.last_message_text || '–';
    const time  = fmtSidebarTime(c.last_ts);
    const sel   = c.phone===phone ? 'sel' : '';
    const hcls  = c.human_takeover ? 'hum' : '';
    const tag   = c.human_takeover
      ? '<span class="tag tag-hum">👤 Human</span>'
      : c.registration_active
        ? '<span class="tag tag-reg">📝 Registering</span>'
        : '<span class="tag tag-bot">🤖 Bot</span>';
    return `<div class="ci ${sel} ${hcls}" onclick="selectConv('${esc(c.phone)}')">
      <div class="av" style="background:${col}">${esc(init)}</div>
      <div class="cb">
        <div class="ctop">
          <div class="cname">${esc(name)}</div>
          ${time ? `<div class="ctime">${esc(time)}</div>` : ''}
        </div>
        <div class="cprev" dir="auto">${esc(prev)}</div>
        <div class="ctags">${tag}</div>
      </div>
    </div>`;
  }).join('');
  document.getElementById('clist').innerHTML = html ||
    '<div class="list-empty">No conversations match this view.<br>Try another filter or search.</div>';
}

// ── Conversation view ──────────────────────────────────────────────
async function selectConv(p) {
  document.body.classList.add('chat-open');
  phone   = p;
  const c = convs.find(x => x.phone===p);
  humanOn = c?.human_takeover || false;
  lastRole= null;
  renderList();
  document.getElementById('empty').style.display = 'none';
  const cv = document.getElementById('cv');
  cv.style.display = 'flex'; cv.style.flexDirection = 'column';

  const name = c?.customer_name || p;
  const col  = avColor(name);
  const init = initials(name);
  document.getElementById('hav').style.background = col;
  document.getElementById('hav').textContent       = init;
  document.getElementById('hname').textContent     = name;
  document.getElementById('hsub').textContent      = '📱 ' + p;

  updateReplyUi();

  const msgs = document.getElementById('msgs');
  msgs.innerHTML = '<div style="text-align:center;color:#9ca3af;font-size:13px;padding:24px 0">Loading…</div>';
  lastRole = null;

  const r = await api(`/support/api/conversations/${enc(p)}`);
  msgs.innerHTML = '';
  lastRole = null;
  if (r) {
    const data = await r.json();
    // Group by date for day separators
    let lastDay = null;
    data.forEach(m => {
      const day = m.ts ? dayLabel(m.ts) : null;
      if (day && day !== lastDay) { appendDaySep(day); lastDay = day; lastRole = null; }
      appendMsg(m);
    });
    msgs.scrollTop = msgs.scrollHeight;
  }
}

function dayLabel(ts) {
  const d = new Date(ts), now = new Date();
  if (d.toDateString()===now.toDateString()) return 'Today';
  const y = new Date(now); y.setDate(now.getDate()-1);
  if (d.toDateString()===y.toDateString()) return 'Yesterday';
  return d.toLocaleDateString([], {weekday:'long', month:'long', day:'numeric'});
}

function appendDaySep(label) {
  const el = document.createElement('div');
  el.className = 'dsep'; el.textContent = label;
  document.getElementById('msgs').appendChild(el);
}

function updateReplyUi() {
  const btn = document.getElementById('tkbtn');
  const ri  = document.getElementById('ri');
  const sb  = document.getElementById('sb');
  const ab  = document.getElementById('abtn');
  const bn  = document.getElementById('bn');

  if (humanOn) {
    btn.className = 'tkbtn hum'; btn.textContent = '👤 Hand back to bot';
    ri.disabled = false; sb.disabled = false; ab.disabled = false;
    ri.placeholder = 'Type a message to the customer… (Enter to send)';
    bn.textContent = '';
  } else {
    btn.className = 'tkbtn bot'; btn.textContent = '🤖 Take over';
    ri.disabled = true; sb.disabled = true; ab.disabled = true;
    ri.placeholder = 'Bot is active — take over to reply';
    bn.textContent = 'The bot is handling this conversation. Click "Take over" to reply as a human agent.';
    clearFile();
  }
}

async function toggleTakeover() {
  if (!phone) return;
  const newActive = !humanOn;
  const r = await api(`/support/api/conversations/${enc(phone)}/takeover`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({active: newActive}),
  });
  if (r) {
    humanOn = newActive;
    const c = convs.find(x => x.phone===phone);
    if (c) c.human_takeover = newActive;
    updateReplyUi(); renderList();
  }
}

// ── Append a message bubble ────────────────────────────────────────
function msgRole(ev) {
  // Normalise the event/message object to one of: 'customer' | 'bot' | 'agent'
  const isSupport = ev.from_support === true;
  const role = ev.role || (ev.direction==='inbound' ? 'user' : 'assistant');
  if (role === 'user')                   return 'customer';
  if (role === 'assistant' && isSupport) return 'agent';
  return 'bot';
}

function appendMsg(ev) {
  const msgs = document.getElementById('msgs');
  const role  = msgRole(ev);                    // 'customer' | 'bot' | 'agent'
  const cls   = {customer:'from-c', bot:'from-b', agent:'from-a'}[role];
  const lbl   = {customer:'Customer', bot:'Bot 🤖', agent:'Support Agent 👤'}[role];

  // Group: mark as "same sender" if previous message is from the same role
  const prevKey = {customer:'c', bot:'b', agent:'a'}[role];
  const sameClass = (lastRole === prevKey) ? `same-${prevKey}` : '';
  lastRole = prevKey;

  // Build bubble content
  const text     = ev.content || ev.text || '';
  const hasMedia = ev.media_type && ev.media_type !== 'text';
  let   inner    = '';
  if (hasMedia) {
    inner = renderMedia(ev, role==='agent');
    if (text && !text.startsWith('[')) inner += `<div style="margin-top:5px" dir="auto">${esc(text)}</div>`;
  } else {
    inner = `<span class="msg-text" dir="auto">${esc(text)}</span>`;
  }

  const hasTime = isValidDate(ev.ts);
  const timeStr = hasTime ? fmtTime(ev.ts) : 'Time unavailable';
  const dateTxt = hasTime ? `<span style="margin-right:6px;opacity:.7">${fmtDate(ev.ts)}</span>` : '';

  const row = document.createElement('div');
  row.className = `mrow ${cls} ${sameClass}`;
  row.innerHTML =
    `<div class="mi">` +
      `<div class="mlbl">${lbl}</div>` +
      `<div class="bbl">${inner}</div>` +
      `<div class="mts${hasTime ? '' : ' missing'}">${dateTxt}${timeStr}</div>` +
    `</div>`;
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
}

function fmtTime(ts) {
  if (!ts) return '';
  return new Date(ts).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}
function isValidDate(ts) {
  return Boolean(ts) && !Number.isNaN(new Date(ts).getTime());
}
function fmtDate(ts) {
  if (!ts) return '';
  const d = new Date(ts), now = new Date();
  if (d.toDateString() === now.toDateString()) return 'Today';
  const y = new Date(now); y.setDate(now.getDate()-1);
  if (d.toDateString() === y.toDateString()) return 'Yesterday';
  return d.toLocaleDateString([], {weekday:'short', month:'short', day:'numeric'});
}

function renderMedia(ev, isAgent) {
  const {media_type:mt, media_id, filename} = ev;
  if ((mt==='image'||mt==='sticker') && media_id)
    return `<img src="/support/api/media/${enc(media_id)}?token=${enc(tok)}" alt="Image"
                 onclick="window.open(this.src,'_blank')" loading="lazy">`;
  if (media_id && filename)
    return `<a class="doc-link" href="/support/api/media/${enc(media_id)}?token=${enc(tok)}"
               target="_blank" download="${esc(filename)}">${dico(mt)} <span>${esc(filename)}</span></a>`;
  const lbl = filename ? `📎 ${filename}` : mt ? `[${mt}]` : '[file]';
  return `<span>${esc(lbl)}</span>`;
}
function dico(t) { return {document:'📄',audio:'🎵',video:'🎬',sticker:'🎭'}[t]||'📎'; }

// ── Send ───────────────────────────────────────────────────────────
async function sendMsg() {
  if (!phone || !humanOn) return;
  if (pendingFile) { await sendFile(); return; }
  const ri   = document.getElementById('ri');
  const text = ri.value.trim();
  if (!text) return;
  ri.value = ''; resetH(ri);
  const r = await api(`/support/api/conversations/${enc(phone)}/reply`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text}),
  });
  if (!r) { ri.value = text; toast('Failed to send — check connection', true); }
}

function onFile(input) {
  const file = input.files[0]; if (!file) return;
  const isImg = file.type.startsWith('image/');
  const fprev = document.getElementById('fprev');
  const fpico = document.getElementById('fpico');
  const fname = document.getElementById('fname');
  if (isImg) {
    const url = URL.createObjectURL(file);
    fpico.innerHTML = `<img class="thumb" src="${url}" alt="">`;
    pendingFile = {file, objectUrl: url};
  } else {
    fpico.textContent = '📄';
    pendingFile = {file};
  }
  fname.textContent = file.name;
  fprev.style.display = 'flex';
  input.value = '';
}
function clearFile() {
  if (pendingFile?.objectUrl) URL.revokeObjectURL(pendingFile.objectUrl);
  pendingFile = null;
  document.getElementById('fprev').style.display = 'none';
  document.getElementById('fpico').textContent = '📄';
}
async function sendFile() {
  if (!pendingFile || !phone) return;
  const fd = new FormData(); fd.append('file', pendingFile.file);
  clearFile();
  const r = await fetch(`/support/api/conversations/${enc(phone)}/media`, {
    method:'POST', headers:{Authorization:`Bearer ${tok}`}, body:fd,
  });
  if (!r || !r.ok) toast('File upload failed', true);
}

// Input auto-grow + Enter to send
document.getElementById('ri').addEventListener('input', function() {
  this.style.height = '40px';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});
document.getElementById('ri').addEventListener('keydown', e => {
  if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});
function resetH(el) { el.style.height = '40px'; }

// ── Helpers ────────────────────────────────────────────────────────
async function api(path, opts={}) {
  try {
    opts.headers = {Authorization:`Bearer ${tok}`, ...(opts.headers||{})};
    const r = await fetch(path, opts);
    if (r.status===401) { showLoginErr(); return null; }
    if (!r.ok) return null;
    return r;
  } catch { return null; }
}
let tTimer;
function toast(msg, warn=false) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = warn?'warn':'';
  t.style.display = 'block';
  clearTimeout(tTimer); tTimer = setTimeout(()=>t.style.display='none', 5000);
}
function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function enc(s) { return encodeURIComponent(s||''); }
function closeChat() { document.body.classList.remove('chat-open'); }
</script>
</body>
</html>"""
