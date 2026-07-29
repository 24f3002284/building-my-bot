"""
Telegram data-analysis agent bot.

Architecture (see README.md for full explanation):
  - FastAPI app exposes /health and /run.jsonl
  - Background thread long-polls Telegram getUpdates
  - Each incoming message runs an agent loop against AI Pipe (OpenAI-compatible)
    with a single `run_python` tool
  - Every step (tool calls + final answers) is logged as JSONL and served publicly

Env vars required:
  BOT_TOKEN        - Telegram bot token from BotFather
  AIPIPE_TOKEN      - your aipipe.org token
  BASE_URL          - public URL of this service, e.g. https://yourapp.onrender.com
  AIPIPE_BASE_URL   - optional, defaults to https://aipipe.org/openai/v1
  MODEL_NAME        - optional, defaults to openai/gpt-4o
"""

import os
import io
import re
import sys
import json
import time
import threading
import traceback
import contextlib
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
AIPIPE_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4o")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
LOG_PATH = os.path.join(os.path.dirname(__file__), "run.jsonl")
MAX_STEPS = 10
WALL_CLOCK_BUDGET_SECONDS = 210  # stay under the grader's 300s timeout

client = OpenAI(base_url=AIPIPE_BASE_URL, api_key=AIPIPE_TOKEN)

# per-chat message history: chat_id -> list of {"role":..., "content":...}
HISTORY = {}
HISTORY_LOCK = threading.Lock()
MAX_HISTORY_TURNS = 20

app = FastAPI()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()


def log_event(event: dict):
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/run.jsonl", response_class=PlainTextResponse)
def run_log():
    if not os.path.exists(LOG_PATH):
        return ""
    with open(LOG_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# The run_python tool
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code and return whatever it prints to stdout. "
                "Use this to download datasets (requests, pandas, openpyxl, "
                "BeautifulSoup are available), compute statistics, etc. "
                "Always print() the values you need to see — nothing is "
                "returned except captured stdout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"],
            },
        },
    }
]

_EXEC_GLOBALS_TEMPLATE = {"__name__": "__main__"}


def run_python_tool(code: str, max_chars: int = 8000) -> str:
    buf = io.StringIO()
    g = dict(_EXEC_GLOBALS_TEMPLATE)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, g)
    except Exception:
        buf.write("\n--- EXCEPTION ---\n")
        buf.write(traceback.format_exc())
    out = buf.getvalue()
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


# ---------------------------------------------------------------------------
# JSON extraction (defensive)
# ---------------------------------------------------------------------------
def extract_json(text: str):
    """Find the first balanced {...} in text and parse it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a data-analysis agent answering questions sent over Telegram.

Rules:
1. Answer the LATEST user message. Earlier messages in this chat are context
   for a possibly multi-turn question (e.g. the user may send data first,
   then ask a question about it).
2. Use the run_python tool to fetch and compute anything you can - never
   guess a number you could calculate. Only fall back to answering from your
   own knowledge if fetching/computing genuinely fails after a real attempt.
3. When you have the final answer, respond with ONLY a single JSON object,
   and nothing else - no prose, no markdown code fences, no explanation.
4. The JSON object must match exactly whatever shape the question asks for
   (same keys, same nesting, correct type - number vs string vs list vs
   object). Always include a "log_url" key (any placeholder value is fine,
   it will be overwritten by the caller).
5. Never add extra keys beyond what the question asked for plus log_url.
6. If the latest message is only setup / doesn't ask a question yet (e.g.
   "I will send data next"), still reply with a small JSON acknowledgement
   such as {"answer": "ok", "log_url": "placeholder"} - you must reply to
   every message.
"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent(chat_id: int, user_text: str) -> str:
    deadline = time.time() + WALL_CLOCK_BUDGET_SECONDS

    with HISTORY_LOCK:
        history = HISTORY.setdefault(chat_id, [])
        history.append({"role": "user", "content": user_text})
        history = history[-MAX_HISTORY_TURNS:]
        HISTORY[chat_id] = history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event({"chat_id": chat_id, "type": "incoming", "text": user_text})

    final_text = None
    for step in range(MAX_STEPS):
        time_left = deadline - time.time()
        use_tools = time_left > 15  # stop giving it tools near the deadline

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS if use_tools else None,
                tool_choice="auto" if use_tools else None,
                max_tokens=1000,
            )
        except Exception as e:
            log_event({"chat_id": chat_id, "type": "llm_error", "error": str(e)})
            final_text = json.dumps({"answer": "internal error", "log_url": "placeholder"})
            break

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls and use_tools:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                code = args.get("code", "")
                result = run_python_tool(code)
                log_event(
                    {
                        "chat_id": chat_id,
                        "type": "tool_call",
                        "step": step,
                        "code": code,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            continue  # loop again with tool results appended

        # No tool call -> this is the final answer
        final_text = msg.content or ""
        log_event({"chat_id": chat_id, "type": "final_raw", "text": final_text})
        break

    if final_text is None:
        final_text = json.dumps({"answer": "internal error", "log_url": "placeholder"})

    parsed = extract_json(final_text)
    if parsed is None:
        parsed = {"answer": final_text.strip()}
    if "answer" not in parsed:
        parsed = {"answer": parsed}

    parsed["log_url"] = f"{BASE_URL}/run.jsonl" if BASE_URL else "placeholder"

    with HISTORY_LOCK:
        HISTORY[chat_id].append({"role": "assistant", "content": json.dumps(parsed)})

    log_event({"chat_id": chat_id, "type": "final_reply", "reply": parsed})

    return json.dumps(parsed, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Telegram plumbing
# ---------------------------------------------------------------------------
def send_message(chat_id: int, text: str):
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )
    except Exception as e:
        log_event({"chat_id": chat_id, "type": "send_error", "error": str(e)})


def handle_update(update: dict):
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "")
    if not text:
        return

    try:
        reply = run_agent(chat_id, text)
    except Exception:
        log_event({"chat_id": chat_id, "type": "unhandled_exception", "trace": traceback.format_exc()})
        reply = json.dumps({"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl" if BASE_URL else "placeholder"})

    send_message(chat_id, reply)


def poll_loop():
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            r.raise_for_status()
            data = r.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                # handle each update in its own thread so slow questions
                # don't block replies to other chats / messages
                threading.Thread(target=handle_update, args=(update,), daemon=True).start()
        except Exception as e:
            log_event({"type": "poll_error", "error": str(e)})
            time.sleep(3)


def keep_warm_loop():
    if not BASE_URL:
        return
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=20)
        except Exception:
            pass


@app.on_event("startup")
def startup():
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keep_warm_loop, daemon=True).start()
    log_event({"type": "startup", "base_url": BASE_URL, "model": MODEL_NAME})
