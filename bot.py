"""
TDS P1 Q5 - Data-Analyst Telegram Bot (Gemini edition)

Single-file FastAPI + Telegram long-polling bot that answers data-analysis
questions with {"answer": ..., "log_url": ...} JSON, using the Gemini API
(function calling) with a run_python tool for real computation.

Env vars required:
  BOT_TOKEN        - from @BotFather
  GEMINI_API_KEY   - from https://aistudio.google.com/apikey
  BASE_URL         - public URL of this deployment (e.g. https://x.onrender.com)
  GEMINI_MODEL     - optional, default "gemini-2.5-flash"
"""

import asyncio
import json
import os
import subprocess
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
GEMINI_API = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

LOG_PATH = Path(__file__).parent / "run.jsonl"
LOG_URL = f"{BASE_URL}/run.jsonl"

WALL_CLOCK_BUDGET = 190  # seconds — force a final answer once this elapses
PYTHON_TIMEOUT = 30      # seconds — per run_python subprocess call
HISTORY_TURNS = 20       # per-chat turns to retain

SYSTEM_PROMPT = (
    "You are a data-analysis assistant. You have a tool called run_python "
    "that executes Python 3 code in a fresh subprocess and returns its "
    "stdout/stderr. Use it whenever you need to compute, parse, transform, "
    "or verify something numerically instead of guessing.\n\n"
    "When you have the final answer, respond with ONLY a single JSON object "
    'of the form {"answer": <value>} — no prose, no markdown code fences, '
    "no explanation outside the JSON. <value> should be the most natural "
    "type for the answer (number, string, list, or object)."
)

FORCE_ANSWER_SUFFIX = (
    "\n\nIMPORTANT: You are nearly out of time. Do NOT call any more tools. "
    'Respond right now with ONLY the JSON object {"answer": <value>} '
    "representing your best current answer."
)

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

# chat_id -> deque of Gemini "contents" entries ({"role": ..., "parts": [...]})
history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))
locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

app = FastAPI()


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def log_event(event: dict):
    event["ts"] = time.time()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# --------------------------------------------------------------------------
# JSON extraction (defensive)
# --------------------------------------------------------------------------

def extract_json_object(text: str) -> dict:
    """Strip code fences, find the first *balanced* {...} pair, parse it.
    Falls back to wrapping the raw text under "answer" if nothing parses."""
    if not text:
        return {"answer": ""}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break  # try next '{'
        start = cleaned.find("{", start + 1)

    return {"answer": cleaned}


# --------------------------------------------------------------------------
# run_python tool
# --------------------------------------------------------------------------

RUN_PYTHON_DECLARATION = {
    "name": "run_python",
    "description": (
        "Execute Python 3 code in a fresh, isolated subprocess and return "
        "its stdout and stderr. Use print() to surface any values you need "
        "to see. No internet access. Times out after "
        f"{PYTHON_TIMEOUT}s."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python source code to execute.",
            }
        },
        "required": ["code"],
    },
}


def _run_python_sync(code: str) -> dict:
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
        )
        return {
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT: exceeded time limit", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": f"EXCEPTION: {e}", "returncode": -1}


async def run_python_tool(code: str) -> dict:
    return await asyncio.to_thread(_run_python_sync, code)


# --------------------------------------------------------------------------
# Gemini agent loop
# --------------------------------------------------------------------------

async def call_gemini(client: httpx.AsyncClient, contents: list, force_answer: bool) -> dict:
    system_text = SYSTEM_PROMPT + (FORCE_ANSWER_SUFFIX if force_answer else "")
    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": contents,
    }
    if not force_answer:
        payload["tools"] = [{"functionDeclarations": [RUN_PYTHON_DECLARATION]}]

    resp = await client.post(
        GEMINI_API,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


async def agent_answer(chat_id: int, user_text: str) -> dict:
    """Run the tool-calling loop for one incoming message. Returns the
    final {"answer": ...} dict (never raises)."""
    deadline = time.time() + WALL_CLOCK_BUDGET
    contents = list(history[chat_id])
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    async with httpx.AsyncClient() as client:
        try:
            while True:
                force = time.time() >= deadline
                data = await call_gemini(client, contents, force_answer=force)
                candidate = data["candidates"][0]
                parts = candidate["content"]["parts"]

                function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
                text_parts = [p["text"] for p in parts if "text" in p]

                # record the model's turn
                contents.append({"role": "model", "parts": parts})

                if function_calls and not force:
                    fn = function_calls[0]
                    code = fn.get("args", {}).get("code", "")
                    log_event({"chat_id": chat_id, "event": "tool_call",
                               "tool": fn["name"], "code": code})
                    result = await run_python_tool(code)
                    log_event({"chat_id": chat_id, "event": "tool_result",
                               "result": result})
                    contents.append({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": fn["name"],
                                "response": result,
                            }
                        }],
                    })
                    continue

                final_text = "\n".join(text_parts)
                answer = extract_json_object(final_text)
                if "answer" not in answer:
                    answer = {"answer": answer}

                history[chat_id].append({"role": "user", "parts": [{"text": user_text}]})
                history[chat_id].append({"role": "model", "parts": [{"text": final_text}]})
                return answer

        except Exception:
            err = traceback.format_exc()
            log_event({"chat_id": chat_id, "event": "error", "trace": err})
            return {"answer": "internal error"}


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

async def telegram_send(client: httpx.AsyncClient, chat_id: int, text: str):
    await client.post(f"{TELEGRAM_API}/sendMessage",
                       json={"chat_id": chat_id, "text": text})


async def handle_message(msg: dict):
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "")
    if not text:
        return

    lock = locks[chat_id]
    async with lock:
        result = await agent_answer(chat_id, text)
        result["log_url"] = LOG_URL
        reply = json.dumps(result)
        log_event({"chat_id": chat_id, "event": "final_answer", "reply": result})

        async with httpx.AsyncClient() as client:
            try:
                await telegram_send(client, chat_id, reply)
            except Exception:
                log_event({"chat_id": chat_id, "event": "send_error",
                           "trace": traceback.format_exc()})


async def telegram_poll_loop():
    offset = None
    async with httpx.AsyncClient(timeout=40) as client:
        while True:
            try:
                params = {"timeout": 30}
                if offset is not None:
                    params["offset"] = offset
                resp = await client.get(f"{TELEGRAM_API}/getUpdates", params=params)
                resp.raise_for_status()
                data = resp.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message")
                    if msg:
                        asyncio.create_task(handle_message(msg))
            except Exception:
                log_event({"event": "poll_error", "trace": traceback.format_exc()})
                await asyncio.sleep(5)


async def self_ping_loop():
    if "localhost" in BASE_URL or "127.0.0.1" in BASE_URL:
        return
    async with httpx.AsyncClient(timeout=20) as client:
        while True:
            await asyncio.sleep(600)
            try:
                await client.get(f"{BASE_URL}/health")
            except Exception:
                pass


# --------------------------------------------------------------------------
# FastAPI routes
# --------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    LOG_PATH.touch(exist_ok=True)
    asyncio.create_task(telegram_poll_loop())
    asyncio.create_task(self_ping_loop())


@app.get("/health")
async def health():
    return {"ok": True, "model": GEMINI_MODEL}


@app.get("/run.jsonl", response_class=PlainTextResponse)
async def get_log():
    if not LOG_PATH.exists():
        return ""
    return LOG_PATH.read_text()
