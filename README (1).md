# Telegram Data-Analysis Agent Bot

## What this is
A FastAPI service that:
- long-polls Telegram for messages
- answers each one using AI Pipe (OpenAI-compatible) with a `run_python` tool
- replies with exactly one JSON object: `{"answer": ..., "log_url": "..."}`
- logs every step to `/run.jsonl` (publicly readable)

## 1. Local setup
```bash
pip install -r requirements.txt
export BOT_TOKEN="123456:ABC..."          # from BotFather
export AIPIPE_TOKEN="your-aipipe-token"    # from aipipe.org after institute-email login
export BASE_URL="http://localhost:8000"    # replace with real host once deployed
export MODEL_NAME="openai/gpt-4o"          # check aipipe.org docs for exact model name string
uvicorn bot:app --host 0.0.0.0 --port 8000
```
Then message your bot on Telegram from your own account and confirm you get
back a single clean JSON reply.

## 2. AI Pipe specifics
- Sign in at https://aipipe.org with your institute email, copy the token.
- AI Pipe exposes an OpenAI-compatible endpoint (`AIPIPE_BASE_URL`, default
  `https://aipipe.org/openai/v1`) — the OpenAI SDK just needs its `base_url`
  pointed there, which `bot.py` already does.
- **Check the exact model name string AI Pipe expects** (often prefixed,
  e.g. `openai/gpt-4o`) in their docs/dashboard before deploying — an
  incorrect model name will fail every request.
- AI Pipe tokens/credits can be limited or expire — confirm your token will
  still be valid and funded at grading time (this killed bots in past runs
  per the failure-mode table).

## 3. Push to GitHub
```bash
git init
git add bot.py requirements.txt README.md
git commit -m "telegram data-analysis agent bot"
git branch -M main
git remote add origin https://github.com/<you>/<your-repo>.git
git push -u origin main
```
Repo must be **public**.

## 4. Deploy on Render
1. New → Web Service → connect your GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
4. Environment variables:
   - `BOT_TOKEN`
   - `AIPIPE_TOKEN`
   - `BASE_URL` = `https://<your-service>.onrender.com` (set this AFTER
     Render gives you the URL, then redeploy)
   - `MODEL_NAME` (optional override)
5. Deploy. Render's free tier spins down after ~15 min idle — the bot's own
   `keep_warm_loop` pings `/health` every 10 minutes to counter this, but an
   external pinger (UptimeRobot) is a good backup.
6. **Remember:** changing env vars on Render does not auto-restart the
   service — trigger a manual deploy after editing them.

## 5. Verify
```bash
curl https://<your-host>/health
wget https://<your-host>/run.jsonl
```
Both must work from an outside network.

## 6. Test like the grader
- Message the bot fresh from Telegram, confirm exactly one JSON reply, no
  prose/fences around it.
- Send a multi-turn sequence (e.g. "I will send data next." then the actual
  question) and confirm it replies to **both** messages.
- Ask a question that needs real computation and time it — must return well
  under 300s.

## 7. Register on SEEK
```
https://github.com/<you>/<your-repo>, your_bot_username
```
(bot username, no `@`, must end in `bot`)

## Notes on the code
- `run_python` executes model-generated code with `exec()` server-side —
  this is expected/required by the task, but be aware it has no sandboxing.
  Only run this on a throwaway free-tier instance, not anywhere sensitive.
- `WALL_CLOCK_BUDGET_SECONDS = 210` leaves a 90s safety margin under the
  grader's 300s timeout; past that the agent is forced to answer without
  further tool calls.
- History is kept in memory per `chat_id` (last 20 turns) — this resets if
  the service restarts, which is fine since each grading run is short.
