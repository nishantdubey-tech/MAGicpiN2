# Magicpin VERA AI Challenge — Deployable Bot

This folder is the **deployable source code** for the Magicpin VERA AI Challenge.

Open this folder directly in **Antigravity**, review/edit the code, run it locally, then deploy the same folder to a public web service.

## 1. Project structure

```text
magicpin_vera_solution/
├── bot.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container deployment option
├── Procfile               # Web-process deployment option
├── render.yaml            # Render-style deployment configuration
├── judge_simulator.py     # Supplied challenge simulator
├── local_smoke_test.py    # Local API smoke test
├── submission.jsonl       # Generated canonical outputs
├── dataset/
│   ├── categories/
│   ├── merchants_seed.json
│   ├── customers_seed.json
│   └── triggers_seed.json
└── README.md
```

## 2. What the bot exposes

Required API:

```text
GET  /v1/healthz
GET  /v1/metadata
POST /v1/context
POST /v1/tick
POST /v1/reply
```

Optional:

```text
POST /v1/teardown
```

The main decision engine is in `bot.py`.

## 3. Open in Antigravity

1. Download/extract this ZIP.
2. Open the extracted `magicpin_vera_solution` folder in Antigravity.
3. Let Antigravity index the project.
4. Read `README.md` and `bot.py`.
5. Keep the API paths exactly as specified unless the challenge portal says otherwise.

Do not hard-code the 30 canonical examples. The challenge judge can send new/updated contexts.

## 4. Run locally

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Check:

```bash
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/metadata
```

Then, in another terminal:

```bash
python local_smoke_test.py
```

Expected health response is HTTP 200.

## 5. Important deployment rule

The challenge judge needs a **publicly reachable URL**.

After deployment, your base URL should look like:

```text
https://YOUR-BOT-DOMAIN
```

The judge endpoints are then:

```text
https://YOUR-BOT-DOMAIN/v1/healthz
https://YOUR-BOT-DOMAIN/v1/metadata
https://YOUR-BOT-DOMAIN/v1/context
https://YOUR-BOT-DOMAIN/v1/tick
https://YOUR-BOT-DOMAIN/v1/reply
```

Submit the **base bot URL** in the challenge portal if the portal asks for a bot URL. If the portal gives a more specific submission format, follow the portal's instructions.

## 6. Deploy using Render-style web hosting

This repository contains both `render.yaml` and a `Dockerfile`.

### Option A — Python build

Use:

```text
Build command:
pip install -r requirements.txt

Start command:
uvicorn bot:app --host 0.0.0.0 --port $PORT
```

Health check:

```text
/v1/healthz
```

### Option B — Docker

The included `Dockerfile` starts:

```bash
uvicorn bot:app --host 0.0.0.0 --port ${PORT:-8080}
```

If the host provides a `PORT` environment variable, the app uses it automatically.

## 7. Deploy from a Git repository

Recommended flow:

```text
Antigravity
   ↓
Edit / test
   ↓
git init
   ↓
git add .
   ↓
git commit
   ↓
Push to GitHub
   ↓
Connect repository to your hosting provider
   ↓
Deploy
   ↓
Copy public HTTPS URL
   ↓
Test /v1/healthz
   ↓
Test /v1/metadata
   ↓
Run challenge simulator
   ↓
Submit URL
```

Example Git commands:

```bash
git init
git add .
git commit -m "Magicpin VERA challenge bot"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not put API keys, passwords, or other secrets into Git.

## 8. Verify the public URL

After deployment:

```bash
curl https://YOUR-BOT-DOMAIN/v1/healthz
```

Then:

```bash
curl https://YOUR-BOT-DOMAIN/v1/metadata
```

Then test the POST API using the supplied challenge examples/simulator.

A public health endpoint returning 200 is necessary but not sufficient; also test `/v1/context`, `/v1/tick`, and `/v1/reply`.

## 9. Run the supplied judge simulator

Put the supplied challenge simulator and dataset in the same project, as included here.

Run:

```bash
python judge_simulator.py
```

If the supplied simulator requires an LLM provider/API key, configure those variables exactly as described inside the supplied simulator/challenge instructions.

Set the bot URL to your deployed service, for example:

```text
BOT_URL=https://YOUR-BOT-DOMAIN
```

Do not assume the simulator's environment-variable names beyond what its own instructions specify.

## 10. What to improve before submission

Use Antigravity to inspect the bot against the actual challenge rubric.

Focus on:

### Decision quality

Pick the strongest current signal rather than responding to every trigger.

### Specificity

Use actual context values:

```text
merchant name
owner
locality
metric
offer
date
category
trigger
```

Never invent facts.

### Category fit

Keep dentist, salon, restaurant, gym and pharmacy messaging meaningfully different.

### Merchant fit

Use merchant-specific history, performance, offers and context.

### Engagement

Prefer:

```text
one clear reason
+
one easy CTA
```

Avoid multiple questions/actions in one send.

### Conversation state

Handle:

```text
YES / commitment
NO / not interested
STOP
auto-reply
off-topic
duplicate message
```

## 11. Determinism

For the same context and simulator settings, the bot should produce stable behavior.

Avoid random selection such as:

```python
random.choice(...)
```

Do not make the core decision dependent on an uncontrolled LLM response.

If you add an LLM later, keep decision selection and factual grounding deterministic.

## 12. Versioned context

The challenge sends contexts incrementally.

For the same:

```text
scope + context_id
```

the implementation should not replace a newer version with an older/equal version.

The current implementation returns a conflict for a stale/equal version.

## 13. Before final submission

Run this checklist:

```text
[ ] Code opens and runs in Antigravity
[ ] requirements.txt installs successfully
[ ] /v1/healthz returns 200
[ ] /v1/metadata returns 200
[ ] /v1/context works
[ ] /v1/tick works
[ ] /v1/reply works
[ ] Public HTTPS URL works
[ ] No hard-coded canonical examples
[ ] No fabricated merchant/customer facts
[ ] One primary CTA
[ ] Duplicate suppression works
[ ] Customer consent is respected
[ ] Auto-reply handling works
[ ] STOP / opt-out ends outreach
[ ] "Let's do it" moves to action handling
[ ] Updated contexts are accepted
[ ] Bot stays within the judge timeout
[ ] Final judge simulator run completed
[ ] README is included
[ ] Public URL is submitted
```

## 14. Important

This package is a **starter implementation**, not a guarantee of a particular challenge score.

Before submitting, use the supplied challenge simulator and the actual challenge portal instructions to validate the implementation.

Also, do not copy challenge case-study wording directly into your output generator. Use the cases to understand the expected behavior and generate grounded responses dynamically.

