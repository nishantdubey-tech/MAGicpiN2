#!/usr/bin/env python3
"""
Comprehensive HTTP Smoke & Intent Test Suite for Magicpin VERA AI Bot.
Start the bot first with: uvicorn bot:app --port 8080
"""
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

BASE = "http://localhost:8080"
passed = 0
failed = 0

def check(name: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f" ({detail})" if detail else ""))

def get(path: str) -> dict:
    req = Request(BASE + path)
    with urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())

def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())

print("\n" + "=" * 60)
print("     MAGICPIN VERA AI BOT — TEST SUITE & TEST PAIRS")
print("=" * 60 + "\n")

# 1. Health & Metadata
print("--- 1. Health & Metadata ---")
try:
    hz = get("/v1/healthz")
    check("Health check", hz.get("status") == "ok", f"uptime: {hz.get('uptime_seconds')}s")
except Exception as e:
    check("Health check", False, str(e))
    sys.exit(1)

meta = get("/v1/metadata")
check("Metadata probe", bool(meta.get("team_name")), f"team: {meta.get('team_name')}, model: {meta.get('model')}")

# 2. Context Ingestion
print("\n--- 2. Context Ingestion ---")
cat_res = post("/v1/context", {
    "scope": "category",
    "context_id": "restaurants",
    "version": 1,
    "payload": {
        "slug": "restaurants",
        "voice": {"tone": "operator-practical", "vocab_taboo": ["synergy", "paradigm"]},
        "peer_stats": {"avg_ctr": 0.03},
        "digest": [],
        "offer_catalog": []
    },
    "delivered_at": "2026-04-26T10:00:00Z"
})
check("Category context push", cat_res.get("accepted") is True)

merch_res = post("/v1/context", {
    "scope": "merchant",
    "context_id": "m_test_cafe",
    "version": 1,
    "payload": {
        "merchant_id": "m_test_cafe",
        "category_slug": "restaurants",
        "identity": {
            "name": "Delhi Spice Junction",
            "owner_first_name": "Ravi",
            "city": "Delhi",
            "locality": "Saket",
            "languages": ["en", "hi"]
        },
        "performance": {"views": 1200, "calls": 24, "ctr": 0.02, "window_days": 30},
        "offers": [{"title": "Weekday Thali @ ₹149", "status": "active"}],
        "conversation_history": [],
        "customer_aggregate": {},
        "signals": []
    },
    "delivered_at": "2026-04-26T10:00:00Z"
})
check("Merchant context push", merch_res.get("accepted") is True)

trg_res = post("/v1/context", {
    "scope": "trigger",
    "context_id": "trg_test_planning",
    "version": 1,
    "payload": {
        "id": "trg_test_planning",
        "scope": "merchant",
        "kind": "active_planning_intent",
        "source": "internal",
        "merchant_id": "m_test_cafe",
        "customer_id": None,
        "payload": {"topic": "corporate_catering", "confirmed_offer": "Weekday Thali @ ₹149"},
        "urgency": 3,
        "suppression_key": "planning:m_test_cafe:corp_thali:2026-W17",
        "expires_at": "2026-05-03T00:00:00Z"
    },
    "delivered_at": "2026-04-26T10:00:00Z"
})
check("Trigger context push", trg_res.get("accepted") is True)

# 3. Message Composition via /v1/tick
print("\n--- 3. Message Composition (Tick Engine) ---")
tick_res = post("/v1/tick", {"now": "2026-04-26T10:05:00Z", "available_triggers": ["trg_test_planning"]})
actions = tick_res.get("actions", [])
check("Tick returned composition", len(actions) > 0)
if actions:
    act = actions[0]
    check("Action contains grounded body", "Delhi Spice Junction" in act["body"] or "Ravi" in act["body"] or "₹149" in act["body"])
    check("Action has valid CTA", act.get("cta") in ("binary_yes_no", "open_ended"))
    check("Action has suppression key", bool(act.get("suppression_key")))

# 4. Multi-Turn Reply Intent Test Pairs
print("\n--- 4. Multi-Turn Reply Intent Pairs ---")
test_pairs = [
    {
        "name": "Hard STOP (English)",
        "msg": "STOP! Do not contact me again.",
        "expected_action": "end",
        "desc": "Hard opt-out ends conversation"
    },
    {
        "name": "Hinglish STOP",
        "msg": "band karo bhai mat bhejo ab",
        "expected_action": "end",
        "desc": "Hindi/Hinglish opt-out detection"
    },
    {
        "name": "Commitment / Proceed",
        "msg": "Haan bilkul, please prepare the draft now!",
        "expected_action": "send",
        "desc": "Action transition on commitment"
    },
    {
        "name": "Soft Decline / Later",
        "msg": "Abhi bohot busy hoon, baad mein dekhenge",
        "expected_action": "wait",
        "desc": "Soft decline backs off politely"
    },
    {
        "name": "Scheduling Intent",
        "msg": "Can we connect next Monday instead?",
        "expected_action": "send",
        "desc": "Schedules follow-up for specific day"
    },
    {
        "name": "Pricing Inquiry",
        "msg": "Kitna charges lagega iska? Is it free?",
        "expected_action": "send",
        "desc": "Clarifies pricing uses existing offers"
    },
    {
        "name": "Information Request",
        "msg": "Tell me more details about how this works",
        "expected_action": "send",
        "desc": "Explains workflow grounded in merchant context"
    },
    {
        "name": "Out-of-Scope Request",
        "msg": "Can you file my GST tax return for April?",
        "expected_action": "send",
        "desc": "Gracefully redirects to merchant-growth"
    },
    {
        "name": "Auto-Reply Backoff (Turn 1)",
        "msg": "Thank you for contacting us! Our team will respond shortly.",
        "turn": 2,
        "expected_action": "wait",
        "desc": "Waits on first canned responder"
    },
    {
        "name": "Auto-Reply Loop Termination (Turn 2)",
        "msg": "Thank you for contacting us! Our team will respond shortly.",
        "turn": 3,
        "expected_action": "end",
        "desc": "Ends on repeated canned responder"
    },
]

for idx, tp in enumerate(test_pairs, start=1):
    conv_id = f"test_pair_conv_{idx}"
    res = post("/v1/reply", {
        "conversation_id": conv_id,
        "merchant_id": "m_test_cafe",
        "customer_id": None,
        "from_role": "merchant",
        "message": tp["msg"],
        "received_at": "2026-04-26T10:10:00Z",
        "turn_number": tp.get("turn", 2)
    })
    action = res.get("action")
    matches = (action == tp["expected_action"])
    check(f"Test Pair {idx}: {tp['name']}", matches, f"action={action} expected={tp['expected_action']}")

print("\n" + "=" * 60)
print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED (Total: {passed + failed})")
if failed == 0:
    print("SUCCESS: ALL TEST PAIRS PASSED! (100% Accuracy)")
else:
    print("WARNING: Some test pairs failed!")
print("=" * 60 + "\n")
