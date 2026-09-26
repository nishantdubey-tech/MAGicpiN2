#!/usr/bin/env python3
"""Minimal HTTP smoke test. Start the bot first with uvicorn bot:app --port 8080."""
import json
from urllib.request import Request, urlopen

BASE = "http://localhost:8080"

def get(path):
    with urlopen(BASE + path, timeout=5) as r:
        print(path, r.status, r.read().decode())

def post(path, body):
    data = json.dumps(body).encode()
    req = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as r:
        print(path, r.status, r.read().decode())

get("/v1/healthz")
get("/v1/metadata")

post("/v1/context", {
    "scope": "category",
    "context_id": "restaurants",
    "version": 1,
    "payload": {
        "slug": "restaurants",
        "voice": {"tone": "warm_busy_practical"},
        "peer_stats": {"avg_ctr": 0.03},
        "digest": [],
        "offer_catalog": []
    },
    "delivered_at": "2026-04-26T10:00:00Z"
})

post("/v1/context", {
    "scope": "merchant",
    "context_id": "m_test",
    "version": 1,
    "payload": {
        "merchant_id": "m_test",
        "category_slug": "restaurants",
        "identity": {
            "name": "Test Cafe",
            "owner_first_name": "Ravi",
            "city": "Delhi",
            "locality": "Saket",
            "languages": ["en", "hi"]
        },
        "performance": {"views": 1000, "calls": 20, "ctr": 0.02, "window_days": 30},
        "offers": [{"title": "Lunch Thali @ ₹149", "status": "active"}],
        "conversation_history": [],
        "customer_aggregate": {},
        "signals": []
    },
    "delivered_at": "2026-04-26T10:00:00Z"
})

post("/v1/context", {
    "scope": "trigger",
    "context_id": "t_test",
    "version": 1,
    "payload": {
        "id": "t_test",
        "scope": "merchant",
        "kind": "curious_ask_due",
        "source": "internal",
        "merchant_id": "m_test",
        "customer_id": None,
        "payload": {"ask_template": "what_service_in_demand_this_week"},
        "urgency": 1,
        "suppression_key": "test:t_test",
        "expires_at": "2026-05-03T00:00:00Z"
    },
    "delivered_at": "2026-04-26T10:00:00Z"
})

post("/v1/tick", {"now": "2026-04-26T10:05:00Z", "available_triggers": ["t_test"]})

post("/v1/reply", {
    "conversation_id": "conv_test",
    "merchant_id": "m_test",
    "customer_id": None,
    "from_role": "merchant",
    "message": "Not interested. Stop messaging me.",
    "received_at": "2026-04-26T10:06:00Z",
    "turn_number": 2
})
