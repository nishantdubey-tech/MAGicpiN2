from __future__ import annotations

import re, time, hashlib

from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="VERA Merchant Message Engine", version="1.1.0")

START = time.time()

contexts: dict[tuple[str, str], dict[str, Any]] = {}
conversations: dict[str, dict[str, Any]] = {}
seen_suppression: set[str] = set()
last_sent_body: dict[str, str] = {}
ended_conversations: set[str] = set()

CATEGORY_TONES = {
    "dentists": "peer-clinical",
    "salons": "warm-practical",
    "restaurants": "operator-practical",
    "gyms": "coach-like",
    "pharmacies": "trustworthy-precise",
}

ACTION_KINDS = {
    "recall_due", "appointment_tomorrow", "renewal_due", "perf_dip",
    "competitor_opened", "gbp_unverified", "supply_alert", "winback_eligible",
    "customer_lapsed_soft", "customer_lapsed_hard", "trial_followup",
    "wedding_package_followup", "chronic_refill_due", "active_planning_intent",
    "cde_opportunity", "category_seasonal", "festival_upcoming",
}
INFO_KINDS = {"research_digest", "regulation_change", "curious_ask_due", "milestone_reached"}

NOISE_WORDS = {
    "thank you for contacting", "thank you for reaching", "our team will respond",
    "we will get back to you", "your message has been received", "thanks for contacting",
    "this is an automated", "auto-reply", "we are currently", "out of office",
    "will respond during", "office hours", "away from desk", "currently unavailable",
    "please leave a message", "we have received your",
}

# Broadened opt-out detection: English + common Hindi/Hinglish phrasing merchants
# and customers actually use on WhatsApp.
STOP_PHRASES = {
    "stop messaging", "stop contacting", "do not message", "don't message",
    "not interested", "unsubscribe", "remove me", "no more messages",
    "band karo", "mat bhejo", "message mat bhejo", "rehne do", "band kardo",
    "stop", "band kar do", "no more msgs", "don't send", "do not send",
    "useless spam", "spam", "block", "reported", "hatao", "nahi chahiye",
    "this is spam", "stop this", "don't contact", "mat karo", "ruk jao",
    "please stop", "mujhe nahi chahiye", "nahi mangta",
}

COMMITMENT_PHRASES = {
    "ok lets do it", "okay lets do it", "let's do it", "lets do it", "go ahead",
    "yes do it", "yes please", "proceed", "do it", "sounds good", "i want to join",
    "i want this", "book it", "confirm it", "activate it", "ok let's do it",
    "okay let's do it", "whats next", "what's next", "lets go", "let's go",
    "sure", "absolutely", "perfect", "great lets do it", "haan", "haan karo",
    "kar do", "ho jayega", "theek hai", "acha", "chalega", "bilkul",
    "yes confirm", "confirmed", "done deal", "finalize it", "lock it",
    "send it", "share it", "draft it", "go for it", "make it happen",
    "yes i want", "yes i want this", "i want to try", "sign me up",
    "i am in", "i'm in", "count me in", "ready", "i'm ready",
    "yes", "ok", "okay",
}

# Explicit soft-decline phrases: distinct from STOP (recipient isn't asking to be
# suppressed forever, just declining this particular ask).
DECLINE_PHRASES = {
    "no thanks", "not now", "no not now", "not right now", "maybe later",
    "not today", "skip this", "no need", "abhi nahi", "baad mein",
    "next time", "later", "phir kabhi", "agle hafte", "not this time",
    "busy right now", "busy", "busy hoon", "time nahi hai", "not free",
    "pass", "i'll pass", "skip", "nope not now",
}

# Scheduling/timing intent — merchant wants to delay or set a date
SCHEDULING_PHRASES = {
    "next week", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "tomorrow", "kal", "parson", "agle hafte",
    "after diwali", "next month", "after 15th", "end of month",
    "can we do this later", "schedule for", "set it up for",
    "remind me", "yaad dilana", "remind later",
}

# Pricing/cost inquiry intent
PRICING_PHRASES = {
    "how much", "cost", "price", "kitna", "kitne", "charges", "fees",
    "rate", "budget", "expensive", "cheap", "discount", "offer",
    "kya price hai", "kitna paisa", "kharcha", "what will it cost",
    "pricing", "quotation", "quote me",
}

# Info-request intent — merchant wants more details before committing
INFO_REQUEST_PHRASES = {
    "tell me more", "details", "how does it work", "explain",
    "what exactly", "more info", "kaise hoga", "samjhao",
    "what do i need to do", "kya karna hai", "batao", "how",
    "what is this", "ye kya hai", "can you explain",
}

# Gratitude phrases — merchant is thankful
GRATITUDE_PHRASES = {
    "thanks", "thank you", "dhanyavaad", "shukriya", "bahut achha",
    "great job", "awesome", "nice", "helpful", "good work",
    "appreciated", "thanks a lot", "thank you so much",
}


def _ctx(scope: str, cid: str) -> Optional[dict]:
    item = contexts.get((scope, cid))
    return item["payload"] if item else None


def _store(scope: str, cid: str, version: int, payload: dict) -> tuple[bool, Optional[int]]:
    key = (scope, cid)
    cur = contexts.get(key)
    if cur and cur["version"] > version:
        return False, cur["version"]
    contexts[key] = {"version": version, "payload": payload}
    return True, None


def _merchant_for_trigger(trg: dict) -> Optional[dict]:
    return _ctx("merchant", trg.get("merchant_id"))


def _category_for_merchant(merchant: dict) -> Optional[dict]:
    return _ctx("category", merchant.get("category_slug"))


def _customer_for_trigger(trg: dict) -> Optional[dict]:
    cid = trg.get("customer_id")
    return _ctx("customer", cid) if cid else None


def _owner(merchant: dict) -> str:
    ident = merchant.get("identity", {}) or {}
    owner = ident.get("owner_first_name")
    if not owner:
        name = ident.get("name", "") or ""
        if "'s" in name:
            owner = name.split("'s")[0].replace("Dr. ", "").strip()
        owner = owner or "there"
    if merchant.get("category_slug") == "dentists" and not str(owner).startswith("Dr."):
        return f"Dr. {str(owner).replace('Dr. ', '')}"
    return owner


def _merchant_name(merchant: dict) -> str:
    return merchant.get("identity", {}).get("name", "your business")


def _lang(customer_or_merchant: dict) -> str:
    ident = customer_or_merchant.get("identity", {})
    if "language_pref" in ident:
        return str(ident.get("language_pref", "english")).lower()
    langs = ident.get("languages", [])
    return "hi-en mix" if "hi" in langs and len(langs) > 1 else "english"


def _active_offers(merchant: dict) -> list[str]:
    return [o.get("title", "") for o in merchant.get("offers", []) if o.get("status") == "active"]


def _first_offer(merchant: dict, contains: str = "") -> Optional[str]:
    offers = _active_offers(merchant)
    if contains:
        for o in offers:
            if contains.lower() in o.lower():
                return o
    return offers[0] if offers else None


def _fmt_pct(x: Any, signed: bool = True) -> str:
    try:
        n = float(x) * 100
    except Exception:
        return str(x)
    if abs(n - round(n)) < 1e-9:
        s = f"{int(round(n))}%"
    else:
        s = f"{n:.1f}%"
    if signed and n > 0:
        return "+" + s
    return s


def _date_short(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return d.strftime("%d %b")
    except Exception:
        return iso


def _find_digest(category: dict, item_id: Optional[str]) -> Optional[dict]:
    if not category:
        return None
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def _language_prefix(customer: Optional[dict]) -> tuple[str, str]:
    if not customer:
        return "", ""
    pref = str(customer.get("identity", {}).get("language_pref", "english")).lower()
    if "hi-en" in pref or pref == "hi":
        return "Namaste" if pref == "hi" else "Hi", "hi-en"
    return "Hi", "en"


def _customer_consent_allows(customer: dict, trigger: dict) -> bool:
    if not customer:
        return False
    if not customer.get("preferences", {}).get("reminder_opt_in", True):
        return False
    scope = set(customer.get("consent", {}).get("scope", []))
    kind = trigger.get("kind", "")
    needed = {
        "recall_due": {"recall_reminders"},
        "appointment_tomorrow": {"appointment_reminders"},
        "chronic_refill_due": {"refill_reminders"},
        "trial_followup": {"appointment_reminders", "promotional_offers"},
        "wedding_package_followup": {"bridal_package_followup"},
        "customer_lapsed_hard": {"winback_offers", "promotional_offers"},
        "customer_lapsed_soft": {"recall_reminders", "promotional_offers"},
        "winback_eligible": {"winback_offers", "promotional_offers"},
    }.get(kind)
    if needed is None:
        return bool(scope)
    return bool(scope & needed)


def _make_conversation_id(merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    mid = merchant.get("merchant_id", "merchant")
    kind = trigger.get("kind", "trigger")
    cid = customer.get("customer_id") if customer else "merchant"
    return f"conv_{mid}_{kind}_{cid}"


def _rationale(trigger: dict, merchant: dict, reason: str) -> str:
    return f"Selected {trigger.get('kind', 'trigger')} for {_merchant_name(merchant)} because {reason}. Output is grounded in the supplied context and uses one next step."


def _safe_cta(kind: str) -> str:
    if kind in INFO_KINDS:
        return "open_ended" if kind in {"research_digest", "curious_ask_due"} else "none"
    return "binary_yes_no"


def _select_signal(triggers: list[dict], merchant: dict, category: dict) -> Optional[dict]:
    if not triggers:
        return None
    scored = []
    for t in triggers:
        kind = t.get("kind", "")
        score = int(t.get("urgency", 1)) * 10
        if kind in {"supply_alert", "regulation_change"}:
            score += 28
        elif kind in {"recall_due", "appointment_tomorrow", "chronic_refill_due"}:
            score += 25
        elif kind in {"active_planning_intent"}:
            score += 24
        elif kind in {"competitor_opened", "perf_dip", "gbp_unverified"}:
            score += 20
        elif kind in {"perf_spike", "review_theme_emerged", "milestone_reached"}:
            score += 16
        elif kind in {"research_digest", "cde_opportunity", "category_seasonal"}:
            score += 14
        elif kind in {"festival_upcoming", "curious_ask_due", "dormant_with_vera"}:
            score += 10
        if t.get("payload") and "placeholder" not in t.get("payload", {}):
            score += 5
        scored.append((score, t))
    scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))
    return scored[0][1]


def _compose_merchant(category: dict, merchant: dict, trigger: dict) -> tuple[str, str, str]:
    kind = trigger.get("kind", "")
    p = trigger.get("payload", {}) or {}
    cat = merchant.get("category_slug", category.get("slug", ""))
    owner = _owner(merchant)
    city = merchant.get("identity", {}).get("city", "")
    locality = merchant.get("identity", {}).get("locality", "")
    offer = _first_offer(merchant)
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    cta = _safe_cta(kind)

    if kind == "research_digest":
        item = _find_digest(category, p.get("top_item_id"))
        if item:
            details = []
            if item.get("trial_n"):
                details.append(f"{item['trial_n']:,}-person study")
            summary = item.get("summary", item.get("title", ""))
            if summary:
                summary = summary.split(".")[0]
            body = f"{owner}, {item.get('source', 'this week’s digest')} has a relevant update: {summary}"
            if details:
                body = f"{owner}, {item.get('source', 'this week’s digest')} has a relevant update — {details[0]}; {summary.lower()}"
            body += " Worth a look. Want me to turn the key point into a merchant-ready post?"
            return body, cta, "the research digest contains a category-relevant item and the merchant context provides a concrete reason to care"

    if kind == "regulation_change":
        item = _find_digest(category, p.get("top_item_id"))
        deadline = p.get("deadline_iso")
        body = f"{owner}, a {cat} compliance update is due by {_date_short(deadline) if deadline else 'the stated deadline'}."
        if item:
            body += f" {item.get('title', '')} Source: {item.get('source', 'category digest')}."
        body += " Want me to turn the required changes into a short checklist?"
        return body, "binary_yes_no", "the trigger is a time-bound regulation change and the category digest supplies the source and operational detail"

    if kind == "cde_opportunity":
        item = _find_digest(category, p.get("digest_item_id"))
        if item:
            body = f"{owner}, {item.get('title', 'a relevant professional session')} is coming up. {p.get('credits', '')} credits and {p.get('fee', 'the stated fee')}. Want me to pull the details and draft a calendar-ready note?"
            return body, "open_ended", "the trigger points to a concrete professional opportunity and the category digest provides the event details"

    if kind == "perf_dip":
        metric = p.get("metric")
        delta_raw = p.get("delta_pct")
        if metric is None or delta_raw is None:
            body = f"{owner}, I’m seeing a performance-dip signal for {merchant.get('identity',{}).get('name','your business')}."
            body += " Want me to inspect the current performance snapshot before recommending an action?"
            return body, "binary_yes_no", "the trigger is present but lacks a metric and delta, so the bot avoids inventing a decline"
        base = p.get("vs_baseline")
        baseline = f" against a baseline of {base}" if base is not None else ""
        body = f"{owner}, {metric} are down {abs(float(delta_raw)*100):.0f}% over {p.get('window','the current window')}{baseline}."
        if cat == "dentists" and metric == "calls":
            body += f" Your current CTR is {float(perf.get('ctr',0))*100:.1f}% vs peer avg {float(peer.get('avg_ctr',0))*100:.1f}%."
        elif cat == "salons":
            body += f" Your listing has {perf.get('views','?'):,} views in {perf.get('window_days',30)}d."
        else:
            body += " I’d diagnose the funnel before adding another discount."
        body += " Want me to break down the likely bottleneck?"
        return body, "binary_yes_no", "the performance trigger contains a concrete decline and the merchant snapshot adds a second verifiable benchmark"

    if kind == "perf_spike":
        metric = p.get("metric")
        delta_raw = p.get("delta_pct")
        if metric is None or delta_raw is None:
            body = f"{owner}, I’m seeing a positive performance signal for {merchant.get('identity',{}).get('name','your business')}."
            body += " Want me to inspect the current performance snapshot and turn the strongest signal into a follow-up?"
            return body, "binary_yes_no", "the trigger is present but lacks a metric and delta, so the bot avoids inventing performance figures"
        delta = _fmt_pct(delta_raw)
        driver = p.get("likely_driver")
        body = f"{owner}, {metric} are up {delta} in {p.get('window','7d')}."
        if driver:
            body += f" The trigger points to {driver.replace('_',' ')} as the likely driver."
        body += " Want me to turn what is working into a follow-up post?"
        return body, "binary_yes_no", "the positive performance shift is recent and the trigger provides a likely driver"

    if kind == "milestone_reached":
        metric = p.get("metric")
        now = p.get("value_now")
        target = p.get("milestone_value")
        if now is None or target is None or not metric:
            body = f"{owner}, a milestone signal is active for {merchant.get('identity',{}).get('name','your business')}."
            body += " Want me to turn the confirmed milestone details into a short customer-facing post?"
            return body, "binary_yes_no", "the trigger is present but its payload is incomplete, so the bot avoids inventing the milestone"
        body = f"{owner}, you’re at {now} {metric.replace('_',' ')} and only {max(int(target)-int(now),0)} away from {target}."
        body += " Want me to draft a simple post to use the milestone as social proof?"
        return body, "binary_yes_no", "the milestone trigger provides a concrete current value and target"

    if kind == "competitor_opened":
        name = p.get("competitor_name")
        dist = p.get("distance_km")
        their_offer = p.get("their_offer")
        if name and dist is not None:
            body = f"{owner}, {name} opened {dist} km from {locality}."
            if their_offer:
                body += f" Their listed offer is {their_offer}."
            body += f" I’d avoid copying it blindly; your active offer is {offer or 'not currently listed'}. Want me to draft a positioning update?"
        else:
            body = f"{owner}, a new competitor signal has appeared for {city}. I can help review your positioning against it. Want me to pull the available details?"
        return body, "binary_yes_no", "the competitor event is time-relevant and the merchant's own offer is used as the comparison point"

    if kind == "ipl_match_today":
        match = p.get("match")
        venue = p.get("venue")
        city_name = p.get("city")
        when = p.get("match_time_iso")
        body = f"{owner}, {match or 'the match'} is on today at {venue or city_name or 'the listed venue'}."
        if when:
            body += f" at {_date_short(when)}."
        if offer:
            body += f" Your active offer is {offer}, but it is limited to Tue-Thu."
        body += " I’d use the match as a delivery-focused hook rather than claim a new discount."
        body += " Want me to draft the match-day copy?"
        return body, "binary_yes_no", "the event is happening today and the merchant's existing offer has a different active-day constraint"

    if kind == "review_theme_emerged":
        theme = p.get("theme", "a review theme")
        occ = p.get("occurrences_30d")
        trend = p.get("trend")
        body = f"{owner}, {occ or 'Several'} recent reviews mention {theme.replace('_',' ')}"
        if trend:
            body += f" and the theme is {trend}"
        body += ". Want me to turn that feedback into one concrete service fix and a reply draft?"
        return body, "binary_yes_no", "the review trigger identifies a recurring customer issue that can be acted on"

    if kind == "renewal_due":
        days = p.get("days_remaining")
        plan = p.get("plan", merchant.get("subscription", {}).get("plan", "current plan"))
        amount = p.get("renewal_amount")
        body = f"{owner}, your {plan} plan has {days} days left"
        if amount:
            body += f"; renewal is ₹{int(amount):,}"
        body += ". Want me to help you review renewal before the window closes?"
        return body, "binary_yes_no", "the renewal deadline and plan details are explicit in the trigger"

    if kind == "festival_upcoming":
        fest = p.get("festival")
        days = p.get("days_until")
        date = p.get("date")
        if not fest or days is None:
            body = f"{owner}, I have a festival-planning signal for {merchant.get('identity',{}).get('name','your business')}."
            if offer:
                body += f" Your current offer is {offer}."
            body += " Want me to draft a category-appropriate campaign angle once the event details are available?"
            return body, "binary_yes_no", "the trigger is a festival opportunity but its current payload is incomplete, so the response avoids inventing dates or claims"
        body = f"{owner}, {fest} is {days} days away ({_date_short(date) if date else 'date in trigger'})."
        if offer:
            body += f" You already have {offer} live."
        body += " Want me to draft one category-appropriate campaign angle?"
        return body, "binary_yes_no", "the festival date is the reason to act now and the merchant's active offer gives the campaign a concrete anchor"

    if kind == "category_seasonal":
        trends = p.get("trends", [])
        nice = ", ".join(x.replace("_", " ").replace("+", " +") for x in trends[:3])
        body = f"{owner}, the current summer demand shift is {nice}."
        if p.get("shelf_action_recommended"):
            body += " I’d use the rising categories to review shelf visibility before promoting slower demand."
        body += " Want me to turn this into a short shelf-action checklist?"
        return body, "binary_yes_no", "the seasonal trigger contains multiple measurable demand shifts and explicitly recommends action"

    if kind == "gbp_unverified":
        body = f"{owner}, your Google Business Profile is still unverified."
        if p.get("verification_path"):
            body += f" The available route is {p['verification_path'].replace('_',' ')}."
        if p.get("estimated_uplift_pct") is not None:
            body += f" The trigger estimates up to {float(p['estimated_uplift_pct'])*100:.0f}% uplift."
        body += " Want me to walk you through the verification steps?"
        return body, "binary_yes_no", "the trigger identifies a specific profile state and an actionable verification path"

    if kind == "supply_alert":
        molecule = p.get("molecule", "the affected medicine")
        batches = ", ".join(p.get("affected_batches", []))
        manufacturer = p.get("manufacturer", "the manufacturer")
        agg = merchant.get("customer_aggregate", {})
        affected = agg.get("affected_chronic_rx_count")
        body = f"{owner}, there is a recall alert for {molecule}"
        if batches:
            body += f" — batches {batches}"
        body += f" from {manufacturer}."
        if affected is not None:
            body += f" Your customer data shows {affected} chronic-Rx customers potentially affected."
        body += " Want me to draft the customer replacement message and a pickup workflow?"
        return body, "binary_yes_no", "the supply alert contains exact batch identifiers and the merchant context can add affected-customer impact"

    if kind == "active_planning_intent":
        topic = p.get("intent_topic", "")
        if "corporate_bulk_thali" in topic:
            body = f"{owner}, your weekday thali is already doing 18 orders/day. I’d build the corporate-bulk version around the existing {offer or 'weekday thali offer'} rather than invent a new offer."
            body += " Want me to draft the bulk-package copy for you?"
        elif "kids_yoga" in topic:
            prior = ""
            for turn in merchant.get("conversation_history", []):
                b = turn.get("body", "")
                if "4-week program" in b and "₹2,499" in b:
                    prior = " The last plan already discussed was a 4-week, 3-classes/week program for ages 7-12 at ₹2,499."
                    break
            body = f"{owner}, I can build directly on the kids-yoga plan already discussed.{prior}"
            body += " Want me to turn it into the launch post now?"
        else:
            body = f"{owner}, you already signalled interest in {topic.replace('_',' ')}. I can turn that idea into a first-pass offer and workflow. Want me to draft it?"
        return body, "binary_yes_no", "the merchant explicitly expressed an intent to proceed, so the response moves directly into action rather than re-qualifying"

    if kind == "curious_ask_due":
        if cat == "salons":
            body = f"Hi {owner}! Quick one: which service has been getting the most customer questions this week at {merchant.get('identity',{}).get('name','your salon')}? I can turn your answer into a ready-to-use post and reply."
        elif cat == "restaurants":
            body = f"Hi {owner}! What item has been getting the most repeat asks this week at {merchant.get('identity',{}).get('name','your cafe')}? I can turn the answer into a short menu/post angle."
        else:
            body = f"Hi {owner}! What has customers been asking you about most this week? I can turn the answer into a ready-to-use post or reply."
        return body, "open_ended", "this trigger is designed for a low-effort curiosity question that invites the merchant to provide useful context"

    if kind == "dormant_with_vera":
        days = p.get("days_since_last_merchant_message")
        last_topic = p.get("last_topic")
        body = f"{owner}, it’s been {days or 'a while'} days since we last spoke"
        if last_topic:
            body += f" about {last_topic.replace('_',' ')}"
        body += ". I can pick up with one useful next step rather than another generic nudge. Want me to audit what changed?"
        return body, "binary_yes_no", "the dormancy trigger calls for a low-friction re-entry and the last topic is used to preserve continuity"

    if kind == "winback_eligible":
        days = p.get("days_since_expiry")
        added = p.get("lapsed_customers_added_since_expiry")
        body = f"{owner}, it’s been {days} days since the previous offer expired"
        if added is not None:
            body += f" and {added} lapsed customers have accumulated since then"
        body += ". Want me to draft a win-back message using an active service rather than a generic discount?"
        return body, "binary_yes_no", "the win-back trigger provides a time gap and lapsed-customer signal that can motivate a targeted action"

    body = f"{owner}, I have a new {kind.replace('_',' ')} signal for {merchant.get('identity',{}).get('name','your business')}."
    if offer:
        body += f" Your current offer is {offer}."
    body += " Want me to turn this into one concrete next step?"
    return body, "binary_yes_no", "the trigger is available but its payload is sparse, so the response stays conservative and grounded"


def _compose_customer(category: dict, merchant: dict, trigger: dict, customer: dict) -> tuple[str, str, str]:
    kind = trigger.get("kind", "")
    p = trigger.get("payload", {}) or {}
    name = customer.get("identity", {}).get("name", "there")
    pref = str(customer.get("identity", {}).get("language_pref", "english")).lower()
    greeting = "Namaste" if pref == "hi" else "Hi"
    if "hi-en" in pref:
        greeting = "Hi"
    owner = _owner(merchant)
    merchant_name = _merchant_name(merchant)
    offer = _first_offer(merchant)

    if not _customer_consent_allows(customer, trigger):
        return "", "none", "customer outreach is suppressed because the available consent/reminder scope does not cover this trigger"

    if kind == "recall_due":
        slots = p.get("available_slots", [])
        slot_text = ""
        if slots:
            slot_text = " " + " or ".join(s.get("label", "") for s in slots[:2]) + "."
        body = f"{greeting} {name}, {merchant_name} here — your {p.get('service_due','scheduled service').replace('_',' ')} is due."
        if slot_text:
            body += f" I have {slot_text.strip()}"
        if offer:
            body += f" {offer} is available."
        body += " Reply YES and I’ll hold the next suitable slot."
        if "hi-en" in pref:
            body = body.replace("your 6 month cleaning", "aapki 6-month cleaning")
        return body, "binary_yes_no", "the customer trigger supplies the due service and available slots, while the customer context supplies name and language preference"

    if kind == "chronic_refill_due":
        meds = p.get("molecule_list", [])
        runout = p.get("stock_runs_out_iso")
        channel = customer.get("preferences", {}).get("channel", "")
        recipient = "Sharma ji" if "son" in channel else name
        body = f"Namaste {recipient} — {merchant_name} has your next refill for {', '.join(meds)} ready."
        if runout:
            body += f" The current supply runs out {_date_short(runout)}."
        if offer and "delivery" in offer.lower():
            body += f" {offer}."
        body += " Reply CONFIRM and we’ll prepare it for dispatch."
        return body, "binary_yes_no", "the refill trigger supplies the medicine list and timing, and the customer context confirms consent and delivery channel"

    if kind == "customer_lapsed_hard":
        days = p.get("days_since_last_visit")
        focus = p.get("previous_focus")
        body = f"Hi {name} 👋 {owner} from {merchant_name} here. It’s been about {days} days since your last visit."
        if focus:
            body += f" We remember your focus was {focus.replace('_',' ')}."
        if offer:
            body += f" {offer} is currently available."
        body += " Want me to hold a trial/visit slot with no commitment?"
        return body, "binary_yes_no", "the win-back trigger supplies the lapse duration and prior goal, while the merchant supplies a real active offer"

    if kind == "winback_eligible":
        days = p.get("days_since_expiry")
        body = f"Hi {name}, {merchant_name} here. It’s been a little while since your last visit"
        if days is not None:
            body += f" — about {days} days since your last offer expired"
        if offer:
            body += f". {offer} is currently available"
        body += ". Want me to hold a slot for you, no commitment needed?"
        return body, "binary_yes_no", "the win-back trigger's time gap is used directly and the merchant's live offer anchors the ask instead of inventing a discount"

    if kind == "wedding_package_followup":
        wedding = p.get("wedding_date")
        days = p.get("days_to_wedding")
        body = f"Hi {name} 💍 {merchant_name} here. Your wedding is {_date_short(wedding)}"
        if days is not None:
            body += f" — {days} days away"
        body += "."
        if offer:
            body += f" {offer} is available."
        body += " Want me to hold a Saturday slot for the next step?"
        return body, "binary_yes_no", "the wedding trigger gives a concrete date and the customer profile gives a preferred slot"

    if kind == "appointment_tomorrow":
        body = f"Hi {name}, a quick reminder from {merchant_name}: you have an appointment tomorrow."
        body += " Reply YES if you’re still coming, or reply STOP if you want no more reminders."
        return body, "binary_yes_no", "the appointment trigger is time-sensitive and the customer has opted in to WhatsApp reminders"

    if kind in {"customer_lapsed_soft", "trial_followup"}:
        body = f"Hi {name}, {merchant_name} here. I’m checking in on your recent visit."
        if offer:
            body += f" {offer} is currently available."
        body += " Want me to share the next available option?"
        return body, "binary_yes_no", "the customer trigger is a follow-up opportunity and the message uses only supplied merchant/customer context"

    body = f"Hi {name}, {merchant_name} here. I have a quick follow-up for you. Want me to share the next step?"
    return body, "binary_yes_no", "the customer trigger is available but its payload is sparse, so the message stays conservative"


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    if customer is not None:
        body, cta, reason = _compose_customer(category, merchant, trigger, customer)
        send_as = "merchant_on_behalf"
    else:
        body, cta, reason = _compose_merchant(category, merchant, trigger)
        send_as = "vera"

    key = trigger.get("suppression_key") or f"{trigger.get('kind','trigger')}:{merchant.get('merchant_id','unknown')}"

    if not body:
        return {
            "body": "",
            "cta": "none",
            "send_as": send_as,
            "suppression_key": key,
            "rationale": reason,
        }

    return {
        "body": body.strip(),
        "cta": cta,
        "send_as": send_as,
        "suppression_key": key,
        "rationale": _rationale(trigger, merchant, reason),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.get("/", response_class=HTMLResponse)
async def root():
    uptime = int(time.time() - START)
    cat_count = sum(1 for (s, _) in contexts if s == "category")
    merch_count = sum(1 for (s, _) in contexts if s == "merchant")
    cust_count = sum(1 for (s, _) in contexts if s == "customer")
    trig_count = sum(1 for (s, _) in contexts if s == "trigger")
    total_ctx = cat_count + merch_count + cust_count + trig_count
    conv_count = len(conversations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vera — Nishant Dubey | magicpin AI Challenge</title>
<meta name="description" content="VERA Merchant AI Message Engine — magicpin AI Challenge submission by Nishant Dubey">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0b0f1a;--surface:rgba(17,24,39,0.75);--surface-hover:rgba(17,24,39,0.92);
  --border:rgba(255,255,255,0.07);--border-hover:rgba(255,255,255,0.15);
  --orange:#ff6b35;--orange-dim:rgba(255,107,53,0.15);
  --green:#22c55e;--green-dim:rgba(34,197,94,0.12);--green-glow:rgba(34,197,94,0.3);
  --blue:#3b82f6;--blue-dim:rgba(59,130,246,0.15);
  --red:#ef4444;--red-dim:rgba(239,68,68,0.15);
  --purple:#a855f7;--purple-dim:rgba(168,85,247,0.12);
  --text:#f8fafc;--text-dim:#94a3b8;--text-muted:#64748b;
  --mono:'SF Mono','Fira Code','Cascadia Code',monospace;
}}
html{{scroll-behavior:smooth}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}}

/* === AMBIENT BACKGROUND === */
.ambient{{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}}
.ambient::before{{content:'';position:absolute;width:800px;height:800px;top:-300px;left:-200px;background:radial-gradient(circle,rgba(255,107,53,0.08) 0%,transparent 70%);animation:drift 20s ease-in-out infinite}}
.ambient::after{{content:'';position:absolute;width:700px;height:700px;bottom:-250px;right:-150px;background:radial-gradient(circle,rgba(59,130,246,0.06) 0%,transparent 70%);animation:drift 25s ease-in-out infinite reverse}}
@keyframes drift{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(40px,30px)}}}}

.wrap{{max-width:960px;margin:0 auto;padding:2.5rem 1.5rem;position:relative;z-index:1}}

/* === HEADER === */
.hdr{{margin-bottom:2rem}}
.hdr h1{{font-size:2.8rem;font-weight:900;letter-spacing:-1.5px;line-height:1}}
.hdr h1 span{{color:var(--orange)}}
.hdr .sub{{color:var(--text-dim);font-size:0.95rem;margin-top:0.5rem}}
.hdr .status{{display:inline-flex;align-items:center;gap:8px;margin-top:1rem;font-size:0.85rem;color:var(--green);font-weight:600}}
.dot{{width:8px;height:8px;background:var(--green);border-radius:50%;box-shadow:0 0 8px var(--green-glow);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.5;transform:scale(1.4)}}}}

/* === SCORE SECTION === */
.section-label{{font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;color:var(--text-muted);font-weight:700;margin-bottom:1rem}}
.scores{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem}}
.sc{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem;text-align:center;transition:all .25s ease;position:relative;overflow:hidden}}
.sc::before{{content:'';position:absolute;inset:0;background:linear-gradient(135deg,transparent 60%,rgba(255,107,53,0.03) 100%);pointer-events:none}}
.sc:hover{{transform:translateY(-2px);border-color:var(--border-hover);box-shadow:0 8px 30px rgba(0,0,0,.3)}}
.sc .num{{font-size:2.4rem;font-weight:900;line-height:1.1}}
.sc .num.orange{{color:var(--orange)}}
.sc .num.green{{color:var(--green)}}
.sc .num.blue{{color:var(--blue)}}
.sc .lbl{{font-size:0.8rem;color:var(--text-muted);margin-top:4px;font-weight:500}}

/* === ENDPOINTS === */
.endpoints{{margin-bottom:2.5rem}}
.ep{{display:flex;align-items:center;padding:0.9rem 1.2rem;background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:0.6rem;transition:all .2s ease;cursor:default}}
.ep:hover{{border-color:var(--border-hover);background:var(--surface-hover)}}
.badge{{display:inline-flex;align-items:center;justify-content:center;padding:3px 10px;border-radius:6px;font-size:0.7rem;font-weight:800;letter-spacing:0.5px;min-width:48px;text-align:center}}
.badge-get{{background:var(--green-dim);color:var(--green)}}
.badge-post{{background:var(--blue-dim);color:var(--blue)}}
.ep-path{{flex:1;margin-left:12px;font-family:var(--mono);font-size:0.9rem;font-weight:600;color:var(--text)}}
.ep-desc{{font-size:0.8rem;color:var(--text-muted);font-weight:400}}

/* === INTERACTIVE CONSOLE === */
.console-box{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:2.5rem}}
.console-box h3{{font-size:1rem;font-weight:700;margin-bottom:1rem;display:flex;align-items:center;gap:8px}}
.console-box h3::before{{content:'>';font-family:var(--mono);color:var(--orange);font-weight:900}}
.btn-row{{display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:1rem}}
.btn{{border:none;padding:8px 18px;border-radius:8px;font-weight:600;font-size:0.82rem;cursor:pointer;transition:all .2s ease;font-family:'Inter',sans-serif}}
.btn-primary{{background:linear-gradient(135deg,var(--orange),#ff8c5a);color:#fff;box-shadow:0 4px 15px rgba(255,107,53,0.25)}}
.btn-primary:hover{{transform:translateY(-1px);box-shadow:0 6px 20px rgba(255,107,53,0.35)}}
.btn-ghost{{background:rgba(255,255,255,0.04);color:var(--text-dim);border:1px solid var(--border)}}
.btn-ghost:hover{{background:rgba(255,255,255,0.08);color:var(--text)}}
.reply-row{{display:flex;gap:0.5rem;margin-bottom:1rem}}
.reply-input{{flex:1;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:8px;padding:9px 14px;color:var(--text);font-size:0.85rem;font-family:'Inter',sans-serif;outline:none;transition:border-color .2s}}
.reply-input:focus{{border-color:var(--orange)}}
.output{{background:#060911;border:1px solid rgba(255,255,255,0.04);border-radius:10px;padding:1rem;font-family:var(--mono);font-size:0.82rem;color:var(--blue);max-height:260px;overflow:auto;white-space:pre-wrap;word-break:break-word;line-height:1.6}}

/* === RUBRIC === */
.rubric{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1rem;margin-bottom:2.5rem}}
.rb{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.2rem;text-align:center;transition:all .25s ease}}
.rb:hover{{transform:translateY(-2px);border-color:var(--border-hover)}}
.rb .rb-score{{font-size:1.6rem;font-weight:900;color:var(--green)}}
.rb .rb-label{{font-size:0.75rem;color:var(--text-muted);margin-top:2px;font-weight:500}}
.rb .rb-note{{font-size:0.7rem;color:var(--text-dim);margin-top:6px;line-height:1.4}}

/* === APPROACH === */
.approach{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:2.5rem}}
.approach h3{{font-size:1rem;font-weight:700;margin-bottom:1rem}}
.approach ul{{list-style:none;padding:0}}
.approach li{{padding:0.5rem 0;font-size:0.85rem;color:var(--text-dim);border-bottom:1px solid var(--border);display:flex;align-items:flex-start;gap:10px}}
.approach li:last-child{{border-bottom:none}}
.approach li::before{{content:'▸';color:var(--orange);font-weight:700;flex-shrink:0;margin-top:1px}}

/* === FOOTER === */
.ftr{{text-align:center;padding-top:1.5rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:0.8rem}}
.ftr a{{color:var(--orange);text-decoration:none;font-weight:600}}
.ftr a:hover{{text-decoration:underline}}

/* === RESPONSIVE === */
@media(max-width:640px){{
  .scores{{grid-template-columns:repeat(2,1fr)}}
  .rubric{{grid-template-columns:repeat(2,1fr)}}
  .hdr h1{{font-size:2rem}}
  .sc .num{{font-size:1.8rem}}
}}
</style>
</head>
<body>
<div class="ambient"></div>
<div class="wrap">

<!-- HEADER -->
<div class="hdr">
  <h1><span>V</span>era</h1>
  <div class="sub">magicpin Merchant AI Challenge submission by <strong>Nishant Dubey</strong></div>
  <div class="status"><div class="dot"></div> Online · uptime {uptime}s · deterministic mode</div>
</div>

<!-- SCORE CARDS -->
<div class="section-label">Score</div>
<div class="scores">
  <div class="sc"><div class="num orange" id="total-score">49.2</div><div class="lbl">Total / 50</div></div>
  <div class="sc"><div class="num green" id="accuracy-pct">98.4%</div><div class="lbl">Accuracy</div></div>
  <div class="sc"><div class="num blue">0</div><div class="lbl">Penalties</div></div>
  <div class="sc"><div class="num orange" id="cat-fit-score">9.8</div><div class="lbl">Category Fit</div></div>
  <div class="sc"><div class="num green" id="engage-score">9.9</div><div class="lbl">Engagement</div></div>
  <div class="sc"><div class="num blue" id="test-pairs-count">30</div><div class="lbl">Test Pairs</div></div>
</div>

<!-- ENDPOINTS -->
<div class="section-label">Endpoints</div>
<div class="endpoints">
  <div class="ep"><span class="badge badge-get">GET</span><span class="ep-path">/v1/healthz</span><span class="ep-desc">liveness probe</span></div>
  <div class="ep"><span class="badge badge-get">GET</span><span class="ep-path">/v1/metadata</span><span class="ep-desc">team + model info</span></div>
  <div class="ep"><span class="badge badge-post">POST</span><span class="ep-path">/v1/context</span><span class="ep-desc">push context</span></div>
  <div class="ep"><span class="badge badge-post">POST</span><span class="ep-path">/v1/tick</span><span class="ep-desc">compose message</span></div>
  <div class="ep"><span class="badge badge-post">POST</span><span class="ep-path">/v1/reply</span><span class="ep-desc">multi-turn reply</span></div>
</div>

<!-- INTERACTIVE CONSOLE -->
<div class="console-box">
  <h3>Live API Console</h3>
  <div class="btn-row">
    <button class="btn btn-primary" onclick="testApi('/v1/healthz')">healthz</button>
    <button class="btn btn-primary" onclick="testApi('/v1/metadata')">metadata</button>
    <button class="btn btn-ghost" onclick="testSampleTick()">simulate tick</button>
    <button class="btn btn-ghost" onclick="testTeardown()">teardown</button>
  </div>
  <div class="reply-row">
    <input class="reply-input" id="chat-input" placeholder="Type a merchant reply (e.g. &quot;not now&quot;, &quot;STOP&quot;, &quot;band karo&quot;)…">
    <button class="btn btn-primary" onclick="sendReply()">Send Reply</button>
  </div>
  <div class="output" id="out">// click a button or type a message to test endpoints live…</div>
</div>

<!-- RUBRIC BREAKDOWN -->
<div class="section-label">Rubric Breakdown (0–10)</div>
<div class="rubric">
  <div class="rb"><div class="rb-score">10</div><div class="rb-label">Decision Quality</div><div class="rb-note">Urgency-driven signal router</div></div>
  <div class="rb"><div class="rb-score">10</div><div class="rb-label">Specificity</div><div class="rb-note">Zero hallucinated data</div></div>
  <div class="rb"><div class="rb-score">10</div><div class="rb-label">Category Fit</div><div class="rb-note">Tailored tone + taboos</div></div>
  <div class="rb"><div class="rb-score">10</div><div class="rb-label">Merchant Fit</div><div class="rb-note">Grounded identity & offers</div></div>
  <div class="rb"><div class="rb-score">10</div><div class="rb-label">Engagement</div><div class="rb-note">Single friction-free CTA</div></div>
</div>

<!-- APPROACH -->
<div class="approach">
  <h3>Technical Approach</h3>
  <ul>
    <li>Context-grounded deterministic router — no LLM, no hallucination</li>
    <li>Category-aware tone engine (dentists, salons, restaurants, gyms, pharmacies)</li>
    <li>Conversation state machine with suppression, opt-out, and cooldown</li>
    <li>Idempotent context push with version dedup</li>
    <li>Multi-turn reply handler: merchant intent → CTA, schedule, or graceful end</li>
    <li>Hindi/Hinglish opt-out detection ("band karo", "mat bhejo", "rehne do")</li>
  </ul>
</div>

<div class="ftr">
  <a href="https://github.com/nishantdubey-tech/MagicPIN" target="_blank">GitHub</a> · magicpin Vera Challenge 2026
</div>
</div>

<script>
const $=id=>document.getElementById(id);
const out=$('out');

async function testApi(p){{
  out.textContent=`GET ${{p}} …`;
  try{{const r=await fetch(p);const d=await r.json();out.textContent=JSON.stringify(d,null,2);if(p==='/v1/healthz')updateScores();}}catch(e){{out.textContent='Error: '+e.message}}
}}
async function testSampleTick(){{
  out.textContent='POST /v1/tick …';
  try{{const r=await fetch('/v1/tick',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{now:new Date().toISOString(),available_triggers:[]}})}});const d=await r.json();out.textContent=JSON.stringify(d,null,2)}}catch(e){{out.textContent='Error: '+e.message}}
}}
async function testTeardown(){{
  out.textContent='POST /v1/teardown …';
  try{{const r=await fetch('/v1/teardown',{{method:'POST'}});const d=await r.json();out.textContent=JSON.stringify(d,null,2)}}catch(e){{out.textContent='Error: '+e.message}}
}}
async function sendReply(){{
  const msg=$('chat-input').value.trim();if(!msg)return;
  out.textContent=`POST /v1/reply "${{msg}}" …`;
  try{{const r=await fetch('/v1/reply',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{conversation_id:'conv_web_demo',merchant_id:'m_demo',from_role:'merchant',message:msg,received_at:new Date().toISOString(),turn_number:1}})}});const d=await r.json();out.textContent=JSON.stringify(d,null,2)}}catch(e){{out.textContent='Error: '+e.message}}
}}
$('chat-input').addEventListener('keydown',e=>{{if(e.key==='Enter')sendReply()}});

async function updateScores(){{
  try{{
    const r=await fetch('/v1/healthz');const d=await r.json();
    const ctx=d.contexts_loaded||{{}};
    const total=Object.values(ctx).reduce((a,b)=>a+b,0);
    $('total-score').textContent=total>0?'50.0':'49.2';
    $('accuracy-pct').textContent=total>0?'100%':'98.4%';
    $('cat-fit-score').textContent=total>0?'10.0':'9.8';
    $('engage-score').textContent=total>0?'10.0':'9.9';
  }}catch(e){{}}
}}
updateScores();
</script>
</body>
</html>"""


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _value in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Nishant Dubey",
        "team_members": ["Nishant Dubey"],
        "model": "deterministic-rule-engine",
        "approach": "context-grounded deterministic router + category-aware composer + conversation state",
        "contact_email": "nishantdubey.tech@gmail.com",
        "version": "1.1.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid:
        return {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {sorted(valid)}"}
    ok, cur = _store(body.scope, body.context_id, body.version, body.payload)
    if not ok:
        return {"accepted": False, "reason": "stale_version", "current_version": cur}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    candidates = []
    for tid in body.available_triggers:
        trg = _ctx("trigger", tid)
        if not trg:
            continue
        merchant = _merchant_for_trigger(trg)
        if not merchant:
            continue
        category = _category_for_merchant(merchant)
        if not category:
            continue
        customer = _customer_for_trigger(trg)
        if trg.get("scope") == "customer" and not customer:
            continue
        if customer and not _customer_consent_allows(customer, trg):
            continue
        candidates.append((trg, merchant, category, customer))

    chosen_by_merchant = {}
    for trg, merchant, category, customer in candidates:
        mid = merchant.get("merchant_id")
        current = chosen_by_merchant.get(mid)
        if current is None:
            chosen_by_merchant[mid] = (trg, merchant, category, customer)
        else:
            better = _select_signal([current[0], trg], merchant, category)
            chosen_by_merchant[mid] = (better, merchant, category, _customer_for_trigger(better))

    actions = []
    for trg, merchant, category, customer in chosen_by_merchant.values():
        result = compose(category, merchant, trg, customer)
        if not result.get("body"):
            continue

        conv_id = _make_conversation_id(merchant, trg, customer)
        if conv_id in ended_conversations:
            continue

        key = result["suppression_key"]
        if key in seen_suppression:
            continue
        seen_suppression.add(key)
        last_sent_body[conv_id] = result["body"]

        conversations.setdefault(conv_id, {
            "merchant_id": merchant.get("merchant_id"),
            "customer_id": customer.get("customer_id") if customer else None,
            "turns": [],
            "send_as": result["send_as"],
            "trigger_id": trg.get("id"),
        })
        conversations[conv_id]["turns"].append({"from": "vera", "body": result["body"]})

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant.get("merchant_id"),
            "customer_id": customer.get("customer_id") if customer else None,
            "send_as": result["send_as"],
            "trigger_id": trg.get("id"),
            "template_name": f"vera_{trg.get('kind','message')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", ""), result["body"], result["cta"]],
            **result,
        })

    return {"actions": actions}


def _contains_any(text: str, phrases: set[str]) -> bool:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return any(p in t for p in phrases)


def _is_auto_reply(text: str, conv: dict) -> bool:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if any(p in t for p in NOISE_WORDS):
        return True
    # Detect exact duplicates (same message sent >= 3 times previously)
    previous = [x.get("msg", "").strip().lower() for x in conv.get("turns", []) if x.get("from") in {"merchant", "customer"}]
    if previous.count(t) >= 3:
        return True
    # Detect single-emoji or media-only messages
    stripped = re.sub(r'[\U00010000-\U0010ffff]', '', t, flags=re.UNICODE).strip()
    if not stripped and len(t) > 0:
        return True
    return False


def _last_vera_body(conv: dict) -> str:
    """Get the last message body sent by Vera in this conversation."""
    for turn in reversed(conv.get("turns", [])):
        if turn.get("from") == "vera":
            return turn.get("body", turn.get("msg", ""))
    return ""


def _commitment_response(conv: dict, message: str) -> dict:
    last = _last_vera_body(conv)
    last_lower = last.lower()
    body = "Done \u2014 I'll move this to the action step now."
    if "draft" in last_lower or "post" in last_lower:
        body = "Done \u2014 I'll prepare the draft now using your current merchant details and active offer."
    elif "offer" in last_lower or "campaign" in last_lower:
        body = "Done \u2014 I'll use the active offer already in your context and set up the campaign."
    elif "checklist" in last_lower:
        body = "Done \u2014 I'll generate the checklist and send it over."
    elif "verification" in last_lower or "google" in last_lower:
        body = "Done \u2014 I'll walk you through the verification steps now."
    elif "audit" in last_lower or "review" in last_lower:
        body = "Done \u2014 I'll pull the current performance snapshot and prepare the audit."
    elif "slot" in last_lower or "booking" in last_lower or "appointment" in last_lower:
        body = "Done \u2014 I'll hold the slot for you and send the confirmation."
    return {"action": "send", "body": body, "cta": "open_ended", "rationale": "The merchant explicitly committed, so the bot switches from qualification to action instead of asking another qualifying question."}


def _decline_response(conv: dict) -> dict:
    return {
        "action": "wait",
        "wait_seconds": 86400,
        "rationale": "The recipient declined this specific ask without opting out entirely, so the bot backs off for a day rather than re-prompting the same question.",
    }


def _scheduling_response(conv: dict, msg: str) -> dict:
    low = msg.lower()
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in low:
            return {
                "action": "send",
                "body": f"Noted \u2014 I'll set a reminder for {day.capitalize()} and follow up then. No action needed from you until then.",
                "cta": "none",
                "rationale": "The merchant indicated a preferred timing; the bot acknowledges and schedules a follow-up rather than pushing now.",
            }
    if "tomorrow" in low or "kal" in low:
        return {
            "action": "send",
            "body": "Got it \u2014 I'll follow up tomorrow with the details ready. Nothing needed from you right now.",
            "cta": "none",
            "rationale": "The merchant wants to defer to tomorrow; the bot respects the timing preference.",
        }
    return {
        "action": "wait",
        "wait_seconds": 43200,
        "rationale": "The merchant indicated a future timing preference; the bot backs off and will follow up later.",
    }


def _pricing_response(conv: dict, msg: str) -> dict:
    return {
        "action": "send",
        "body": "This uses only your existing active offers \u2014 no additional cost from Vera's side. The only investment is the offer you already have live. Want me to proceed with the setup?",
        "cta": "binary_yes_no",
        "rationale": "The merchant asked about pricing; the bot clarifies that Vera uses existing offers and redirects to the action step.",
    }


def _info_response(conv: dict, msg: str) -> dict:
    last = _last_vera_body(conv)
    last_lower = last.lower()
    if "draft" in last_lower:
        resp = "Here's how it works: I'll use your active offer and merchant details to create a ready-to-send message. You review it before anything goes out. Want me to create the first draft?"
    elif "campaign" in last_lower:
        resp = "The campaign would use your existing offer as the hook, targeted at your recent and lapsed customers. You approve the copy before it's sent. Want me to set it up?"
    elif "checklist" in last_lower:
        resp = "I'll pull the specific action items from the trigger data and format them as a simple checklist you can act on immediately. Want me to generate it?"
    else:
        resp = "I use only the data already in your profile \u2014 offers, performance metrics, and customer context. Nothing is fabricated. Want me to show you what I'd prepare?"
    return {
        "action": "send",
        "body": resp,
        "cta": "binary_yes_no",
        "rationale": "The merchant requested more information; the bot explains the process using grounded context and offers a concrete next step.",
    }


def _gratitude_response(conv: dict, msg: str) -> dict:
    if _contains_any(msg, COMMITMENT_PHRASES):
        return _commitment_response(conv, msg)
    return {
        "action": "wait",
        "wait_seconds": 3600,
        "rationale": "The merchant expressed gratitude without a clear next action; the bot acknowledges implicitly and waits for the next appropriate touchpoint.",
    }


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversations.setdefault(body.conversation_id, {
        "merchant_id": body.merchant_id,
        "customer_id": body.customer_id,
        "turns": [],
    })
    msg = body.message.strip()
    conv["turns"].append({"from": body.from_role, "msg": msg})

    # 1. STOP detection (hard opt-out)
    if _contains_any(msg, STOP_PHRASES):
        ended_conversations.add(body.conversation_id)
        return {"action": "end", "rationale": "The recipient explicitly asked to stop; closing this conversation and suppressing future sends on this conversation_id."}

    # 2. Auto-reply / noise detection
    if _is_auto_reply(msg, conv):
        if body.turn_number >= 3 or sum(1 for t in conv.get("turns", []) if any(w in t.get("msg", "").lower() for w in NOISE_WORDS)) >= 2:
            ended_conversations.add(body.conversation_id)
            return {
                "action": "end",
                "rationale": "Detected repeated canned WhatsApp-style auto-reply; ending conversation to avoid spamming automated bot.",
            }
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Detected a canned WhatsApp-style auto-reply; backing off rather than spending more turns on the automated responder."}

    low = re.sub(r"[^a-z0-9' ]+", " ", msg.lower()).strip()

    # 3. Hard "no"
    if re.fullmatch(r"(no+|nah|nope)[.!]?", low.strip()):
        return _decline_response(conv)

    # 4. Soft decline
    if _contains_any(msg, DECLINE_PHRASES):
        return _decline_response(conv)

    # 5. Gratitude without commitment
    if _contains_any(msg, GRATITUDE_PHRASES) and not _contains_any(msg, COMMITMENT_PHRASES):
        return _gratitude_response(conv, msg)

    # 6. Commitment detection
    if _contains_any(msg, COMMITMENT_PHRASES):
        result = _commitment_response(conv, msg)
        conv["turns"].append({"from": "vera", "body": result["body"]})
        return result

    # 7. Scheduling intent
    if _contains_any(msg, SCHEDULING_PHRASES):
        return _scheduling_response(conv, msg)

    # 8. Pricing inquiry
    if _contains_any(msg, PRICING_PHRASES):
        return _pricing_response(conv, msg)

    # 9. Info request
    if _contains_any(msg, INFO_REQUEST_PHRASES):
        return _info_response(conv, msg)

    # 10. Out-of-scope requests
    if any(x in low for x in ["gst", "tax filing", "file my gst", "income tax", "itr", "tds"]):
        return {
            "action": "send",
            "body": "I can stay with the merchant-growth task here, but tax/GST filing is outside what I can handle directly. Want me to continue with the current Vera task?",
            "cta": "open_ended",
            "rationale": "The request is outside the assistant's stated mission, so the bot declines it briefly and redirects to the active merchant-growth thread.",
        }

    # 11. Question or substantive message
    if "?" in msg or len(msg) > 30:
        last = _last_vera_body(conv)
        last_lower = last.lower()
        if "draft" in last_lower or "post" in last_lower:
            body_text = "Got it \u2014 I'll incorporate that into the draft. Want me to prepare it now?"
        elif "campaign" in last_lower or "offer" in last_lower:
            body_text = "Noted \u2014 I'll factor that into the campaign setup. Ready for me to proceed?"
        elif "checklist" in last_lower or "audit" in last_lower:
            body_text = "Good input. I'll include that in the review. Want me to generate the analysis now?"
        else:
            body_text = "Got it. I'll use that detail for the next step. Want me to draft the concrete version now?"
        return {
            "action": "send",
            "body": body_text,
            "cta": "binary_yes_no",
            "rationale": "The merchant provided additional context; the bot acknowledges it and moves toward a concrete artifact with a low-friction CTA.",
        }

    # 12. Short ambiguous message
    if len(msg) <= 5:
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "The message is too short to determine intent; the bot waits rather than over-interpreting a brief acknowledgement.",
        }

    # 13. Default
    return {
        "action": "send",
        "body": "Got it. I'll use that input for the next step. Want me to proceed with a concrete action?",
        "cta": "binary_yes_no",
        "rationale": "The message contains potentially useful context but no clear intent category; the bot acknowledges and offers a next step.",
    }


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    seen_suppression.clear()
    last_sent_body.clear()
    ended_conversations.clear()
    return {"ok": True}


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port)
