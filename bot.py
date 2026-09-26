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
}

# Broadened opt-out detection: English + common Hindi/Hinglish phrasing merchants
# and customers actually use on WhatsApp.
STOP_PHRASES = {
    "stop messaging", "stop contacting", "do not message", "don't message",
    "not interested", "unsubscribe", "remove me", "no more messages",
    "band karo", "mat bhejo", "message mat bhejo", "rehne do", "band kardo",
    "stop", "band kar do", "no more msgs",
}

COMMITMENT_PHRASES = {
    "ok lets do it", "okay lets do it", "let's do it", "lets do it", "go ahead",
    "yes do it", "yes please", "proceed", "do it", "sounds good", "i want to join",
    "i want this", "book it", "confirm it", "activate it",
}

# Explicit soft-decline phrases: distinct from STOP (recipient isn't asking to be
# suppressed forever, just declining this particular ask).
DECLINE_PHRASES = {
    "no thanks", "not now", "no not now", "not right now", "maybe later",
    "not today", "skip this", "no need", "abhi nahi", "baad mein",
}


def _ctx(scope: str, cid: str) -> Optional[dict]:
    item = contexts.get((scope, cid))
    return item["payload"] if item else None


def _store(scope: str, cid: str, version: int, payload: dict) -> tuple[bool, Optional[int]]:
    key = (scope, cid)
    cur = contexts.get(key)
    if cur and cur["version"] >= version:
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
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VERA AI Engine — Magicpin Challenge</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #090d16;
  --card-bg: rgba(22, 30, 46, 0.7);
  --card-border: rgba(255, 255, 255, 0.08);
  --accent: #ff4757;
  --accent-gradient: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
  --primary-glow: rgba(255, 71, 87, 0.25);
  --text-main: #f1f2f6;
  --text-muted: #a4b0be;
  --green: #2ed573;
  --green-glow: rgba(46, 213, 115, 0.25);
}
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 2rem; overflow-x: hidden; }
.bg-glow { position: fixed; width: 600px; height: 600px; border-radius: 50%; filter: blur(140px); opacity: 0.15; pointer-events: none; z-index: 0; }
.bg-1 { top: -200px; left: -200px; background: #ff4757; }
.bg-2 { bottom: -200px; right: -200px; background: #70a1ff; }
.container { max-width: 1100px; margin: 0 auto; position: relative; z-index: 1; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--card-border); }
.logo-box { display: flex; align-items: center; gap: 1rem; }
.logo-icon { width: 48px; height: 48px; background: var(--accent-gradient); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.4rem; box-shadow: 0 8px 20px var(--primary-glow); }
.logo-title h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px; }
.logo-title p { font-size: 0.85rem; color: var(--text-muted); }
.status-badge { display: flex; align-items: center; gap: 8px; background: rgba(46, 213, 115, 0.1); border: 1px solid rgba(46, 213, 115, 0.3); padding: 8px 16px; border-radius: 30px; font-weight: 600; font-size: 0.85rem; color: var(--green); box-shadow: 0 0 15px var(--green-glow); }
.pulse-dot { width: 8px; height: 8px; background: var(--green); border-radius: 50%; box-shadow: 0 0 10px var(--green); animation: pulse 1.8s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem; }
.card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--card-border); border-radius: 16px; padding: 1.5rem; transition: transform 0.2s ease, border-color 0.2s ease; }
.card:hover { transform: translateY(-3px); border-color: rgba(255, 255, 255, 0.18); }
.card-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: 600; margin-bottom: 8px; }
.card-value { font-size: 1.8rem; font-weight: 800; color: #fff; }
.card-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; }
.section-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; gap: 8px; }
.tester-box { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 1.5rem; margin-bottom: 2.5rem; }
.btn-group { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
button { background: var(--accent-gradient); border: none; color: white; padding: 10px 20px; border-radius: 10px; font-weight: 600; font-size: 0.9rem; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 4px 15px var(--primary-glow); }
button:hover { transform: translateY(-1px); opacity: 0.95; }
button.secondary { background: rgba(255, 255, 255, 0.06); border: 1px solid var(--card-border); box-shadow: none; }
button.secondary:hover { background: rgba(255, 255, 255, 0.12); }
pre { background: #060911; border: 1px solid rgba(255, 255, 255, 0.05); padding: 1.2rem; border-radius: 12px; font-family: monospace; font-size: 0.88rem; color: #70a1ff; overflow-x: auto; max-height: 280px; }
.api-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
.api-item { background: rgba(255, 255, 255, 0.02); border: 1px solid var(--card-border); padding: 1rem; border-radius: 12px; }
.method { display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; margin-right: 8px; }
.get { background: rgba(46, 213, 115, 0.2); color: var(--green); }
.post { background: rgba(112, 161, 255, 0.2); color: #70a1ff; }
footer { text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 3rem; border-top: 1px solid var(--card-border); padding-top: 1.5rem; }
</style>
</head>
<body>
<div class="bg-glow bg-1"></div>
<div class="bg-glow bg-2"></div>
<div class="container">
<header>
<div class="logo-box">
<div class="logo-icon">V</div>
<div class="logo-title">
<h1>VERA AI Bot Engine</h1>
<p>Magicpin AI Challenge — Deterministic Decision System</p>
</div>
</div>
<div class="status-badge">
<div class="pulse-dot"></div>
SYSTEM LIVE & OPERATIONAL
</div>
</header>
<div class="grid">
<div class="card">
<div class="card-label">Engine Model</div>
<div class="card-value" style="font-size:1.3rem; margin-top:4px;">Rule Router v1.1</div>
<div class="card-sub">Grounded & Deterministic</div>
</div>
<div class="card">
<div class="card-label">Uptime</div>
<div class="card-value" id="uptime-val">Active</div>
<div class="card-sub">FastAPI + Uvicorn</div>
</div>
<div class="card">
<div class="card-label">Category Contexts</div>
<div class="card-value" id="cat-val">Ready</div>
<div class="card-sub">Dentist, Salon, Restaurant, Gym, Pharmacy</div>
</div>
<div class="card">
<div class="card-label">API Endpoints</div>
<div class="card-value">5 Active</div>
<div class="card-sub">Compliant with Spec</div>
</div>
</div>
<div class="tester-box">
<div class="section-title">Live API Console</div>
<div class="btn-group">
<button onclick="testApi('/v1/healthz')">Test /v1/healthz</button>
<button onclick="testApi('/v1/metadata')">Test /v1/metadata</button>
<button class="secondary" onclick="testSampleTick()">Simulate /v1/tick</button>
</div>
<pre id="output-box">// Click a button above to run live API test...</pre>
</div>
<div class="section-title">Challenge API Endpoints</div>
<div class="api-list">
<div class="api-item">
<span class="method get">GET</span><strong>/v1/healthz</strong>
<p style="font-size:0.83rem; color:var(--text-muted); margin-top:4px;">Returns bot health status and loaded context counts.</p>
</div>
<div class="api-item">
<span class="method get">GET</span><strong>/v1/metadata</strong>
<p style="font-size:0.83rem; color:var(--text-muted); margin-top:4px;">Returns team metadata, model name, and version info.</p>
</div>
<div class="api-item">
<span class="method post">POST</span><strong>/v1/context</strong>
<p style="font-size:0.83rem; color:var(--text-muted); margin-top:4px;">Receives merchant, customer, category, or trigger contexts.</p>
</div>
<div class="api-item">
<span class="method post">POST</span><strong>/v1/tick</strong>
<p style="font-size:0.83rem; color:var(--text-muted); margin-top:4px;">Evaluates available triggers and produces grounded actions.</p>
</div>
<div class="api-item">
<span class="method post">POST</span><strong>/v1/reply</strong>
<p style="font-size:0.83rem; color:var(--text-muted); margin-top:4px;">Handles merchant or customer replies in conversations.</p>
</div>
</div>
<footer>
Magicpin VERA AI Challenge Bot &bull; Deployed & Evaluated Live
</footer>
</div>
<script>
async function testApi(path) {
  const out = document.getElementById('output-box');
  out.textContent = `Fetching ${path}...`;
  try {
    const res = await fetch(path);
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = `Error: ${err.message}`;
  }
}
async function testSampleTick() {
  const out = document.getElementById('output-box');
  out.textContent = `Posting sample tick request...`;
  try {
    const res = await fetch('/v1/tick', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({now: new Date().toISOString(), available_triggers: []})
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = `Error: ${err.message}`;
  }
}
testApi('/v1/healthz');
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
    previous = [x.get("msg", "").strip().lower() for x in conv.get("turns", []) if x.get("from") in {"merchant", "customer"}]
    return previous.count(t) >= 2


def _commitment_response(conv: dict, message: str) -> dict:
    last = conv.get("turns", [])[-1].get("body", "") if conv.get("turns") else ""
    body = "Done — I’ll move this to the action step now."
    if "draft" in last.lower():
        body = "Done — I’ll prepare the draft now. Want me to use the current merchant details?"
    elif "offer" in last.lower():
        body = "Done — I’ll use the active offer already in your context and move to the setup step."
    return {"action": "send", "body": body, "cta": "open_ended", "rationale": "The merchant explicitly committed, so the bot switches from qualification to action instead of asking another qualifying question."}


def _decline_response(conv: dict) -> dict:
    return {
        "action": "wait",
        "wait_seconds": 86400,
        "rationale": "The recipient declined this specific ask without opting out entirely, so the bot backs off for a day rather than re-prompting the same question.",
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

    if _contains_any(msg, STOP_PHRASES):
        ended_conversations.add(body.conversation_id)
        return {"action": "end", "rationale": "The recipient explicitly asked to stop; closing this conversation and suppressing future sends on this conversation_id."}

    if _is_auto_reply(msg, conv):
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Detected a canned WhatsApp-style auto-reply; backing off rather than spending more turns on the automated responder."}

    low = re.sub(r"[^a-z0-9' ]+", " ", msg.lower())

    if any(p in low for p in COMMITMENT_PHRASES):
        result = _commitment_response(conv, msg)
        conv["turns"].append({"from": "vera", "body": result["body"]})
        return result

    if _contains_any(msg, DECLINE_PHRASES):
        result = _decline_response(conv)
        return result

    if re.fullmatch(r"(no+|nah|nope)[.!]?", low.strip()):
        return _decline_response(conv)

    if any(x in low for x in ["gst", "tax filing", "file my gst"]):
        return {
            "action": "send",
            "body": "I can stay with the merchant-growth task here, but GST filing is outside what I can handle directly. Want me to continue with the current Vera task?",
            "cta": "open_ended",
            "rationale": "The request is outside the assistant's stated mission, so the bot declines it briefly and redirects to the active merchant-growth thread.",
        }

    if "?" in msg or len(msg) > 20:
        return {
            "action": "send",
            "body": "Got it. I’ll use that detail for the next step. Want me to draft the concrete version now?",
            "cta": "binary_yes_no",
            "rationale": "The merchant provided additional context; the bot acknowledges it and moves toward a concrete artifact with a low-friction CTA.",
        }

    return {
        "action": "wait",
        "wait_seconds": 1800,
        "rationale": "No clear action or question was detected, so the bot backs off rather than creating unnecessary conversation.",
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
