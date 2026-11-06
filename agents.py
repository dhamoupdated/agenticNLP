"""Agents: preprocessing, classifier, verifier, arbitrator, rule-based priority."""
import json, re
from config import *
from llm_client import call_llm, parse_json


def preprocess(text: str) -> str:
    """Normalize whitespace and mask obvious personal data (emails, phone numbers)."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[PHONE]", text)
    return text


def _clean(d: dict) -> dict:
    """Force labels into the allowed taxonomy."""
    if d.get("category") not in CATEGORIES:
        d["category"] = "Other"
    if d.get("sentiment") not in SENTIMENTS:
        d["sentiment"] = "Neutral"
    return d


def classify(text: str, cfg: dict) -> dict:
    system = ("You classify customer feedback. Reply ONLY with JSON: "
              '{"category": str, "sentiment": str, "confidence": 0-1, "evidence": "short quote from text"}.\n'
              f"Allowed categories: {CATEGORIES}\nAllowed sentiments: {SENTIMENTS}\n{LABEL_GUIDE}")
    try:
        out = _clean(parse_json(call_llm(cfg, "classify", system, text)))
        out["confidence"] = float(out.get("confidence", 0.5))
    except Exception as e:  # never crash the pipeline on a bad reply
        out = {"category": "Other", "sentiment": "Neutral", "confidence": 0.0, "evidence": f"error: {e}"}
    out["model"] = cfg["name"]
    return out


def verify(text: str, pred: dict, cfg: dict) -> dict:
    system = ("You verify a proposed label for customer feedback against the text and the label guide. "
              'Reply ONLY with JSON: {"agrees": bool, "category": str|null, "sentiment": str|null, "reason": str}. '
              "If you disagree, give corrected labels.\n" + LABEL_GUIDE)
    user = f"Feedback: {text}\nProposed: category={pred['category']}, sentiment={pred['sentiment']}, evidence={pred['evidence']}"
    try:
        v = parse_json(call_llm(cfg, "verify", system, user))
    except Exception as e:
        return {"agrees": True, "reason": f"verifier error: {e}"}
    if not v.get("agrees"):
        pred = {**pred, "category": v.get("category") or pred["category"],
                "sentiment": v.get("sentiment") or pred["sentiment"]}
        _clean(pred)
    return {"agrees": bool(v.get("agrees")), "category": pred["category"],
            "sentiment": pred["sentiment"], "reason": v.get("reason", "")}


def arbitrate(text: str, candidates: list[dict], cfg: dict) -> dict:
    system = ("You are the arbitrator. Several analysts disagree on how to label customer feedback. "
              "Pick the best labels using only evidence in the text. If the text is genuinely ambiguous, "
              'set resolved=false. Reply ONLY with JSON: {"category": str, "sentiment": str, "resolved": bool, "reason": str}.\n'
              f"Allowed categories: {CATEGORIES}\nAllowed sentiments: {SENTIMENTS}\n{LABEL_GUIDE}")
    cand = "\n".join(f"- {c['model']}: {c['category']} / {c['sentiment']} (evidence: {c['evidence']})" for c in candidates)
    try:
        a = _clean(parse_json(call_llm(cfg, "arbitrate", system, f"Feedback: {text}\nCandidates:\n{cand}")))
        a["resolved"] = bool(a.get("resolved", False))
        return a
    except Exception as e:
        return {"category": "Other", "sentiment": "Neutral", "resolved": False, "reason": f"error: {e}"}


def priority_rules(text: str, sentiment: str) -> str:
    """Explicit business rules - replace with your organization's real escalation policy."""
    t = text.lower()
    if any(k in t for k in URGENT_KEYWORDS):
        return "Urgent"
    if any(k in t for k in HIGH_KEYWORDS):
        return "High"
    return "Low" if sentiment == "Positive" else "Normal"
