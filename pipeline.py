"""Pipelines: single LLM (Baseline A), majority vote (Baseline B), full multi-agent with routing."""
import time
from collections import Counter
from config import *
from agents import preprocess, classify, verify, arbitrate, priority_rules


def _vote(preds, field):
    counts = Counter(p[field] for p in preds)
    top = counts.most_common()
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else preds[0][field]  # tie -> first model


def _result(text, cat, sent, status, conf, calls, t0, trace):
    return {"text": text, "category": cat, "sentiment": sent, "priority": priority_rules(text, sent),
            "status": status, "confidence": conf, "llm_calls": calls,
            "latency": round(time.time() - t0, 2), "trace": trace}


def run_single(raw: str) -> dict:
    t0, text = time.time(), preprocess(raw)
    p = classify(text, MODELS[0])
    return _result(text, p["category"], p["sentiment"], "auto", p["confidence"], 1, t0, {"preds": [p]})


def run_vote(raw: str) -> dict:
    t0, text = time.time(), preprocess(raw)
    preds = [classify(text, m) for m in MODELS]
    conf = sum(p["confidence"] for p in preds) / len(preds)
    return _result(text, _vote(preds, "category"), _vote(preds, "sentiment"), "auto", conf, len(preds), t0, {"preds": preds})


def run_full(raw: str, use_routing: bool = True) -> dict:
    """use_routing=True tests the proposal's hypothesis: skip verify/arbitrate when models agree confidently."""
    t0, text = time.time(), preprocess(raw)
    calls, trace = 0, {}
    preds = [classify(text, m) for m in MODELS]; calls += len(preds)
    trace["preds"] = preds
    agree = len({(p["category"], p["sentiment"]) for p in preds}) == 1
    conf = min(p["confidence"] for p in preds)
    trace["agree"], trace["min_conf"] = agree, conf

    if use_routing and agree and conf >= CONFIDENCE_THRESHOLD:
        r = _result(text, preds[0]["category"], preds[0]["sentiment"], "auto", conf, calls, t0, trace)
    else:
        # Verification: each prediction is checked against the text.
        verified = [verify(text, p, VERIFIER_MODEL) for p in preds]; calls += len(preds)
        trace["verified"] = verified
        labels = {(v["category"], v["sentiment"]) for v in verified}
        if len(labels) == 1 and all(v["agrees"] for v in verified):
            cat, sent = next(iter(labels)); status = "verified"
        else:
            arb = arbitrate(text, preds, ARBITRATOR_MODEL); calls += 1
            trace["arbitration"] = arb
            cat, sent = arb["category"], arb["sentiment"]
            status = "arbitrated" if arb["resolved"] else "human_review"
        r = _result(text, cat, sent, status, conf, calls, t0, trace)

    # High-impact items always get a human look, whatever the models said.
    if r["priority"] in ("High", "Urgent") and r["status"] != "human_review":
        r["status"] = "human_review"
    return r


MODES = {"single": run_single, "vote": run_vote,
         "full": lambda t: run_full(t, use_routing=False),
         "routed": lambda t: run_full(t, use_routing=True)}
