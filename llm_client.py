"""Thin provider-agnostic wrapper around LLM APIs, plus an offline mock for testing."""
import os, json, re


def call_llm(cfg: dict, task: str, system: str, user: str) -> str:
    p = cfg["provider"]
    if p == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
        r = client.messages.create(model=cfg["model"], max_tokens=400, temperature=0,
                                   system=system, messages=[{"role": "user", "content": user}])
        return r.content[0].text
    if p == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=cfg.get("base_url"),
                        api_key=os.getenv(cfg.get("api_key_env", "OPENAI_API_KEY"), "none"))
        r = client.chat.completions.create(model=cfg["model"], temperature=0, max_tokens=400,
                                           messages=[{"role": "system", "content": system},
                                                     {"role": "user", "content": user}])
        return r.choices[0].message.content
    if p == "mock":
        return _mock(cfg, task, user)
    raise ValueError(f"Unknown provider: {p}")


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in reply: {text[:200]}")
    return json.loads(m.group(0))


# ---------- offline mock (keyword heuristic) ----------
_KW = {"Delivery": ["deliver", "shipping", "parcel", "arrived", "late"],
       "Billing": ["charge", "bill", "invoice", "payment"],
       "Customer support": ["support", "agent", "service team", "chat"],
       "Returns/refunds": ["refund", "return"],
       "App/website": ["app", "website", "login", "checkout"],
       "Product quality": ["quality", "broke", "broken", "defect", "works", "stopped"]}
_POS = ["great", "love", "quick", "quickly", "resolved", "excellent", "good", "happy"]
_NEG = ["late", "broke", "broken", "bad", "terrible", "worst", "charged twice", "never", "slow", "stopped"]


def _mock(cfg, task, user):
    t = user.lower()
    if task == "classify":
        cat = next((c for c, ws in _KW.items() if any(w in t for w in ws)), "Other")
        pos, neg = any(w in t for w in _POS), any(w in t for w in _NEG)
        sent = "Mixed" if pos and neg else "Positive" if pos else "Negative" if neg else "Neutral"
        return json.dumps({"category": cat, "sentiment": sent, "confidence": 0.8, "evidence": "keyword match"})
    if task == "verify":
        return json.dumps({"agrees": True, "category": None, "sentiment": None, "reason": "mock"})
    return json.dumps({"category": "Other", "sentiment": "Neutral", "resolved": False, "reason": "mock"})
