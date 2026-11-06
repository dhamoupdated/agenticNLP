"""Central configuration: taxonomy, models, thresholds. Edit this to fit your organization."""

CATEGORIES = ["Product quality", "Delivery", "Billing", "Customer support",
              "Returns/refunds", "App/website", "Other"]
SENTIMENTS = ["Positive", "Negative", "Neutral", "Mixed"]

# Short label definitions shared by classifier / verifier / arbitrator (your "annotation guide").
LABEL_GUIDE = """
Categories:
- Product quality: defects, durability, how well the product works.
- Delivery: shipping speed, damaged in transit, lost parcels.
- Billing: charges, invoices, payment errors, pricing.
- Customer support: interactions with support staff/chat/phone.
- Returns/refunds: return process, refund status.
- App/website: bugs, usability, login, checkout issues.
- Other: none of the above.
Sentiment: Positive, Negative, Neutral, or Mixed (clear positive AND negative parts).
If feedback touches several categories, choose the MAIN complaint/topic.
"""

# Use 2+ different models/providers. provider: "anthropic" | "openai" | "mock"
# For an open-source model via Ollama: provider="openai", base_url="http://localhost:11434/v1"
MODELS = [
    {"name": "claude", "provider": "anthropic", "model": "claude-haiku-5-5"},
    {"name": "gpt", "provider": "openai", "model": "gpt-4o-mini"},
]
VERIFIER_MODEL = MODELS[0]
ARBITRATOR_MODEL = {"name": "arbiter", "provider": "anthropic", "model": "claude-sonnet-5-5"}

# To test offline with no API keys, uncomment:
# MODELS = [{"name": "mock-a", "provider": "mock", "model": ""},
#           {"name": "mock-b", "provider": "mock", "model": ""}]
# VERIFIER_MODEL = MODELS[0]; ARBITRATOR_MODEL = MODELS[0]

CONFIDENCE_THRESHOLD = 0.75   # below this -> not trusted without verification
DB_PATH = "feedback.db"

# Explicit, auditable priority rules (not LLM-decided).
URGENT_KEYWORDS = ["injur", "fire", "unsafe", "dangerous", "allergic", "fraud", "stolen", "lawsuit"]
HIGH_KEYWORDS = ["charged twice", "overcharged", "no refund", "cancel", "never arrived", "urgent", "asap"]
