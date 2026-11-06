# Feedback Classification - starter

1. `pip install -r requirements.txt`
2. Set keys: `export ANTHROPIC_API_KEY=...` and `export OPENAI_API_KEY=...`
   (or switch `MODELS` in `config.py` to the offline `mock` provider to test with no keys)
3. Compare approaches: `python evaluate.py sample_data.csv`
4. Dashboard: `streamlit run app.py`

Modes: `single` (Baseline A), `vote` (Baseline B), `full` (always verify+arbitrate), `routed`
(confidence-based routing - the hypothesis to test). Replace sample_data.csv with a real annotated set,
and keep a held-out test split that you never use to tune prompts.
