"""Streamlit dashboard. Run: streamlit run app.py"""
import pandas as pd, streamlit as st
from config import CATEGORIES, SENTIMENTS
from pipeline import MODES
import store

st.title("Customer Feedback Classification")
tab1, tab2, tab3 = st.tabs(["Classify", "Human review", "Dashboard"])

with tab1:
    mode = st.selectbox("Mode", list(MODES), index=3, help="single / vote / full / routed")
    text = st.text_area("Customer feedback")
    if st.button("Classify") and text.strip():
        r = MODES[mode](text)
        r["id"] = store.save(r)
        st.json({k: r[k] for k in ("category", "sentiment", "priority", "status", "confidence", "llm_calls", "latency")})
        with st.expander("Agent trace"):
            st.json(r["trace"])

with tab2:
    with store.conn() as c:
        q = pd.read_sql("SELECT id,text,category,sentiment,priority FROM results "
                        "WHERE status='human_review' AND reviewed=0", c)
    st.write(f"{len(q)} item(s) waiting")
    for _, row in q.iterrows():
        st.markdown(f"**#{row.id}** — {row.text}")
        cat = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(row.category), key=f"c{row.id}")
        sen = st.selectbox("Sentiment", SENTIMENTS, index=SENTIMENTS.index(row.sentiment), key=f"s{row.id}")
        if st.button("Save correction", key=f"b{row.id}"):
            store.correct(row.id, cat, sen); st.rerun()

with tab3:
    with store.conn() as c:
        df = pd.read_sql("SELECT * FROM results", c)
    if df.empty:
        st.info("No data yet.")
    else:
        st.metric("Items", len(df)); st.metric("Avg latency (s)", round(df.latency.mean(), 2))
        st.metric("Human-review rate", f"{100 * (df.status.isin(['human_review', 'human_corrected'])).mean():.0f}%")
        st.bar_chart(df.category.value_counts()); st.bar_chart(df.sentiment.value_counts())
        st.bar_chart(df.status.value_counts())
