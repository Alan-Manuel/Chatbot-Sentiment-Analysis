import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Review Analyzer", layout="wide")
st.title("Review Analyzer 📝")
st.caption("Upload your reviews CSV → get a quick, interpretable analysis (no API keys).")

# --- Your CSV expected columns ---
# review text: content
# rating: score (1-5)
# optional: at (datetime), reviewCreatedVersion, thumbsUpCount, appId

STOPWORDS = {
    "the","a","an","and","or","but","if","then","this","that","these","those","to","of","in","on","for","with","at",
    "is","are","was","were","be","been","being","it","its","as","by","from","so","very","too","just","not","no",
    "i","me","my","we","our","you","your","they","them","their","he","she","his","her","him",
    "app","use","using","used","really","still","also","can","could","would","will","im","ive","dont","didnt","cant"
}

def clean_tokens(text: str):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove links
    text = re.sub(r"[^a-z\s]", " ", text)          # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return [t for t in text.split(" ") if len(t) >= 3 and t not in STOPWORDS]

def rating_to_sentiment(r: int) -> str:
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"

def top_terms(text_series: pd.Series, n=15):
    c = Counter()
    for t in text_series.dropna().astype(str).tolist():
        c.update(clean_tokens(t))
    return c.most_common(n)

def safe_parse_datetime(series: pd.Series) -> pd.Series:
    # Your `at` column typically looks ISO-ish; coerce errors safely
    return pd.to_datetime(series, errors="coerce", utc=True)

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if not uploaded:
    st.info("Upload your reviews CSV to get started.")
    st.stop()

# --- Load ---
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# --- Validate required columns for YOUR file ---
required = ["content", "score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Your CSV must include: {required}")
    st.stop()

# --- Normalize core fields ---
df["content"] = df["content"].astype(str).fillna("").str.strip()
df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

# drop empty reviews
df = df[df["content"].ne("")].copy()

# sentiment label
df["sentiment"] = df["score"].apply(rating_to_sentiment)

# Parse date if present
has_date = "at" in df.columns
if has_date:
    df["at_parsed"] = safe_parse_datetime(df["at"])
    df["date"] = df["at_parsed"].dt.date.astype("string")

# Version/app optional
has_version = "reviewCreatedVersion" in df.columns
has_appid = "appId" in df.columns
has_thumbs = "thumbsUpCount" in df.columns

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")

    sentiment_opt = ["ALL"] + sorted(df["sentiment"].dropna().unique().tolist())
    sentiment_filter = st.selectbox("Sentiment", sentiment_opt)

    if has_version:
        vers = df["reviewCreatedVersion"].fillna("").astype(str)
        version_opt = ["ALL"] + sorted([v for v in vers.unique().tolist() if v.strip() != ""])
        version_filter = st.selectbox("Version", version_opt)
    else:
        version_filter = "ALL"
        st.caption("No `reviewCreatedVersion` column found.")

    if has_date:
        # date slider using parsed dates
        dmin = df["at_parsed"].min()
        dmax = df["at_parsed"].max()
        if pd.notna(dmin) and pd.notna(dmax):
            start_date, end_date = st.date_input(
                "Date range (UTC)",
                value=(dmin.date(), dmax.date())
            )
        else:
            start_date, end_date = None, None
            st.caption("Could not parse `at` dates.")
    else:
        start_date, end_date = None, None
        st.caption("No `at` (date) column found.")

# --- Apply filters ---
fdf = df.copy()

if sentiment_filter != "ALL":
    fdf = fdf[fdf["sentiment"] == sentiment_filter]

if has_version and version_filter != "ALL":
    fdf = fdf[fdf["reviewCreatedVersion"].astype(str) == str(version_filter)]

if has_date and start_date and end_date:
    # filter on parsed UTC datetime
    fdf = fdf[
        (fdf["at_parsed"].notna()) &
        (fdf["at_parsed"].dt.date >= start_date) &
        (fdf["at_parsed"].dt.date <= end_date)
    ]

# --- Metrics ---
total = len(fdf)
if total == 0:
    st.warning("No rows match your filters. Try widening the filters.")
    st.stop()

avg_rating = float(fdf["score"].mean())
sent_counts = fdf["sentiment"].value_counts().to_dict()
neg = sent_counts.get("negative", 0)
neu = sent_counts.get("neutral", 0)
pos = sent_counts.get("positive", 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews (filtered)", f"{total:,}")
c2.metric("Avg score", f"{avg_rating:.2f}")
c3.metric("Negative", f"{neg:,}")
c4.metric("Positive", f"{pos:,}")

# --- Brief analysis (rule-based, explainable) ---
neg_pct = (neg / total) * 100
pos_pct = (pos / total) * 100
neu_pct = (neu / total) * 100

if neg_pct >= 40:
    tone = "Strong negative signal in the filtered set."
elif neg_pct >= 25:
    tone = "Noticeable dissatisfaction in the filtered set."
else:
    tone = "Overall sentiment looks fairly healthy in the filtered set."

if avg_rating >= 4.0:
    rating_take = "Average score is high, suggesting generally positive experience."
elif avg_rating >= 3.0:
    rating_take = "Average score is moderate, suggesting mixed experiences."
else:
    rating_take = "Average score is low, suggesting significant issues."

st.subheader("Brief analysis")
st.markdown(
    f"- **Overall:** {tone}\n"
    f"- **Scores:** {rating_take}\n"
    f"- **Split:** {pos_pct:.1f}% positive, {neu_pct:.1f}% neutral, {neg_pct:.1f}% negative\n"
    f"- **Suggested next step:** Inspect the top negative terms and sample low-score reviews below."
)

st.divider()

# --- Terms ---
left, right = st.columns(2)

with left:
    st.subheader("Top terms in negative reviews (score 1–2)")
    neg_terms = top_terms(fdf.loc[fdf["sentiment"] == "negative", "content"], n=15)
    if neg_terms:
        st.dataframe(pd.DataFrame(neg_terms, columns=["term", "count"]), use_container_width=True)
    else:
        st.info("No negative reviews in this filtered set.")

with right:
    st.subheader("Top terms in positive reviews (score 4–5)")
    pos_terms = top_terms(fdf.loc[fdf["sentiment"] == "positive", "content"], n=15)
    if pos_terms:
        st.dataframe(pd.DataFrame(pos_terms, columns=["term", "count"]), use_container_width=True)
    else:
        st.info("No positive reviews in this filtered set.")

st.divider()

# --- Optional: Most helpful reviews (thumbsUpCount) ---
if has_thumbs:
    st.subheader("Most helpful reviews (by thumbsUpCount)")
    show_cols = ["score", "sentiment", "thumbsUpCount", "content"]
    if has_version: show_cols.insert(0, "reviewCreatedVersion")
    if has_date: show_cols.insert(0, "date")
    st.dataframe(
        fdf.sort_values("thumbsUpCount", ascending=False).head(10)[show_cols],
        use_container_width=True
    )
    st.divider()

# --- Sample reviews ---
st.subheader("Sample reviews for quick triage")
col1, col2 = st.columns(2)

base_cols = ["score", "sentiment", "content"]
if has_version: base_cols.insert(0, "reviewCreatedVersion")
if has_date: base_cols.insert(0, "date")

with col1:
    st.markdown("**Lowest-score samples**")
    st.dataframe(fdf.sort_values("score").head(10)[base_cols], use_container_width=True)

with col2:
    st.markdown("**Highest-score samples**")
    st.dataframe(fdf.sort_values("score", ascending=False).head(10)[base_cols], use_container_width=True)
