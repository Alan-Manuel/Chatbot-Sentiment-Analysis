import re
from collections import Counter
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ==========================================================
# Streamlit Setup
# ==========================================================
st.set_page_config(page_title="Review Analyzer Chatbot", layout="wide")
st.title("Review Analyzer 📝🤖")
st.caption("Upload your reviews CSV → chat with your dataset + interpretable charts (no API keys).")

# ==========================================================
# Landing Page / Instructions
# ==========================================================
with st.expander("📌 How to use this app", expanded=True):
    st.markdown(
        """
**1) Upload your reviews CSV** using the uploader below.  
**2) Use the sidebar filters** to narrow down by sentiment, version, and date range (if available).  
**3) Review the charts** to spot changes in volume, rating, and sentiment share over time.  
**4) Chat with the dataset** at the bottom to answer questions like:
- *"Give me a summary"*
- *"What’s driving negative reviews?"*
- *"Negative themes"*
- *"Trend monthly"*
- *"Show flagged periods"*
- *"Show worst reviews"*

✅ **No API keys required.** The chatbot is rule-based + retrieval (explainable).
        """
    )

with st.expander("🧾 Expected CSV columns", expanded=False):
    st.markdown(
        """
Required:
- **content**: review text  
- **score**: rating (1–5)

Optional (enables extra features):
- **at**: datetime timestamp (enables trends + flagged periods)
- **reviewCreatedVersion**: app version (enables version comparison)
- **thumbsUpCount**: helpful votes (enables “most helpful reviews” table)
- **appId**: app identifier (not required, but ok to include)
        """
    )

st.markdown("### 💬 Quick prompts you can try")
st.code(
    "Give me a summary\n"
    "What's driving negative reviews?\n"
    "Negative themes\n"
    "Trend monthly\n"
    "Show flagged periods\n"
    "Show worst reviews",
    language="text",
)

# ==========================================================
# Stopwords / Regex / Helpers
# ==========================================================
STOPWORDS = {
    "the","a","an","and","or","but","if","then","this","that","these","those","to","of","in","on","for","with","at",
    "is","are","was","were","be","been","being","it","its","as","by","from","so","very","too","just","not","no",
    "i","me","my","we","our","you","your","they","them","their","he","she","his","her","him",
    "app","use","using","used","really","still","also","can","could","would","will","im","ive","dont","didnt","cant"
}

RE_LINKS = re.compile(r"http\S+|www\.\S+")
RE_NON_ALPHA = re.compile(r"[^a-z\s]")
RE_SPACES = re.compile(r"\s+")

def clean_tokens(text: str) -> List[str]:
    """Lowercase → remove links/punct → split → drop short tokens and stopwords."""
    text = str(text).lower()
    text = RE_LINKS.sub(" ", text)
    text = RE_NON_ALPHA.sub(" ", text)
    text = RE_SPACES.sub(" ", text).strip()
    return [t for t in text.split(" ") if len(t) >= 3 and t not in STOPWORDS]

def rating_to_sentiment(r: int) -> str:
    """
    Explainable mapping:
      1-2 => negative
      3   => neutral
      4-5 => positive
    """
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"

def safe_parse_datetime(series: pd.Series) -> pd.Series:
    """Parse `at` to datetime; invalid parses become NaT (safe for messy CSVs)."""
    return pd.to_datetime(series, errors="coerce", utc=True)

def top_terms(text_series: pd.Series, n=15) -> List[Tuple[str, int]]:
    """Top token counts using the same cleaning logic."""
    c = Counter()
    for t in text_series.dropna().astype(str).tolist():
        c.update(clean_tokens(t))
    return c.most_common(n)

def clamp_int(x, lo, hi):
    """Safely clamp rating values into [lo, hi]."""
    try:
        x = int(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))

# ==========================================================
# Upload + Load
# ==========================================================
# This uploader expects a CSV. We load it into a DataFrame and validate required columns.
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
if not uploaded:
    st.info("Upload your reviews CSV to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Clean column names (extra whitespace can break lookups)
df.columns = [c.strip() for c in df.columns]

# Ensure required fields exist
required = ["content", "score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Your CSV must include: {required}")
    st.stop()

# ==========================================================
# Normalize & Feature Engineering
# ==========================================================
# - Ensure review content is string + non-empty
# - Ensure score is numeric and clamped into a safe range
# - Create a simple sentiment label from rating (explainable rule-based mapping)
df["content"] = df["content"].astype(str).fillna("").str.strip()
df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
df["score"] = df["score"].apply(lambda x: clamp_int(x, 0, 5))
df = df[df["content"].ne("")].copy()

df["sentiment"] = df["score"].apply(rating_to_sentiment)

# Optional columns unlock extra features (trends, version charts, helpful reviews)
has_date = "at" in df.columns
if has_date:
    df["at_parsed"] = safe_parse_datetime(df["at"])
    df["date"] = df["at_parsed"].dt.date.astype("string")

has_version = "reviewCreatedVersion" in df.columns
has_thumbs = "thumbsUpCount" in df.columns

# ==========================================================
# Sidebar Filters
# ==========================================================
with st.sidebar:
    st.header("Filters")

    # Sentiment filter (always available)
    sentiment_opt = ["ALL"] + sorted(df["sentiment"].dropna().unique().tolist())
    sentiment_filter = st.selectbox("Sentiment", sentiment_opt)

    # Version filter (optional)
    if has_version:
        vers = df["reviewCreatedVersion"].fillna("").astype(str)
        version_opt = ["ALL"] + sorted([v for v in vers.unique().tolist() if v.strip() != ""])
        version_filter = st.selectbox("Version", version_opt)
    else:
        version_filter = "ALL"
        st.caption("No `reviewCreatedVersion` column found.")

    # Date range filter (optional)
    if has_date:
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

# Apply filters
fdf = df.copy()

if sentiment_filter != "ALL":
    fdf = fdf[fdf["sentiment"] == sentiment_filter]

if has_version and version_filter != "ALL":
    fdf = fdf[fdf["reviewCreatedVersion"].astype(str) == str(version_filter)]

if has_date and start_date and end_date:
    fdf = fdf[
        (fdf["at_parsed"].notna()) &
        (fdf["at_parsed"].dt.date >= start_date) &
        (fdf["at_parsed"].dt.date <= end_date)
    ]

total = len(fdf)
if total == 0:
    st.warning("No rows match your filters. Try widening the filters.")
    st.stop()

# ==========================================================
# Core Metrics
# ==========================================================
avg_rating = float(fdf["score"].mean())
sent_counts = fdf["sentiment"].value_counts().to_dict()
neg = int(sent_counts.get("negative", 0))
neu = int(sent_counts.get("neutral", 0))
pos = int(sent_counts.get("positive", 0))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews (filtered)", f"{total:,}")
c2.metric("Avg score", f"{avg_rating:.2f}")
c3.metric("Negative", f"{neg:,}")
c4.metric("Positive", f"{pos:,}")

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
    f"- **Suggested next step:** Ask the chatbot what’s driving negatives or inspect the flagged periods below."
)
st.divider()

# ==========================================================
# Trend Prep (shared across charts + chatbot)
# ==========================================================
def build_trends(_df: pd.DataFrame, gran: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Build time-series aggregates for:
      - vol: count of reviews per period
      - avg: average score per period
      - share: sentiment share per period (negative/neutral/positive)
    Returns None if no parseable 'at' dates exist.
    """
    if "at_parsed" not in _df.columns or _df["at_parsed"].isna().all():
        return None

    tdf = _df[_df["at_parsed"].notna()].copy().sort_values("at_parsed")

    # Convert timestamps to buckets
    if gran == "Daily":
        tdf["period"] = tdf["at_parsed"].dt.to_period("D").dt.to_timestamp()
    elif gran == "Weekly":
        tdf["period"] = tdf["at_parsed"].dt.to_period("W-MON").dt.start_time
    else:
        tdf["period"] = tdf["at_parsed"].dt.to_period("M").dt.to_timestamp()

    # Volume
    vol = tdf.groupby("period").size().rename("reviews").reset_index()

    # Avg rating
    avg = tdf.groupby("period")["score"].mean().rename("avg_score").reset_index()

    # Sentiment share (counts -> proportions)
    sent = (
        tdf.groupby(["period", "sentiment"])
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = sent.pivot(index="period", columns="sentiment", values="count").fillna(0)
    for col in ["negative", "neutral", "positive"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["negative", "neutral", "positive"]]

    share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0).reset_index()
    return {"vol": vol, "avg": avg, "share": share, "tdf": tdf}

def rolling_z_flags(series: pd.Series, window: int = 6, z: float = 2.5) -> pd.Series:
    """
    Rolling z-score anomaly flags.
    Useful for spotting unusual spikes (volume, negative share, etc.).
    """
    if len(series) < max(8, window + 2):
        return pd.Series([False] * len(series), index=series.index)
    roll_mean = series.rolling(window, min_periods=window).mean()
    roll_std = series.rolling(window, min_periods=window).std().replace(0, np.nan)
    zscore = (series - roll_mean) / roll_std
    return zscore.abs().fillna(0) >= z

# ==========================================================
# Charts + Thresholds + Flagged Periods
# ==========================================================
st.subheader("Trends")

# If no dates exist, we can’t show time-series charts
if not has_date or ("at_parsed" not in fdf.columns) or fdf["at_parsed"].isna().all():
    st.info("Trend charts need the `at` column with parseable dates.")
    trends = None
else:
    gran = st.radio("Trend granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)
    trends = build_trends(fdf, gran)

    if trends:
        vol = trends["vol"]
        avg = trends["avg"]
        share = trends["share"].set_index("period")

        # 1) Review volume over time
        fig1, ax1 = plt.subplots()
        ax1.plot(vol["period"], vol["reviews"])
        ax1.set_title("Review Volume Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Number of reviews")
        ax1.tick_params(axis="x", rotation=30)
        st.pyplot(fig1, clear_figure=True)

        # 2) Average rating over time + threshold lines
        fig2, ax2 = plt.subplots()
        ax2.plot(avg["period"], avg["avg_score"])
        ax2.set_title("Average Rating Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Average score")
        ax2.set_ylim(0, 5)

        # Thresholds (interpretable "guardrails")
        ax2.axhline(4.0, linestyle="--", linewidth=1)
        ax2.axhline(3.0, linestyle="--", linewidth=1)
        if len(avg) > 0:
            ax2.text(avg["period"].iloc[0], 4.02, "Healthy ≥ 4.0", fontsize=9)
            ax2.text(avg["period"].iloc[0], 3.02, "Watchlist ≤ 3.0", fontsize=9)

        ax2.tick_params(axis="x", rotation=30)
        st.pyplot(fig2, clear_figure=True)

        # 3) Sentiment share over time (stacked area)
        fig3, ax3 = plt.subplots()
        ax3.stackplot(
            share.index,
            share["negative"],
            share["neutral"],
            share["positive"],
            labels=["negative", "neutral", "positive"],
        )
        ax3.set_title("Sentiment Share Over Time")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Share of reviews")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis="x", rotation=30)
        ax3.legend(loc="upper left")
        st.pyplot(fig3, clear_figure=True)

        # Flagged periods (triage)
        # - Threshold-based flags: avg < 3.5 OR negative share > 0.35
        # - Anomaly flags: rolling z-score spikes (volume / negative share)
        neg_share = share["negative"].copy()
        avg_series = (
            avg.set_index("period")["avg_score"]
            .reindex(share.index)
            .ffill()
            .bfill()
        )

        flag_thresh = (avg_series < 3.5) | (neg_share > 0.35)
        flag_anom_vol = rolling_z_flags(
            vol.set_index("period")["reviews"].reindex(share.index).fillna(0),
            window=6,
            z=2.5,
        )
        flag_anom_neg = rolling_z_flags(neg_share, window=6, z=2.5)

        flagged = pd.DataFrame({
            "period": share.index,
            "avg_score": avg_series.values,
            "neg_share": neg_share.values,
            "flag_threshold": flag_thresh.values,
            "flag_anom_volume": flag_anom_vol.values,
            "flag_anom_neg_share": flag_anom_neg.values,
        })

        flagged["any_flag"] = flagged[
            ["flag_threshold", "flag_anom_volume", "flag_anom_neg_share"]
        ].any(axis=1)

        st.subheader("Flagged periods (quick triage)")
        st.caption(
            "Flags trigger when avg score < 3.5, negative share > 35%, "
            "or unusual spikes appear (rolling z-score)."
        )
        st.dataframe(
            flagged[flagged["any_flag"]].sort_values("period", ascending=False),
            use_container_width=True
        )

        st.divider()

# ==========================================================
# Version Comparison (optional)
# ==========================================================
if has_version and fdf["reviewCreatedVersion"].astype(str).str.strip().ne("").any():
    st.subheader("Version Comparison")

    vdf = fdf.copy()
    vdf["reviewCreatedVersion"] = vdf["reviewCreatedVersion"].astype(str).str.strip()
    vdf = vdf[vdf["reviewCreatedVersion"].ne("")]

    # Aggregate by version
    v_agg = (
        vdf.groupby("reviewCreatedVersion")
        .agg(
            reviews=("reviewCreatedVersion", "size"),
            avg_score=("score", "mean"),
            neg_share=("sentiment", lambda s: (s == "negative").mean())
        )
        .reset_index()
    )

    top_n = st.slider("Show top N versions (by review count)", 5, 30, 10)
    v_agg = v_agg.sort_values("reviews", ascending=False).head(top_n)
    v_agg = v_agg.sort_values("reviewCreatedVersion")

    # Avg score by version + thresholds
    fig4, ax4 = plt.subplots()
    ax4.bar(v_agg["reviewCreatedVersion"], v_agg["avg_score"])
    ax4.set_title("Average Rating by Version")
    ax4.set_xlabel("Version")
    ax4.set_ylabel("Average score")
    ax4.set_ylim(0, 5)
    ax4.axhline(4.0, linestyle="--", linewidth=1)
    ax4.axhline(3.0, linestyle="--", linewidth=1)
    ax4.tick_params(axis="x", rotation=30)
    st.pyplot(fig4, clear_figure=True)

    # Volume by version
    fig5, ax5 = plt.subplots()
    ax5.bar(v_agg["reviewCreatedVersion"], v_agg["reviews"])
    ax5.set_title("Review Volume by Version")
    ax5.set_xlabel("Version")
    ax5.set_ylabel("Number of reviews")
    ax5.tick_params(axis="x", rotation=30)
    st.pyplot(fig5, clear_figure=True)

    # Negative share by version + thresholds
    fig6, ax6 = plt.subplots()
    ax6.bar(v_agg["reviewCreatedVersion"], v_agg["neg_share"])
    ax6.set_title("Negative Share by Version")
    ax6.set_xlabel("Version")
    ax6.set_ylabel("Negative share")
    ax6.set_ylim(0, 1)
    ax6.axhline(0.25, linestyle="--", linewidth=1)
    ax6.axhline(0.40, linestyle="--", linewidth=1)
    ax6.tick_params(axis="x", rotation=30)
    st.pyplot(fig6, clear_figure=True)
else:
    st.caption("No usable `reviewCreatedVersion` values found for version charts.")

st.divider()

# ==========================================================
# Topic Modeling (NMF) for Negative Themes
# ==========================================================
st.subheader("Negative themes (Topic modeling)")

neg_df = fdf[fdf["sentiment"] == "negative"].copy()

def nmf_topics(texts: List[str], n_topics: int = 5, n_terms: int = 8) -> List[Dict[str, Any]]:
    """
    Extract themes from negative reviews using:
      TF-IDF -> NMF -> top words per topic

    This is an explainable ML method: "topics" are just groups of terms that co-occur.
    """
    vect = TfidfVectorizer(
        tokenizer=clean_tokens,
        lowercase=True,
        min_df=3,
        max_df=0.90,
        max_features=4000
    )
    X = vect.fit_transform(texts)

    # If the data is too small/sparse, topics won't be stable.
    if X.shape[0] < 10 or X.shape[1] < 20:
        return []

    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=300)
    model.fit(X)
    H = model.components_
    terms = np.array(vect.get_feature_names_out())

    topics = []
    for k in range(n_topics):
        top_idx = np.argsort(H[k])[::-1][:n_terms]
        topics.append({"topic": k + 1, "terms": [terms[i] for i in top_idx]})
    return topics

if len(neg_df) < 30:
    st.info("Not enough negative reviews in the filtered set to generate stable topics. Try widening filters.")
else:
    n_topics = st.slider("Number of topics", 3, 8, 5)
    topics = nmf_topics(neg_df["content"].astype(str).tolist(), n_topics=n_topics, n_terms=8)

    if not topics:
        st.info("Could not generate topics (text too sparse after filtering). Try widening filters.")
    else:
        topic_rows = [
            {"Topic": f"Topic {t['topic']}", "Top terms": ", ".join([f"`{x}`" for x in t["terms"]])}
            for t in topics
        ]
        st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)

st.divider()

# ==========================================================
# Top Terms (Classic keyword counts)
# ==========================================================
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

# ==========================================================
# Helpful Reviews + Samples
# ==========================================================
if has_thumbs:
    st.subheader("Most helpful reviews (by thumbsUpCount)")

    show_cols = ["score", "sentiment", "thumbsUpCount", "content"]
    if has_version:
        show_cols.insert(0, "reviewCreatedVersion")
    if has_date:
        show_cols.insert(0, "date")

    st.dataframe(
        fdf.sort_values("thumbsUpCount", ascending=False).head(10)[show_cols],
        use_container_width=True
    )
    st.divider()

st.subheader("Sample reviews for quick triage")
col1, col2 = st.columns(2)

base_cols = ["score", "sentiment", "content"]
if has_version:
    base_cols.insert(0, "reviewCreatedVersion")
if has_date:
    base_cols.insert(0, "date")

with col1:
    st.markdown("**Lowest-score samples**")
    st.dataframe(fdf.sort_values("score").head(10)[base_cols], use_container_width=True)

with col2:
    st.markdown("**Highest-score samples**")
    st.dataframe(fdf.sort_values("score", ascending=False).head(10)[base_cols], use_container_width=True)

st.divider()

# ==========================================================
# Chatbot Layer (No API keys) — Intent + Retrieval + Templates
# ==========================================================
# The chatbot is intentionally "explainable":
# - We detect user intent using keywords (summary / drivers / trend / themes / etc.)
# - We retrieve relevant slices/aggregates from the already-filtered dataframe (fdf)
# - We generate a short response using templates (no hallucination risk)
st.subheader("Chat with your dataset 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Ask me about sentiment, trends, versions, negative themes, or examples. "
                    "Try: 'Give me a summary' or 'What’s driving negative reviews?'."}
    ]

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def detect_intent(q: str) -> str:
    """Keyword-based intent classification (simple + robust)."""
    ql = q.lower()
    if any(k in ql for k in ["summary", "overall", "health", "status", "snapshot"]):
        return "summary"
    if any(k in ql for k in ["why", "drivers", "reason", "cause", "complaint", "issues", "problem"]):
        return "drivers_negative"
    if any(k in ql for k in ["topic", "themes", "theme", "cluster"]):
        return "themes"
    if any(k in ql for k in ["trend", "over time", "monthly", "weekly", "daily"]):
        return "trend"
    if any(k in ql for k in ["flag", "spike", "anomaly", "alert"]):
        return "flags"
    if any(k in ql for k in ["version", "release", "compare", "vs"]):
        return "version"
    if any(k in ql for k in ["examples", "sample", "show reviews", "worst", "best", "lowest", "highest"]):
        return "samples"
    return "fallback"

def extract_granularity(q: str) -> str:
    """Infer desired trend bucket from user prompt."""
    ql = q.lower()
    if "daily" in ql or "today" in ql:
        return "Daily"
    if "weekly" in ql or "this week" in ql:
        return "Weekly"
    if "monthly" in ql or "this month" in ql:
        return "Monthly"
    return "Monthly"

def chatbot_answer(q: str, fdf: pd.DataFrame) -> str:
    """Main chatbot router: intent -> retrieve -> template response."""
    intent = detect_intent(q)
    total = len(fdf)
    neg = int((fdf["sentiment"] == "negative").sum())
    neu = int((fdf["sentiment"] == "neutral").sum())
    pos = int((fdf["sentiment"] == "positive").sum())
    avg = float(fdf["score"].mean())

    if total == 0:
        return "No reviews match the current filters."

    if intent == "summary":
        return (
            f"Here’s the snapshot for your current filters:\n\n"
            f"- Reviews: **{total:,}**\n"
            f"- Avg score: **{avg:.2f}/5**\n"
            f"- Split: **{(pos/total)*100:.1f}% positive**, **{(neu/total)*100:.1f}% neutral**, **{(neg/total)*100:.1f}% negative**\n\n"
            f"Next: ask **'what’s driving negative reviews?'** or **'show flagged periods'**."
        )

    if intent == "drivers_negative":
        neg_df_local = fdf[fdf["sentiment"] == "negative"]
        if len(neg_df_local) < 10:
            return "I don’t have enough negative reviews in the filtered set to identify reliable drivers. Try widening filters."
        terms = top_terms(neg_df_local["content"], n=8)
        top_list = ", ".join([f"`{t}`" for t, _ in terms[:6]])
        return (
            f"Top recurring terms in negative reviews: {top_list}.\n\n"
            f"If you want deeper themes, ask **'negative themes'**. "
            f"If you want examples, ask **'show worst reviews'**."
        )

    if intent == "themes":
        neg_df_local = fdf[fdf["sentiment"] == "negative"]
        if len(neg_df_local) < 30:
            return "Not enough negative reviews for stable theme extraction. Try widening filters or removing version/date filters."
        topics = nmf_topics(neg_df_local["content"].astype(str).tolist(), n_topics=5, n_terms=6)
        if not topics:
            return "Couldn’t generate themes (text too sparse after filtering). Try widening filters."
        lines = []
        for t in topics:
            lines.append(f"- **Topic {t['topic']}**: " + ", ".join([f"`{x}`" for x in t["terms"][:6]]))
        return "Here are common negative themes:\n\n" + "\n".join(lines)

    if intent == "trend":
        gran = extract_granularity(q)
        tr = build_trends(fdf, gran)
        if tr is None:
            return "Trend answers need the `at` column with parseable dates."
        avg_df = tr["avg"]
        share_df = tr["share"].set_index("period")

        last_avg = float(avg_df["avg_score"].iloc[-1]) if len(avg_df) else avg
        last_neg = float(share_df["negative"].iloc[-1]) if len(share_df) else (neg/total)

        return (
            f"Trend ({gran}) quick read:\n\n"
            f"- Latest avg score: **{last_avg:.2f}**\n"
            f"- Latest negative share: **{last_neg*100:.1f}%**\n\n"
            f"Check the trend charts above. Ask **'show flagged periods'** to see risk windows."
        )

    if intent == "flags":
        if trends is None:
            return "Flagged periods require the `at` column with parseable dates."
        return (
            "I flagged time windows where avg score drops below **3.5**, negative share rises above **35%**, "
            "or unusual spikes appear (rolling z-score). Scroll to **Flagged periods (quick triage)** above."
        )

    if intent == "version":
        if "reviewCreatedVersion" not in fdf.columns or fdf["reviewCreatedVersion"].astype(str).str.strip().eq("").all():
            return "I can’t compare versions because `reviewCreatedVersion` is missing or empty in this dataset."
        top_versions = (
            fdf["reviewCreatedVersion"].astype(str).str.strip().replace("", np.nan).dropna()
            .value_counts().head(5).index.tolist()
        )
        return (
            "I can compare versions by **avg score**, **review volume**, and **negative share**.\n\n"
            f"Top versions in this filtered set: {', '.join([f'`{v}`' for v in top_versions])}\n\n"
            "Tip: set the Version filter in the sidebar to focus on one release at a time."
        )

    if intent == "samples":
        low = fdf.sort_values("score").head(5)
        examples = "\n".join([f"- ({int(r.score)}/5) {str(r.content)[:140]}..." for r in low.itertuples()])
        return f"Here are a few low-score examples (trimmed):\n\n{examples}"

    return (
        "Try one of these:\n\n"
        "- **Give me a summary**\n"
        "- **What’s driving negative reviews?**\n"
        "- **Negative themes**\n"
        "- **Trend monthly / weekly**\n"
        "- **Show flagged periods**\n"
        "- **Show worst reviews**"
    )

# Chat input -> append -> answer -> append
user_q = st.chat_input("Ask: summary • drivers • themes • trend monthly • flagged periods • worst reviews")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    ans = chatbot_answer(user_q, fdf)

    st.session_state.messages.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)
