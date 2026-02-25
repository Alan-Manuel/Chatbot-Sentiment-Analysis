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
st.caption("Upload your reviews CSV → instructions + dashboard + chat (no API keys).")


# ==========================================================
# Helpers: Tokenization / Labels / Parsing
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
    text = str(text).lower()
    text = RE_LINKS.sub(" ", text)
    text = RE_NON_ALPHA.sub(" ", text)
    text = RE_SPACES.sub(" ", text).strip()
    return [t for t in text.split(" ") if len(t) >= 3 and t not in STOPWORDS]

def rating_to_sentiment(r: int) -> str:
    # Explainable mapping
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"

def safe_parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def top_terms(text_series: pd.Series, n=15) -> List[Tuple[str, int]]:
    c = Counter()
    for t in text_series.dropna().astype(str).tolist():
        c.update(clean_tokens(t))
    return c.most_common(n)

def clamp_int(x, lo, hi):
    try:
        x = int(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


# ==========================================================
# Upload + Load
# ==========================================================
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
if not uploaded:
    st.info("Upload your reviews CSV to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]

required = ["content", "score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Your CSV must include: {required}")
    st.stop()

# Normalize
df["content"] = df["content"].astype(str).fillna("").str.strip()
df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
df["score"] = df["score"].apply(lambda x: clamp_int(x, 0, 5))
df = df[df["content"].ne("")].copy()

df["sentiment"] = df["score"].apply(rating_to_sentiment)

has_date = "at" in df.columns
if has_date:
    df["at_parsed"] = safe_parse_datetime(df["at"])
    df["date"] = df["at_parsed"].dt.date.astype("string")

has_version = "reviewCreatedVersion" in df.columns
has_thumbs = "thumbsUpCount" in df.columns


# ==========================================================
# Sidebar: Filters + Threshold Controls (makes charts intuitive)
# ==========================================================
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
        dmin = df["at_parsed"].min()
        dmax = df["at_parsed"].max()
        if pd.notna(dmin) and pd.notna(dmax):
            start_date, end_date = st.date_input("Date range (UTC)", value=(dmin.date(), dmax.date()))
        else:
            start_date, end_date = None, None
            st.caption("Could not parse `at` dates.")
    else:
        start_date, end_date = None, None
        st.caption("No `at` column found (trend charts disabled).")

    st.divider()
    st.header("Thresholds (for readability)")

    # User-controlled thresholds
    thr_healthy = st.slider("Healthy score threshold", 3.0, 5.0, 4.0, 0.1)
    thr_watch = st.slider("Watchlist score threshold", 1.0, 4.5, 3.0, 0.1)
    thr_neg_share = st.slider("Negative share threshold", 0.10, 0.80, 0.35, 0.05)

    z_window = st.slider("Anomaly window (periods)", 4, 16, 6, 1)
    z_cutoff = st.slider("Anomaly sensitivity (z)", 1.5, 4.0, 2.5, 0.1)


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
# Core metrics (always visible)
# ==========================================================
avg_rating = float(fdf["score"].mean())
sent_counts = fdf["sentiment"].value_counts().to_dict()
neg = int(sent_counts.get("negative", 0))
neu = int(sent_counts.get("neutral", 0))
pos = int(sent_counts.get("positive", 0))

neg_pct = (neg / total) * 100
pos_pct = (pos / total) * 100
neu_pct = (neu / total) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews (filtered)", f"{total:,}")
c2.metric("Avg score", f"{avg_rating:.2f}")
c3.metric("Negative", f"{neg:,}")
c4.metric("Positive", f"{pos:,}")

# A simple “status badge” based on thresholds (feels intuitive)
if avg_rating >= thr_healthy and (neg / total) <= thr_neg_share:
    st.success("Status: Healthy ✅ (based on your thresholds)")
elif avg_rating <= thr_watch or (neg / total) >= thr_neg_share:
    st.warning("Status: Watchlist ⚠️ (based on your thresholds)")
else:
    st.info("Status: Mixed / Monitor ℹ️ (based on your thresholds)")


# ==========================================================
# Trends helpers
# ==========================================================
def build_trends(_df: pd.DataFrame, gran: str) -> Optional[Dict[str, pd.DataFrame]]:
    if "at_parsed" not in _df.columns or _df["at_parsed"].isna().all():
        return None

    tdf = _df[_df["at_parsed"].notna()].copy().sort_values("at_parsed")

    if gran == "Daily":
        tdf["period"] = tdf["at_parsed"].dt.to_period("D").dt.to_timestamp()
    elif gran == "Weekly":
        tdf["period"] = tdf["at_parsed"].dt.to_period("W-MON").dt.start_time
    else:
        tdf["period"] = tdf["at_parsed"].dt.to_period("M").dt.to_timestamp()

    vol = tdf.groupby("period").size().rename("reviews").reset_index()
    avg = tdf.groupby("period")["score"].mean().rename("avg_score").reset_index()

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

def rolling_z_flags(series: pd.Series, window: int, z: float) -> pd.Series:
    if len(series) < max(8, window + 2):
        return pd.Series([False] * len(series), index=series.index)
    roll_mean = series.rolling(window, min_periods=window).mean()
    roll_std = series.rolling(window, min_periods=window).std().replace(0, np.nan)
    zscore = (series - roll_mean) / roll_std
    return zscore.abs().fillna(0) >= z


# ==========================================================
# Topic modeling helper (NMF)
# ==========================================================
def nmf_topics(texts: List[str], n_topics: int = 5, n_terms: int = 8) -> List[Dict[str, Any]]:
    vect = TfidfVectorizer(
        tokenizer=clean_tokens,
        lowercase=True,
        min_df=3,
        max_df=0.90,
        max_features=4000
    )
    X = vect.fit_transform(texts)
    if X.shape[0] < 10 or X.shape[1] < 20:
        return []

    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=300)
    model.fit(X)
    H = model.components_
    terms = np.array(vect.get_feature_names_out())

    out = []
    for k in range(n_topics):
        top_idx = np.argsort(H[k])[::-1][:n_terms]
        out.append({"topic": k + 1, "terms": [terms[i] for i in top_idx]})
    return out


# ==========================================================
# Tabs: Home / Chat / Dashboard
# ==========================================================
tab_home, tab_chat, tab_dash = st.tabs(["🏠 Home", "💬 Chat", "📊 Dashboard"])


# ==========================================================
# HOME TAB (landing page)
# ==========================================================
with tab_home:
    st.subheader("Welcome 👋")
    st.markdown(
        """
This app helps you quickly understand review sentiment and *chat* with your dataset using explainable logic.

### What you can do
- **Dashboard:** trends, thresholds, flagged periods, version comparisons, top terms, topic modeling.
- **Chat:** ask questions and get answers grounded in the filtered data.

### Expected columns
- Required: `content`, `score`
- Optional: `at`, `reviewCreatedVersion`, `thumbsUpCount`

➡️ Next: go to **Chat** tab and try “Give me a summary”.
        """
    )

    st.markdown("### Quick prompts")
    st.code(
        "Give me a summary\n"
        "What's driving negative reviews?\n"
        "Negative themes\n"
        "Trend monthly\n"
        "Show flagged periods\n"
        "Show worst reviews",
        language="text"
    )


# ==========================================================
# CHAT TAB (conversational layer)
# ==========================================================
with tab_chat:
    st.subheader("Chat with your dataset 🤖")
    st.caption("This is rule-based + retrieval (no API keys). Answers are computed from your filtered data.")

    # Quick prompt buttons to make it feel like a “chatbot”
    p1, p2, p3, p4, p5 = st.columns(5)
    if p1.button("Summary"):
        st.session_state["_quick_prompt"] = "Give me a summary"
    if p2.button("Drivers"):
        st.session_state["_quick_prompt"] = "What's driving negative reviews?"
    if p3.button("Themes"):
        st.session_state["_quick_prompt"] = "Negative themes"
    if p4.button("Trend"):
        st.session_state["_quick_prompt"] = "Trend monthly"
    if p5.button("Flags"):
        st.session_state["_quick_prompt"] = "Show flagged periods"

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hey! Upload your CSV and set filters in the sidebar. Ask me about sentiment, trends, versions, themes, or examples."
            }
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    def detect_intent(q: str) -> str:
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
        ql = q.lower()
        if "daily" in ql:
            return "Daily"
        if "weekly" in ql:
            return "Weekly"
        if "monthly" in ql:
            return "Monthly"
        return "Monthly"

    def chatbot_answer(q: str, fdf_local: pd.DataFrame) -> str:
        intent = detect_intent(q)
        total_local = len(fdf_local)

        neg_local = int((fdf_local["sentiment"] == "negative").sum())
        neu_local = int((fdf_local["sentiment"] == "neutral").sum())
        pos_local = int((fdf_local["sentiment"] == "positive").sum())
        avg_local = float(fdf_local["score"].mean())

        if intent == "summary":
            return (
                f"Snapshot (current filters):\n\n"
                f"- Reviews: **{total_local:,}**\n"
                f"- Avg score: **{avg_local:.2f}/5**\n"
                f"- Split: **{(pos_local/total_local)*100:.1f}% positive**, "
                f"**{(neu_local/total_local)*100:.1f}% neutral**, "
                f"**{(neg_local/total_local)*100:.1f}% negative**\n\n"
                f"Thresholds: Healthy ≥ **{thr_healthy:.1f}**, Watch ≤ **{thr_watch:.1f}**, Neg share > **{thr_neg_share:.2f}** triggers alerts."
            )

        if intent == "drivers_negative":
            neg_df_local = fdf_local[fdf_local["sentiment"] == "negative"]
            if len(neg_df_local) < 10:
                return "Not enough negative reviews in the filtered set to identify stable drivers. Widen filters."
            terms = top_terms(neg_df_local["content"], n=8)
            top_list = ", ".join([f"`{t}`" for t, _ in terms[:6]])
            return f"Most common terms in negative reviews: {top_list}.\n\nAsk **'Negative themes'** for clustered topics."

        if intent == "themes":
            neg_df_local = fdf_local[fdf_local["sentiment"] == "negative"]
            if len(neg_df_local) < 30:
                return "Not enough negative reviews for stable topic modeling (need ~30+). Widen filters."
            topics = nmf_topics(neg_df_local["content"].astype(str).tolist(), n_topics=5, n_terms=6)
            if not topics:
                return "Could not generate themes (text too sparse). Widen filters."
            lines = [f"- **Topic {t['topic']}**: " + ", ".join([f"`{x}`" for x in t["terms"]]) for t in topics]
            return "Negative themes:\n\n" + "\n".join(lines)

        if intent == "trend":
            gran = extract_granularity(q)
            tr = build_trends(fdf_local, gran)
            if tr is None:
                return "Trend requires an `at` column with parseable dates."
            avg_df = tr["avg"]
            share_df = tr["share"].set_index("period")
            last_avg = float(avg_df["avg_score"].iloc[-1]) if len(avg_df) else avg_local
            last_neg = float(share_df["negative"].iloc[-1]) if len(share_df) else (neg_local / total_local)
            return (
                f"Trend ({gran}) latest:\n\n"
                f"- Avg score: **{last_avg:.2f}**\n"
                f"- Negative share: **{last_neg*100:.1f}%**\n\n"
                "Go to **Dashboard** tab to see the charts and the flagged periods table."
            )

        if intent == "flags":
            tr = build_trends(fdf_local, "Weekly")
            if tr is None:
                return "Flags require an `at` column with parseable dates."
            share = tr["share"].set_index("period")
            avg_df = tr["avg"].set_index("period")
            avg_series = avg_df["avg_score"].reindex(share.index).ffill().bfill()
            neg_share = share["negative"]

            flag_thresh = (avg_series < (thr_watch + 0.5)) | (neg_share > thr_neg_share)
            flagged_periods = share.index[flag_thresh].tolist()[-8:]
            if not flagged_periods:
                return "No flagged weekly periods found using your current thresholds."
            formatted = ", ".join([str(p.date()) if hasattr(p, "date") else str(p) for p in flagged_periods])
            return f"Recent flagged weekly periods: {formatted}\n\nSee **Dashboard → Flagged periods** for full details."

        if intent == "samples":
            low = fdf_local.sort_values("score").head(5)
            examples = "\n".join([f"- ({int(r.score)}/5) {str(r.content)[:160]}..." for r in low.itertuples()])
            return f"Low-score examples (trimmed):\n\n{examples}"

        if intent == "version":
            if "reviewCreatedVersion" not in fdf_local.columns or fdf_local["reviewCreatedVersion"].astype(str).str.strip().eq("").all():
                return "No version column found (`reviewCreatedVersion`)."
            top_versions = (
                fdf_local["reviewCreatedVersion"].astype(str).str.strip().replace("", np.nan).dropna()
                .value_counts().head(5).index.tolist()
            )
            return "Top versions in current filters: " + ", ".join([f"`{v}`" for v in top_versions])

        return (
            "Try:\n"
            "- Give me a summary\n"
            "- What's driving negative reviews?\n"
            "- Negative themes\n"
            "- Trend monthly\n"
            "- Show flagged periods\n"
            "- Show worst reviews"
        )

    # Allow quick prompt injection
    default_prompt = st.session_state.pop("_quick_prompt", None)
    user_q = st.chat_input("Ask something…") if default_prompt is None else default_prompt

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        ans = chatbot_answer(user_q, fdf)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)


# ==========================================================
# DASHBOARD TAB (charts + thresholds + tables)
# ==========================================================
with tab_dash:
    st.subheader("Trends (with thresholds)")

    if (not has_date) or ("at_parsed" not in fdf.columns) or fdf["at_parsed"].isna().all():
        st.info("Trend charts need the `at` column with parseable dates.")
        trends = None
    else:
        gran = st.radio("Trend granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)
        trends = build_trends(fdf, gran)

        if trends:
            vol = trends["vol"]
            avg = trends["avg"]
            share = trends["share"].set_index("period")

            # 1) Volume
            fig1, ax1 = plt.subplots()
            ax1.plot(vol["period"], vol["reviews"])
            ax1.set_title("Review Volume Over Time")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Number of reviews")
            ax1.tick_params(axis="x", rotation=30)
            st.pyplot(fig1, clear_figure=True)

            # 2) Avg rating + threshold lines
            fig2, ax2 = plt.subplots()
            ax2.plot(avg["period"], avg["avg_score"])
            ax2.set_title("Average Rating Over Time")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Average score")
            ax2.set_ylim(0, 5)

            ax2.axhline(thr_healthy, linestyle="--", linewidth=1)
            ax2.axhline(thr_watch, linestyle="--", linewidth=1)
            if len(avg) > 0:
                ax2.text(avg["period"].iloc[0], thr_healthy + 0.02, f"Healthy ≥ {thr_healthy:.1f}", fontsize=9)
                ax2.text(avg["period"].iloc[0], thr_watch + 0.02, f"Watch ≤ {thr_watch:.1f}", fontsize=9)

            ax2.tick_params(axis="x", rotation=30)
            st.pyplot(fig2, clear_figure=True)

            # 3) Sentiment share + negative threshold note
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

            st.caption(f"Note: negative-share alert threshold is set to **{thr_neg_share:.2f}** in the sidebar.")

            # Flagged periods table
            neg_share = share["negative"].copy()
            avg_series = avg.set_index("period")["avg_score"].reindex(share.index).ffill().bfill()

            flag_thresh = (avg_series < (thr_watch + 0.5)) | (neg_share > thr_neg_share)

            flag_anom_vol = rolling_z_flags(
                vol.set_index("period")["reviews"].reindex(share.index).fillna(0),
                window=z_window,
                z=z_cutoff,
            )
            flag_anom_neg = rolling_z_flags(neg_share, window=z_window, z=z_cutoff)

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
            st.caption("Flags are driven by your threshold sliders + anomaly sensitivity controls.")
            st.dataframe(flagged[flagged["any_flag"]].sort_values("period", ascending=False), use_container_width=True)

    st.divider()

    # Version comparison
    if has_version and fdf["reviewCreatedVersion"].astype(str).str.strip().ne("").any():
        st.subheader("Version Comparison")

        vdf = fdf.copy()
        vdf["reviewCreatedVersion"] = vdf["reviewCreatedVersion"].astype(str).str.strip()
        vdf = vdf[vdf["reviewCreatedVersion"].ne("")]

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
        v_agg = v_agg.sort_values("reviews", ascending=False).head(top_n).sort_values("reviewCreatedVersion")

        fig4, ax4 = plt.subplots()
        ax4.bar(v_agg["reviewCreatedVersion"], v_agg["avg_score"])
        ax4.set_title("Average Rating by Version")
        ax4.set_ylim(0, 5)
        ax4.axhline(thr_healthy, linestyle="--", linewidth=1)
        ax4.axhline(thr_watch, linestyle="--", linewidth=1)
        ax4.tick_params(axis="x", rotation=30)
        st.pyplot(fig4, clear_figure=True)

        fig5, ax5 = plt.subplots()
        ax5.bar(v_agg["reviewCreatedVersion"], v_agg["reviews"])
        ax5.set_title("Review Volume by Version")
        ax5.tick_params(axis="x", rotation=30)
        st.pyplot(fig5, clear_figure=True)

        fig6, ax6 = plt.subplots()
        ax6.bar(v_agg["reviewCreatedVersion"], v_agg["neg_share"])
        ax6.set_title("Negative Share by Version")
        ax6.set_ylim(0, 1)
        ax6.axhline(thr_neg_share, linestyle="--", linewidth=1)
        ax6.tick_params(axis="x", rotation=30)
        st.pyplot(fig6, clear_figure=True)
    else:
        st.caption("No usable `reviewCreatedVersion` values found for version charts.")

    st.divider()

    # Topics
    st.subheader("Negative themes (Topic modeling)")
    neg_df = fdf[fdf["sentiment"] == "negative"]

    if len(neg_df) < 30:
        st.info("Not enough negative reviews for stable topics (need ~30+). Widen filters.")
    else:
        n_topics = st.slider("Number of topics", 3, 8, 5)
        topics = nmf_topics(neg_df["content"].astype(str).tolist(), n_topics=n_topics, n_terms=8)
        if not topics:
            st.info("Could not generate topics (text too sparse).")
        else:
            rows = [{"Topic": f"Topic {t['topic']}", "Top terms": ", ".join(t["terms"])} for t in topics]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # Terms
    l, r = st.columns(2)
    with l:
        st.subheader("Top terms in negative reviews (score 1–2)")
        neg_terms = top_terms(fdf.loc[fdf["sentiment"] == "negative", "content"], n=15)
        if neg_terms:
            st.dataframe(pd.DataFrame(neg_terms, columns=["term", "count"]), use_container_width=True)
        else:
            st.info("No negative reviews in this filtered set.")
    with r:
        st.subheader("Top terms in positive reviews (score 4–5)")
        pos_terms = top_terms(fdf.loc[fdf["sentiment"] == "positive", "content"], n=15)
        if pos_terms:
            st.dataframe(pd.DataFrame(pos_terms, columns=["term", "count"]), use_container_width=True)
        else:
            st.info("No positive reviews in this filtered set.")

    st.divider()

    # Helpful + Samples
    if has_thumbs:
        st.subheader("Most helpful reviews (by thumbsUpCount)")
        cols = ["score", "sentiment", "thumbsUpCount", "content"]
        if has_version: cols.insert(0, "reviewCreatedVersion")
        if has_date: cols.insert(0, "date")
        st.dataframe(fdf.sort_values("thumbsUpCount", ascending=False).head(10)[cols], use_container_width=True)
        st.divider()

    st.subheader("Samples")
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
