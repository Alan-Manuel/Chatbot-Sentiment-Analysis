import re
from collections import Counter
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# ==========================================================
# Streamlit Setup
# ==========================================================
st.set_page_config(page_title="Review Analyzer Chatbot", layout="wide")
st.title("Review Analyzer 📝🤖")
st.caption("Kaggle reviews CSV → landing instructions + dashboard + chat (no API keys).")


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
uploaded = st.file_uploader("Upload your Kaggle reviews CSV", type=["csv"])
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

# Kaggle dataset usually has these columns:
has_date = "at" in df.columns
if has_date:
    df["at_parsed"] = safe_parse_datetime(df["at"])
    df["date"] = df["at_parsed"].dt.date.astype("string")

has_version = "reviewCreatedVersion" in df.columns
has_thumbs = "thumbsUpCount" in df.columns


# ==========================================================
# Sidebar: Filters + Threshold Controls (more intuitive charts)
# ==========================================================
with st.sidebar:
    st.header("Filters")

    sentiment_opt = ["ALL"] + sorted(df["sentiment"].dropna().unique().tolist())
    sentiment_filter = st.selectbox("Sentiment", sentiment_opt)

    if has_version:
        vers = df["reviewCreatedVersion"].fillna("").astype(str).str.strip()
        version_opt = ["ALL"] + sorted([v for v in vers.unique().tolist() if v != ""])
        version_filter = st.selectbox("Version", version_opt)
    else:
        version_filter = "ALL"
        st.caption("No `reviewCreatedVersion` column found.")

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
        st.caption("No `at` column found (trend charts disabled).")

    st.divider()
    st.header("Thresholds")

    thr_healthy = st.slider("Healthy avg score ≥", 3.0, 5.0, 4.0, 0.1)
    thr_watch = st.slider("Watchlist avg score ≤", 1.0, 4.5, 3.0, 0.1)
    thr_neg_share = st.slider("Alert if negative share >", 0.10, 0.80, 0.35, 0.05)

    z_window = st.slider("Anomaly window (periods)", 4, 16, 6, 1)
    z_cutoff = st.slider("Anomaly sensitivity (z)", 1.5, 4.0, 2.5, 0.1)


# ==========================================================
# Apply filters
# ==========================================================
fdf = df.copy()

if sentiment_filter != "ALL":
    fdf = fdf[fdf["sentiment"] == sentiment_filter]

if has_version and version_filter != "ALL":
    fdf = fdf[fdf["reviewCreatedVersion"].astype(str).str.strip() == str(version_filter)]

if has_date and start_date and end_date:
    fdf = fdf[
        (fdf["at_parsed"].notna()) &
        (fdf["at_parsed"].dt.date >= start_date) &
        (fdf["at_parsed"].dt.date <= end_date)
    ]

if len(fdf) == 0:
    st.warning("No rows match your filters. Try widening the filters.")
    st.stop()


# ==========================================================
# Core metrics (always visible)
# ==========================================================
total = len(fdf)
avg_rating = float(fdf["score"].mean())
sent_counts = fdf["sentiment"].value_counts().to_dict()
neg = int(sent_counts.get("negative", 0))
neu = int(sent_counts.get("neutral", 0))
pos = int(sent_counts.get("positive", 0))

neg_share_overall = neg / total if total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews (filtered)", f"{total:,}")
c2.metric("Avg score", f"{avg_rating:.2f}")
c3.metric("Negative", f"{neg:,}")
c4.metric("Positive", f"{pos:,}")

# Status badge (intuitive)
if avg_rating >= thr_healthy and neg_share_overall <= thr_neg_share:
    st.success("Status: Healthy ✅ (based on your thresholds)")
elif avg_rating <= thr_watch or neg_share_overall >= thr_neg_share:
    st.warning("Status: Watchlist ⚠️ (based on your thresholds)")
else:
    st.info("Status: Mixed / Monitor ℹ️ (based on your thresholds)")


# ==========================================================
# Trends + Flags helpers
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

    return {"tdf": tdf, "vol": vol, "avg": avg, "share": share}

def rolling_z_flags(series: pd.Series, window: int, z: float) -> pd.Series:
    if len(series) < max(8, window + 2):
        return pd.Series([False] * len(series), index=series.index)
    roll_mean = series.rolling(window, min_periods=window).mean()
    roll_std = series.rolling(window, min_periods=window).std().replace(0, np.nan)
    zscore = (series - roll_mean) / roll_std
    return zscore.abs().fillna(0) >= z

def make_trend_figs(tr: Dict[str, pd.DataFrame], title_prefix: str = "") -> Dict[str, Any]:
    vol = tr["vol"]
    avg = tr["avg"]
    share = tr["share"].set_index("period")

    # Volume
    fig_vol = px.line(vol, x="period", y="reviews", title=f"{title_prefix}Review Volume Over Time")
    fig_vol.update_layout(xaxis_title="Date", yaxis_title="Reviews")

    # Avg rating + threshold lines
    fig_avg = px.line(avg, x="period", y="avg_score", title=f"{title_prefix}Average Rating Over Time")
    fig_avg.update_layout(xaxis_title="Date", yaxis_title="Avg score", yaxis_range=[0, 5])
    fig_avg.add_hline(y=thr_healthy, line_dash="dash", annotation_text=f"Healthy ≥ {thr_healthy:.1f}")
    fig_avg.add_hline(y=thr_watch, line_dash="dash", annotation_text=f"Watch ≤ {thr_watch:.1f}")

    # Negative share line (more readable than stacked share for “alerts”)
    neg_line = share["negative"].reset_index().rename(columns={"negative": "neg_share"})
    fig_neg = px.line(neg_line, x="period", y="neg_share", title=f"{title_prefix}Negative Share Over Time")
    fig_neg.update_layout(xaxis_title="Date", yaxis_title="Negative share", yaxis_range=[0, 1])
    fig_neg.add_hline(y=thr_neg_share, line_dash="dash", annotation_text=f"Alert > {thr_neg_share:.2f}")

    # Flags table
    avg_series = avg.set_index("period")["avg_score"].reindex(share.index).ffill().bfill()
    neg_share = share["negative"].copy()
    vol_series = vol.set_index("period")["reviews"].reindex(share.index).fillna(0)

    flag_thresh = (avg_series <= (thr_watch + 0.5)) | (neg_share >= thr_neg_share)
    flag_anom_vol = rolling_z_flags(vol_series, window=z_window, z=z_cutoff)
    flag_anom_neg = rolling_z_flags(neg_share, window=z_window, z=z_cutoff)

    flagged = pd.DataFrame({
        "period": share.index,
        "avg_score": avg_series.values,
        "neg_share": neg_share.values,
        "reviews": vol_series.values,
        "flag_threshold": flag_thresh.values,
        "flag_anom_volume": flag_anom_vol.values,
        "flag_anom_neg_share": flag_anom_neg.values,
    })
    flagged["any_flag"] = flagged[["flag_threshold","flag_anom_volume","flag_anom_neg_share"]].any(axis=1)

    return {"fig_vol": fig_vol, "fig_avg": fig_avg, "fig_neg": fig_neg, "flagged": flagged}


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
# HOME TAB (landing instructions)
# ==========================================================
with tab_home:
    st.subheader("Welcome 👋")
    st.markdown(
        """
This app helps you understand Kaggle review sentiment **and** chat with your dataset.

### What you can do
- **Dashboard:** interactive charts + thresholds + flagged periods + versions + themes.
- **Chat:** type questions like a chatbot and get answers grounded in the filtered data.

### Expected CSV columns (Kaggle format works)
Required:
- `content` (review text)
- `score` (1–5)

Optional:
- `at` (enables trends + flags)
- `reviewCreatedVersion` (version comparisons)
- `thumbsUpCount` (most helpful reviews)

### Try these prompts
"""
    )
    st.code(
        "Give me a summary\n"
        "What's driving negative reviews?\n"
        "Trend weekly\n"
        "Trend monthly\n"
        "Show flagged periods\n"
        "Compare versions\n"
        "Show worst reviews",
        language="text"
    )


# ==========================================================
# CHAT TAB (conversation + charts in chat)
# ==========================================================
with tab_chat:
    st.subheader("Chat with your dataset 🤖")
    st.caption("Type anything. The assistant can also render charts/tables in-chat.")

    # Quick prompt buttons
    b1, b2, b3, b4, b5, b6, b7 = st.columns(7)
    if b1.button("Summary"):
        st.session_state["_quick_prompt"] = "Give me a summary"
    if b2.button("Drivers"):
        st.session_state["_quick_prompt"] = "What's driving negative reviews?"
    if b3.button("Trend (W)"):
        st.session_state["_quick_prompt"] = "Trend weekly"
    if b4.button("Trend (M)"):
        st.session_state["_quick_prompt"] = "Trend monthly"
    if b5.button("Flags"):
        st.session_state["_quick_prompt"] = "Show flagged periods"
    if b6.button("Versions"):
        st.session_state["_quick_prompt"] = "Compare versions"
    if b7.button("Worst"):
        st.session_state["_quick_prompt"] = "Show worst reviews"

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey! Ask me about sentiment, trends, flagged periods, versions, themes, or examples."}
        ]

    # Render prior messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    def detect_intent(q: str) -> str:
        ql = q.lower()
        if any(k in ql for k in ["summary", "overall", "health", "status", "snapshot"]):
            return "summary"
        if any(k in ql for k in ["why", "drivers", "reason", "cause", "complaint", "issues", "problem"]):
            return "drivers_negative"
        if any(k in ql for k in ["topic", "themes", "theme", "clusters"]):
            return "themes"
        if any(k in ql for k in ["trend", "over time", "monthly", "weekly", "daily"]):
            return "trend"
        if any(k in ql for k in ["flag", "flags", "spike", "anomaly", "alert"]):
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
        return "Monthly"

    def chatbot_answer_and_artifacts(q: str, fdf_local: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a dict:
          - text: str
          - figs: list[plotly_fig]
          - tables: list[pd.DataFrame]
        """
        intent = detect_intent(q)
        total_local = len(fdf_local)
        if total_local == 0:
            return {"text": "No reviews match the current filters.", "figs": [], "tables": []}

        neg_local = int((fdf_local["sentiment"] == "negative").sum())
        neu_local = int((fdf_local["sentiment"] == "neutral").sum())
        pos_local = int((fdf_local["sentiment"] == "positive").sum())
        avg_local = float(fdf_local["score"].mean())
        neg_share_local = neg_local / total_local

        if intent == "summary":
            text = (
                f"**Snapshot (current filters)**\n\n"
                f"- Reviews: **{total_local:,}**\n"
                f"- Avg score: **{avg_local:.2f}/5**\n"
                f"- Split: **{(pos_local/total_local)*100:.1f}% positive**, "
                f"**{(neu_local/total_local)*100:.1f}% neutral**, "
                f"**{(neg_local/total_local)*100:.1f}% negative**\n\n"
                f"**Your thresholds:** Healthy ≥ **{thr_healthy:.1f}**, Watch ≤ **{thr_watch:.1f}**, "
                f"Alert if negative share > **{thr_neg_share:.2f}**."
            )
            return {"text": text, "figs": [], "tables": []}

        if intent == "drivers_negative":
            neg_df_local = fdf_local[fdf_local["sentiment"] == "negative"]
            if len(neg_df_local) < 10:
                return {"text": "Not enough negative reviews to find stable drivers. Widen filters.", "figs": [], "tables": []}
            terms = top_terms(neg_df_local["content"], n=12)
            top_list = ", ".join([f"`{t}`" for t, _ in terms[:8]])
            text = (
                f"Top recurring terms in negative reviews: {top_list}\n\n"
                f"Try: **Negative themes** (to cluster), or **Show worst reviews** (examples)."
            )
            return {"text": text, "figs": [], "tables": []}

        if intent == "themes":
            neg_df_local = fdf_local[fdf_local["sentiment"] == "negative"]
            if len(neg_df_local) < 30:
                return {"text": "Not enough negative reviews for stable topics (need ~30+). Widen filters.", "figs": [], "tables": []}
            topics = nmf_topics(neg_df_local["content"].astype(str).tolist(), n_topics=5, n_terms=6)
            if not topics:
                return {"text": "Could not generate themes (text too sparse). Widen filters.", "figs": [], "tables": []}
            lines = [f"- **Topic {t['topic']}**: " + ", ".join([f"`{x}`" for x in t["terms"]]) for t in topics]
            return {"text": "**Negative themes:**\n\n" + "\n".join(lines), "figs": [], "tables": []}

        if intent == "trend":
            if "at_parsed" not in fdf_local.columns or fdf_local["at_parsed"].isna().all():
                return {"text": "Trend requires a usable `at` column (parseable dates).", "figs": [], "tables": []}
            gran = extract_granularity(q)
            tr = build_trends(fdf_local, gran)
            figs = []
            tables = []
            pack = make_trend_figs(tr, title_prefix=f"({gran}) ")
            figs.extend([pack["fig_vol"], pack["fig_avg"], pack["fig_neg"]])

            # quick read
            avg_df = tr["avg"]
            share_df = tr["share"].set_index("period")
            last_avg = float(avg_df["avg_score"].iloc[-1]) if len(avg_df) else avg_local
            last_neg = float(share_df["negative"].iloc[-1]) if len(share_df) else neg_share_local

            text = (
                f"**Trend ({gran})**\n\n"
                f"- Latest avg score: **{last_avg:.2f}**\n"
                f"- Latest negative share: **{last_neg*100:.1f}%**\n\n"
                f"I rendered the charts below with your thresholds."
            )
            return {"text": text, "figs": figs, "tables": []}

        if intent == "flags":
            if "at_parsed" not in fdf_local.columns or fdf_local["at_parsed"].isna().all():
                return {"text": "Flags require a usable `at` column (parseable dates).", "figs": [], "tables": []}
            tr = build_trends(fdf_local, "Weekly")
            pack = make_trend_figs(tr, title_prefix="(Weekly) ")
            flagged = pack["flagged"]
            flagged_show = flagged[flagged["any_flag"]].sort_values("period", ascending=False).head(30)

            text = (
                "Here are **recent flagged weekly periods** (thresholds + anomaly detection). "
                "Use this as your quick triage list."
            )
            return {"text": text, "figs": [pack["fig_avg"], pack["fig_neg"]], "tables": [flagged_show]}

        if intent == "version":
            if "reviewCreatedVersion" not in fdf_local.columns or fdf_local["reviewCreatedVersion"].astype(str).str.strip().eq("").all():
                return {"text": "No usable `reviewCreatedVersion` values found to compare versions.", "figs": [], "tables": []}

            vdf = fdf_local.copy()
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
                .sort_values("reviews", ascending=False)
                .head(12)
                .sort_values("reviewCreatedVersion")
            )

            fig_avg_v = px.bar(v_agg, x="reviewCreatedVersion", y="avg_score", title="Average Rating by Version")
            fig_avg_v.update_layout(xaxis_title="Version", yaxis_title="Avg score", yaxis_range=[0, 5])
            fig_avg_v.add_hline(y=thr_healthy, line_dash="dash", annotation_text=f"Healthy ≥ {thr_healthy:.1f}")
            fig_avg_v.add_hline(y=thr_watch, line_dash="dash", annotation_text=f"Watch ≤ {thr_watch:.1f}")

            fig_neg_v = px.bar(v_agg, x="reviewCreatedVersion", y="neg_share", title="Negative Share by Version")
            fig_neg_v.update_layout(xaxis_title="Version", yaxis_title="Negative share", yaxis_range=[0, 1])
            fig_neg_v.add_hline(y=thr_neg_share, line_dash="dash", annotation_text=f"Alert > {thr_neg_share:.2f}")

            text = "Here’s a **version comparison** (top versions by review volume in current filters)."
            return {"text": text, "figs": [fig_avg_v, fig_neg_v], "tables": [v_agg]}

        if intent == "samples":
            low = fdf_local.sort_values("score").head(10)
            show_cols = ["score", "sentiment", "content"]
            if "reviewCreatedVersion" in fdf_local.columns:
                show_cols.insert(0, "reviewCreatedVersion")
            if "date" in fdf_local.columns:
                show_cols.insert(0, "date")

            text = "Here are **low-score review samples** (trimmed for quick triage)."
            return {"text": text, "figs": [], "tables": [low[show_cols]]}

        return {
            "text": (
                "Try:\n"
                "- Give me a summary\n"
                "- What's driving negative reviews?\n"
                "- Trend weekly / Trend monthly\n"
                "- Show flagged periods\n"
                "- Compare versions\n"
                "- Show worst reviews"
            ),
            "figs": [],
            "tables": []
        }

    # Always keep a real text input
    user_q = st.chat_input("Ask something… (e.g., 'Trend weekly', 'Show flagged periods', 'Compare versions')")

    # Quick prompt injection
    quick = st.session_state.pop("_quick_prompt", None)
    if quick:
        user_q = quick

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        out = chatbot_answer_and_artifacts(user_q, fdf)
        st.session_state.messages.append({"role": "assistant", "content": out["text"]})

        with st.chat_message("assistant"):
            st.markdown(out["text"])
            for fig in out.get("figs", []):
                st.plotly_chart(fig, use_container_width=True)
            for t in out.get("tables", []):
                st.dataframe(t, use_container_width=True)


# ==========================================================
# DASHBOARD TAB (all charts visible normally)
# ==========================================================
with tab_dash:
    st.subheader("Dashboard")

    # Trends
    st.markdown("### Trends (interactive + thresholds)")
    if (not has_date) or ("at_parsed" not in fdf.columns) or fdf["at_parsed"].isna().all():
        st.warning("No usable `at` column → time trends disabled. (Your Kaggle file should have `at`.)")

        # fallback charts
        figA = px.histogram(fdf, x="score", nbins=6, title="Score Distribution")
        st.plotly_chart(figA, use_container_width=True)

        sent = fdf["sentiment"].value_counts().reset_index()
        sent.columns = ["sentiment", "count"]
        figB = px.bar(sent, x="sentiment", y="count", title="Sentiment Counts")
        st.plotly_chart(figB, use_container_width=True)
    else:
        gran = st.radio("Trend granularity", ["Daily", "Weekly", "Monthly"], horizontal=True, key="dash_gran")
        tr = build_trends(fdf, gran)
        pack = make_trend_figs(tr, title_prefix=f"({gran}) ")

        st.plotly_chart(pack["fig_vol"], use_container_width=True)
        st.plotly_chart(pack["fig_avg"], use_container_width=True)
        st.plotly_chart(pack["fig_neg"], use_container_width=True)

        st.markdown("### Flagged periods (quick triage)")
        st.caption("Flags are driven by your threshold sliders + anomaly sensitivity controls.")
        flagged_show = pack["flagged"][pack["flagged"]["any_flag"]].sort_values("period", ascending=False)
        st.dataframe(flagged_show, use_container_width=True)

    st.divider()

    # Version comparison
    st.markdown("### Version comparison")
    if has_version and fdf["reviewCreatedVersion"].astype(str).str.strip().ne("").any():
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

        top_n = st.slider("Show top N versions (by review count)", 5, 30, 12, key="dash_top_n")
        v_agg = v_agg.sort_values("reviews", ascending=False).head(top_n).sort_values("reviewCreatedVersion")

        fig_avg_v = px.bar(v_agg, x="reviewCreatedVersion", y="avg_score", title="Average Rating by Version")
        fig_avg_v.update_layout(yaxis_range=[0, 5])
        fig_avg_v.add_hline(y=thr_healthy, line_dash="dash", annotation_text=f"Healthy ≥ {thr_healthy:.1f}")
        fig_avg_v.add_hline(y=thr_watch, line_dash="dash", annotation_text=f"Watch ≤ {thr_watch:.1f}")
        st.plotly_chart(fig_avg_v, use_container_width=True)

        fig_neg_v = px.bar(v_agg, x="reviewCreatedVersion", y="neg_share", title="Negative Share by Version")
        fig_neg_v.update_layout(yaxis_range=[0, 1])
        fig_neg_v.add_hline(y=thr_neg_share, line_dash="dash", annotation_text=f"Alert > {thr_neg_share:.2f}")
        st.plotly_chart(fig_neg_v, use_container_width=True)

        st.dataframe(v_agg, use_container_width=True)
    else:
        st.info("No usable `reviewCreatedVersion` values found.")

    st.divider()

    # Themes + Terms
    st.markdown("### Negative themes (Topic modeling)")
    neg_df = fdf[fdf["sentiment"] == "negative"]
    if len(neg_df) < 30:
        st.info("Not enough negative reviews for stable topics (need ~30+). Widen filters.")
    else:
        n_topics = st.slider("Number of topics", 3, 8, 5, key="dash_topics")
        topics = nmf_topics(neg_df["content"].astype(str).tolist(), n_topics=n_topics, n_terms=8)
        if not topics:
            st.info("Could not generate themes (text too sparse).")
        else:
            rows = [{"Topic": f"Topic {t['topic']}", "Top terms": ", ".join(t["terms"])} for t in topics]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    st.markdown("### Top terms")
    l, r = st.columns(2)
    with l:
        st.subheader("Top terms in negative reviews")
        neg_terms = top_terms(fdf.loc[fdf["sentiment"] == "negative", "content"], n=15)
        st.dataframe(pd.DataFrame(neg_terms, columns=["term", "count"]), use_container_width=True)
    with r:
        st.subheader("Top terms in positive reviews")
        pos_terms = top_terms(fdf.loc[fdf["sentiment"] == "positive", "content"], n=15)
        st.dataframe(pd.DataFrame(pos_terms, columns=["term", "count"]), use_container_width=True)

    st.divider()

    if has_thumbs:
        st.markdown("### Most helpful reviews")
        cols = ["score", "sentiment", "thumbsUpCount", "content"]
        if has_version:
            cols.insert(0, "reviewCreatedVersion")
        if has_date:
            cols.insert(0, "date")
        st.dataframe(
            fdf.sort_values("thumbsUpCount", ascending=False).head(15)[cols],
            use_container_width=True
        )

    st.divider()
    st.markdown("### Samples")
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
