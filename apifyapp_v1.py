import os
import re
from datetime import datetime
from collections import Counter
from itertools import chain, combinations
from typing import Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from apify_client import ApifyClient

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

# Configure the Streamlit page
st.set_page_config(
    page_title="Instagram Reach & Impressions Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APIFY_TOKEN = None
if hasattr(st, "secrets"):
    APIFY_TOKEN = st.secrets.get("APIFY_TOKEN")
if not APIFY_TOKEN:
    load_dotenv()
    APIFY_TOKEN = os.getenv("APIFY_TOKEN")

if not APIFY_TOKEN:
    st.error(
        "APIFY_TOKEN not found. Set it via Streamlit secrets or environment variables before running the dashboard."
    )
    st.stop()

PRODUCT_TYPE_MAP = {
    "feed": "Feed Post (Photo/Video)",
    "feed_single": "Feed Post (Photo/Video)",
    "feed_video": "Feed Post (Photo/Video)",
    "carousel_container": "Carousel (Mixed Media)",
    "carousel_child": "Carousel Slide",
    "reels": "Reel",
    "clip": "Reel",
    "clips": "Reel",
    "story": "Story",
    "igtv": "IGTV",
    "live": "Live Broadcast",
    "ad": "Sponsored Post",
    "sponsored": "Sponsored Post",
    "shopping": "Shoppable Post",
    "product_tag": "Shoppable Post",
    "guide": "Guide",
    "other": "Other",
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def format_number(value: Union[float, int, None], is_percent: bool = False) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "–"
    if is_percent:
        return f"{value:,.1f}%"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.0f}"


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0:
        return np.nan
    return numerator / denominator


def extract_hashtags(raw_hashtags) -> list[str]:
    if isinstance(raw_hashtags, list):
        tags = raw_hashtags
    elif isinstance(raw_hashtags, str):
        tags = [tag.strip() for tag in raw_hashtags.split(",")]
    else:
        tags = []
    clean_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag = tag.strip().replace("#", "").lower()
        if tag:
            clean_tags.append(tag)
    return clean_tags


def tokenize_caption(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def flatten_comments(row) -> str:
    comments = []
    fields = ["latestComments", "comments", "firstComment", "lastComment"]
    for field in fields:
        value = row.get(field)
        if isinstance(value, list):
            for comment in value:
                if isinstance(comment, dict):
                    text = comment.get("text") or comment.get("content")
                    if text:
                        comments.append(str(text))
                elif isinstance(comment, str):
                    comments.append(comment)
        elif isinstance(value, dict):
            text = value.get("text") or value.get("content")
            if text:
                comments.append(str(text))
        elif isinstance(value, str):
            comments.append(value)
    return " ".join(comments)


def analyze_sentiment(text: str) -> Optional[float]:
    if TextBlob is None or not text:
        return np.nan
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return np.nan


def sentiment_label(score: Optional[float], neutral_window: float = 0.05) -> str:
    if score is None or np.isnan(score):
        return "neutral"
    if score > neutral_window:
        return "positive"
    if score < -neutral_window:
        return "negative"
    return "neutral"


def get_image_url(post: pd.Series) -> Optional[str]:
    candidate_fields = [
        "displayUrl",
        "display_url",
        "thumbnailUrl",
        "thumbnailSrc",
        "imageUrl",
        "image_url",
        "previewImage",
    ]
    for field in candidate_fields:
        value = post.get(field)
        if isinstance(value, str) and value.startswith("http"):
            return value
    images_field = post.get("images")
    if isinstance(images_field, list) and images_field:
        first_img = images_field[0]
        if isinstance(first_img, dict):
            src = first_img.get("url") or first_img.get("src")
            if isinstance(src, str) and src.startswith("http"):
                return src
        if isinstance(first_img, str) and first_img.startswith("http"):
            return first_img
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_instagram_posts(
    mode: str,
    query: str,
    max_posts: int,
    owner_lock: Optional[str],
) -> pd.DataFrame:
    client = ApifyClient(APIFY_TOKEN)
    if mode == "Hashtag":
        actor_id = "apify/instagram-hashtag-scraper"
        run_input = {
            "hashtags": [query.replace("#", "").strip()],
            "resultsLimit": int(max_posts),
        }
    else:
        actor_id = "apify/instagram-post-scraper"
        run_input = {
            "username": [query.replace("@", "").strip()],
            "resultsLimit": int(max_posts),
        }
    run = client.actor(actor_id).call(run_input=run_input)
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        return pd.DataFrame()
    dataset = client.dataset(dataset_id)
    items = list(dataset.list_items().items)
    df = pd.DataFrame(items)
    if owner_lock and "ownerUsername" in df.columns:
        df = df[df["ownerUsername"].astype(str).str.lower() == owner_lock.lower()]
    return df.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def preprocess_posts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    processed = df.copy()
    numeric_cols = [
        "likesCount",
        "commentsCount",
        "videoViewCount",
        "playsCount",
        "saveCount",
        "impressions",
        "reach",
    ]
    for col in numeric_cols:
        if col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors="coerce").fillna(0.0)
        else:
            processed[col] = 0.0
    processed["ownerUsername"] = processed.get("ownerUsername", "").astype(str).replace("nan", "Unknown")
    processed["productType"] = processed.get("productType", "other").astype(str).str.lower()
    processed["productTypeDisplay"] = processed["productType"].map(PRODUCT_TYPE_MAP).fillna("Other")
    processed["caption"] = processed.get("caption", "").astype(str)
    processed["caption_preview"] = processed["caption"].apply(
        lambda text: (text[:140] + " …") if isinstance(text, str) and len(text) > 140 else text
    )
    processed["caption_tokens"] = processed["caption"].apply(tokenize_caption)
    processed["caption_length"] = processed["caption"].apply(lambda text: len(text) if isinstance(text, str) else 0)
    processed["hashtag_list"] = processed.get("hashtags", []).apply(extract_hashtags)
    processed["mentions"] = processed.get("mentions", []).apply(
        lambda mentions: [m.lower() for m in mentions] if isinstance(mentions, list) else []
    )
    processed["comment_text"] = processed.apply(flatten_comments, axis=1)
    processed["caption_sentiment"] = processed["caption"].apply(analyze_sentiment)
    processed["comment_sentiment"] = processed["comment_text"].apply(analyze_sentiment)
    processed["caption_sentiment_label"] = processed["caption_sentiment"].apply(sentiment_label)
    processed["comment_sentiment_label"] = processed["comment_sentiment"].apply(sentiment_label)
    processed["timestamp"] = pd.to_datetime(processed.get("timestamp", pd.NaT), errors="coerce", utc=True)
    processed["posted_date"] = processed["timestamp"].dt.date
    processed["posted_hour"] = processed["timestamp"].dt.hour
    processed["day_of_week"] = processed["timestamp"].dt.day_name()
    processed["engagement_interactions"] = processed["likesCount"] + processed["commentsCount"]
    processed["views_count"] = processed[["videoViewCount", "playsCount"]].max(axis=1)
    processed["impressions_proxy"] = processed[["impressions", "reach"]].max(axis=1)
    processed["impressions_proxy"] = processed["impressions_proxy"].mask(processed["impressions_proxy"] == 0)
    processed["reach_estimate"] = processed["impressions_proxy"].fillna(
        processed["likesCount"] + processed["commentsCount"] + processed["views_count"]
    )
    denominator = processed["impressions_proxy"].replace(0, np.nan)
    fallback_denominator = processed["reach_estimate"].replace(0, np.nan)
    processed["engagement_rate"] = processed["engagement_interactions"] / denominator
    processed.loc[processed["engagement_rate"].isna(), "engagement_rate"] = (
        processed["engagement_interactions"] / fallback_denominator
    )
    processed["engagement_rate_pct"] = processed["engagement_rate"] * 100
    processed["engagement_rate_pct"] = processed["engagement_rate_pct"].fillna(0.0)
    processed["engagement_volume"] = (
        processed["likesCount"] + processed["commentsCount"] + processed["views_count"]
    )
    processed = processed.replace([np.inf, -np.inf], np.nan)
    return processed


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple,
    product_types: list[str],
    hashtags: list[str],
    usernames: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if date_range:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["posted_date"] >= start_date)
            & (filtered["posted_date"] <= end_date)
        ]
    if product_types:
        filtered = filtered[filtered["productTypeDisplay"].isin(product_types)]
    if hashtags:
        hashtag_set = set(hashtags)
        filtered = filtered[filtered["hashtag_list"].apply(lambda tags: bool(set(tags) & hashtag_set))]
    if usernames:
        filtered = filtered[filtered["ownerUsername"].isin(usernames)]
    return filtered.reset_index(drop=True)


def compute_growth(series: pd.Series) -> Optional[float]:
    clean_series = series.dropna()
    if clean_series.empty or len(clean_series) < 2:
        return np.nan
    first = clean_series.iloc[0]
    last = clean_series.iloc[-1]
    if first == 0:
        return np.nan
    return ((last - first) / first) * 100


def render_overview_kpis(df: pd.DataFrame) -> None:
    total_likes = df["likesCount"].sum()
    total_comments = df["commentsCount"].sum()
    total_impressions = df["reach_estimate"].sum()
    avg_engagement_rate = df["engagement_rate_pct"].mean()
    daily_reach = (
        df.sort_values("timestamp")
        .groupby("posted_date")["reach_estimate"]
        .sum()
    )
    reach_growth_pct = compute_growth(daily_reach)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Likes", format_number(total_likes))
    col2.metric("Total Comments", format_number(total_comments))
    col3.metric("Avg Engagement Rate", format_number(avg_engagement_rate, is_percent=True))
    col4.metric(
        "Estimated Reach Growth",
        format_number(reach_growth_pct, is_percent=True),
        help="Percentage change in estimated reach between the first and latest posting dates in view.",
    )
    st.caption(
        "KPIs aggregate the filtered dataset. Engagement rate is calculated as (likes + comments) ÷ impressions (or best available proxy)."
    )
    st.divider()


def render_hashtag_analysis(df: pd.DataFrame) -> None:
    st.subheader("Hashtag Analysis")
    all_hashtags = list(chain.from_iterable(df["hashtag_list"].tolist()))
    if not all_hashtags:
        st.info("No hashtags found in the current selection.")
        st.divider()
        return
    hashtag_counts = Counter(all_hashtags).most_common(20)
    hashtag_df = pd.DataFrame(hashtag_counts, columns=["Hashtag", "Frequency"])
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            hashtag_df,
            x="Hashtag",
            y="Frequency",
            color="Frequency",
            color_continuous_scale="Purples",
        )
        fig.update_layout(
            height=350,
            xaxis_title="",
            yaxis_title="Usage Count",
            xaxis_tickangle=-35,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(
            hashtag_df.set_index("Hashtag"),
            use_container_width=True,
            height=300,
        )
    st.caption("Top hashtags ranked by frequency within the filtered posts.")
    if nx is None:
        st.info("Install `networkx` to view the hashtag co-occurrence network graph.")
        st.divider()
        return
    edge_weights = Counter()
    for tags in df["hashtag_list"]:
        unique_tags = sorted(set(tags))
        for combo in combinations(unique_tags, 2):
            edge_weights[tuple(sorted(combo))] += 1
    filtered_edges = {edge: weight for edge, weight in edge_weights.items() if weight >= 2}
    if not filtered_edges:
        st.info("Not enough repeated hashtag combinations to generate a network graph.")
        st.divider()
        return
    graph = nx.Graph()
    for (tag_a, tag_b), weight in filtered_edges.items():
        graph.add_edge(tag_a, tag_b, weight=weight)
    degrees = dict(graph.degree())
    node_sizes = [degrees[node] * 400 for node in graph.nodes()]
    pos = nx.spring_layout(graph, k=0.6, seed=42)
    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="rebeccapurple", alpha=0.8, ax=ax)
    edges = nx.draw_networkx_edges(
        graph,
        pos,
        width=[graph[u][v]["weight"] for u, v in graph.edges()],
        alpha=0.4,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="white", ax=ax)
    ax.set_axis_off()
    st.pyplot(fig, use_container_width=True)
    st.caption("Network of frequently co-occurring hashtags (edges represent repeated co-mentions).")
    st.divider()


def render_audience_engagement(df: pd.DataFrame) -> None:
    st.subheader("Audience Engagement Insights")
    col1, col2 = st.columns(2)
    with col1:
        product_metrics = (
            df.groupby("productTypeDisplay")
            .agg(
                Avg_Engagement_Rate=("engagement_rate_pct", "mean"),
                Median_Reach=("reach_estimate", "median"),
            )
            .reset_index()
        )
        if product_metrics.empty:
            st.info("No product type data available.")
        else:
            fig = px.bar(
                product_metrics,
                x="productTypeDisplay",
                y="Avg_Engagement_Rate",
                color="Avg_Engagement_Rate",
                color_continuous_scale="Teal",
            )
            fig.update_layout(
                height=360,
                xaxis_title="Product Type",
                yaxis_title="Avg Engagement Rate (%)",
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Average engagement rate by content format (hover to compare median reach).")
    with col2:
        sentiment_records = []
        for column, label in [
            ("caption_sentiment_label", "Captions"),
            ("comment_sentiment_label", "Comments"),
        ]:
            counts = df[column].value_counts()
            for sentiment, value in counts.items():
                sentiment_records.append(
                    {
                        "Sentiment": sentiment.title(),
                        "Source": label,
                        "Posts": int(value),
                    }
                )
        sentiment_df = pd.DataFrame(sentiment_records)
        if sentiment_df.empty:
            st.info("Sentiment analysis unavailable (TextBlob not installed or no text).")
        else:
            fig = px.bar(
                sentiment_df,
                x="Sentiment",
                y="Posts",
                color="Source",
                barmode="group",
                color_discrete_sequence=px.colors.sequential.RdPu,
            )
            fig.update_layout(
                height=360,
                xaxis_title="Sentiment",
                yaxis_title="Post Count",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Sentiment distribution for captions and recent comments (TextBlob polarity scoring).")
    scatter_df = df[df["engagement_rate_pct"].notnull()]
    if scatter_df.empty:
        st.info("Not enough engagement data to render the caption length scatterplot.")
    else:
        scatter_df = scatter_df.copy()
        scatter_df["bubble_size"] = scatter_df["likesCount"].clip(lower=0)
        fig = px.scatter(
            scatter_df,
            x="caption_length",
            y="engagement_rate_pct",
            size="bubble_size",
            color="productTypeDisplay",
            hover_data={
                "ownerUsername": True,
                "engagement_rate_pct": ":.2f",
                "caption_preview": True,
                "url": True,
            },
            labels={
                "caption_length": "Caption Length (characters)",
                "engagement_rate_pct": "Engagement Rate (%)",
            },
        )
        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Longer captions often correlate with deeper storytelling—use this scatter to find the sweet spot.")
    funnel_totals = {
        "Impressions (est.)": df["reach_estimate"].sum(),
        "Views": df["views_count"].sum(),
        "Likes": df["likesCount"].sum(),
        "Comments": df["commentsCount"].sum(),
        "Engagements": df["engagement_volume"].sum(),
    }
    funnel_df = pd.DataFrame(
        {
            "Stage": list(funnel_totals.keys()),
            "Value": list(funnel_totals.values()),
        }
    )
    fig = px.funnel(funnel_df, y="Stage", x="Value", color="Stage", color_discrete_sequence=px.colors.sequential.Purples)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Funnel illustrates the drop-off from impressions through to core engagement actions.")
    st.divider()


def render_temporal_trends(df: pd.DataFrame) -> None:
    st.subheader("Temporal Trends")
    timeline = (
        df.sort_values("timestamp")
        .groupby("posted_date")
        .agg(
            Likes=("likesCount", "sum"),
            Comments=("commentsCount", "sum"),
            Views=("views_count", "sum"),
            Reach=("reach_estimate", "sum"),
        )
        .reset_index()
    )
    col1, col2 = st.columns([3, 2])
    with col1:
        if timeline.empty:
            st.info("Insufficient chronological data to display trend lines.")
        else:
            melted = timeline.melt(id_vars="posted_date", var_name="Metric", value_name="Value")
            fig = px.line(
                melted,
                x="posted_date",
                y="Value",
                color="Metric",
                markers=True,
                color_discrete_sequence=px.colors.sequential.Magma,
            )
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Total",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Daily totals show growth momentum across likes, comments, views, and reach.")
    with col2:
        heatmap = (
            df.groupby(["day_of_week", "posted_hour"])["engagement_volume"]
            .mean()
            .reset_index()
        )
        if heatmap.empty:
            st.info("Insufficient posting timestamps to render the engagement heatmap.")
        else:
            heatmap["day_of_week"] = pd.Categorical(heatmap["day_of_week"], categories=DAY_ORDER, ordered=True)
            pivot = heatmap.pivot_table(
                index="day_of_week",
                columns="posted_hour",
                values="engagement_volume",
            ).fillna(0)
            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="Plasma",
                )
            )
            fig.update_layout(
                height=400,
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Spot the best days/times to post using engagement intensity across the week.")
    st.divider()


def render_top_posts(df: pd.DataFrame) -> None:
    st.subheader("Top Posts Gallery")
    if df.empty:
        st.info("No posts available in the filtered dataset.")
        return
    top_n = st.slider("Number of highlighted posts", min_value=3, max_value=12, value=6, step=1)
    ranking_metric = st.selectbox(
        "Ranking metric",
        options=["engagement_rate_pct", "reach_estimate", "likesCount", "commentsCount"],
        format_func=lambda option: {
            "engagement_rate_pct": "Engagement Rate",
            "reach_estimate": "Estimated Reach",
            "likesCount": "Likes",
            "commentsCount": "Comments",
        }[option],
        help="Choose how to rank top-performing posts in the gallery.",
    )
    top_posts = (
        df.sort_values(ranking_metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    columns = st.columns(3)
    for idx, post in top_posts.iterrows():
        target_column = columns[idx % 3]
        with target_column:
            image_url = get_image_url(post)
            if image_url:
                st.image(image_url, use_container_width=True)
            else:
                st.markdown("`Image unavailable`")
            st.markdown(f"**@{post.get('ownerUsername', 'unknown')}** · {post.get('productTypeDisplay', 'Unknown')}")
            st.markdown(
                f"❤️ {format_number(post.get('likesCount', 0))} · 💬 {format_number(post.get('commentsCount', 0))} · "
                f"ER {format_number(post.get('engagement_rate_pct', 0), is_percent=True)}"
            )
            st.caption(post.get("caption_preview", ""))
            if isinstance(post.get("url"), str) and post["url"].startswith("http"):
                st.markdown(f"[Open post]({post['url']})")
    st.caption("Gallery showcases the highest-performing posts based on your selected ranking metric.")
    st.divider()


def main() -> None:
    st.title("📈 Instagram Reach & Impressions Command Center")
    st.markdown(
        """
        Gain a 360° view of hashtag and profile performance. Fetch fresh posts from Apify, apply granular filters,
        and explore the reach, impressions, and engagement dynamics powering your top-performing content.
        """
    )
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = pd.DataFrame()
    if "last_params" not in st.session_state:
        st.session_state["last_params"] = {}
    with st.sidebar:
        st.header("Data Controls")
        search_mode = st.radio("Source Type", ["Hashtag", "Username"], horizontal=True)
        if search_mode == "Hashtag":
            query = st.text_input("Target hashtag", placeholder="socialmedia", help="Leave out the # symbol.")
            owner_lock = st.text_input(
                "Restrict to owner username (optional)",
                placeholder="brandhandle",
                help="Only keep posts from this creator when fetching hashtag results.",
            )
        else:
            query = st.text_input("Owner username", placeholder="brandhandle", help="You can enter one username per fetch.")
            owner_lock = ""
        max_posts = st.slider("Max posts to fetch", min_value=10, max_value=200, value=80, step=10)
        fetch_button = st.button("Fetch / Refresh Data", type="primary")
        st.markdown("---")
    if fetch_button:
        if not query.strip():
            st.sidebar.error("Please enter a hashtag or username before fetching.")
        else:
            try:
                with st.spinner("Fetching posts from Apify…"):
                    raw_df = fetch_instagram_posts(search_mode, query.strip(), int(max_posts), owner_lock.strip() or None)
                if raw_df.empty:
                    st.warning("No posts found for the supplied parameters.")
                else:
                    st.success(f"Fetched {len(raw_df)} posts from Apify.")
                st.session_state["raw_df"] = raw_df
                st.session_state["last_params"] = {
                    "mode": search_mode,
                    "query": query,
                    "max_posts": max_posts,
                    "owner_lock": owner_lock,
                }
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")
    raw_df = st.session_state.get("raw_df", pd.DataFrame())
    if raw_df.empty:
        st.info("Use the sidebar to fetch Instagram data by hashtag or username.")
        return
    processed_df = preprocess_posts(raw_df)
    with st.sidebar:
        st.header("Filters")
        if processed_df["posted_date"].notnull().any():
            min_date = processed_df["posted_date"].min()
            max_date = processed_df["posted_date"].max()
            default_range = (min_date, max_date)
            selected_dates = st.date_input(
                "Date range",
                value=default_range,
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(selected_dates, tuple):
                date_range = selected_dates
            else:
                date_range = (selected_dates, selected_dates)
        else:
            selected_dates = None
            date_range = None
        product_options = sorted(processed_df["productTypeDisplay"].dropna().unique().tolist())
        selected_products = st.multiselect(
            "Product types",
            options=product_options,
            default=product_options,
        )
        hashtag_options = sorted({tag for tags in processed_df["hashtag_list"] for tag in tags})
        selected_hashtags = st.multiselect("Hashtags", options=hashtag_options)
        user_options = sorted(processed_df["ownerUsername"].dropna().unique().tolist())
        selected_users = st.multiselect("Usernames", options=user_options)
    filtered_df = apply_filters(processed_df, date_range, selected_products, selected_hashtags, selected_users)
    if filtered_df.empty:
        st.warning("No posts match the current filters. Adjust the filters in the sidebar to see results.")
        return
    st.caption(
        f"Showing {len(filtered_df)} posts (filtered from {len(processed_df)} total fetched). "
        f"Last fetch params: {st.session_state['last_params']}"
    )
    render_overview_kpis(filtered_df)
    render_hashtag_analysis(filtered_df)
    render_audience_engagement(filtered_df)
    render_temporal_trends(filtered_df)
    render_top_posts(filtered_df)
    with st.expander("View filtered dataset"):
        st.dataframe(filtered_df, use_container_width=True)
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_data,
        file_name="instagram_filtered_posts.csv",
        mime="text/csv",
    )
    st.caption("Export includes all filtered rows and calculated metrics for further analysis.")


if __name__ == "__main__":
    main()
