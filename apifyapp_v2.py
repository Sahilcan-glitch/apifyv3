import os
import io
import json
import textwrap
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Any, Dict, Iterable, Optional, Union
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
from apify_client import ApifyClient

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

try:
    from email.message import EmailMessage
    import smtplib
except ImportError:
    EmailMessage = None
    smtplib = None

try:
    import plotly.io as pio
except ImportError:
    pio = None


st.set_page_config(
    page_title="Instagram Campaign Intelligence V3",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


APIFY_TOKEN: Optional[str] = None
if hasattr(st, "secrets"):
    APIFY_TOKEN = st.secrets.get("APIFY_TOKEN")
if not APIFY_TOKEN:
    load_dotenv()
    APIFY_TOKEN = os.getenv("APIFY_TOKEN")

if not APIFY_TOKEN:
    st.error(
        "APIFY_TOKEN not found. Configure it via Streamlit secrets or environment variables before running V3."
    )
    st.stop()


DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "instagram_metrics.db"


PRODUCT_TYPE_MAP: Dict[str, str] = {
    "feed": "Feed Post (Photo/Video)",
    "feed_single": "Feed Post (Photo/Video)",
    "feed_video": "Feed Post (Photo/Video)",
    "carousel_container": "Carousel",
    "carousel_child": "Carousel Slide",
    "carousel_album": "Carousel",
    "reels": "Reel",
    "clip": "Reel",
    "clips": "Reel",
    "story": "Story",
    "igtv": "Long-form Video",
    "live": "Live Broadcast",
    "ad": "Sponsored Post",
    "sponsored": "Sponsored Post",
    "shopping": "Shoppable Post",
    "product_tag": "Shoppable Post",
    "guide": "Guide",
    "other": "Other",
}


CONTENT_CATEGORY_MAP: Dict[str, str] = {
    "feed": "Photo / Video",
    "feed_single": "Photo / Video",
    "feed_video": "Photo / Video",
    "carousel_container": "Carousel",
    "carousel_child": "Carousel",
    "carousel_album": "Carousel",
    "reels": "Reel",
    "clip": "Reel",
    "clips": "Reel",
    "story": "Story",
    "igtv": "Long-form Video",
    "live": "Live",
    "ad": "Paid",
    "sponsored": "Paid",
    "shopping": "Shoppable",
    "product_tag": "Shoppable",
}


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ENGAGEMENT_COLUMNS = {
    "likes": "Likes",
    "comments": "Comments",
    "shares": "Shares",
    "saves": "Saves",
}

METRIC_HELP: Dict[str, str] = {
    "Total Posts": "Unique Instagram posts included in the filtered dataset.",
    "Reach": "Reach uses reported reach/impressions, or a proxy when unavailable.",
    "Impressions": "Reported impressions aggregated across the filtered dataset.",
    "Engagement": "Likes + comments + shares + saves.",
    "Engagement Rate": "Total Engagement ÷ Reach × 100.",
}


def initialize_database() -> None:
    """Ensure the SQLite persistence layer exists."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                timestamp TEXT,
                posted_date TEXT,
                posted_hour INTEGER,
                owner_username TEXT,
                caption TEXT,
                hashtags TEXT,
                source_query TEXT,
                product_type TEXT,
                product_display TEXT,
                content_category TEXT,
                country TEXT,
                city TEXT,
                latitude REAL,
                longitude REAL,
                likes REAL,
                comments REAL,
                shares REAL,
                saves REAL,
                impressions REAL,
                reach REAL,
                reach_final REAL,
                video_views REAL,
                engagement_total REAL,
                engagement_rate REAL,
                engagement_rate_pct REAL,
                raw_json TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_date ON posts (posted_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_hashtags ON posts (hashtags)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS report_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                recipients TEXT NOT NULL,
                frequency TEXT NOT NULL,
                include_formats TEXT NOT NULL,
                last_run TEXT,
                filters TEXT NOT NULL,
                active INTEGER DEFAULT 1
            )
            """
        )
        conn.commit()


def get_sql_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def format_number(value: Union[float, int, None], is_percent: bool = False, digits: int = 1) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "–"
    if is_percent:
        return f"{value:.{digits}f}%"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.{digits}f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.{digits}f}K"
    return f"{value:.0f}"


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator in (None, 0) or (isinstance(denominator, float) and np.isnan(denominator)):
        return np.nan
    return numerator / denominator


def parse_query_input(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [segment.strip().replace("#", "").replace("@", "") for segment in raw.replace("\n", ",").split(",")]
    return [segment for segment in parts if segment]


def extract_hashtags(raw_hashtags: Any) -> list[str]:
    if isinstance(raw_hashtags, list):
        tags = raw_hashtags
    elif isinstance(raw_hashtags, str):
        tags = [tag.strip() for tag in raw_hashtags.split(",")]
    else:
        tags = []
    clean = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag = tag.strip().lower().replace("#", "")
        if tag:
            clean.append(tag)
    return clean


def parse_location_fields(row: pd.Series) -> Dict[str, Optional[Union[str, float]]]:
    location_candidate = row.get("location") or row.get("locationDict") or row.get("locationObject")
    result = {"country": None, "city": None, "latitude": None, "longitude": None}
    if isinstance(location_candidate, dict):
        result["country"] = (
            location_candidate.get("country_name")
            or location_candidate.get("country")
            or location_candidate.get("countryCode")
        )
        result["city"] = (
            location_candidate.get("city_name")
            or location_candidate.get("city")
            or location_candidate.get("name")
            or location_candidate.get("address_json", {}).get("city")
            if isinstance(location_candidate.get("address_json"), dict)
            else None
        )
        result["latitude"] = location_candidate.get("lat") or location_candidate.get("latitude")
        result["longitude"] = location_candidate.get("lng") or location_candidate.get("longitude")
    else:
        result["country"] = row.get("locationName") or row.get("locationSlug")
    if isinstance(result["country"], str):
        result["country"] = result["country"].strip()
    if isinstance(result["city"], str):
        result["city"] = result["city"].strip()
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_instagram_posts(
    mode: str,
    queries: Iterable[str],
    max_posts: int,
    owner_lock: Optional[str],
) -> pd.DataFrame:
    client = ApifyClient(APIFY_TOKEN)
    records: list[pd.DataFrame] = []
    for query in queries:
        if not query:
            continue
        if mode == "Hashtag":
            actor_id = "apify/instagram-hashtag-scraper"
            run_input = {"hashtags": [query], "resultsLimit": int(max_posts)}
        else:
            actor_id = "apify/instagram-post-scraper"
            run_input = {"username": [query], "resultsLimit": int(max_posts)}
        run = client.actor(actor_id).call(run_input=run_input)
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            continue
        dataset = client.dataset(dataset_id)
        items = list(dataset.list_items().items)
        if not items:
            continue
        df = pd.DataFrame(items)
        df["source_query"] = query
        if owner_lock and mode == "Hashtag" and "ownerUsername" in df.columns:
            df = df[df["ownerUsername"].astype(str).str.lower() == owner_lock.lower()]
        records.append(df.reset_index(drop=True))
    if not records:
        return pd.DataFrame()
    combined = pd.concat(records, ignore_index=True)
    return combined


def preprocess_posts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    processed = df.copy()
    id_candidates = ["id", "postId", "post_id", "shortCode", "shortcode"]
    post_id_series = pd.Series(pd.NA, index=processed.index, dtype="object")
    for candidate in id_candidates:
        if candidate in processed.columns:
            candidate_values = processed[candidate]
            candidate_str = candidate_values.astype(str)
            valid_mask = (
                candidate_values.notna()
                & candidate_str.str.strip().ne("")
                & candidate_str.str.lower().ne("nan")
            )
            fill_mask = post_id_series.isna() & valid_mask
            post_id_series = post_id_series.where(~fill_mask, candidate_str)
    missing_mask = (
        post_id_series.isna()
        | post_id_series.astype(str).str.strip().eq("")
        | post_id_series.astype(str).str.lower().eq("nan")
    )
    fallback_ids = processed.index.astype(str)
    post_id_series.loc[missing_mask] = fallback_ids[missing_mask.to_numpy()]
    processed["post_id"] = post_id_series.astype(str)
    processed["timestamp"] = pd.to_datetime(processed.get("timestamp", pd.NaT), errors="coerce", utc=True)
    processed["posted_date"] = processed["timestamp"].dt.date
    processed["posted_hour"] = processed["timestamp"].dt.hour
    processed["day_of_week"] = processed["timestamp"].dt.day_name()
    numeric_fields = {
        "likes": ["likesCount", "edge_liked_by", "likeCount"],
        "comments": ["commentsCount", "edge_media_to_comment", "commentCount"],
        "video_views": ["videoViewCount", "playsCount", "video_count"],
        "saves": ["saveCount", "savedCount"],
        "impressions": ["impressions"],
        "reach": ["reach"],
    }
    for target, source_candidates in numeric_fields.items():
        values = None
        for candidate in source_candidates:
            if candidate in processed.columns:
                values = pd.to_numeric(processed[candidate], errors="coerce")
                break
        processed[target] = values.fillna(0.0) if values is not None else 0.0
    share_candidates = ["shareCount", "sharesCount", "reshareCount", "sendCount"]
    share_values = None
    for candidate in share_candidates:
        if candidate in processed.columns:
            share_values = pd.to_numeric(processed[candidate], errors="coerce")
            break
    processed["shares"] = share_values.fillna(0.0) if share_values is not None else 0.0
    processed["ownerUsername"] = processed.get("ownerUsername", "Unknown").astype(str).replace("nan", "Unknown")
    processed["productType"] = processed.get("productType", "other").astype(str).str.lower()
    processed["product_display"] = processed["productType"].map(PRODUCT_TYPE_MAP).fillna("Other")
    processed["content_category"] = processed["productType"].map(CONTENT_CATEGORY_MAP).fillna("Other")
    processed["caption"] = processed.get("caption", processed.get("title", "")).astype(str)
    processed["hashtag_list"] = processed.get("hashtags", []).apply(extract_hashtags)
    processed["primary_hashtag"] = processed["hashtag_list"].apply(lambda tags: tags[0] if tags else None)
    processed["reach_final"] = processed["reach"]
    mask_zero_reach = processed["reach_final"] == 0
    processed.loc[mask_zero_reach, "reach_final"] = processed.loc[mask_zero_reach, "impressions"]
    mask_zero_reach = processed["reach_final"] == 0
    if mask_zero_reach.any():
        processed.loc[mask_zero_reach, "reach_final"] = (
            processed.loc[mask_zero_reach, ["likes", "comments", "shares", "saves"]].sum(axis=1)
        )
    processed["reach_final"] = processed["reach_final"].replace(0, np.nan)
    processed["engagement_total"] = processed["likes"] + processed["comments"] + processed["shares"] + processed["saves"]
    processed["engagement_rate"] = processed["engagement_total"].div(processed["reach_final"])
    processed["engagement_rate_pct"] = processed["engagement_rate"] * 100
    processed["source_query"] = processed.get("source_query", "").astype(str)
    location_details = processed.apply(parse_location_fields, axis=1, result_type="expand")
    processed[["country", "city", "latitude", "longitude"]] = location_details
    processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    return processed


def persist_posts(df: pd.DataFrame) -> None:
    if df.empty:
        return
    records = []
    for _, row in df.iterrows():
        hashtags_json = json.dumps(row.get("hashtag_list", []), ensure_ascii=False)
        record = (
            str(row["post_id"]),
            row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            row["posted_date"].isoformat() if isinstance(row["posted_date"], date) else None,
            int(row["posted_hour"]) if pd.notna(row["posted_hour"]) else None,
            row.get("ownerUsername"),
            row.get("caption"),
            hashtags_json,
            row.get("source_query"),
            row.get("productType"),
            row.get("product_display"),
            row.get("content_category"),
            row.get("country"),
            row.get("city"),
            float(row.get("latitude")) if pd.notna(row.get("latitude")) else None,
            float(row.get("longitude")) if pd.notna(row.get("longitude")) else None,
            float(row.get("likes", 0.0)),
            float(row.get("comments", 0.0)),
            float(row.get("shares", 0.0)),
            float(row.get("saves", 0.0)),
            float(row.get("impressions", 0.0)),
            float(row.get("reach", 0.0)),
            float(row.get("reach_final", 0.0)) if pd.notna(row.get("reach_final")) else None,
            float(row.get("video_views", 0.0)),
            float(row.get("engagement_total", 0.0)),
            float(row.get("engagement_rate", 0.0)) if pd.notna(row.get("engagement_rate")) else None,
            float(row.get("engagement_rate_pct", 0.0)) if pd.notna(row.get("engagement_rate_pct")) else None,
            json.dumps(row.to_dict(), default=str, ensure_ascii=False),
        )
        records.append(record)
    with get_sql_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO posts (
                post_id,
                timestamp,
                posted_date,
                posted_hour,
                owner_username,
                caption,
                hashtags,
                source_query,
                product_type,
                product_display,
                content_category,
                country,
                city,
                latitude,
                longitude,
                likes,
                comments,
                shares,
                saves,
                impressions,
                reach,
                reach_final,
                video_views,
                engagement_total,
                engagement_rate,
                engagement_rate_pct,
                raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            records,
        )
        conn.commit()


def load_posts(start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    with get_sql_connection() as conn:
        query = "SELECT * FROM posts WHERE 1=1"
        params: list[Any] = []
        if start:
            query += " AND posted_date >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND posted_date <= ?"
            params.append(end.isoformat())
        rows = conn.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        record = dict(row)
        record["posted_date"] = date.fromisoformat(record["posted_date"]) if record["posted_date"] else None
        record["timestamp"] = pd.to_datetime(record["timestamp"]) if record["timestamp"] else pd.NaT
        record["hashtags"] = json.loads(record["hashtags"]) if record.get("hashtags") else []
        record["engagement_rate"] = record.get("engagement_rate")
        record["engagement_rate_pct"] = record.get("engagement_rate_pct")
        records.append(record)
    df = pd.DataFrame(records)
    df.rename(columns={"hashtags": "hashtag_list"}, inplace=True)
    if not df.empty and "posted_hour" in df.columns:
        df["posted_hour"] = pd.to_numeric(df["posted_hour"], errors="coerce")
    numeric_cols = [
        "likes",
        "comments",
        "shares",
        "saves",
        "impressions",
        "reach",
        "reach_final",
        "video_views",
        "engagement_total",
        "engagement_rate",
        "engagement_rate_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.day_name()
    return df


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    daily = (
        df.groupby("posted_date")
        .agg(
            posts=("post_id", "nunique"),
            reach=("reach_final", "sum"),
            impressions=("impressions", "sum"),
            engagement=("engagement_total", "sum"),
            likes=("likes", "sum"),
            comments=("comments", "sum"),
            shares=("shares", "sum"),
            saves=("saves", "sum"),
        )
        .reset_index()
        .sort_values("posted_date")
    )
    daily["engagement_rate_pct"] = daily.apply(
        lambda row: safe_divide(row["engagement"], row["reach"]) * 100, axis=1
    )
    return daily


def compute_hashtag_rollup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    exploded = df.explode("hashtag_list")
    exploded = exploded[exploded["hashtag_list"].notnull()]
    if exploded.empty:
        return pd.DataFrame()
    metrics = (
        exploded.groupby("hashtag_list")
        .agg(
            posts=("post_id", "nunique"),
            reach=("reach_final", "sum"),
            impressions=("impressions", "sum"),
            engagement=("engagement_total", "sum"),
            likes=("likes", "sum"),
            comments=("comments", "sum"),
            shares=("shares", "sum"),
            saves=("saves", "sum"),
            avg_engagement_rate=("engagement_rate_pct", "mean"),
        )
        .reset_index()
        .rename(columns={"hashtag_list": "hashtag"})
    )
    return metrics.sort_values("engagement", ascending=False)


def compute_content_rollup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = (
        df.groupby("content_category")
        .agg(
            posts=("post_id", "nunique"),
            reach=("reach_final", "sum"),
            impressions=("impressions", "sum"),
            engagement=("engagement_total", "sum"),
            avg_reach=("reach_final", "mean"),
            avg_engagement_rate=("engagement_rate_pct", "mean"),
        )
        .reset_index()
    )
    return metrics.sort_values("engagement", ascending=False)


def compute_location_rollup(df: pd.DataFrame, level: str = "country") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    column = "country" if level == "country" else "city"
    metrics = (
        df[df[column].notnull()]
        .groupby(column)
        .agg(
            posts=("post_id", "nunique"),
            reach=("reach_final", "sum"),
            impressions=("impressions", "sum"),
            engagement=("engagement_total", "sum"),
            avg_engagement_rate=("engagement_rate_pct", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
        .rename(columns={column: "location"})
    )
    return metrics.sort_values("engagement", ascending=False)


def compute_period_summary(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"total_posts": 0, "reach": 0.0, "impressions": 0.0, "engagement": 0.0, "engagement_rate": np.nan}
    reach = df["reach_final"].sum()
    engagement = df["engagement_total"].sum()
    impressions = df["impressions"].sum()
    total_posts = df["post_id"].nunique()
    engagement_rate = safe_divide(engagement, reach) * 100
    return {
        "total_posts": total_posts,
        "reach": reach,
        "impressions": impressions,
        "engagement": engagement,
        "engagement_rate": engagement_rate,
    }


def compute_period_comparison(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    current = compute_period_summary(current_df)
    previous = compute_period_summary(previous_df)

    def delta(current_value: float, previous_value: float) -> float:
        if previous_value in (0, None) or (isinstance(previous_value, float) and np.isnan(previous_value)):
            return np.nan
        return ((current_value - previous_value) / previous_value) * 100

    comparison = {}
    for key in ["total_posts", "reach", "impressions", "engagement", "engagement_rate"]:
        comparison[key] = {
            "current": current.get(key),
            "previous": previous.get(key),
            "delta_pct": delta(current.get(key, 0.0), previous.get(key, 0.0)),
        }
    return comparison


def detect_anomalies(daily_df: pd.DataFrame, field: str, z_threshold: float = 2.5) -> list[Dict[str, Any]]:
    if daily_df.empty or field not in daily_df.columns or len(daily_df) < 3:
        return []
    values = daily_df[field].astype(float)
    mean = values.mean()
    std = values.std()
    if std == 0 or np.isnan(std):
        return []
    anomalies = []
    for _, row in daily_df.iterrows():
        value = row[field]
        if value in (None, 0) or np.isnan(value):
            continue
        z_score = (value - mean) / std
        if abs(z_score) >= z_threshold:
            anomalies.append(
                {
                    "date": row["posted_date"],
                    "value": value,
                    "z_score": z_score,
                }
            )
    return anomalies


def find_emerging_hashtags(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    tracked_hashtags: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    if current_df.empty:
        return pd.DataFrame()
    current_counter = Counter()
    previous_counter = Counter()
    tracked_set = {tag.lower() for tag in tracked_hashtags}
    for tags in current_df["hashtag_list"]:
        if not isinstance(tags, list):
            continue
        tags_lower = [tag.lower() for tag in tags]
        if tracked_set.intersection(tags_lower):
            for tag in tags_lower:
                if tag not in tracked_set:
                    current_counter[tag] += 1
    for tags in previous_df["hashtag_list"]:
        if not isinstance(tags, list):
            continue
        tags_lower = [tag.lower() for tag in tags]
        if tracked_set.intersection(tags_lower):
            for tag in tags_lower:
                if tag not in tracked_set:
                    previous_counter[tag] += 1
    deltas = []
    for tag, count in current_counter.items():
        previous_count = previous_counter.get(tag, 0)
        growth = count - previous_count
        if growth > 0:
            deltas.append({"hashtag": tag, "current": count, "previous": previous_count, "growth": growth})
    emerging = pd.DataFrame(deltas)
    if emerging.empty:
        return emerging
    return emerging.sort_values(["growth", "current"], ascending=False).head(top_n)


def suggest_post_times(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if df.empty or "posted_hour" not in df.columns:
        return pd.DataFrame()
    hourly = (
        df.groupby("posted_hour")
        .agg(
            avg_engagement_rate=("engagement_rate_pct", "mean"),
            median_engagement=("engagement_total", "median"),
            posts=("post_id", "nunique"),
        )
        .reset_index()
    )
    hourly = hourly.dropna(subset=["avg_engagement_rate"])
    if hourly.empty:
        return hourly
    return hourly.sort_values("avg_engagement_rate", ascending=False).head(top_n)


def top_post(df: pd.DataFrame, target_date: Optional[date] = None) -> Optional[pd.Series]:
    if df.empty:
        return None
    subset = df[df["posted_date"] == target_date] if target_date else df
    if subset.empty:
        return None
    return subset.sort_values("engagement_total", ascending=False).iloc[0]


def top_hashtag_of_week(df: pd.DataFrame, end_date: date) -> Optional[str]:
    if df.empty:
        return None
    start_date = end_date - timedelta(days=6)
    window = df[(df["posted_date"] >= start_date) & (df["posted_date"] <= end_date)]
    if window.empty:
        return None
    rollup = compute_hashtag_rollup(window)
    if rollup.empty:
        return None
    return rollup.iloc[0]["hashtag"]


def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[tuple[date, date]],
    hashtags: list[str],
    content_types: list[str],
    countries: list[str],
    cities: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if date_range:
        start, end = date_range
        filtered = filtered[
            (filtered["posted_date"] >= start)
            & (filtered["posted_date"] <= end)
        ]
    if hashtags:
        hashtag_set = {tag.lower() for tag in hashtags}
        filtered = filtered[
            filtered["hashtag_list"].apply(lambda tags: isinstance(tags, list) and bool(set(tags) & hashtag_set))
        ]
    if content_types:
        filtered = filtered[filtered["content_category"].isin(content_types)]
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if cities:
        filtered = filtered[filtered["city"].isin(cities)]
    return filtered.reset_index(drop=True)


def build_time_series_chart(
    current_daily: pd.DataFrame,
    previous_daily: pd.DataFrame,
    metric: str,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    fig = go.Figure()
    if not current_daily.empty:
        fig.add_trace(
            go.Scatter(
                x=current_daily["posted_date"],
                y=current_daily[metric],
                mode="lines+markers",
                name="Current period",
            )
        )
    if not previous_daily.empty:
        fig.add_trace(
            go.Scatter(
                x=previous_daily["posted_date"],
                y=previous_daily[metric],
                mode="lines",
                name="Previous period",
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        title=title,
        height=340,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title=yaxis_title,
        xaxis_title="Date",
    )
    return fig


def build_hashtag_comparison_chart(rollup: pd.DataFrame, metric: str) -> go.Figure:
    if rollup.empty:
        return go.Figure()
    chart_df = rollup.head(15)
    fig = px.bar(
        chart_df,
        x="hashtag",
        y=metric,
        color=metric,
        color_continuous_scale="Teal",
        title=f"Hashtag comparison by {metric}",
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=80), xaxis_tickangle=-30)
    return fig


def build_heatmap(df: pd.DataFrame, metric: str) -> go.Figure:
    if df.empty:
        return go.Figure()
    heat_df = (
        df.groupby(["day_of_week", "posted_hour"])
        .agg(value=(metric, "mean"))
        .reset_index()
    )
    heat_df["day_of_week"] = pd.Categorical(heat_df["day_of_week"], categories=DAY_ORDER, ordered=True)
    pivot = heat_df.pivot(index="day_of_week", columns="posted_hour", values="value").fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Viridis",
            colorbar_title=metric,
        )
    )
    fig.update_layout(
        title=f"Engagement by day & hour ({metric})",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Hour of day",
        yaxis_title="Day of week",
    )
    return fig


def build_geo_map(location_df: pd.DataFrame) -> go.Figure:
    if location_df.empty:
        return go.Figure()
    valid = location_df.dropna(subset=["latitude", "longitude"])
    if valid.empty:
        return go.Figure()
    fig = px.scatter_geo(
        valid,
        lat="latitude",
        lon="longitude",
        size="engagement",
        color="engagement",
        hover_name="location",
        hover_data={"reach": True, "engagement": True, "posts": True},
        projection="natural earth",
        height=420,
        title="Geo engagement heatmap",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def _excel_safe_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def _pdf_safe_text(text: Any, max_chunk: int = 24) -> str:
    """Ensure PDF strings only contain Latin-1 chars and wrap long tokens."""
    if not isinstance(text, str):
        text = str(text)
    text = (
        text.replace("\r", " ")
        .replace("\n", " ")
        .replace("•", "-")
    )
    sanitized_tokens: list[str] = []
    for token in text.split(" "):
        if not token:
            sanitized_tokens.append("")
            continue
        try:
            token.encode("latin-1")
        except UnicodeEncodeError:
            token = token.encode("latin-1", "ignore").decode("latin-1")
        if len(token) <= max_chunk:
            sanitized_tokens.append(token)
            continue
        wrapped = textwrap.wrap(token, width=max_chunk, break_long_words=True, break_on_hyphens=False)
        sanitized_tokens.append(" ".join(wrapped))
    sanitized = " ".join(sanitized_tokens)
    return " ".join(sanitized.split())


def export_dataframe_to_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, frame in sheets.items():
            sanitized = frame.copy()
            datetime_tz_cols = sanitized.select_dtypes(include=["datetimetz"]).columns
            for column in datetime_tz_cols:
                try:
                    sanitized[column] = sanitized[column].dt.tz_convert(None)
                except (TypeError, AttributeError):
                    sanitized[column] = sanitized[column].dt.tz_localize(None)
            for column in sanitized.columns:
                if sanitized[column].dtype == "O":
                    sanitized[column] = sanitized[column].apply(_excel_safe_value)
            sanitized.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buffer.seek(0)
    return buffer.read()


def export_figure_to_png(fig: go.Figure) -> Optional[bytes]:
    if fig is None or fig.data == ():
        return None
    if pio is None:
        return None
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def generate_pdf_summary(
    headline_metrics: Dict[str, Dict[str, float]],
    insights: list[str],
    recommendations: list[str],
) -> Optional[bytes]:
    if FPDF is None:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Instagram Marketing Intelligence Summary", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 12)
    for label, payload in headline_metrics.items():
        current_value = payload.get("current")
        delta = payload.get("delta_pct")
        text = _pdf_safe_text(f"{label}: {format_number(current_value)}")
        if delta is not None and not np.isnan(delta):
            text = f"{text} ({delta:+.1f}% vs prev.)"
        pdf.multi_cell(0, 7, text)
    if insights:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Highlights & Alerts", ln=True)
        pdf.set_font("Helvetica", "", 12)
        for insight in insights:
            pdf.multi_cell(0, 6, _pdf_safe_text(f"• {insight}"))
    if recommendations:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Recommended Actions", ln=True)
        pdf.set_font("Helvetica", "", 12)
        for rec in recommendations:
            pdf.multi_cell(0, 6, _pdf_safe_text(f"• {rec}"))
    return pdf.output(dest="S").encode("latin-1")


def generate_html_summary(
    headline_metrics: Dict[str, Dict[str, float]],
    insights: list[str],
    recommendations: list[str],
) -> str:
    rows = []
    for label, payload in headline_metrics.items():
        current_value = payload.get("current")
        delta = payload.get("delta_pct")
        delta_text = f"{delta:+.1f}%" if delta is not None and not np.isnan(delta) else "n/a"
        rows.append(
            f"<tr><td>{label}</td><td>{format_number(current_value)}</td><td>{delta_text}</td></tr>"
        )
    insights_html = "".join(f"<li>{text}</li>" for text in insights) or "<li>No anomalies detected.</li>"
    rec_html = "".join(f"<li>{text}</li>" for text in recommendations) or "<li>Keep monitoring performance.</li>"
    return f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 24px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f4f4f4; }}
            </style>
        </head>
        <body>
            <h1>Instagram Marketing Intelligence Summary</h1>
            <p>Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
            <table>
                <thead>
                    <tr><th>Metric</th><th>Current</th><th>Δ vs Previous</th></tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
            <h2>Highlights & Alerts</h2>
            <ul>{insights_html}</ul>
            <h2>Recommended Actions</h2>
            <ul>{rec_html}</ul>
        </body>
    </html>
    """


def send_email_report(
    recipients: list[str],
    subject: str,
    html_body: str,
    attachments: Dict[str, bytes],
) -> tuple[bool, Optional[str]]:
    if EmailMessage is None or smtplib is None:
        return False, "Email dependencies not installed (email.message, smtplib)."
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_from = os.getenv("EMAIL_FROM")
    if not all([smtp_host, smtp_port, smtp_user, smtp_password, email_from]):
        return False, "SMTP environment variables are incomplete."
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = email_from
    message["To"] = ", ".join(recipients)
    message.set_content("Plain-text version unavailable. View the HTML portion.")
    message.add_alternative(html_body, subtype="html")
    for filename, content in attachments.items():
        message.add_attachment(content, maintype="application", subtype="octet-stream", filename=filename)
    try:
        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)
        return True, None
    except Exception as exc:
        return False, str(exc)


def load_report_schedules() -> pd.DataFrame:
    with get_sql_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, recipients, frequency, include_formats, last_run, filters, active FROM report_schedules"
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["recipients"] = frame["recipients"].apply(lambda x: ", ".join(json.loads(x)))
    frame["include_formats"] = frame["include_formats"].apply(lambda x: ", ".join(json.loads(x)))
    return frame


def save_report_schedule(
    name: str,
    recipients: list[str],
    frequency: str,
    include_formats: list[str],
    filters: Dict[str, Any],
) -> None:
    payload = (
        name,
        json.dumps(recipients),
        frequency,
        json.dumps(include_formats),
        None,
        json.dumps(filters),
        1,
    )
    with get_sql_connection() as conn:
        conn.execute(
            """
            INSERT INTO report_schedules (
                name, recipients, frequency, include_formats, last_run, filters, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()


def update_schedule_last_run(schedule_id: int) -> None:
    with get_sql_connection() as conn:
        conn.execute(
            "UPDATE report_schedules SET last_run = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), schedule_id),
        )
        conn.commit()


def delete_schedule(schedule_id: int) -> None:
    with get_sql_connection() as conn:
        conn.execute("DELETE FROM report_schedules WHERE id = ?", (schedule_id,))
        conn.commit()


initialize_database()


def main() -> None:
    st.title("🚀 Instagram Campaign Intelligence V3")
    st.caption(
        "Daily refreshed analytics for Instagram hashtags and profiles so marketing teams can amplify awareness and leads."
    )

    if "last_fetch_info" not in st.session_state:
        st.session_state["last_fetch_info"] = None

    with st.sidebar:
        st.header("Data Ingestion")
        mode = st.radio("Source mode", options=["Hashtag", "Username"], horizontal=True)
        query_input = st.text_area(
            "Hashtags or usernames",
            placeholder="socialmedia, brandawareness",
            help="Comma or newline separated values. Prefixes (#/@) are optional.",
        )
        queries = parse_query_input(query_input)
        owner_lock = ""
        if mode == "Hashtag":
            owner_lock = st.text_input(
                "Restrict to owner (optional)",
                help="Only store posts from this creator when fetching hashtag data.",
            )
        max_posts = st.slider("Max posts per query", min_value=10, max_value=200, step=10, value=80)
        fetch_btn = st.button("Fetch & Update", type="primary")
        st.markdown("---")

    if fetch_btn:
        if not queries:
            st.sidebar.error("Provide at least one hashtag or username.")
        else:
            with st.spinner("Calling Apify and updating the analytics store..."):
                fetched = fetch_instagram_posts(mode, queries, max_posts, owner_lock.strip() or None)
                if fetched.empty:
                    st.sidebar.warning("No posts retrieved for the supplied parameters.")
                else:
                    processed = preprocess_posts(fetched)
                    persist_posts(processed)
                    st.session_state["last_fetch_info"] = {
                        "mode": mode,
                        "queries": queries,
                        "fetched": len(fetched),
                        "stored": processed["post_id"].nunique(),
                        "timestamp": datetime.utcnow(),
                    }
                    st.sidebar.success(f"Ingested {processed['post_id'].nunique()} posts into the trend store.")

    default_end = date.today()
    default_start = default_end - timedelta(days=29)
    full_df = load_posts()
    if full_df.empty:
        st.info("No data is stored yet. Use the sidebar to fetch posts and populate the dashboard.")
        return

    with st.sidebar:
        st.header("Analytics Filters")
        available_dates = full_df["posted_date"].dropna()
        min_date = available_dates.min() if not available_dates.empty else default_start
        max_date = available_dates.max() if not available_dates.empty else default_end
        use_date_filter = st.checkbox(
            "Filter by date range",
            value=False,
            help="Toggle on to narrow the dataset to a specific date window.",
        )
        current_range = None
        if use_date_filter and min_date is not None and max_date is not None:
            date_selection = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_selection, tuple) and len(date_selection) == 2:
                current_range = (date_selection[0], date_selection[1])
            elif isinstance(date_selection, date):
                current_range = (date_selection, date_selection)
        else:
            current_range = None

        comparison_mode = st.selectbox(
            "Compare to",
            options=["Previous period", "Same period last year", "None"],
            help="Period-over-period comparisons power KPI deltas and insights.",
        )
        hashtag_options = sorted({tag for tags in full_df["hashtag_list"] for tag in (tags or [])})
        selected_hashtags = st.multiselect(
            "Tracked hashtags",
            options=hashtag_options,
            help="Select multiple hashtags to combine insights or compare performance.",
        )
        hashtag_view = st.radio(
            "Hashtag view",
            options=["Combined", "Side-by-side"],
            horizontal=True,
        )
        content_options = sorted(full_df["content_category"].dropna().unique())
        selected_content = st.multiselect(
            "Content types",
            options=content_options,
            default=content_options,
        )
        countries = sorted(full_df["country"].dropna().unique())
        selected_countries = st.multiselect("Countries", options=countries)
        city_options = sorted(
            full_df[full_df["country"].isin(selected_countries)]["city"].dropna().unique()
            if selected_countries
            else full_df["city"].dropna().unique()
        )
        selected_cities = st.multiselect("Cities", options=city_options)
        engagement_focus = st.multiselect(
            "Highlight engagement types",
            options=list(ENGAGEMENT_COLUMNS.values()),
            default=list(ENGAGEMENT_COLUMNS.values()),
        )

    current_df = apply_filters(full_df, current_range, selected_hashtags, selected_content, selected_countries, selected_cities)

    if current_df.empty:
        st.warning("No posts match the current filters. Adjust filters or ingest additional data.")
        return

    if comparison_mode == "Previous period" and current_range:
        period_days = (current_range[1] - current_range[0]).days + 1
        previous_end = current_range[0] - timedelta(days=1)
        previous_start = previous_end - timedelta(days=period_days - 1)
        previous_range = (previous_start, previous_end)
    elif comparison_mode == "Same period last year" and current_range:
        delta_year = timedelta(days=365)
        previous_range = (current_range[0] - delta_year, current_range[1] - delta_year)
    else:
        previous_range = None

    previous_df = (
        apply_filters(full_df, previous_range, selected_hashtags, selected_content, selected_countries, selected_cities)
        if previous_range
        else pd.DataFrame()
    )

    current_daily = compute_daily_metrics(current_df)
    previous_daily = compute_daily_metrics(previous_df) if not previous_df.empty else pd.DataFrame()
    comparison_stats = compute_period_comparison(current_df, previous_df) if comparison_mode != "None" else {}
    hashtag_rollup = compute_hashtag_rollup(current_df)
    content_rollup = compute_content_rollup(current_df)
    country_rollup = compute_location_rollup(current_df, level="country")
    city_rollup = compute_location_rollup(current_df, level="city")

    chart_registry: Dict[str, go.Figure] = {}

    current_summary = compute_period_summary(current_df)
    st.markdown("### KPI Command Center")
    kpi_cols = st.columns(5)
    for idx, (label, help_text) in enumerate(METRIC_HELP.items()):
        column = kpi_cols[idx]
        metric_key = label.replace(" ", "_").lower()
        if metric_key == "reach":
            current_value = current_summary["reach"]
        elif metric_key == "impressions":
            current_value = current_summary["impressions"]
        elif metric_key == "engagement":
            current_value = current_summary["engagement"]
        elif metric_key == "total_posts":
            current_value = current_summary["total_posts"]
        elif metric_key == "engagement_rate":
            current_value = current_summary["engagement_rate"]
        else:
            current_value = 0.0
        delta_pct = None
        if comparison_stats and metric_key in comparison_stats:
            delta_pct = comparison_stats[metric_key]["delta_pct"]
        column.metric(
            label,
            format_number(current_value, is_percent=(metric_key == "engagement_rate"), digits=1),
            f"{delta_pct:+.1f}%" if delta_pct not in (None, np.nan) else None,
            help=help_text,
        )

    st.markdown("---")
    overview_tab, hashtag_tab, audience_tab, content_tab, insights_tab, reporting_tab = st.tabs(
        [
            "Overview",
            "Hashtag Details",
            "Audience Geography",
            "Content & Timing",
            "Insights & Alerts",
            "Reporting",
        ]
    )

    with overview_tab:
        st.subheader("Daily Trends")
        reach_fig = build_time_series_chart(current_daily, previous_daily, "reach", "Reach trend", "Reach")
        engagement_fig = build_time_series_chart(
            current_daily, previous_daily, "engagement", "Engagement trend", "Engagement"
        )
        rate_fig = build_time_series_chart(
            current_daily, previous_daily, "engagement_rate_pct", "Engagement rate trend", "Engagement rate (%)"
        )
        chart_registry["Reach trend"] = reach_fig
        chart_registry["Engagement trend"] = engagement_fig
        chart_registry["Engagement rate trend"] = rate_fig
        st.plotly_chart(reach_fig, use_container_width=True)
        st.plotly_chart(engagement_fig, use_container_width=True)
        st.plotly_chart(rate_fig, use_container_width=True)
        focus_cols = [col for col, label in ENGAGEMENT_COLUMNS.items() if label in engagement_focus and col in current_daily.columns]
        if focus_cols:
            breakdown_fig = go.Figure()
            for idx, col in enumerate(focus_cols):
                breakdown_fig.add_trace(
                    go.Scatter(
                        x=current_daily["posted_date"],
                        y=current_daily[col],
                        mode="lines",
                        name=ENGAGEMENT_COLUMNS[col],
                        stackgroup="one",
                    )
                )
            breakdown_fig.update_layout(
                title="Engagement mix by interaction type",
                height=320,
                margin=dict(l=20, r=20, t=60, b=30),
                yaxis_title="Interactions",
                xaxis_title="Date",
            )
            chart_registry["Engagement mix"] = breakdown_fig
            st.plotly_chart(breakdown_fig, use_container_width=True)
        st.caption("Charts update daily using persistent data; previous period overlays accelerate context.")

    with hashtag_tab:
        st.subheader("Hashtag performance")
        if hashtag_rollup.empty:
            st.info("No hashtags detected for the current filters.")
        else:
            metric_choice = st.selectbox(
                "Metric",
                options=["engagement", "reach", "impressions", "likes", "comments", "shares", "saves", "avg_engagement_rate"],
                format_func=lambda key: key.replace("_", " ").title(),
            )
            hashtag_chart = build_hashtag_comparison_chart(hashtag_rollup, metric_choice)
            chart_registry[f"Hashtag comparison ({metric_choice})"] = hashtag_chart
            st.plotly_chart(hashtag_chart, use_container_width=True)
            if hashtag_view == "Side-by-side" and selected_hashtags:
                comparison_df = hashtag_rollup[hashtag_rollup["hashtag"].isin([tag.lower() for tag in selected_hashtags])]
                st.dataframe(comparison_df, use_container_width=True)
            st.caption("Ranked view highlights which hashtags are compounding reach and engagement.")
        st.subheader("Top-performing hashtags")
        top_tags_df = hashtag_rollup.head(20)
        if not top_tags_df.empty:
            st.dataframe(top_tags_df, use_container_width=True, hide_index=True)

    with audience_tab:
        st.subheader("Country breakdown")
        if country_rollup.empty:
            st.info("No country data available for the current selection.")
        else:
            st.dataframe(country_rollup, use_container_width=True)
            geo_fig = build_geo_map(country_rollup)
            chart_registry["Geo engagement heatmap"] = geo_fig
            st.plotly_chart(geo_fig, use_container_width=True)
        st.subheader("City drill-down")
        if city_rollup.empty:
            st.info("Add posts with city-level data to unlock this view.")
        else:
            st.dataframe(city_rollup, use_container_width=True, hide_index=True)

    with content_tab:
        st.subheader("Content type performance")
        if content_rollup.empty:
            st.info("No content type breakdown available.")
        else:
            content_chart = px.bar(
                content_rollup,
                x="content_category",
                y="avg_engagement_rate",
                hover_data=["engagement", "reach"],
                color="avg_engagement_rate",
                color_continuous_scale="Blues",
                title="Average engagement rate by content type",
            )
            content_chart.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=40))
            chart_registry["Content type engagement"] = content_chart
            st.plotly_chart(content_chart, use_container_width=True)
            st.dataframe(content_rollup, use_container_width=True, hide_index=True)

        st.subheader("Posting cadence heatmap")
        heatmap_fig = build_heatmap(current_df, "engagement_total")
        if heatmap_fig.data:
            chart_registry["Engagement heatmap"] = heatmap_fig
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Need more data across days and hours to render the heatmap.")

    with insights_tab:
        st.subheader("Automated insights")
        insights = []
        recommendations = []
        anomaly_records = detect_anomalies(current_daily, "engagement")
        if anomaly_records:
            for anomaly in anomaly_records:
                direction = "spiked" if anomaly["z_score"] > 0 else "dipped"
                insights.append(
                    f"Engagement {direction} on {anomaly['date']} ({format_number(anomaly['value'])}; z-score {anomaly['z_score']:.1f})."
                )
        else:
            insights.append("Engagement levels are stable during the selected period.")

        comparison_reference_date = current_range[1] if current_range else current_df["posted_date"].max()
        if isinstance(comparison_reference_date, pd.Timestamp):
            comparison_reference_date = comparison_reference_date.date()
        if isinstance(comparison_reference_date, date):
            reference_label = comparison_reference_date.isoformat()
            today_top = top_post(current_df, comparison_reference_date)
            weekly_top = top_hashtag_of_week(current_df, comparison_reference_date)
        else:
            reference_label = "current selection"
            today_top = top_post(current_df, None)
            weekly_top = None
        if today_top is not None:
            insights.append(
                f"Top post {reference_label}: @{today_top['owner_username']} drove "
                f"{format_number(today_top['engagement_total'])} engagements ("
                f"{format_number(today_top['engagement_rate_pct'], is_percent=True)} rate)."
            )
        if weekly_top:
            insights.append(f"Top hashtag of the week: #{weekly_top}.")

        emerging = find_emerging_hashtags(current_df, previous_df, selected_hashtags) if selected_hashtags else pd.DataFrame()
        if emerging.empty:
            recommendations.append("Continue monitoring paired hashtags to spot emerging opportunities.")
        else:
            formatted = ", ".join(f"#{row['hashtag']} (+{row['growth']})" for _, row in emerging.iterrows())
            insights.append(f"Emerging paired hashtags: {formatted}.")

        if comparison_stats:
            engagement_delta = comparison_stats["engagement"]["delta_pct"]
            if engagement_delta is not None and not np.isnan(engagement_delta) and engagement_delta < -15:
                insights.append("Alert: Engagement dropped more than 15% vs the comparison period.")
                recommendations.append("Review creative mix and posting cadence for underperforming segments.")

        optimal_times = suggest_post_times(current_df)
        if not optimal_times.empty:
            rec_lines = [
                f"{int(row['posted_hour']):02d}:00 (avg rate {row['avg_engagement_rate']:.1f}%)"
                for _, row in optimal_times.iterrows()
            ]
            recommendations.append(f"Optimal posting windows: {', '.join(rec_lines)}.")

        st.write("**Highlights**")
        for item in insights:
            st.markdown(f"- {item}")
        st.write("**Recommended actions**")
        for item in recommendations or ["Maintain current strategy and monitor upcoming campaigns."]:
            st.markdown(f"- {item}")

        st.subheader("Top-performing posts")
        top_posts = current_df.sort_values("engagement_total", ascending=False).head(25)
        display_cols = [
            "posted_date",
            "owner_username",
            "content_category",
            "engagement_total",
            "reach_final",
            "engagement_rate_pct",
            "hashtag_list",
        ]
        st.dataframe(top_posts[display_cols], use_container_width=True, hide_index=True)

    with reporting_tab:
        st.subheader("Exports")
        csv_bytes = current_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data (CSV)",
            csv_bytes,
            file_name="instagram_filtered_posts.csv",
            mime="text/csv",
        )
        excel_bytes = export_dataframe_to_excel(
            {
                "Posts": current_df,
                "Daily metrics": current_daily,
                "Hashtags": hashtag_rollup,
                "Content": content_rollup,
            }
        )
        st.download_button(
            "Download analytics workbook (Excel)",
            excel_bytes,
            file_name="instagram_insights.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        chart_choice = st.selectbox(
            "Chart to export as PNG",
            options=list(chart_registry.keys()),
            help="Generates a high-resolution PNG using Plotly. Requires `kaleido`.",
        )
        if st.button("Download chart PNG"):
            selected_fig = chart_registry.get(chart_choice)
            png_bytes = export_figure_to_png(selected_fig)
            if png_bytes:
                st.download_button(
                    f"Download {chart_choice} PNG",
                    png_bytes,
                    file_name=f"{chart_choice.replace(' ', '_').lower()}.png",
                    mime="image/png",
                )
            else:
                st.warning("Unable to export chart. Install `kaleido` package for Plotly static images.")

        pdf_bytes = generate_pdf_summary(comparison_stats or {}, insights, recommendations)
        html_summary = generate_html_summary(comparison_stats or {}, insights, recommendations)
        if pdf_bytes:
            st.download_button(
                "Download summary (PDF)",
                pdf_bytes,
                file_name="instagram_summary.pdf",
                mime="application/pdf",
            )
        st.download_button(
            "Download summary (HTML)",
            html_summary.encode("utf-8"),
            file_name="instagram_summary.html",
            mime="text/html",
        )

        st.subheader("Scheduled email reports")
        schedule_df = load_report_schedules()
        if not schedule_df.empty:
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)
        with st.form("schedule_report_form"):
            st.markdown("Configure automated HTML/PDF deliveries. Requires SMTP credentials via environment.")
            schedule_name = st.text_input("Schedule name")
            recipient_input = st.text_input("Recipients", placeholder="email@brand.com, marketing@agency.com")
            frequency = st.selectbox("Frequency", options=["Daily", "Weekly", "Monthly"])
            formats = st.multiselect("Include formats", options=["HTML", "PDF"], default=["HTML", "PDF"])
            submitted = st.form_submit_button("Add schedule")
            if submitted:
                recipients = [email.strip() for email in recipient_input.split(",") if email.strip()]
                if not schedule_name or not recipients:
                    st.warning("Provide a schedule name and at least one recipient.")
                else:
                    if current_range:
                        date_filter_snapshot = [current_range[0].isoformat(), current_range[1].isoformat()]
                    else:
                        date_filter_snapshot = None
                    filter_snapshot = {
                        "date_range": date_filter_snapshot,
                        "hashtags": selected_hashtags,
                        "content_types": selected_content,
                        "countries": selected_countries,
                        "cities": selected_cities,
                    }
                    save_report_schedule(schedule_name, recipients, frequency, formats, filter_snapshot)
                    st.success("Schedule stored. A background process can read from `report_schedules` to dispatch emails.")

        if schedule_df.empty:
            st.info("No scheduled reports yet. Add one above once SMTP settings are configured.")
        else:
            st.markdown("Trigger a schedule manually (useful for testing SMTP).")
            schedule_ids = schedule_df["id"].tolist()
            selected_schedule = st.selectbox("Schedule", options=schedule_ids, format_func=lambda sid: f"ID {sid}")
            if st.button("Send selected schedule now"):
                schedule_record = schedule_df[schedule_df["id"] == selected_schedule].iloc[0]
                filters = json.loads(schedule_record["filters"])
                recipients = json.loads(schedule_record["recipients"])
                formats = json.loads(schedule_record["include_formats"])
                attachments: Dict[str, bytes] = {}
                if "HTML" in formats:
                    attachments["instagram_summary.html"] = html_summary.encode("utf-8")
                if "PDF" in formats and pdf_bytes:
                    attachments["instagram_summary.pdf"] = pdf_bytes
                success, error = send_email_report(
                    recipients,
                    f"Instagram report: {schedule_record['name']}",
                    html_summary,
                    attachments,
                )
                if success:
                    update_schedule_last_run(int(schedule_record["id"]))
                    st.success("Email dispatched successfully.")
                else:
                    st.error(f"Failed to send email: {error}")


if __name__ == "__main__":
    main()
