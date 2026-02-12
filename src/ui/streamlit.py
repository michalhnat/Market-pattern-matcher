import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Market Pattern Matcher")
st.title("Market Pattern Matcher")

API_ADDRESS = os.getenv("API_ADDRESS", "http://localhost:8000")


def search_request(
    ticker: str, window_size: int, date: datetime.date, top_k: int
) -> list[dict]:
    url = f"{API_ADDRESS}/search"
    params = {
        "ticker": ticker,
        "window_size": window_size,
        "start_date": f"{date.isoformat()}T00:00:00",
        "top_k": top_k,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_raw_window_with_margins(
    window_idx: int,
    metadata_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    left_margin_days: int = 0,
    right_margin_days: int = 0,
) -> tuple[pd.DataFrame, int, int]:
    meta = metadata_df.iloc[window_idx]
    start_date = pd.to_datetime(meta["start_date"])
    end_date = pd.to_datetime(meta["end_date"])

    try:
        start_idx = raw_df.index.get_loc(start_date)
        end_idx = raw_df.index.get_loc(end_date)
    except KeyError:
        start_idx = raw_df.index.searchsorted(start_date)
        end_idx = raw_df.index.searchsorted(end_date)

    margin_start_idx = max(0, start_idx - left_margin_days)
    margin_end_idx = min(len(raw_df) - 1, end_idx + right_margin_days)

    window_data = raw_df.iloc[margin_start_idx : margin_end_idx + 1]

    main_start_idx = start_idx - margin_start_idx
    main_end_idx = main_start_idx + (end_idx - start_idx) + 1

    return window_data, main_start_idx, main_end_idx


def find_query_index(start_date: datetime.date, metadata_df: pd.DataFrame) -> int:
    target_date = pd.to_datetime(start_date)
    dates = pd.to_datetime(metadata_df["start_date"])
    return int((dates - target_date).abs().idxmin())


def create_chart(
    query_idx: int,
    results: list[dict],
    metadata_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    left_margin: int,
    right_margin: int,
) -> None:
    if query_idx + 1 >= len(metadata_df):
        st.error("No future data available for this date")
        st.stop()

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    next_idx = query_idx + 1
    next_window, main_start, main_end = get_raw_window_with_margins(
        next_idx, metadata_df, raw_df, left_margin, right_margin
    )
    next_info = metadata_df.iloc[next_idx]

    st.subheader(f"Query Next: {next_info['start_date']}")
    fig_query = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume"),
    )

    x_vals = list(range(len(next_window)))

    fig_query.add_trace(
        go.Candlestick(
            x=x_vals,
            open=next_window["Open"],
            high=next_window["High"],
            low=next_window["Low"],
            close=next_window["Close"],
            showlegend=False,
            opacity=0.4,
            name="Context",
        ),
        row=1,
        col=1,
    )

    fig_query.add_trace(
        go.Candlestick(
            x=x_vals[main_start:main_end],
            open=next_window["Open"].iloc[main_start:main_end],
            high=next_window["High"].iloc[main_start:main_end],
            low=next_window["Low"].iloc[main_start:main_end],
            close=next_window["Close"].iloc[main_start:main_end],
            showlegend=False,
            name="Main",
        ),
        row=1,
        col=1,
    )

    fig_query.add_trace(
        go.Bar(
            x=x_vals,
            y=next_window["Volume"],
            marker_color=colors[0],
            opacity=0.3,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig_query.add_trace(
        go.Bar(
            x=x_vals[main_start:main_end],
            y=next_window["Volume"].iloc[main_start:main_end],
            marker_color=colors[0],
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig_query.add_vrect(
        x0=main_start,
        x1=main_end,
        fillcolor="lightgray",
        opacity=0.2,
        layer="below",
        line_width=0,
        row=1,
        col=1,
    )

    fig_query.update_layout(height=400, xaxis_rangeslider_visible=False)
    fig_query.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_query.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig_query, use_container_width=True)

    st.subheader("Similar Historical Patterns")
    for i, res in enumerate(results):
        match_next_idx = res["window_index"] + 1
        if match_next_idx >= len(metadata_df):
            continue

        match_window, match_start, match_end = get_raw_window_with_margins(
            match_next_idx, metadata_df, raw_df, left_margin, right_margin
        )
        match_info = metadata_df.iloc[match_next_idx]

        st.write(
            f"**Rank {res['rank']}**: {match_info['start_date']} "
            f"(distance: {res['distance']:.2f})"
        )

        fig_match = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume"),
        )

        x_vals = list(range(len(match_window)))

        fig_match.add_trace(
            go.Candlestick(
                x=x_vals,
                open=match_window["Open"],
                high=match_window["High"],
                low=match_window["Low"],
                close=match_window["Close"],
                showlegend=False,
                opacity=0.4,
            ),
            row=1,
            col=1,
        )

        fig_match.add_trace(
            go.Candlestick(
                x=x_vals[match_start:match_end],
                open=match_window["Open"].iloc[match_start:match_end],
                high=match_window["High"].iloc[match_start:match_end],
                low=match_window["Low"].iloc[match_start:match_end],
                close=match_window["Close"].iloc[match_start:match_end],
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig_match.add_trace(
            go.Bar(
                x=x_vals,
                y=match_window["Volume"],
                marker_color=colors[(i + 1) % len(colors)],
                opacity=0.3,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig_match.add_trace(
            go.Bar(
                x=x_vals[match_start:match_end],
                y=match_window["Volume"].iloc[match_start:match_end],
                marker_color=colors[(i + 1) % len(colors)],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig_match.add_vrect(
            x0=match_start,
            x1=match_end,
            fillcolor="lightgray",
            opacity=0.2,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )

        fig_match.update_layout(height=350, xaxis_rangeslider_visible=False)
        fig_match.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_match.update_yaxes(title_text="Volume", row=2, col=1)
        st.plotly_chart(fig_match, use_container_width=True)


with st.sidebar:
    start_date = st.date_input("Start date", value=datetime.date(2020, 1, 1))
    st.divider()
    num_results = st.slider("Number of similar patterns", 1, 10, 5)
    st.divider()
    st.subheader("Context Margins")
    left_margin = st.slider("Left margin (trading days)", 0, 30, 5)
    right_margin = st.slider("Right margin (trading days)", 0, 30, 5)

if st.sidebar.button("Search", type="primary", use_container_width=True):
    with st.spinner("Searching for similar patterns..."):
        try:
            results = search_request(
                ticker="SPY",
                window_size=30,
                date=start_date,
                top_k=num_results,
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    if not results:
        st.warning("No similar patterns found.")
        st.stop()

    metadata_path = results[0]["metadata_path"]
    raw_data_path = results[0]["raw_data_path"]

    try:
        metadata_df = pd.read_csv(metadata_path)
        raw_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    query_idx = find_query_index(start_date, metadata_df)
    create_chart(query_idx, results, metadata_df, raw_df, left_margin, right_margin)
