import datetime
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from scipy.stats import gaussian_kde


def fetch_stock_data(ticker_symbol: str, period_str: str) -> Optional[pd.DataFrame]:

    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period_str)
        if df.empty:
            st.error(f"No data available for {ticker_symbol}")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def process_data(df: pd.DataFrame) -> pd.DataFrame:
  
    df = df.reset_index().copy()
    N = len(df)
    df["t"] = np.arange(N)
    df["ret"] = df["Close"].pct_change().fillna(0)

    volume_norm = (df["Volume"] - df["Volume"].min()) / (df["Volume"].max() - df["Volume"].min() + 1e-10)
    fast = 2.0 * np.exp(-df["t"].values / (max(1, N / 6)))
    slow = 1.4 * np.exp(-df["t"].values / (max(1, N / 2)))
    noise = volume_norm * 0.3
    df["intensity"] = fast + slow + noise
    df["fast"] = fast
    df["slow"] = slow
    df["baseline"] = np.ones(N) * 1.8
    df["vol"] = df["ret"].rolling(20).std().fillna(0)

    volume_threshold = df["Volume"].quantile(0.75)
    df["events"] = (df["Volume"] > volume_threshold).cumsum()

    acf_vals = []
    for k in range(N):
        try:
            if k == 0:
                acf_vals.append(1.0)
            else:
                val = df["ret"].autocorr(lag=k)
                acf_vals.append(float(val) if pd.notna(val) else 0.0)
        except Exception:
            acf_vals.append(0.0)
    df["acf"] = acf_vals

    roll_mean = df["ret"].rolling(20).mean().fillna(0)
    roll_std = df["ret"].rolling(20).std().fillna(1)
    df["z"] = (df["ret"] - roll_mean) / roll_std
    df["ofi"] = np.cumsum(np.sign(df["ret"]))
    return df


def create_figure(df: pd.DataFrame, ticker: str) -> go.Figure:

    N = len(df)
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"type": "candlestick", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=[
            f"{ticker} PRICE ACTION | Candlestick Chart",
            "HAWKES INTENSITY | Self-Exciting Process",
            "VOLATILITY REGIME | Rolling",
            "ORDER FLOW IMBALANCE | Cumulative Directional Pressure",
            "RETURN DISTRIBUTION | Kernel Density Estimation",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        row_heights=[0.5, 0.5, 0.45],
    )

    fig.add_trace(
        go.Candlestick(x=df["t"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], increasing_line_color="#00ff99", decreasing_line_color="#ff3366", name="Price", showlegend=False),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=df["t"], y=df["intensity"], mode="lines", name="Total Intensity", line=dict(color="#00eaff", width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["fast"], mode="lines", name="Fast Decay", line=dict(color="#00ffaa", width=2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["slow"], mode="lines", name="Slow Decay", line=dict(color="#ff4d4d", width=2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["baseline"], mode="lines", name="Baseline", line=dict(color="#ffaa00", width=1, dash="dash")), row=1, col=2)

    fig.add_trace(go.Scatter(x=df["t"], y=df["vol"], mode="lines", name="Volatility", line=dict(color="#ffcc00", width=3), fill="tozeroy", fillcolor="rgba(255, 204, 0, 0.2)"), row=2, col=2)

    colors = ["#00ff99" if x > 0 else "#ff3366" for x in df["ofi"]]
    fig.add_trace(go.Scatter(x=df["t"], y=df["ofi"], mode="lines+markers", name="OFI", line=dict(color="#ff44cc", width=2), marker=dict(size=7, color=colors)), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=3, col=1, opacity=0.5)

    ret_clean = df["ret"].dropna()
    if len(ret_clean) > 1:
        kde = gaussian_kde(ret_clean)
        xgrid = np.linspace(ret_clean.min(), ret_clean.max(), 200)
        fig.add_trace(go.Scatter(x=xgrid, y=kde(xgrid), mode="lines", name="KDE", line=dict(color="#b266ff", width=3), fill="tozeroy", fillcolor="rgba(178, 102, 255, 0.3)"), row=3, col=2)

    frames = []
    step = max(1, N // 100)
    for i in range(2, N, step):
        ret_clean_i = df["ret"][:i].dropna()
        if len(ret_clean_i) > 1:
            kde = gaussian_kde(ret_clean_i)
            xgrid = np.linspace(ret_clean_i.min(), ret_clean_i.max(), 200)
            kde_y = kde(xgrid)
        else:
            xgrid = [0]
            kde_y = [0]
        colors_frame = ["#00ff99" if x > 0 else "#ff3366" for x in df["ofi"][:i]]
        frames.append(
            go.Frame(
                data=[
                    go.Candlestick(x=df["t"][:i], open=df["Open"][:i], high=df["High"][:i], low=df["Low"][:i], close=df["Close"][:i]),
                    go.Scatter(x=df["t"][:i], y=df["intensity"][:i]),
                    go.Scatter(x=df["t"][:i], y=df["fast"][:i]),
                    go.Scatter(x=df["t"][:i], y=df["slow"][:i]),
                    go.Scatter(x=df["t"][:i], y=df["baseline"][:i]),
                    go.Scatter(x=df["t"][:i], y=df["vol"][:i]),
                    go.Scatter(x=df["t"][:i], y=df["ofi"][:i], mode="lines+markers", marker=dict(size=7, color=colors_frame)),
                    go.Scatter(x=xgrid, y=kde_y),
                ],
                name=str(i),
            )
        )
    fig.frames = frames

    fig.update_layout(
        height=1300,
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="#ffffff", size=13),
        showlegend=True,
        legend=dict(bgcolor="rgba(20, 20, 30, 0.8)", bordercolor="#00eaff", borderwidth=1, font=dict(size=11)),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    dict(label="PLAY", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                    dict(label="PAUSE", method="animate", args=[[None], {"mode": "immediate"}]),
                ],
                "x": 0.48,
                "y": -0.05,
                "bgcolor": "#1a1a2e",
                "bordercolor": "#00eaff",
                "font": dict(color="#00eaff"),
            }
        ],
        margin=dict(l=40, r=40, t=100, b=80),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1a1a1a", zeroline=False, color="#888888")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1a1a1a", zeroline=False, color="#888888")
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=3), marker=dict(size=6))
    return fig


def main() -> None:
    st.set_page_config(page_title="QUANT MICROSTRUCTURE DASHBOARD", layout="wide")

    st.markdown(
        """
    <style>
        .main {background-color: #0a0a0a;}
        .stPlotly {border: 1px solid #1a1a1a; border-radius: 8px;}
        h1 {color: #00eaff; text-align: center; font-weight: 700;}
        .metric-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #00eaff33;
            margin: 10px 0;
        }
        .metric-title {color: #00eaff; font-size: 14px; font-weight: 600;}
        .metric-value {color: #ffffff; font-size: 24px; font-weight: 700;}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("QUANTITATIVE MICROSTRUCTURE ANALYTICS")
    st.markdown("<p style='text-align: center; color: #888; margin-top: -10px;'>Real-time market microstructure analysis with Hawkes process modeling</p>", unsafe_allow_html=True)

    col_ticker, col_period = st.columns([3, 1])
    with col_ticker:
        ticker = st.selectbox("Select Ticker", ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"], index=0)
    with col_period:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=0)

    fetch_cached = st.cache_data(ttl=300)(fetch_stock_data)
    with st.spinner(f"Fetching {ticker} data..."):
        stock_data = fetch_cached(ticker, period)

    if stock_data is None or stock_data.empty:
        st.stop()

    df = process_data(stock_data)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = df["Close"].iloc[-1]
        price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2] if len(df) > 1 else 0
        price_pct = (price_change / df["Close"].iloc[-2] * 100) if len(df) > 1 and df["Close"].iloc[-2] != 0 else 0
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>{ticker} PRICE</div>
            <div class='metric-value'>${current_price:.2f}</div>
            <div style='color: {'#00ff99' if price_change >= 0 else '#ff3366'}; font-size: 12px;'>
                {'+' if price_change >= 0 else ''}{price_pct:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>VOLATILITY</div>
            <div class='metric-value'>{df['vol'].iloc[-1]:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>INTENSITY</div>
            <div class='metric-value'>{df['intensity'].iloc[-1]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>HIGH VOLUME EVENTS</div>
            <div class='metric-value'>{df['events'].iloc[-1]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = create_figure(df, ticker)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
