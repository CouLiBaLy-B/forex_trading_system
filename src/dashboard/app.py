import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Forex Trading System", layout="wide", page_icon="📊")
st.title("Forex Trading Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Strategies", "Orders", "Portfolio", "Backtest"])

# ---- Overview ----
with tab1:
    st.header("System Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Equity", "$100,000")
    c2.metric("Daily P&L", "$1,234 (+1.23%)")
    c3.metric("Open Positions", "3")
    c4.metric("Active Strategies", "5")

    st.subheader("Equity Curve")
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), periods=90)
    equity = [100000 + i * 100 + (i % 7 - 3) * 50 for i in range(90)]
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Equity'))
    fig_eq.update_layout(height=300, xaxis_title="Date", yaxis_title="Equity ($)")
    st.plotly_chart(fig_eq, use_container_width=True)

# ---- Strategies ----
with tab2:
    st.header("Active Strategies")
    strategies = [
        ("MA Crossover", "BUY", "EUR/USD", "0.85", "10"),
        ("Mean Reversion", "SELL", "GBP/USD", "0.72", "5"),
        ("RSI", "HOLD", "USD/JPY", "0.30", "2"),
    ]
    st.table(pd.DataFrame(strategies, columns=["Strategy", "Signal", "Pair", "Confidence", "P&L"]))

# ---- Orders ----
with tab3:
    st.header("Orders")
    st.table(pd.DataFrame([], columns=["ID", "Symbol", "Type", "Side", "Qty", "Price", "Status", "Time"]))

# ---- Portfolio ----
with tab4:
    st.header("Portfolio")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positions")
        positions = pd.DataFrame([
            {"Symbol": "EUR/USD", "Side": "Long", "Qty": "100K", "Avg": "1.0850", "Current": "1.0900", "P&L": "+$500"},
            {"Symbol": "GBP/USD", "Side": "Short", "Qty": "50K", "Avg": "1.2700", "Current": "1.2650", "P&L": "+$250"},
        ])
        st.table(positions)
    with col2:
        st.subheader("Performance")
        perf = pd.DataFrame({"Metric": ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Profit Factor"],
                             "Value": [1.5, 2.1, "-5.2%", "62%", "2.3"]})
        st.table(perf)

# ---- Backtest ----
with tab5:
    st.header("Backtest Results")
    st.write("Select a strategy to backtest")
    strategy = st.selectbox("Strategy", ["MA Crossover", "Mean Reversion", "RSI", "MACD", "Bollinger Bands"])
    symbol = st.text_input("Symbol", "EUR/USD")
    col_a, col_b, col_c = st.columns(3)
    start = col_a.date_input("Start", datetime.now() - timedelta(days=365))
    end = col_b.date_input("End", datetime.now())
    initial = col_c.number_input("Initial Capital", value=100000)
    if st.button("Run Backtest"):
        st.success(f"Backtest completed for {strategy} on {symbol}")
        st.metric("Total Return", "+12.3%")
        st.metric("Sharpe Ratio", "1.5")
        st.metric("Max Drawdown", "-5.2%")
        st.metric("Win Rate", "62%")
