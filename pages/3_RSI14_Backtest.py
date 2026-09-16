# ============================================================
# RSI 14 CROSSOVER BACKTESTING STRATEGY
# ============================================================
# BUY  : RSI(14) crosses ABOVE 50
# SELL : RSI(14) crosses BELOW 40
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STREAMLIT USER SETTINGS
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="RSI 14 Backtesting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 RSI 14 Crossover Backtesting Strategy")
st.caption("BUY: RSI(14) crosses above 50 | SELL: RSI(14) crosses below 40")

# User inputs
col1, col2, col3 = st.columns(3)

with col1:
    NAME = st.text_input(
        "Stock Name / NSE Symbol",
        value="RELIANCE",
        placeholder="e.g. RELIANCE, TCS, INFY"
    ).strip().upper()

with col2:
    START_DATE_INPUT = st.date_input(
        "START_DATE",
        value=pd.Timestamp("2022-01-01").date()
    )

with col3:
    END_DATE_INPUT = st.date_input(
        "END_DATE",
        value=pd.Timestamp("2026-09-08").date()
    )

SYMBOL = NAME + ".NS"

INITIAL_CAPITAL = 100000     # ₹1,00,000
BROKERAGE = 0.001            # 0.10% per side
SLIPPAGE = 0.0005            # 0.05% per side

RSI_PERIOD = 14
BUY_LEVEL = 50
SELL_LEVEL = 40

# yfinance END date is exclusive, so add one day to include the
# date selected by the user.
START_DATE = START_DATE_INPUT.strftime("%Y-%m-%d")
END_DATE = (
    pd.Timestamp(END_DATE_INPUT) + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

if not NAME:
    st.warning("Please enter an NSE stock symbol.")
    st.stop()

if START_DATE_INPUT >= END_DATE_INPUT:
    st.error("END_DATE must be later than START_DATE.")
    st.stop()

RUN_BACKTEST = st.button(
    "🚀 RUN BACKTEST",
    type="primary",
    use_container_width=True
)

if not RUN_BACKTEST:
    st.info("Enter Stock Name, START_DATE and END_DATE, then click RUN BACKTEST.")
    st.stop()


# ============================================================
# RSI CALCULATION
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# DOWNLOAD DATA
# ============================================================

with st.spinner(f"Downloading {SYMBOL} data..."):
    df = yf.download(
        SYMBOL,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False
    )

if df.empty:
    raise ValueError("No data downloaded. Check SYMBOL or dates.")

# Handle MultiIndex returned by newer yfinance versions
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

df.dropna(inplace=True)


# ============================================================
# CALCULATE RSI
# ============================================================

df["RSI"] = calculate_rsi(
    df["Close"],
    RSI_PERIOD
)


# ============================================================
# CROSSOVER CONDITIONS
# ============================================================

# BUY:
# Previous RSI <= 50
# Current RSI > 50

df["BUY_SIGNAL"] = (
    (df["RSI"].shift(1) <= BUY_LEVEL) &
    (df["RSI"] > BUY_LEVEL)
)


# SELL:
# Previous RSI >= 40
# Current RSI < 40

df["SELL_SIGNAL"] = (
    (df["RSI"].shift(1) >= SELL_LEVEL) &
    (df["RSI"] < SELL_LEVEL)
)


# ============================================================
# BACKTEST
# ============================================================

capital = INITIAL_CAPITAL

position = 0
shares = 0
entry_price = 0
entry_date = None

trades = []

equity_curve = []


for i in range(1, len(df)):

    date = df.index[i]

    open_price = float(df["Open"].iloc[i])
    close_price = float(df["Close"].iloc[i])

    buy_signal = bool(df["BUY_SIGNAL"].iloc[i])
    sell_signal = bool(df["SELL_SIGNAL"].iloc[i])


    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if position == 0 and buy_signal:

        # Buy at next candle Open
        buy_price = open_price * (1 + SLIPPAGE)

        shares = int(
            capital /
            (buy_price * (1 + BROKERAGE))
        )

        if shares > 0:

            investment = shares * buy_price

            brokerage = investment * BROKERAGE

            total_cost = investment + brokerage

            capital -= total_cost

            position = 1

            entry_price = buy_price
            entry_date = date

            print(
                f"BUY  | {date.date()} | "
                f"₹{buy_price:.2f} | "
                f"Shares: {shares}"
            )


    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    elif position == 1 and sell_signal:

        # Sell at next candle Open
        sell_price = open_price * (1 - SLIPPAGE)

        sale_value = shares * sell_price

        brokerage = sale_value * BROKERAGE

        net_sale = sale_value - brokerage

        capital += net_sale

        pnl = (
            (sell_price - entry_price)
            * shares
        )

        pnl -= (
            (entry_price * shares * BROKERAGE)
            + brokerage
        )

        pnl_percent = (
            pnl /
            (entry_price * shares)
        ) * 100

        trades.append({

            "Entry Date": entry_date,
            "Exit Date": date,

            "Entry Price": round(entry_price, 2),
            "Exit Price": round(sell_price, 2),

            "Shares": shares,

            "P&L": round(pnl, 2),
            "P&L %": round(pnl_percent, 2),

            "Result":
                "WIN" if pnl > 0 else "LOSS"
        })

        print(
            f"SELL | {date.date()} | "
            f"₹{sell_price:.2f} | "
            f"P&L: ₹{pnl:.2f}"
        )

        position = 0
        shares = 0
        entry_price = 0
        entry_date = None


    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    if position == 1:

        current_value = (
            capital +
            shares * close_price
        )

    else:

        current_value = capital

    equity_curve.append(
        [date, current_value]
    )


# ============================================================
# CLOSE OPEN POSITION AT LAST PRICE
# ============================================================

if position == 1:

    final_price = float(df["Close"].iloc[-1])

    sale_value = shares * final_price

    brokerage = sale_value * BROKERAGE

    net_sale = sale_value - brokerage

    capital += net_sale

    pnl = (
        (final_price - entry_price)
        * shares
    )

    pnl -= (
        (entry_price * shares * BROKERAGE)
        + brokerage
    )

    pnl_percent = (
        pnl /
        (entry_price * shares)
    ) * 100

    trades.append({

        "Entry Date": entry_date,
        "Exit Date": df.index[-1],

        "Entry Price": round(entry_price, 2),
        "Exit Price": round(final_price, 2),

        "Shares": shares,

        "P&L": round(pnl, 2),
        "P&L %": round(pnl_percent, 2),

        "Result":
            "WIN" if pnl > 0 else "LOSS"
    })


# ============================================================
# TRADE DATAFRAME
# ============================================================

trades_df = pd.DataFrame(trades)


# ============================================================
# PERFORMANCE CALCULATIONS
# ============================================================

final_capital = capital

total_return = (
    (final_capital - INITIAL_CAPITAL)
    / INITIAL_CAPITAL
) * 100


if len(trades_df) > 0:

    total_trades = len(trades_df)

    winning_trades = (
        trades_df["P&L"] > 0
    ).sum()

    losing_trades = (
        trades_df["P&L"] <= 0
    ).sum()

    win_rate = (
        winning_trades /
        total_trades
    ) * 100

    gross_profit = trades_df.loc[
        trades_df["P&L"] > 0,
        "P&L"
    ].sum()

    gross_loss = abs(
        trades_df.loc[
            trades_df["P&L"] < 0,
            "P&L"
        ].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit /
            gross_loss
        )
    else:
        profit_factor = np.inf

    average_trade = trades_df["P&L"].mean()

else:

    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    win_rate = 0
    profit_factor = 0
    average_trade = 0


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

equity_df = pd.DataFrame(
    equity_curve,
    columns=["Date", "Equity"]
)

equity_df.set_index(
    "Date",
    inplace=True
)

equity_df["Peak"] = (
    equity_df["Equity"]
    .cummax()
)

equity_df["Drawdown"] = (
    equity_df["Equity"]
    - equity_df["Peak"]
)

equity_df["Drawdown %"] = (
    equity_df["Drawdown"]
    / equity_df["Peak"]
) * 100

max_drawdown = (
    equity_df["Drawdown %"].min()
)


# ============================================================
# BUY & HOLD COMPARISON
# ============================================================

first_close = float(df["Close"].iloc[0])
last_close = float(df["Close"].iloc[-1])

buy_hold_return = (
    (last_close - first_close)
    / first_close
) * 100


# ============================================================
# RESULTS
# ============================================================

st.success(f"Backtest completed: {SYMBOL} | {START_DATE_INPUT} → {END_DATE_INPUT}")

st.subheader("📊 Backtest Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Initial Capital", f"₹{INITIAL_CAPITAL:,.0f}")
m2.metric("Final Capital", f"₹{final_capital:,.0f}")
m3.metric("Net Profit", f"₹{final_capital - INITIAL_CAPITAL:,.0f}")
m4.metric("Total Return", f"{total_return:.2f}%")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Total Trades", f"{total_trades}")
m6.metric("Winning Trades", f"{winning_trades}")
m7.metric("Losing Trades", f"{losing_trades}")
m8.metric("Win Rate", f"{win_rate:.2f}%")

m9, m10, m11 = st.columns(3)
m9.metric(
    "Profit Factor",
    "∞" if np.isinf(profit_factor) else f"{profit_factor:.2f}"
)
m10.metric("Average Trade", f"₹{average_trade:,.2f}")
m11.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.metric("Strategy Return", f"{total_return:.2f}%")
with c2:
    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")

# ============================================================
# TRADE LOG
# ============================================================

if len(trades_df) > 0:
    st.subheader("📋 Trade Log")
    st.dataframe(
        trades_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = trades_df.to_csv(index=False).encode("utf-8")
    filename = SYMBOL.replace(".", "_") + "_RSI14_Backtest.csv"

    st.download_button(
        "⬇️ Download Trade Log CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )
else:
    st.info("No completed trades were generated during the selected period.")

# ============================================================
# RSI CHART
# ============================================================

st.subheader("📈 RSI 14 Chart")

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["RSI"],
    label="RSI 14"
)

plt.axhline(
    BUY_LEVEL,
    linestyle="--",
    label="BUY Level 50"
)

plt.axhline(
    SELL_LEVEL,
    linestyle="--",
    label="SELL Level 40"
)

# Mark BUY signals
buy_points = df[
    df["BUY_SIGNAL"]
]

plt.scatter(
    buy_points.index,
    buy_points["RSI"],
    marker="^",
    s=80,
    color="green",
    label="BUY"
)

# Mark SELL signals
sell_points = df[
    df["SELL_SIGNAL"]
]

plt.scatter(
    sell_points.index,
    sell_points["RSI"],
    marker="v",
    s=80,
    color="red",
    label="SELL"
)

plt.title(
    f"{SYMBOL} - RSI 14 Strategy"
)

plt.ylabel("RSI")

plt.legend()

plt.grid(True)

st.pyplot(plt.gcf())


# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("💰 Equity Curve")

plt.figure(figsize=(14, 6))

plt.plot(
    equity_df.index,
    equity_df["Equity"],
    label="Strategy Equity"
)

plt.axhline(
    INITIAL_CAPITAL,
    linestyle="--",
    label="Initial Capital"
)

plt.title(
    f"{SYMBOL} - RSI Strategy Equity Curve"
)

plt.ylabel("Portfolio Value (₹)")

plt.legend()

plt.grid(True)

st.pyplot(plt.gcf())
