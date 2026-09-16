# ============================================================
# 📊 MINERVINI / WEINSTEIN STAGE ANALYZER
# ============================================================
# 👤 Developed by SUJOY ROY
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Minervini Stage Analyzer",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Minervini / Weinstein Stage Analyzer")

st.markdown(
    """
    Analyze an NSE stock using weekly price data,
    30-week moving average and moving-average slope.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")

symbol = st.sidebar.text_input(
    "NSE Stock Symbol",
    value="CCL"
).strip().upper()

period = st.sidebar.selectbox(
    "Historical Period",
    ["2y", "3y", "5y", "10y"],
    index=1
)

analyze = st.sidebar.button(
    "🔍 ANALYZE STOCK",
    type="primary",
    use_container_width=True
)


# ============================================================
# ROBUST CLOSE EXTRACTION
# ============================================================

def get_close(data):

    if data is None:
        return None

    if data.empty:
        return None

    try:

        # ====================================================
        # CASE 1 — MultiIndex columns
        # ====================================================

        if isinstance(data.columns, pd.MultiIndex):

            # Look for Close at ANY level
            close_positions = []

            for i, col in enumerate(data.columns):

                values = [
                    str(x).strip().lower()
                    for x in col
                ]

                if "close" in values:
                    close_positions.append(i)

            if not close_positions:
                return None

            # Select first Close column by position
            close_data = data.iloc[
                :,
                close_positions[0]
            ]

        # ====================================================
        # CASE 2 — Normal columns
        # ====================================================

        else:

            close_data = data["Close"]

        # ====================================================
        # CASE 3 — DataFrame
        # ====================================================

        if isinstance(
            close_data,
            pd.DataFrame
        ):

            close_data = close_data.iloc[:, 0]

        # ====================================================
        # CASE 4 — Series
        # ====================================================

        if isinstance(
            close_data,
            pd.Series
        ):

            values = close_data.to_numpy(
                dtype="float64"
            ).reshape(-1)

        # ====================================================
        # CASE 5 — numpy / other array
        # ====================================================

        else:

            values = np.asarray(
                close_data,
                dtype="float64"
            ).reshape(-1)

        # ====================================================
        # Make sure index length matches
        # ====================================================

        if len(values) != len(data.index):

            return None

        # ====================================================
        # Create guaranteed 1-D Series
        # ====================================================

        result = pd.Series(
            values,
            index=data.index,
            name="Close",
            dtype="float64"
        )

        result = result.replace(
            [np.inf, -np.inf],
            np.nan
        )

        result = result.dropna()

        return result

    except Exception as e:

        st.error(
            f"Unable to extract Close price: {e}"
        )

        return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    close = get_close(data)

    if close is None or close.empty:

        return pd.DataFrame()

    # ========================================================
    # Create a brand-new DataFrame
    # ========================================================

    stock = pd.DataFrame(
        {
            "Close": close
        }
    )

    # ========================================================
    # Date index
    # ========================================================

    stock.index = pd.to_datetime(
        stock.index
    )

    stock = stock.sort_index()

    # ========================================================
    # 30 WEEK MOVING AVERAGE
    # ========================================================

    stock["MA30"] = (
        stock["Close"]
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # ========================================================
    # MA SLOPE
    # ========================================================

    stock["MA_Slope"] = (
        stock["MA30"]
        .diff()
    )

    # ========================================================
    # MA SLOPE %
    # ========================================================

    stock["MA_Slope_%"] = (
        stock["MA30"]
        .pct_change()
        * 100
    )

    # ========================================================
    # PRICE VS MA
    # ========================================================

    stock["Price_vs_MA_%"] = (
        (
            stock["Close"]
            - stock["MA30"]
        )
        /
        stock["Close"]
        * 100
    )

    return stock


# ============================================================
# STAGE CLASSIFICATION
# ============================================================

def classify_stage(row):

    price = row["Close"]
    ma = row["MA30"]
    slope = row["MA_Slope"]

    if pd.isna(ma) or pd.isna(slope):

        return "Indeterminate"

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    if (
        abs(slope) < 0.1
        and abs(price - ma) < 0.05 * price
    ):

        return "Stage 1: Basing"

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    if (
        slope >= 0.1
        and price > ma
    ):

        return "Stage 2: Advancing"

    # --------------------------------------------------------
    # Stage 4
    # --------------------------------------------------------

    if (
        slope <= -0.1
        and price < ma
    ):

        return "Stage 4: Declining"

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------

    if abs(slope) < 0.1:

        return "Stage 3: Topping"

    # --------------------------------------------------------
    # Indeterminate
    # --------------------------------------------------------

    return "Indeterminate"


# ============================================================
# DISPLAY CURRENT STAGE
# ============================================================

def show_stage(stage):

    if stage == "Stage 1: Basing":

        st.info(
            """
            ## 🔵 STAGE 1

            ### Basing / Accumulation

            **Classification:** Stage 1: Basing
            """
        )

    elif stage == "Stage 2: Advancing":

        st.success(
            """
            ## 🟢 STAGE 2

            ### Advancing / Uptrend

            **Classification:** Stage 2: Advancing
            """
        )

    elif stage == "Stage 3: Topping":

        st.warning(
            """
            ## 🟠 STAGE 3

            ### Topping / Distribution

            **Classification:** Stage 3: Topping
            """
        )

    elif stage == "Stage 4: Declining":

        st.error(
            """
            ## 🔴 STAGE 4

            ### Declining / Downtrend

            **Classification:** Stage 4: Declining
            """
        )

    else:

        st.info(
            """
            ## ⚪ INDETERMINATE

            ### No clear stage

            The current conditions do not satisfy
            the defined Stage 1–4 rules.
            """
        )


# ============================================================
# MAIN
# ============================================================

if analyze:

    if not symbol:

        st.warning(
            "Please enter an NSE stock symbol."
        )

        st.stop()

    ticker = symbol + ".NS"

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.info(
        f"📥 Downloading weekly data for **{symbol}**..."
    )

    try:

        data = yf.download(
            ticker,
            period=period,
            interval="1wk",
            auto_adjust=False,
            progress=False,
            threads=False
        )

    except Exception as e:

        st.error(
            f"❌ Download failed: {e}"
        )

        st.stop()

    # ========================================================
    # CHECK DATA
    # ========================================================

    if data is None or data.empty:

        st.error(
            f"""
            ❌ No data found for **{symbol}**.

            Check whether the NSE symbol is correct.
            """
        )

        st.stop()

    # ========================================================
    # PREPARE
    # ========================================================

    stock = prepare_data(data)

    if stock.empty:

        st.error(
            "❌ Unable to prepare stock data."
        )

        st.stop()

    # ========================================================
    # STAGE
    # ========================================================

    stock["Stage"] = stock.apply(
        classify_stage,
        axis=1
    )

    # ========================================================
    # VALID DATA
    # ========================================================

    valid = stock.dropna(
        subset=["MA30"]
    ).copy()

    if valid.empty:

        st.error(
            """
            ❌ Not enough weekly data to calculate
            the 30-week moving average.
            """
        )

        st.stop()

    # ========================================================
    # LATEST
    # ========================================================

    latest = valid.iloc[-1]

    close = float(
        latest["Close"]
    )

    ma30 = float(
        latest["MA30"]
    )

    slope = float(
        latest["MA_Slope"]
    )

    slope_pct = float(
        latest["MA_Slope_%"]
    )

    price_vs_ma = float(
        latest["Price_vs_MA_%"]
    )

    stage = latest["Stage"]

    # ========================================================
    # CURRENT STAGE
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"📈 {symbol} — Current Stage"
    )

    show_stage(stage)

    # ========================================================
    # METRICS
    # ========================================================

    st.markdown(
        "### 📊 Current Technical Data"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Weekly Close",
            f"₹{close:,.2f}"
        )

    with c2:

        st.metric(
            "30-Week MA",
            f"₹{ma30:,.2f}"
        )

    with c3:

        st.metric(
            "MA Slope",
            f"{slope:.3f}"
        )

    with c4:

        st.metric(
            "MA Slope %",
            f"{slope_pct:.2f}%"
        )

    with c5:

        st.metric(
            "Price vs MA",
            f"{price_vs_ma:.2f}%"
        )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📈 Weekly Price vs 30-Week Moving Average"
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        valid.index,
        valid["Close"],
        label="Weekly Close",
        linewidth=2
    )

    ax.plot(
        valid.index,
        valid["MA30"],
        label="30-Week MA",
        linewidth=2
    )

    ax.scatter(
        valid.index[-1],
        valid["Close"].iloc[-1],
        s=80,
        zorder=5
    )

    ax.set_title(
        f"{symbol} — Minervini / Weinstein Stage"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Price (₹)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ========================================================
    # STAGE HISTORY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Stage History"
    )

    stage_counts = (
        valid["Stage"]
        .value_counts()
    )

    history = pd.DataFrame(
        {
            "Stage": stage_counts.index,
            "Weeks": stage_counts.values
        }
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STAGE BAR CHART
    # ========================================================

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    ax2.bar(
        history["Stage"],
        history["Weeks"]
    )

    ax2.set_title(
        "Weeks Spent in Each Stage"
    )

    ax2.set_ylabel(
        "Number of Weeks"
    )

    ax2.tick_params(
        axis="x",
        rotation=25
    )

    ax2.grid(
        axis="y",
        alpha=0.3
    )

    st.pyplot(
        fig2,
        use_container_width=True
    )

    plt.close(fig2)

    # ========================================================
    # RECENT DATA
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📅 Last 20 Weekly Observations"
    )

    recent = valid.tail(20).copy()

    recent_display = recent[
        [
            "Close",
            "MA30",
            "MA_Slope",
            "MA_Slope_%",
            "Price_vs_MA_%",
            "Stage"
        ]
    ].copy()

    recent_display.insert(
        0,
        "Date",
        recent_display.index.strftime(
            "%d-%m-%Y"
        )
    )

    recent_display = (
        recent_display
        .reset_index(drop=True)
    )

    st.dataframe(
        recent_display.style.format(
            {
                "Close": "₹{:.2f}",
                "MA30": "₹{:.2f}",
                "MA_Slope": "{:.3f}",
                "MA_Slope_%": "{:.2f}%",
                "Price_vs_MA_%": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    st.markdown("---")

    csv_df = valid.copy()

    csv_df.index.name = "Date"

    csv_df = csv_df.reset_index()

    csv = csv_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Full Stage Analysis CSV",
        csv,
        file_name=f"{symbol}_Minervini_Stage.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# INITIAL INFORMATION
# ============================================================

else:

    st.info(
        """
        👈 Enter an NSE stock symbol in the sidebar and
        click **🔍 ANALYZE STOCK**.
        """
    )

    st.markdown("---")

    st.subheader(
        "📚 Stage Definitions"
    )

    definitions = pd.DataFrame(
        {
            "Stage": [
                "Stage 1",
                "Stage 2",
                "Stage 3",
                "Stage 4",
                "Indeterminate"
            ],
            "Description": [
                "Basing / Accumulation",
                "Advancing / Uptrend",
                "Topping / Distribution",
                "Declining / Downtrend",
                "No clear stage"
            ],
            "Rule": [
                "Flat MA + price near MA",
                "Rising MA + price above MA",
                "Flat MA + price away from MA",
                "Declining MA + price below MA",
                "Other combination"
            ]
        }
    )

    st.dataframe(
        definitions,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.caption(
        """
        ⚠️ Stage classification is based on the mathematical
        rules implemented in this scanner. It is a technical
        analysis classification, not a prediction of future prices.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "📊 NSE STOCK ANALYZER BY SUJOY ROY | Minervini / Weinstein Stage Analyzer"
)
