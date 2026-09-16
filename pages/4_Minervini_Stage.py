# ============================================================
# 📊 MINERVINI / WEINSTEIN STAGE ANALYZER
# ============================================================
# 👤 Developed by SUJOY ROY
# 📅 Updated: 2026
#
# Stage Detection:
# Stage 1 → Basing / Accumulation
# Stage 2 → Advancing / Uptrend
# Stage 3 → Topping / Distribution
# Stage 4 → Declining / Downtrend
#
# Uses:
# - Weekly closing price
# - 30-week Moving Average
# - Moving Average slope
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
# CUSTOM TITLE
# ============================================================

st.title("📊 Minervini / Weinstein Stage Analyzer")

st.markdown(
    """
    Identify the current **market stage** of an NSE stock using
    weekly price action and the **30-week moving average**.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Scanner Settings")

symbol = st.sidebar.text_input(
    "Enter NSE Stock Symbol",
    value="RELIANCE"
).upper().strip()

period = st.sidebar.selectbox(
    "Historical Period",
    [
        "2y",
        "3y",
        "5y",
        "10y"
    ],
    index=1
)

run_analysis = st.sidebar.button(
    "🔍 ANALYZE STOCK",
    type="primary",
    use_container_width=True
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_close_series(data):
    """
    Safely extract Close prices from yfinance data.

    Newer versions of yfinance may return:
    - Normal DataFrame
    - MultiIndex DataFrame
    - 2-dimensional Close data

    This function always returns a clean 1-D Series.
    """

    if data is None:
        return pd.Series(dtype="float64")

    if data.empty:
        return pd.Series(dtype="float64")

    try:

        # ----------------------------------------------------
        # Case 1: MultiIndex columns
        # ----------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):

            close_columns = [
                col for col in data.columns
                if "Close" in [str(x) for x in col]
            ]

            if len(close_columns) > 0:

                close_data = data[close_columns]

                # If still DataFrame, take first column
                if isinstance(close_data, pd.DataFrame):

                    close_series = close_data.iloc[:, 0]

                else:

                    close_series = close_data

            else:

                return pd.Series(dtype="float64")

        # ----------------------------------------------------
        # Case 2: Normal columns
        # ----------------------------------------------------

        elif "Close" in data.columns:

            close_data = data["Close"]

            if isinstance(close_data, pd.DataFrame):

                close_series = close_data.iloc[:, 0]

            else:

                close_series = close_data

        else:

            return pd.Series(dtype="float64")

        # ----------------------------------------------------
        # Convert safely to Series
        # ----------------------------------------------------

        close_series = pd.Series(
            close_data,
            index=data.index,
            name="Close"
        )

        close_series = pd.to_numeric(
            close_series,
            errors="coerce"
        )

        close_series = close_series.dropna()

        return close_series

    except Exception as e:

        st.error(
            f"Unable to extract Close price: {e}"
        )

        return pd.Series(dtype="float64")


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    close_series = extract_close_series(data)

    if close_series.empty:
        return pd.DataFrame()

    # Create completely new DataFrame
    # This prevents MultiIndex problems
    stock = pd.DataFrame(
        {
            "Close": close_series
        }
    )

    stock.index = pd.to_datetime(stock.index)

    stock = stock.sort_index()

    # --------------------------------------------------------
    # 30 WEEK MOVING AVERAGE
    # --------------------------------------------------------

    stock["MA30"] = (
        stock["Close"]
        .rolling(window=30)
        .mean()
    )

    # --------------------------------------------------------
    # MA SLOPE
    #
    # Difference between current MA and previous week MA
    # --------------------------------------------------------

    stock["MA_Slope"] = (
        stock["MA30"]
        .diff()
    )

    # --------------------------------------------------------
    # MA SLOPE %
    #
    # Normalized slope makes the indicator easier to compare
    # across different stock prices.
    # --------------------------------------------------------

    stock["MA_Slope_%"] = (
        stock["MA30"]
        .pct_change()
        * 100
    )

    # --------------------------------------------------------
    # PRICE VS MA
    # --------------------------------------------------------

    stock["Price_vs_MA_%"] = (
        (
            stock["Close"]
            - stock["MA30"]
        )
        / stock["Close"]
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
    # Price difference from MA
    # --------------------------------------------------------

    price_difference = abs(price - ma)

    # --------------------------------------------------------
    # Stage 1
    #
    # MA relatively flat
    # Price near MA
    # --------------------------------------------------------

    if (
        abs(slope) < 0.1
        and price_difference < 0.05 * price
    ):

        return "Stage 1: Basing"

    # --------------------------------------------------------
    # Stage 2
    #
    # MA rising
    # Price above MA
    # --------------------------------------------------------

    elif (
        slope >= 0.1
        and price > ma
    ):

        return "Stage 2: Advancing"

    # --------------------------------------------------------
    # Stage 4
    #
    # MA declining
    # Price below MA
    # --------------------------------------------------------

    elif (
        slope <= -0.1
        and price < ma
    ):

        return "Stage 4: Declining"

    # --------------------------------------------------------
    # Stage 3
    #
    # MA relatively flat
    # Price not close enough to MA for Stage 1
    # --------------------------------------------------------

    elif abs(slope) < 0.1:

        return "Stage 3: Topping"

    # --------------------------------------------------------
    # Otherwise
    # --------------------------------------------------------

    else:

        return "Indeterminate"


# ============================================================
# STAGE DISPLAY
# ============================================================

def display_stage(stage):

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

            The current price/MA relationship does not clearly
            satisfy the defined stage conditions.
            """
        )


# ============================================================
# DOWNLOAD DATA
# ============================================================

def create_download_data(stock):

    download_df = stock.copy()

    download_df.index.name = "Date"

    download_df = download_df.reset_index()

    return download_df


# ============================================================
# MAIN ANALYSIS
# ============================================================

if run_analysis:

    if not symbol:

        st.warning(
            "Please enter an NSE stock symbol."
        )

        st.stop()

    # --------------------------------------------------------
    # Yahoo Finance Symbol
    # --------------------------------------------------------

    yf_symbol = f"{symbol}.NS"

    st.info(
        f"📥 Downloading weekly data for **{symbol}**..."
    )

    try:

        data = yf.download(
            yf_symbol,
            period=period,
            interval="1wk",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column"
        )

    except Exception as e:

        st.error(
            f"❌ Yahoo Finance download failed:\n\n{e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if data is None or data.empty:

        st.error(
            f"""
            ❌ No data found for **{symbol}**.

            Please check the NSE symbol.
            
            Example:
            - RELIANCE
            - TCS
            - INFY
            - HDFCBANK
            - ITC
            """
        )

        st.stop()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    stock = prepare_data(data)

    if stock.empty:

        st.error(
            "❌ Unable to prepare stock data."
        )

        st.stop()

    # --------------------------------------------------------
    # Minimum data requirement
    # --------------------------------------------------------

    if len(stock) < 35:

        st.warning(
            f"""
            ⚠️ Only {len(stock)} weekly observations were found.

            At least 30 weeks are required for the 30-week
            moving average. Results may be unreliable.
            """
        )

    # --------------------------------------------------------
    # Classify Stage
    # --------------------------------------------------------

    stock["Stage"] = stock.apply(
        classify_stage,
        axis=1
    )

    # --------------------------------------------------------
    # Remove rows without MA
    # --------------------------------------------------------

    valid_stock = stock.dropna(
        subset=["MA30"]
    ).copy()

    if valid_stock.empty:

        st.error(
            "❌ Not enough data to calculate the 30-week moving average."
        )

        st.stop()

    # ========================================================
    # LATEST DATA
    # ========================================================

    latest = valid_stock.iloc[-1]

    latest_close = float(
        latest["Close"]
    )

    latest_ma = float(
        latest["MA30"]
    )

    latest_slope = float(
        latest["MA_Slope"]
    )

    latest_slope_pct = float(
        latest["MA_Slope_%"]
    )

    latest_price_vs_ma = float(
        latest["Price_vs_MA_%"]
    )

    latest_stage = latest["Stage"]

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"📈 {symbol} — Current Market Stage"
    )

    # ========================================================
    # CURRENT STAGE
    # ========================================================

    display_stage(
        latest_stage
    )

    # ========================================================
    # KEY METRICS
    # ========================================================

    st.markdown("### 📊 Current Data")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Current Price",
            f"₹{latest_close:,.2f}"
        )

    with col2:

        st.metric(
            "30-Week MA",
            f"₹{latest_ma:,.2f}"
        )

    with col3:

        st.metric(
            "MA Slope",
            f"{latest_slope:.2f}"
        )

    with col4:

        st.metric(
            "MA Slope %",
            f"{latest_slope_pct:.2f}%"
        )

    with col5:

        st.metric(
            "Price vs MA",
            f"{latest_price_vs_ma:.2f}%"
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown("### 🔎 Interpretation")

    if latest_stage == "Stage 1: Basing":

        st.write(
            """
            The 30-week moving average is relatively flat and
            the price is close to the moving average.
            This corresponds to a **Stage 1 basing condition**
            under the rules used by this scanner.
            """
        )

    elif latest_stage == "Stage 2: Advancing":

        st.write(
            """
            The 30-week moving average is rising and the price
            is above the moving average. This corresponds to a
            **Stage 2 advancing condition** under the scanner rules.
            """
        )

    elif latest_stage == "Stage 3: Topping":

        st.write(
            """
            The 30-week moving average is relatively flat while
            the price is not sufficiently close to it to meet
            the Stage 1 condition. This corresponds to a
            **Stage 3 topping condition** under the scanner rules.
            """
        )

    elif latest_stage == "Stage 4: Declining":

        st.write(
            """
            The 30-week moving average is declining and the price
            is below the moving average. This corresponds to a
            **Stage 4 declining condition** under the scanner rules.
            """
        )

    else:

        st.write(
            """
            The current price and moving-average conditions do not
            satisfy the defined Stage 1–4 rules.
            """
        )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📈 Weekly Price & 30-Week Moving Average"
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        valid_stock.index,
        valid_stock["Close"],
        label="Weekly Close",
        linewidth=2
    )

    ax.plot(
        valid_stock.index,
        valid_stock["MA30"],
        label="30-Week MA",
        linewidth=2
    )

    # --------------------------------------------------------
    # Current price marker
    # --------------------------------------------------------

    ax.scatter(
        valid_stock.index[-1],
        valid_stock["Close"].iloc[-1],
        s=80,
        zorder=5
    )

    ax.set_title(
        f"{symbol} — Minervini / Weinstein Stage Analysis"
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
    # STAGE DISTRIBUTION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Historical Stage Distribution"
    )

    stage_counts = (
        valid_stock["Stage"]
        .value_counts()
    )

    distribution_df = pd.DataFrame(
        {
            "Stage": stage_counts.index,
            "Weeks": stage_counts.values
        }
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        st.dataframe(
            distribution_df,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        fig2, ax2 = plt.subplots(
            figsize=(8, 5)
        )

        ax2.bar(
            distribution_df["Stage"],
            distribution_df["Weeks"]
        )

        ax2.set_title(
            "Number of Weeks in Each Stage"
        )

        ax2.set_ylabel(
            "Weeks"
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
    # RECENT WEEKLY DATA
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📅 Recent Weekly Observations"
    )

    recent = valid_stock.tail(20).copy()

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

    recent_display = recent_display.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    recent_display["Close"] = (
        recent_display["Close"]
        .round(2)
    )

    recent_display["MA30"] = (
        recent_display["MA30"]
        .round(2)
    )

    recent_display["MA_Slope"] = (
        recent_display["MA_Slope"]
        .round(3)
    )

    recent_display["MA_Slope_%"] = (
        recent_display["MA_Slope_%"]
        .round(2)
    )

    recent_display["Price_vs_MA_%"] = (
        recent_display["Price_vs_MA_%"]
        .round(2)
    )

    st.dataframe(
        recent_display,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💾 Download Analysis"
    )

    download_df = create_download_data(
        valid_stock
    )

    csv_data = download_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ Download Full Analysis CSV",
        data=csv_data,
        file_name=f"{symbol}_Minervini_Stage.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# INFORMATION SECTION
# ============================================================

else:

    st.info(
        """
        👈 Enter an NSE stock symbol in the sidebar and click
        **🔍 ANALYZE STOCK**.
        """
    )

    st.markdown("---")

    st.subheader(
        "📚 Stage Definitions"
    )

    stage_table = pd.DataFrame(
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
            "Scanner Condition": [
                "Flat MA + price near MA",
                "Rising MA + price above MA",
                "Flat MA + price not near MA",
                "Declining MA + price below MA",
                "Other combination"
            ]
        }
    )

    st.dataframe(
        stage_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.caption(
        """
        ⚠️ This scanner is a technical-analysis tool. Stage
        classification depends on the mathematical rules defined
        in this program and should not be treated as a prediction
        of future price movement.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "📊 NSE Stock Analyzer by SUJOY ROY | Minervini / Weinstein Stage Analyzer"
)
