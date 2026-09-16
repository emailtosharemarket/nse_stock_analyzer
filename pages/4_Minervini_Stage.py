# ============================================================
# 📊 MARK MINERVINI / WEINSTEIN STAGE DETECTION
# ============================================================
# NSE STOCK ANALYZER
# Stage 1 / Stage 2 / Stage 3 / Stage 4
#
# Developed for:
# NSE STOCK ANALYZER BY SUJOY ROY
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Minervini Stage Detection",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 16px;
        color: #666666;
        margin-bottom: 20px;
    }

    .stage-box {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 10px;
    }

    .stage1 {
        background-color: #dbeafe;
        border: 2px solid #2563eb;
    }

    .stage2 {
        background-color: #dcfce7;
        border: 2px solid #16a34a;
    }

    .stage3 {
        background-color: #ffedd5;
        border: 2px solid #ea580c;
    }

    .stage4 {
        background-color: #fee2e2;
        border: 2px solid #dc2626;
    }

    .indeterminate {
        background-color: #f3f4f6;
        border: 2px solid #6b7280;
    }

    .stage-number {
        font-size: 34px;
        font-weight: 800;
    }

    .stage-description {
        font-size: 18px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Minervini / Weinstein Stage Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Weekly price and 30-week moving average based stage analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    user_ticker = st.text_input(
        "Enter NSE Stock",
        value="TITAN",
        placeholder="Example: TITAN, VBL, SBIN"
    )

    period = st.selectbox(
        "Historical Period",
        ["2y", "3y", "5y"],
        index=0
    )

    analyze_button = st.button(
        "🔍 ANALYZE STOCK",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")

    st.info(
        """
        **Examples**

        TITAN  
        SBIN  
        VBL  
        RELIANCE  
        TCS  
        INFY  
        HDFCBANK
        """
    )


# ============================================================
# STAGE FUNCTION
# ============================================================

def determine_stage(row):

    price = float(row["Close"])
    ma = float(row["30w_MA"])
    slope = float(row["MA_slope"])

    if abs(slope) < 0.1:

        if abs(price - ma) < 0.05 * price:
            return "Stage 1: Basing"

        else:
            return "Stage 3: Topping"

    elif slope >= 0.1 and price > ma:

        return "Stage 2: Advancing"

    elif slope <= -0.1 and price < ma:

        return "Stage 4: Declining"

    else:

        return "Indeterminate"


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def download_stock_data(ticker, selected_period):

    try:

        data = yf.download(
            ticker,
            period=selected_period,
            interval="1wk",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        return data

    except Exception:

        return None


# ============================================================
# ROBUST CLOSE EXTRACTION
# ============================================================

def extract_close_column(stock):

    """
    Safely extract Close as a 1-D pandas Series.

    Handles:
    1. Normal yfinance DataFrame
    2. MultiIndex DataFrame
    3. DataFrame returned by stock["Close"]
    4. Different yfinance versions
    """

    if stock is None or stock.empty:
        return None

    # --------------------------------------------------------
    # CASE 1: MultiIndex columns
    # --------------------------------------------------------

    if isinstance(stock.columns, pd.MultiIndex):

        # First try level-0 column name = Close
        close_columns = [
            col for col in stock.columns
            if str(col[0]).strip().lower() == "close"
        ]

        if len(close_columns) > 0:

            close_data = stock[close_columns]

            # If DataFrame, take first column
            if isinstance(close_data, pd.DataFrame):

                if close_data.shape[1] > 0:

                    close_data = close_data.iloc[:, 0]

                else:

                    return None

            return pd.to_numeric(
                close_data,
                errors="coerce"
            )

        # ----------------------------------------------------
        # Try level-1
        # ----------------------------------------------------

        close_columns = [
            col for col in stock.columns
            if len(col) > 1
            and str(col[1]).strip().lower() == "close"
        ]

        if len(close_columns) > 0:

            close_data = stock[close_columns]

            if isinstance(close_data, pd.DataFrame):

                close_data = close_data.iloc[:, 0]

            return pd.to_numeric(
                close_data,
                errors="coerce"
            )

        return None

    # ========================================================
    # CASE 2: Normal columns
    # ========================================================

    if "Close" in stock.columns:

        close_data = stock["Close"]

        # Very important:
        # Sometimes this can still be a DataFrame
        if isinstance(close_data, pd.DataFrame):

            if close_data.shape[1] == 0:
                return None

            close_data = close_data.iloc[:, 0]

        return pd.to_numeric(
            close_data,
            errors="coerce"
        )

    # ========================================================
    # CASE 3: lowercase close
    # ========================================================

    for column in stock.columns:

        if str(column).strip().lower() == "close":

            close_data = stock[column]

            if isinstance(close_data, pd.DataFrame):

                close_data = close_data.iloc[:, 0]

            return pd.to_numeric(
                close_data,
                errors="coerce"
            )

    return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(stock):

    if stock is None or stock.empty:

        return None

    stock = stock.copy()

    # --------------------------------------------------------
    # Extract Close safely
    # --------------------------------------------------------

    close_data = extract_close_column(stock)

    if close_data is None:

        return None

    # --------------------------------------------------------
    # Force Series
    # --------------------------------------------------------

    if not isinstance(close_data, pd.Series):

        close_data = pd.Series(
            close_data,
            index=stock.index
        )

    # --------------------------------------------------------
    # Clean index
    # --------------------------------------------------------

    close_data.index = stock.index

    # --------------------------------------------------------
    # Create clean Close column
    # --------------------------------------------------------

    stock["Close"] = close_data

    # --------------------------------------------------------
    # Ensure numeric
    # --------------------------------------------------------

    stock["Close"] = pd.to_numeric(
        stock["Close"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid prices
    # --------------------------------------------------------

    stock = stock[
        stock["Close"].notna()
    ].copy()

    if stock.empty:

        return None

    # --------------------------------------------------------
    # 30 WEEK MOVING AVERAGE
    # --------------------------------------------------------

    stock["30w_MA"] = (
        stock["Close"]
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # --------------------------------------------------------
    # MA SLOPE
    # --------------------------------------------------------

    stock["MA_slope"] = (
        stock["30w_MA"]
        .diff()
    )

    # --------------------------------------------------------
    # Remove rows without indicators
    # --------------------------------------------------------

    stock = stock.dropna(
        subset=[
            "Close",
            "30w_MA",
            "MA_slope"
        ]
    ).copy()

    if stock.empty:

        return None

    # --------------------------------------------------------
    # Stage
    # --------------------------------------------------------

    stock["Stage"] = stock.apply(
        determine_stage,
        axis=1
    )

    return stock


# ============================================================
# STAGE INFORMATION
# ============================================================

stage_information = {

    "Stage 1: Basing": {
        "number": "STAGE 1",
        "description": "Basing / Accumulation",
        "class": "stage1"
    },

    "Stage 2: Advancing": {
        "number": "STAGE 2",
        "description": "Advancing / Uptrend",
        "class": "stage2"
    },

    "Stage 3: Topping": {
        "number": "STAGE 3",
        "description": "Topping / Distribution",
        "class": "stage3"
    },

    "Stage 4: Declining": {
        "number": "STAGE 4",
        "description": "Declining / Downtrend",
        "class": "stage4"
    },

    "Indeterminate": {
        "number": "N/A",
        "description": "Indeterminate",
        "class": "indeterminate"
    }
}


# ============================================================
# MAIN
# ============================================================

if analyze_button:

    ticker_input = user_ticker.strip().upper()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not ticker_input:

        st.error("❌ Please enter an NSE stock symbol.")

        st.stop()

    # --------------------------------------------------------
    # Remove .NS if user entered it twice
    # --------------------------------------------------------

    ticker_input = ticker_input.replace(
        ".NS.NS",
        ".NS"
    )

    # --------------------------------------------------------
    # Add .NS
    # --------------------------------------------------------

    if ticker_input.endswith(".NS"):

        yahoo_ticker = ticker_input

        display_ticker = ticker_input.replace(
            ".NS",
            ""
        )

    else:

        yahoo_ticker = ticker_input + ".NS"

        display_ticker = ticker_input

    # ========================================================
    # DOWNLOAD
    # ========================================================

    with st.spinner(
        f"📥 Downloading weekly data for {display_ticker}..."
    ):

        raw_data = download_stock_data(
            yahoo_ticker,
            period
        )

    # ========================================================
    # DATA CHECK
    # ========================================================

    if raw_data is None or raw_data.empty:

        st.error(
            f"❌ No data found for **{display_ticker}**."
        )

        st.warning(
            "Please check the NSE stock symbol."
        )

        st.stop()

    # ========================================================
    # PREPARE
    # ========================================================

    stock = prepare_data(raw_data)

    if stock is None or stock.empty:

        st.error(
            "❌ Unable to prepare the stock data."
        )

        st.info(
            """
            Possible reasons:

            • Invalid NSE symbol  
            • Yahoo Finance returned incomplete data  
            • Insufficient weekly history  
            • Temporary Yahoo Finance issue
            """
        )

        st.stop()

    # ========================================================
    # LATEST DATA
    # ========================================================

    latest = stock.iloc[-1]

    latest_close = float(
        latest["Close"]
    )

    latest_ma = float(
        latest["30w_MA"]
    )

    latest_slope = float(
        latest["MA_slope"]
    )

    latest_stage = str(
        latest["Stage"]
    )

    latest_date = stock.index[-1]

    # ========================================================
    # STAGE BOX
    # ========================================================

    info = stage_information.get(
        latest_stage,
        stage_information["Indeterminate"]
    )

    st.markdown("---")

    st.subheader(
        f"📌 Current Stage — {display_ticker}"
    )

    st.markdown(
        f"""
        <div class="stage-box {info['class']}">

        <div class="stage-number">
        {info['number']}
        </div>

        <div class="stage-description">
        {info['description']}
        </div>

        <div style="margin-top:8px;font-size:15px;">
        Classification: <b>{latest_stage}</b>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # METRICS
    # ========================================================

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
            f"{latest_slope:.3f}"
        )

    with col4:

        distance = (
            (
                latest_close
                - latest_ma
            )
            / latest_ma
        ) * 100

        st.metric(
            "Price vs MA",
            f"{distance:.2f}%"
        )

    with col5:

        if hasattr(
            latest_date,
            "strftime"
        ):

            date_text = latest_date.strftime(
                "%d-%m-%Y"
            )

        else:

            date_text = str(latest_date)

        st.metric(
            "Data Through",
            date_text
        )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"📈 {display_ticker} — Weekly Stage Chart"
    )

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    ax.plot(
        stock.index,
        stock["Close"],
        label="Weekly Close",
        color="black",
        linewidth=1.8
    )

    # --------------------------------------------------------
    # 30 Week MA
    # --------------------------------------------------------

    ax.plot(
        stock.index,
        stock["30w_MA"],
        label="30-Week MA",
        color="purple",
        linestyle="--",
        linewidth=1.8
    )

    # --------------------------------------------------------
    # Stage colors
    # --------------------------------------------------------

    stage_colors = {

        "Stage 1: Basing": "blue",

        "Stage 2: Advancing": "green",

        "Stage 3: Topping": "orange",

        "Stage 4: Declining": "red",

        "Indeterminate": "gray"
    }

    stage_sizes = {

        "Stage 1: Basing": 30,

        "Stage 2: Advancing": 45,

        "Stage 3: Topping": 40,

        "Stage 4: Declining": 45,

        "Indeterminate": 20
    }

    # --------------------------------------------------------
    # Plot stages
    # --------------------------------------------------------

    for stage_name, color in stage_colors.items():

        stage_data = stock[
            stock["Stage"] == stage_name
        ]

        if not stage_data.empty:

            ax.scatter(
                stage_data.index,
                stage_data["Close"],
                label=stage_name,
                color=color,
                alpha=0.65,
                s=stage_sizes[stage_name]
            )

    ax.set_title(
        f"{display_ticker} - Stage Detection",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Price (₹)")

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

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
        "📊 Stage Distribution"
    )

    stage_counts = (
        stock["Stage"]
        .value_counts()
    )

    d1, d2, d3, d4, d5 = st.columns(5)

    distribution = [
        (d1, "Stage 1: Basing"),
        (d2, "Stage 2: Advancing"),
        (d3, "Stage 3: Topping"),
        (d4, "Stage 4: Declining"),
        (d5, "Indeterminate")
    ]

    for col, stage_name in distribution:

        with col:

            count = int(
                stage_counts.get(
                    stage_name,
                    0
                )
            )

            st.metric(
                stage_name,
                count
            )

    # ========================================================
    # LAST 20 WEEKS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📋 Last 20 Weekly Observations"
    )

    final_rows = stock[
        [
            "Close",
            "30w_MA",
            "MA_slope",
            "Stage"
        ]
    ].tail(20).copy()

    final_rows["Close"] = (
        final_rows["Close"]
        .round(2)
    )

    final_rows["30w_MA"] = (
        final_rows["30w_MA"]
        .round(2)
    )

    final_rows["MA_slope"] = (
        final_rows["MA_slope"]
        .round(4)
    )

    final_rows.index = (
        final_rows.index
        .strftime("%d-%m-%Y")
    )

    final_rows.index.name = "Date"

    st.dataframe(
        final_rows,
        use_container_width=True
    )

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv_data = final_rows.to_csv()

    st.download_button(
        label="⬇️ Download Last 20 Weeks CSV",
        data=csv_data,
        file_name=(
            f"{display_ticker}_Minervini_Stage.csv"
        ),
        mime="text/csv"
    )

    # ========================================================
    # STAGE DEFINITIONS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📖 Stage Definitions"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            ### 🔵 Stage 1 — Basing

            The 30-week MA is relatively flat and
            price is relatively close to the MA.

            ---

            ### 🟢 Stage 2 — Advancing

            Price is above the 30-week MA and the
            30-week MA slope is positive.
            """
        )

    with c2:

        st.markdown(
            """
            ### 🟠 Stage 3 — Topping

            The 30-week MA is relatively flat while
            price is relatively away from the MA.

            ---

            ### 🔴 Stage 4 — Declining

            Price is below the 30-week MA and the
            30-week MA slope is negative.
            """
        )

    st.markdown("---")

    st.caption(
        "⚠️ This tool applies the quantitative rules implemented "
        "in this program. It should not be treated as a complete "
        "implementation of every element of Mark Minervini's "
        "trading methodology."
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👈 Enter an NSE stock symbol and click "
        "**🔍 ANALYZE STOCK**."
    )

    st.markdown(
        """
        ### 📊 How it works

        **1.** Enter an NSE stock symbol.

        **2.** The application automatically adds `.NS`.

        **3.** Weekly historical data is downloaded.

        **4.** A 30-week moving average is calculated.

        **5.** The MA slope is calculated.

        **6.** Each weekly observation is classified.

        **7.** The latest stage is displayed.

        **8.** A stage-colored chart is generated.

        **9.** The latest 20 observations are displayed.

        **10.** The observations can be downloaded as CSV.
        """
    )
