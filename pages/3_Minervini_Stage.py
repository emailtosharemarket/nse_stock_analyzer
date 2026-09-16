# ============================================================
# 📊 MARK MINERVINI STAGE DETECTION
# ============================================================
# NSE Stock Analyzer
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Minervini Stage Detection",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
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
    '<div class="main-title">📊 Mark Minervini Stage Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Identify Stage 1, Stage 2, Stage 3 and Stage 4 using weekly price '
    'and 30-week moving average analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    user_ticker = st.text_input(
        "Enter NSE Stock",
        value="TITAN",
        placeholder="Example: TITAN, SBIN, VBL"
    )

    period = st.selectbox(
        "Historical Period",
        [
            "2y",
            "3y",
            "5y"
        ],
        index=0
    )

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

    analyze_button = st.button(
        "🔍 ANALYZE STOCK",
        type="primary",
        use_container_width=True
    )


# ============================================================
# STAGE CLASSIFICATION FUNCTION
# ============================================================

def determine_stage(row):

    price = float(row["Close"])
    ma = float(row["30w_MA"])
    slope = float(row["MA_slope"])

    # --------------------------------------------------------
    # Stage 1 / Stage 3
    # --------------------------------------------------------

    if abs(slope) < 0.1:

        if abs(price - ma) < 0.05 * price:
            return "Stage 1: Basing"

        else:
            return "Stage 3: Topping"

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    elif slope >= 0.1 and price > ma:

        return "Stage 2: Advancing"

    # --------------------------------------------------------
    # Stage 4
    # --------------------------------------------------------

    elif slope <= -0.1 and price < ma:

        return "Stage 4: Declining"

    # --------------------------------------------------------
    # Indeterminate
    # --------------------------------------------------------

    else:

        return "Indeterminate"


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def download_stock_data(ticker, selected_period):

    try:

        data = yf.download(
            ticker,
            period=selected_period,
            interval="1wk",
            auto_adjust=False,
            progress=False
        )

        return data

    except Exception as e:

        return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(stock):

    if stock is None or stock.empty:
        return None

    stock = stock.copy()

    # --------------------------------------------------------
    # Handle MultiIndex returned by newer yfinance versions
    # --------------------------------------------------------

    if isinstance(stock.columns, pd.MultiIndex):

        # Find Close column
        close_candidates = [
            col for col in stock.columns
            if str(col[0]).lower() == "close"
        ]

        if close_candidates:

            close_col = close_candidates[0]

            stock["Close"] = stock[close_col]

        else:

            return None

    else:

        if "Close" not in stock.columns:
            return None

        stock["Close"] = stock["Close"]

    # --------------------------------------------------------
    # Convert Close to numeric
    # --------------------------------------------------------

    stock["Close"] = pd.to_numeric(
        stock["Close"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 30 Week Moving Average
    # --------------------------------------------------------

    stock["30w_MA"] = (
        stock["Close"]
        .rolling(window=30)
        .mean()
    )

    # --------------------------------------------------------
    # MA Slope
    # --------------------------------------------------------

    stock["MA_slope"] = stock["30w_MA"].diff()

    # --------------------------------------------------------
    # Remove incomplete rows
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
    # Determine Stage
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
        "class": "stage1",
        "color": "blue"
    },

    "Stage 2: Advancing": {
        "number": "STAGE 2",
        "description": "Advancing / Uptrend",
        "class": "stage2",
        "color": "green"
    },

    "Stage 3: Topping": {
        "number": "STAGE 3",
        "description": "Topping / Distribution",
        "class": "stage3",
        "color": "orange"
    },

    "Stage 4: Declining": {
        "number": "STAGE 4",
        "description": "Declining / Downtrend",
        "class": "stage4",
        "color": "red"
    },

    "Indeterminate": {
        "number": "N/A",
        "description": "Indeterminate",
        "class": "indeterminate",
        "color": "gray"
    }
}


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # Clean ticker
    # --------------------------------------------------------

    ticker_input = user_ticker.strip().upper()

    if ticker_input == "":

        st.error("❌ Please enter a stock symbol.")

        st.stop()

    # --------------------------------------------------------
    # Automatically add .NS
    # --------------------------------------------------------

    if not ticker_input.endswith(".NS"):

        yahoo_ticker = ticker_input + ".NS"

    else:

        yahoo_ticker = ticker_input

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    with st.spinner(
        f"📥 Downloading weekly data for {ticker_input}..."
    ):

        raw_data = download_stock_data(
            yahoo_ticker,
            period
        )

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if raw_data is None or raw_data.empty:

        st.error(
            f"❌ No data found for **{ticker_input}**."
        )

        st.warning(
            "Please check whether the NSE symbol is correct."
        )

        st.stop()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    stock = prepare_data(raw_data)

    if stock is None or stock.empty:

        st.error(
            "❌ Unable to prepare sufficient data for analysis."
        )

        st.info(
            "At least 30 weekly observations are required "
            "for the 30-week moving average."
        )

        st.stop()

    # ========================================================
    # LATEST DATA
    # ========================================================

    latest = stock.iloc[-1]

    latest_close = float(latest["Close"])
    latest_ma = float(latest["30w_MA"])
    latest_slope = float(latest["MA_slope"])
    latest_stage = latest["Stage"]

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
        f"📌 Current Stage — {ticker_input}"
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
    # KEY METRICS
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
            (latest_close - latest_ma)
            / latest_ma
        ) * 100

        st.metric(
            "Price vs MA",
            f"{distance:.2f}%"
        )

    with col5:

        st.metric(
            "Data Through",
            latest_date.strftime("%d-%m-%Y")
        )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"📈 {ticker_input} — Weekly Stage Chart"
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

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        f"{ticker_input} - Mark Minervini Stage Detection",
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

    dist_col1, dist_col2, dist_col3, dist_col4, dist_col5 = st.columns(5)

    stages_order = [
        "Stage 1: Basing",
        "Stage 2: Advancing",
        "Stage 3: Topping",
        "Stage 4: Declining",
        "Indeterminate"
    ]

    columns = [
        dist_col1,
        dist_col2,
        dist_col3,
        dist_col4,
        dist_col5
    ]

    for col, stage_name in zip(
        columns,
        stages_order
    ):

        with col:

            count = int(
                stage_counts.get(
                    stage_name,
                    0
                )
            )

            st.metric(
                stage_name.replace(
                    "Stage ",
                    "S"
                ),
                count
            )

    # ========================================================
    # RECENT 20 ROWS
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

    final_rows["Close"] = final_rows[
        "Close"
    ].round(2)

    final_rows["30w_MA"] = final_rows[
        "30w_MA"
    ].round(2)

    final_rows["MA_slope"] = final_rows[
        "MA_slope"
    ].round(4)

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
    # DOWNLOAD CSV
    # ========================================================

    csv_data = final_rows.to_csv()

    st.download_button(
        label="⬇️ Download Last 20 Weeks CSV",
        data=csv_data,
        file_name=(
            f"{ticker_input}_Minervini_Stage.csv"
        ),
        mime="text/csv"
    )

    # ========================================================
    # STAGE EXPLANATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📖 Stage Definitions"
    )

    explanation_col1, explanation_col2 = st.columns(2)

    with explanation_col1:

        st.markdown(
            """
            ### 🔵 Stage 1 — Basing

            Price is relatively close to the 30-week MA
            while the MA slope is relatively flat.

            ---

            ### 🟢 Stage 2 — Advancing

            Price is above the 30-week MA and the
            30-week MA slope is positive.

            """
        )

    with explanation_col2:

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

    st.caption(
        "⚠️ This is a quantitative stage-classification tool based "
        "on the rules implemented above. It is not a complete "
        "implementation of every element of Mark Minervini's "
        "trading methodology."
    )

# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👈 Enter an NSE stock symbol in the sidebar and "
        "click **🔍 ANALYZE STOCK**."
    )

    st.markdown(
        """
        ### How it works

        1. Enter an NSE stock symbol.
        2. The app automatically adds `.NS` for Yahoo Finance.
        3. Weekly data is downloaded.
        4. A 30-week moving average is calculated.
        5. The weekly MA slope is calculated.
        6. Each week is classified into a stage.
        7. The latest stage is displayed at the top.
        8. A visual stage chart is generated.
        9. The latest 20 observations can be downloaded.

        **Example:** Enter `TITAN` rather than `TITAN.NS`.
        """
    )
