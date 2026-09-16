# ============================================================
# 📊 MINERVINI / WEINSTEIN STAGE DETECTION
# ============================================================
# NSE STOCK ANALYZER
#
# Stage 1 - Basing
# Stage 2 - Advancing
# Stage 3 - Topping
# Stage 4 - Declining
#
# Developed for:
# NSE STOCK ANALYZER BY SUJOY ROY
#
# Robust yfinance version
# Compatible with Streamlit Cloud
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
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        margin: 10px 0 20px 0;
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
        font-size: 36px;
        font-weight: 800;
    }

    .stage-description {
        font-size: 19px;
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
    'Weekly stock stage analysis using price, 30-week moving average '
    'and MA slope.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    user_ticker = st.text_input(
        "Enter NSE Stock Symbol",
        value="TITAN",
        placeholder="Example: TITAN"
    )

    period = st.selectbox(
        "Historical Period",
        options=[
            "2y",
            "3y",
            "5y"
        ],
        index=0
    )

    analyze_button = st.button(
        "🔍 ANALYZE STOCK",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown(
        """
        ### Examples

        `TITAN`

        `VBL`

        `SBIN`

        `RELIANCE`

        `TCS`

        `INFY`

        `HDFCBANK`
        """
    )

    st.markdown("---")

    st.caption(
        "NSE symbols are automatically converted to Yahoo Finance "
        "format using .NS"
    )


# ============================================================
# STAGE CLASSIFICATION
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
# DOWNLOAD DATA
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def download_stock_data(
    ticker,
    selected_period
):

    try:

        data = yf.download(
            ticker,
            period=selected_period,
            interval="1wk",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column"
        )

        return data

    except Exception:

        return None


# ============================================================
# EXTRACT CLOSE SERIES
# ============================================================

def extract_close_series(data):

    """
    Extract Close as a guaranteed 1-D pandas Series.

    Handles different yfinance column formats.
    """

    if data is None:

        return None

    if data.empty:

        return None

    try:

        # ====================================================
        # MULTIINDEX COLUMNS
        # ====================================================

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            close_positions = []

            for position, column in enumerate(
                data.columns
            ):

                # Search every level
                for level_value in column:

                    if str(
                        level_value
                    ).strip().lower() == "close":

                        close_positions.append(
                            position
                        )

                        break

            if not close_positions:

                return None

            # ------------------------------------------------
            # Take the FIRST Close column
            # ------------------------------------------------

            close_series = data.iloc[
                :,
                close_positions[0]
            ]

        # ====================================================
        # NORMAL COLUMNS
        # ====================================================

        else:

            close_column = None

            for column in data.columns:

                if str(
                    column
                ).strip().lower() == "close":

                    close_column = column

                    break

            if close_column is None:

                return None

            close_series = data[
                close_column
            ]

        # ====================================================
        # GUARANTEE SERIES
        # ====================================================

        if isinstance(
            close_series,
            pd.DataFrame
        ):

            if close_series.shape[1] == 0:

                return None

            close_series = close_series.iloc[
                :,
                0
            ]

        # ----------------------------------------------------
        # Convert to Series
        # ----------------------------------------------------

        if not isinstance(
            close_series,
            pd.Series
        ):

            close_series = pd.Series(
                close_series,
                index=data.index
            )

        # ====================================================
        # NUMERIC
        # ====================================================

        close_series = pd.to_numeric(
            close_series,
            errors="coerce"
        )

        # ====================================================
        # REMOVE NaN
        # ====================================================

        close_series = close_series.dropna()

        if close_series.empty:

            return None

        return close_series

    except Exception as error:

        st.error(
            f"❌ Error extracting Close price: {error}"
        )

        return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    if data is None:

        return None

    if data.empty:

        return None

    # ========================================================
    # GET CLEAN CLOSE SERIES
    # ========================================================

    close_series = extract_close_series(
        data
    )

    if close_series is None:

        st.error(
            "❌ Close price could not be extracted."
        )

        return None

    # ========================================================
    # IMPORTANT:
    #
    # Create a completely NEW DataFrame.
    #
    # We do NOT modify yfinance's original DataFrame.
    # ========================================================

    stock = pd.DataFrame(
        {
            "Close": close_series
        }
    )

    # ========================================================
    # FORCE CLOSE TO 1-D
    # ========================================================

    if isinstance(
        stock["Close"],
        pd.DataFrame
    ):

        stock["Close"] = stock[
            "Close"
        ].iloc[:, 0]

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    stock["Close"] = pd.to_numeric(
        stock["Close"],
        errors="coerce"
    )

    # ========================================================
    # REMOVE INVALID DATA
    # ========================================================

    stock = stock.dropna(
        subset=[
            "Close"
        ]
    ).copy()

    # ========================================================
    # CHECK DATA LENGTH
    # ========================================================

    if len(stock) < 31:

        st.error(
            f"""
            ❌ Insufficient weekly data.

            Only {len(stock)} valid weekly observations
            were received.

            At least 31 observations are required for
            the 30-week moving average.
            """
        )

        return None

    # ========================================================
    # 30-WEEK MOVING AVERAGE
    # ========================================================

    stock["30w_MA"] = (
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

    stock["MA_slope"] = (
        stock["30w_MA"]
        .diff()
    )

    # ========================================================
    # REMOVE NaN INDICATOR ROWS
    # ========================================================

    stock = stock.dropna(
        subset=[
            "30w_MA",
            "MA_slope"
        ]
    ).copy()

    if stock.empty:

        return None

    # ========================================================
    # DETERMINE STAGE
    # ========================================================

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
# MAIN ANALYSIS
# ============================================================

if analyze_button:

    # ========================================================
    # CLEAN INPUT
    # ========================================================

    ticker_input = (
        user_ticker
        .strip()
        .upper()
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    if not ticker_input:

        st.error(
            "❌ Please enter an NSE stock symbol."
        )

        st.stop()

    # ========================================================
    # NORMALIZE .NS
    # ========================================================

    ticker_input = ticker_input.replace(
        ".NS.NS",
        ".NS"
    )

    if ticker_input.endswith(
        ".NS"
    ):

        yahoo_ticker = ticker_input

        display_ticker = ticker_input[
            :-3
        ]

    else:

        yahoo_ticker = (
            ticker_input + ".NS"
        )

        display_ticker = ticker_input

    # ========================================================
    # DOWNLOAD
    # ========================================================

    with st.spinner(
        f"📥 Downloading weekly data for "
        f"{display_ticker}..."
    ):

        raw_data = download_stock_data(
            yahoo_ticker,
            period
        )

    # ========================================================
    # CHECK DOWNLOAD
    # ========================================================

    if raw_data is None:

        st.error(
            f"❌ Unable to download data for "
            f"{display_ticker}."
        )

        st.stop()

    if raw_data.empty:

        st.error(
            f"❌ No data found for "
            f"{display_ticker}."
        )

        st.warning(
            "Please verify that the NSE symbol is correct."
        )

        st.stop()

    # ========================================================
    # PREPARE
    # ========================================================

    with st.spinner(
        "⚙️ Calculating 30-week MA and stages..."
    ):

        stock = prepare_data(
            raw_data
        )

    if stock is None:

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
    # PRICE VS MA
    # ========================================================

    price_vs_ma = (
        (
            latest_close
            - latest_ma
        )
        / latest_ma
    ) * 100

    # ========================================================
    # STAGE INFORMATION
    # ========================================================

    info = stage_information.get(
        latest_stage,
        stage_information[
            "Indeterminate"
        ]
    )

    # ========================================================
    # CURRENT STAGE
    # ========================================================

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

            <div style="
                margin-top:10px;
                font-size:16px;
            ">
                Classification:
                <b>{latest_stage}</b>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # KEY METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Current Price",
            f"₹{latest_close:,.2f}"
        )

    with c2:

        st.metric(
            "30-Week MA",
            f"₹{latest_ma:,.2f}"
        )

    with c3:

        st.metric(
            "MA Slope",
            f"{latest_slope:.3f}"
        )

    with c4:

        st.metric(
            "Price vs MA",
            f"{price_vs_ma:.2f}%"
        )

    with c5:

        try:

            date_text = latest_date.strftime(
                "%d-%m-%Y"
            )

        except Exception:

            date_text = str(
                latest_date
            )

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
    # WEEKLY CLOSE
    # --------------------------------------------------------

    ax.plot(
        stock.index,
        stock["Close"],
        label="Weekly Close",
        color="black",
        linewidth=1.8
    )

    # --------------------------------------------------------
    # 30 WEEK MA
    # --------------------------------------------------------

    ax.plot(
        stock.index,
        stock["30w_MA"],
        label="30-Week MA",
        color="purple",
        linestyle="--",
        linewidth=1.8
    )

    # ========================================================
    # STAGE COLORS
    # ========================================================

    stage_colors = {

        "Stage 1: Basing": "blue",

        "Stage 2: Advancing": "green",

        "Stage 3: Topping": "orange",

        "Stage 4: Declining": "red",

        "Indeterminate": "gray"
    }

    stage_sizes = {

        "Stage 1: Basing": 35,

        "Stage 2: Advancing": 50,

        "Stage 3: Topping": 45,

        "Stage 4: Declining": 50,

        "Indeterminate": 25
    }

    # ========================================================
    # STAGE POINTS
    # ========================================================

    for stage_name in stage_colors:

        stage_data = stock[
            stock["Stage"]
            == stage_name
        ]

        if not stage_data.empty:

            ax.scatter(
                stage_data.index,
                stage_data["Close"],
                label=stage_name,
                color=stage_colors[
                    stage_name
                ],
                alpha=0.65,
                s=stage_sizes[
                    stage_name
                ]
            )

    # ========================================================
    # CHART FORMATTING
    # ========================================================

    ax.set_title(
        f"{display_ticker} - Stage Detection",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Price (₹)"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.autofmt_xdate()

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

        (
            d1,
            "Stage 1: Basing"
        ),

        (
            d2,
            "Stage 2: Advancing"
        ),

        (
            d3,
            "Stage 3: Topping"
        ),

        (
            d4,
            "Stage 4: Declining"
        ),

        (
            d5,
            "Indeterminate"
        )
    ]

    for column, stage_name in distribution:

        with column:

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

    # --------------------------------------------------------
    # ROUND NUMBERS
    # --------------------------------------------------------

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

    # ========================================================
    # FORMAT DATE
    # ========================================================

    try:

        final_rows.index = (
            final_rows.index
            .strftime(
                "%d-%m-%Y"
            )
        )

    except Exception:

        final_rows.index = (
            final_rows.index
            .astype(str)
        )

    final_rows.index.name = "Date"

    # ========================================================
    # DISPLAY
    # ========================================================

    st.dataframe(
        final_rows,
        use_container_width=True,
        height=500
    )

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv_data = final_rows.to_csv()

    st.download_button(
        label="⬇️ Download Last 20 Weeks CSV",
        data=csv_data,
        file_name=(
            f"{display_ticker}"
            "_Minervini_Stage.csv"
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

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            ### 🔵 Stage 1 — Basing

            The 30-week moving average is relatively
            flat and price is relatively close to the MA.

            ---

            ### 🟢 Stage 2 — Advancing

            Price is above the 30-week moving average
            and the MA slope is positive.
            """
        )

    with right:

        st.markdown(
            """
            ### 🟠 Stage 3 — Topping

            The 30-week moving average is relatively
            flat while price is relatively away from it.

            ---

            ### 🔴 Stage 4 — Declining

            Price is below the 30-week moving average
            and the MA slope is negative.
            """
        )

    # ========================================================
    # METHODOLOGY NOTE
    # ========================================================

    st.markdown("---")

    st.warning(
        """
        **Methodology note:** This program uses the specific
        quantitative rules implemented in this application:
        weekly price, 30-week moving average and MA slope.

        It is not a complete implementation of every element
        of Mark Minervini's broader trading methodology.
        """
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👈 Enter an NSE stock symbol from the sidebar "
        "and click **🔍 ANALYZE STOCK**."
    )

    st.markdown(
        """
        ## 📊 How This Tool Works

        **Step 1 — Enter Stock**

        Enter an NSE symbol such as:

        `TITAN`

        `VBL`

        `SBIN`

        `RELIANCE`

        **Step 2 — Download Weekly Data**

        Yahoo Finance weekly data is used.

        **Step 3 — Calculate 30-Week MA**

        A 30-week simple moving average is calculated.

        **Step 4 — Calculate MA Slope**

        The change in the 30-week MA is calculated.

        **Step 5 — Determine Stage**

        The stock is classified as:

        🔵 **Stage 1 — Basing**

        🟢 **Stage 2 — Advancing**

        🟠 **Stage 3 — Topping**

        🔴 **Stage 4 — Declining**

        ⚪ **Indeterminate**

        **Step 6 — View Chart**

        The weekly price, 30-week MA and stage points
        are shown on the chart.

        **Step 7 — Download Data**

        The last 20 weekly observations can be downloaded
        as a CSV file.
        """
    )

    st.markdown("---")

    st.caption(
        "NSE STOCK ANALYZER BY SUJOY ROY"
    )
