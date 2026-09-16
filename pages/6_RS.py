# ============================================================
# 🔁 RESISTANCE BECAME SUPPORT ANALYZER
# ============================================================
# 👤 Developed by SUJOY ROY
# ============================================================
# Features:
# • NSE Stock Symbol input
# • START DATE input
# • END DATE = TODAY automatically
# • Resistance detection
# • Resistance → Support detection
# • Recent month highlighting
# • Interactive Plotly chart
# • Zoom In / Zoom Out
# • Pan
# • Reset Zoom
# • Range Slider
# • 1M / 3M / 6M / 1Y / ALL buttons
# • Recent month data
# • CSV download
# ============================================================

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.signal import argrelextrema


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resistance Became Support",
    page_icon="🔁",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🔁 Resistance Became Support Analyzer"
)

st.markdown(
    """
    Detect previous **Resistance levels that later become Support**
    using local highs and subsequent price behaviour.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Analysis Settings"
)


# ============================================================
# STOCK SYMBOL
# ============================================================

symbol = st.sidebar.text_input(
    "NSE Stock Symbol",
    value="CCL",
    help=(
        "Enter NSE symbol without .NS "
        "Example: CCL, RELIANCE, TCS, INFY"
    )
).strip().upper()


# ============================================================
# START DATE
# ============================================================

start_date = st.sidebar.date_input(
    "START DATE",
    value=pd.Timestamp(
        "2025-01-01"
    ).date()
)


# ============================================================
# END DATE = TODAY
# ============================================================

end_date = (
    pd.Timestamp
    .today()
    .normalize()
    .date()
)


st.sidebar.write(
    f"**END DATE:** "
    f"{end_date.strftime('%d-%m-%Y')}"
)


# ============================================================
# RESISTANCE DETECTION ORDER
# ============================================================

order = st.sidebar.slider(
    "Resistance Detection Order",
    min_value=2,
    max_value=15,
    value=5,
    help=(
        "Higher value filters out smaller "
        "local highs and identifies stronger levels."
    )
)


# ============================================================
# TOLERANCE
# ============================================================

tolerance_percent = st.sidebar.slider(
    "R → S Tolerance (%)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help=(
        "Price must return within this percentage "
        "of the previous resistance."
    )
)


tolerance = (
    tolerance_percent / 100
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.sidebar.button(
    "🔍 ANALYZE STOCK",
    type="primary",
    use_container_width=True
)


# ============================================================
# FUNCTION
# FIND RESISTANCE → SUPPORT
# ============================================================

def find_resistance_support(
    df,
    order=5,
    tolerance=0.01
):

    df = df.copy()

    # --------------------------------------------------------
    # Local highs
    # --------------------------------------------------------

    local_max_idx = argrelextrema(
        df["High"].values,
        np.greater_equal,
        order=order
    )[0]

    resistance_support = []

    # --------------------------------------------------------
    # Check each resistance
    # --------------------------------------------------------

    for idx in local_max_idx:

        resistance_date = (
            df.index[idx]
        )

        resistance_price = float(
            df.iloc[idx]["High"]
        )

        # ----------------------------------------------------
        # Future data
        # ----------------------------------------------------

        future_df = df.iloc[
            idx + 1:
        ]

        for future_idx in range(
            len(future_df)
        ):

            low = float(
                future_df.iloc[
                    future_idx
                ]["Low"]
            )

            # ------------------------------------------------
            # Avoid division by zero
            # ------------------------------------------------

            if resistance_price == 0:
                continue

            price_diff = (
                abs(
                    low - resistance_price
                )
                /
                resistance_price
            )

            # ------------------------------------------------
            # Price returns near resistance
            # ------------------------------------------------

            if price_diff <= tolerance:

                # --------------------------------------------
                # Get next 3 closes
                # --------------------------------------------

                close_series = (
                    future_df
                    .iloc[
                        future_idx:
                        future_idx + 3
                    ]["Close"]
                    .values
                )

                # --------------------------------------------
                # Confirm 3 rising closes
                # --------------------------------------------

                if len(close_series) == 3:

                    if (
                        close_series[1]
                        >
                        close_series[0]
                        and
                        close_series[2]
                        >
                        close_series[1]
                    ):

                        support_date = (
                            future_df.index[
                                future_idx
                            ]
                        )

                        resistance_support.append(
                            (
                                resistance_date.date(),
                                support_date.date(),
                                round(
                                    resistance_price,
                                    2
                                )
                            )
                        )

                        break

    return resistance_support


# ============================================================
# FUNCTION
# DOWNLOAD STOCK DATA
# ============================================================

def download_stock_data(
    symbol,
    start_date,
    end_date
):

    ticker = (
        symbol + ".NS"
    )

    try:

        data = yf.download(
            ticker,

            start=pd.Timestamp(
                start_date
            ).strftime(
                "%Y-%m-%d"
            ),

            end=(
                pd.Timestamp(
                    end_date
                )
                +
                pd.Timedelta(
                    days=1
                )
            ).strftime(
                "%Y-%m-%d"
            ),

            progress=False,

            auto_adjust=False,

            threads=False
        )

    except Exception as e:

        st.error(
            f"❌ Download failed: {e}"
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Empty data
    # --------------------------------------------------------

    if (
        data is None
        or data.empty
    ):

        return pd.DataFrame()

    # ========================================================
    # HANDLE MULTIINDEX
    # ========================================================

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        extracted = {}

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for column_name in (
            required_columns
        ):

            found = False

            # ------------------------------------------------
            # Search MultiIndex
            # ------------------------------------------------

            for col in data.columns:

                col_values = [
                    str(x)
                    .strip()
                    .lower()
                    for x in col
                ]

                if (
                    column_name.lower()
                    in col_values
                ):

                    extracted[
                        column_name
                    ] = data[col]

                    found = True

                    break

            if not found:

                return pd.DataFrame()

        data = pd.DataFrame(
            extracted,
            index=data.index
        )

    # ========================================================
    # NORMAL DATAFRAME
    # ========================================================

    else:

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        if not all(
            column in data.columns
            for column in required_columns
        ):

            return pd.DataFrame()

        data = data[
            required_columns
        ]


    # ========================================================
    # CONVERT NUMERIC
    # ========================================================

    for column in [
        "Open",
        "High",
        "Low",
        "Close"
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


    # ========================================================
    # CLEAN
    # ========================================================

    data = data.dropna()

    data.index = pd.to_datetime(
        data.index
    )

    data = data.sort_index()

    return data


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze:

    # ========================================================
    # CHECK SYMBOL
    # ========================================================

    if not symbol:

        st.warning(
            "⚠️ Please enter an NSE stock symbol."
        )

        st.stop()


    # ========================================================
    # CHECK DATE
    # ========================================================

    if (
        pd.Timestamp(start_date)
        >=
        pd.Timestamp(end_date)
    ):

        st.error(
            "❌ START DATE must be before END DATE."
        )

        st.stop()


    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    with st.spinner(
        f"📥 Downloading data for {symbol}..."
    ):

        df = download_stock_data(
            symbol,
            start_date,
            end_date
        )


    # ========================================================
    # CHECK DATA
    # ========================================================

    if df.empty:

        st.error(
            f"""
            ❌ No data found for **{symbol}**.

            Please check the NSE symbol
            and START DATE.
            """
        )

        st.stop()


    st.success(
        f"✅ {symbol} data downloaded successfully."
    )


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"📊 {symbol} Analysis"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Start Date",
            df.index.min().strftime(
                "%d-%m-%Y"
            )
        )


    with c2:

        st.metric(
            "Latest Date",
            df.index.max().strftime(
                "%d-%m-%Y"
            )
        )


    with c3:

        st.metric(
            "Trading Days",
            len(df)
        )


    with c4:

        st.metric(
            "Latest Close",
            f"₹{df['Close'].iloc[-1]:,.2f}"
        )


    # ========================================================
    # RECENT MONTH
    # ========================================================

    latest_date = (
        df.index.max()
    )


    recent_month_start = (
        latest_date.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )
    )


    recent_month_end = (
        recent_month_start
        +
        pd.offsets.MonthEnd(1)
    )


    # ========================================================
    # RECENT MONTH DISPLAY
    # ========================================================

    st.info(
        f"""
        📅 **Recent Month:**
        {recent_month_start.strftime('%d-%m-%Y')}
        →
        {latest_date.strftime('%d-%m-%Y')}
        """
    )


    # ========================================================
    # FIND RESISTANCE → SUPPORT
    # ========================================================

    levels = find_resistance_support(
        df,
        order=order,
        tolerance=tolerance
    )


    # ========================================================
    # LEVEL RESULTS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔁 Resistance → Support Levels"
    )


    if not levels:

        st.info(
            "No Resistance → Support conversion "
            "was detected for the selected period."
        )

    else:

        levels_df = pd.DataFrame(
            levels,
            columns=[
                "Resistance Date",
                "Support Date",
                "Price"
            ]
        )


        levels_df[
            "Resistance Date"
        ] = pd.to_datetime(
            levels_df[
                "Resistance Date"
            ]
        )


        levels_df[
            "Support Date"
        ] = pd.to_datetime(
            levels_df[
                "Support Date"
            ]
        )


        # ====================================================
        # RECENT LEVELS
        # ====================================================

        recent_levels_df = (
            levels_df[
                (
                    levels_df[
                        "Support Date"
                    ]
                    >=
                    recent_month_start
                )
                &
                (
                    levels_df[
                        "Support Date"
                    ]
                    <=
                    latest_date
                )
            ]
            .copy()
        )


        # ====================================================
        # RECENT LEVEL MESSAGE
        # ====================================================

        if not recent_levels_df.empty:

            st.success(
                f"""
                🔁 **{len(recent_levels_df)}**
                Resistance → Support conversion(s)
                detected during the recent month.
                """
            )

        else:

            st.info(
                "No new Resistance → Support conversion "
                "detected during the recent month."
            )


        # ====================================================
        # DISPLAY TABLE
        # ====================================================

        display_levels = (
            levels_df.copy()
        )


        display_levels[
            "Resistance Date"
        ] = display_levels[
            "Resistance Date"
        ].dt.strftime(
            "%d-%m-%Y"
        )


        display_levels[
            "Support Date"
        ] = display_levels[
            "Support Date"
        ].dt.strftime(
            "%d-%m-%Y"
        )


        display_levels[
            "Price"
        ] = display_levels[
            "Price"
        ].map(
            lambda x:
            f"₹{x:,.2f}"
        )


        st.dataframe(
            display_levels,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # INTERACTIVE CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📈 Interactive Price Chart"
    )


    fig = go.Figure()


    # ========================================================
    # CLOSE PRICE
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close Price",

            line=dict(
                width=2
            ),

            hovertemplate=
            "<b>Date:</b> %{x|%d-%m-%Y}"
            "<br><b>Close:</b> ₹%{y:.2f}"
            "<extra></extra>"
        )
    )


    # ========================================================
    # RECENT MONTH HIGHLIGHT
    # ========================================================

    fig.add_vrect(

        x0=recent_month_start,

        x1=recent_month_end,

        fillcolor="orange",

        opacity=0.15,

        line_width=0,

        annotation_text=
        "RECENT MONTH",

        annotation_position=
        "top left"
    )


    # ========================================================
    # RESISTANCE → SUPPORT LEVELS
    # ========================================================

    for (
        res_date,
        sup_date,
        price
    ) in levels:


        res_timestamp = (
            pd.Timestamp(
                res_date
            )
        )


        sup_timestamp = (
            pd.Timestamp(
                sup_date
            )
        )


        # ====================================================
        # HORIZONTAL LEVEL
        # ====================================================

        fig.add_hline(

            y=price,

            line_dash="dash",

            opacity=0.50,

            annotation_text=
            f"R→S ₹{price:.2f}",

            annotation_position=
            "right"
        )


        # ====================================================
        # RESISTANCE MARKER
        # ====================================================

        if (
            df.index.min()
            <=
            res_timestamp
            <=
            df.index.max()
        ):

            fig.add_trace(

                go.Scatter(

                    x=[
                        res_timestamp
                    ],

                    y=[
                        price
                    ],

                    mode=
                    "markers+text",

                    name=
                    "Resistance",

                    marker=dict(
                        symbol=
                        "triangle-up",
                        size=12
                    ),

                    text=[
                        f"R ₹{price:.2f}"
                    ],

                    textposition=
                    "top center",

                    hovertemplate=
                    "<b>Resistance</b>"
                    "<br>Date: "
                    "%{x|%d-%m-%Y}"
                    "<br>Price: "
                    "₹%{y:.2f}"
                    "<extra></extra>",

                    showlegend=False
                )
            )


        # ====================================================
        # SUPPORT MARKER
        # ====================================================

        if (
            df.index.min()
            <=
            sup_timestamp
            <=
            df.index.max()
        ):

            fig.add_trace(

                go.Scatter(

                    x=[
                        sup_timestamp
                    ],

                    y=[
                        price
                    ],

                    mode=
                    "markers+text",

                    name=
                    "Resistance → Support",

                    marker=dict(
                        symbol=
                        "triangle-down",
                        size=12
                    ),

                    text=[
                        f"R→S ₹{price:.2f}"
                    ],

                    textposition=
                    "bottom center",

                    hovertemplate=
                    "<b>Resistance → Support</b>"
                    "<br>Date: "
                    "%{x|%d-%m-%Y}"
                    "<br>Price: "
                    "₹%{y:.2f}"
                    "<extra></extra>",

                    showlegend=False
                )
            )


    # ========================================================
    # CHART LAYOUT
    # ========================================================

    fig.update_layout(

        title=dict(

            text=
            f"{symbol} — Resistance Became Support",

            x=0.5
        ),


        # ====================================================
        # X AXIS
        # ====================================================

        xaxis=dict(

            title="Date",

            showgrid=True,

            fixedrange=False,

            rangeslider=dict(
                visible=True
            ),

            rangeselector=dict(

                buttons=[

                    dict(
                        count=1,
                        label="1M",
                        step="month",
                        stepmode="backward"
                    ),

                    dict(
                        count=3,
                        label="3M",
                        step="month",
                        stepmode="backward"
                    ),

                    dict(
                        count=6,
                        label="6M",
                        step="month",
                        stepmode="backward"
                    ),

                    dict(
                        count=1,
                        label="1Y",
                        step="year",
                        stepmode="backward"
                    ),

                    dict(
                        step="all",
                        label="ALL"
                    )
                ]
            )
        ),


        # ====================================================
        # Y AXIS
        # ====================================================

        yaxis=dict(

            title="Price (₹)",

            showgrid=True,

            fixedrange=False
        ),


        # ====================================================
        # HOVER
        # ====================================================

        hovermode=
        "x unified",


        # ====================================================
        # SIZE
        # ====================================================

        height=700,


        margin=dict(
            l=60,
            r=60,
            t=90,
            b=90
        )
    )


    # ========================================================
    # DISPLAY CHART
    # ========================================================

    st.plotly_chart(

        fig,

        use_container_width=True,

        config={

            # Mouse wheel zoom
            "scrollZoom": True,

            # Remove Plotly logo
            "displaylogo": False,

            # Show useful tools
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "eraseshape"
            ],

            # Responsive
            "responsive": True
        }
    )


    # ========================================================
    # RECENT MONTH DATA
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📅 Recent Month Data — Latest First"
    )


    recent_data = df[
        (
            df.index
            >=
            recent_month_start
        )
        &
        (
            df.index
            <=
            latest_date
        )
    ].copy()


    recent_data = (
        recent_data
        .sort_index(
            ascending=False
        )
    )


    recent_display = recent_data[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ].copy()


    st.dataframe(

        recent_display.style.format(

            {
                "Open":
                "₹{:.2f}",

                "High":
                "₹{:.2f}",

                "Low":
                "₹{:.2f}",

                "Close":
                "₹{:.2f}"
            }
        ),

        use_container_width=True
    )


    # ========================================================
    # RECENT R → S TABLE
    # ========================================================

    if not levels_df.empty:

        st.markdown("---")

        st.subheader(
            "🔁 Recent Month R → S Conversions"
        )


        if recent_levels_df.empty:

            st.info(
                "No Resistance → Support conversion "
                "was detected during the recent month."
            )

        else:

            recent_r_s_display = (
                recent_levels_df
                .sort_values(
                    "Support Date",
                    ascending=False
                )
                .copy()
            )


            recent_r_s_display[
                "Resistance Date"
            ] = (
                recent_r_s_display[
                    "Resistance Date"
                ]
                .dt.strftime(
                    "%d-%m-%Y"
                )
            )


            recent_r_s_display[
                "Support Date"
            ] = (
                recent_r_s_display[
                    "Support Date"
                ]
                .dt.strftime(
                    "%d-%m-%Y"
                )
            )


            recent_r_s_display[
                "Price"
            ] = (
                recent_r_s_display[
                    "Price"
                ]
                .map(
                    lambda x:
                    f"₹{x:,.2f}"
                )
            )


            st.dataframe(
                recent_r_s_display,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # DOWNLOAD PRICE DATA
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💾 Download Analysis"
    )


    csv_df = df.copy()

    csv_df.index.name = "Date"

    csv_df = (
        csv_df
        .reset_index()
        .sort_values(
            "Date",
            ascending=False
        )
    )


    csv = (
        csv_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(

        label=
        "⬇️ Download Price Data CSV",

        data=csv,

        file_name=
        f"{symbol}_Resistance_Support.csv",

        mime=
        "text/csv",

        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD R → S DATA
    # ========================================================

    if levels:

        rs_csv_df = (
            levels_df
            .sort_values(
                "Support Date",
                ascending=False
            )
            .copy()
        )


        rs_csv_df[
            "Resistance Date"
        ] = (
            rs_csv_df[
                "Resistance Date"
            ]
            .dt.strftime(
                "%d-%m-%Y"
            )
        )


        rs_csv_df[
            "Support Date"
        ] = (
            rs_csv_df[
                "Support Date"
            ]
            .dt.strftime(
                "%d-%m-%Y"
            )
        )


        rs_csv = (
            rs_csv_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )


        st.download_button(

            label=
            "⬇️ Download R → S Levels CSV",

            data=rs_csv,

            file_name=
            f"{symbol}_R_to_S_Levels.csv",

            mime=
            "text/csv",

            use_container_width=True
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        """
        👈 Enter an **NSE Stock Symbol** and
        **START DATE** in the sidebar.

        Then click:

        **🔍 ANALYZE STOCK**
        """
    )


    st.markdown("---")


    st.subheader(
        "📚 How this scanner works"
    )


    st.markdown(
        """
        ### 1️⃣ Resistance Detection

        Local price highs are identified as potential
        resistance levels.

        ### 2️⃣ Retest

        The program checks whether price subsequently
        returns within the selected tolerance of that
        resistance.

        ### 3️⃣ Support Confirmation

        The next three closing prices must rise
        consecutively.

        ### 4️⃣ Resistance → Support

        When these conditions are satisfied, the previous
        resistance level is classified as having become
        support.

        ### 5️⃣ Recent Month

        The latest calendar month is highlighted on the
        interactive chart.

        ### 6️⃣ Interactive Chart

        You can:

        - 🔍 Zoom in
        - 🔎 Zoom out
        - 🖱️ Pan
        - 🏠 Reset zoom
        - 📅 Select 1M / 3M / 6M / 1Y / ALL
        - ↔️ Drag the range slider
        - 📱 Pinch zoom on mobile
        """
    )


    st.markdown("---")


    st.subheader(
        "⚙️ Current Detection Rules"
    )


    rules_df = pd.DataFrame(

        {
            "Parameter": [
                "Resistance",
                "Tolerance",
                "Support Confirmation"
            ],

            "Rule": [
                "Local High",
                "Default ±1%",
                "3 consecutive rising closes"
            ]
        }
    )


    st.dataframe(
        rules_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")


    st.caption(
        """
        ⚠️ This scanner uses predefined mathematical rules
        for technical analysis. A detected Resistance →
        Support level does not guarantee future price movement.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🔁 NSE STOCK ANALYZER BY SUJOY ROY | "
    "Resistance Became Support Analyzer"
)
