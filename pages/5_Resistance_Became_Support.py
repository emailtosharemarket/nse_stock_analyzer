# ============================================================
# 🔁 RESISTANCE BECAME SUPPORT ANALYZER
# ============================================================
# 👤 Developed by SUJOY ROY
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

st.title("🔁 Resistance Became Support Analyzer")

st.markdown(
    """
    Detect previous **Resistance levels that later become Support**
    using local price highs and subsequent price behaviour.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")


# ------------------------------------------------------------
# STOCK SYMBOL
# ------------------------------------------------------------

symbol = st.sidebar.text_input(
    "NSE Stock Symbol",
    value="CCL",
    help="Example: CCL, RELIANCE, TCS, INFY"
).strip().upper()


# ------------------------------------------------------------
# START DATE
# ------------------------------------------------------------

start_date = st.sidebar.date_input(
    "START DATE",
    value=pd.Timestamp("2025-01-01").date()
)


# ------------------------------------------------------------
# END DATE = TODAY
# ------------------------------------------------------------

end_date = pd.Timestamp.today().normalize().date()

st.sidebar.write(
    f"**END DATE:** {end_date.strftime('%d-%m-%Y')}"
)


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

order = st.sidebar.slider(
    "Resistance Detection Order",
    min_value=2,
    max_value=15,
    value=5,
    help="Higher value = stronger/local resistance filtering"
)


tolerance_percent = st.sidebar.slider(
    "Resistance → Support Tolerance (%)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1
)


tolerance = tolerance_percent / 100


# ------------------------------------------------------------
# ANALYZE BUTTON
# ------------------------------------------------------------

analyze = st.sidebar.button(
    "🔍 ANALYZE STOCK",
    type="primary",
    use_container_width=True
)


# ============================================================
# FUNCTION: FIND RESISTANCE → SUPPORT
# ============================================================

def find_resistance_support(
    df,
    order=5,
    tolerance=0.01
):

    df = df.copy()

    # --------------------------------------------------------
    # Local highs = resistance
    # --------------------------------------------------------

    local_max_idx = argrelextrema(
        df["High"].values,
        np.greater_equal,
        order=order
    )[0]

    resistance_support = []

    for idx in local_max_idx:

        resistance_date = df.index[idx]

        resistance_price = float(
            df.iloc[idx]["High"]
        )

        # ----------------------------------------------------
        # Look ahead
        # ----------------------------------------------------

        future_df = df.iloc[idx + 1:]

        for future_idx in range(
            len(future_df)
        ):

            low = float(
                future_df.iloc[
                    future_idx
                ]["Low"]
            )

            if resistance_price == 0:
                continue

            price_diff = (
                abs(
                    low - resistance_price
                )
                / resistance_price
            )

            # ------------------------------------------------
            # Price comes near resistance
            # ------------------------------------------------

            if price_diff <= tolerance:

                close_series = (
                    future_df
                    .iloc[
                        future_idx:
                        future_idx + 3
                    ]["Close"]
                    .values
                )

                # ------------------------------------------------
                # Three consecutive rising closes
                # ------------------------------------------------

                if len(close_series) == 3:

                    if (
                        close_series[1]
                        > close_series[0]
                        and
                        close_series[2]
                        > close_series[1]
                    ):

                        support_date = (
                            future_df
                            .index[
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
# FUNCTION: GET YFINANCE DATA
# ============================================================

def download_stock_data(
    symbol,
    start_date,
    end_date
):

    ticker = symbol + ".NS"

    try:

        data = yf.download(
            ticker,
            start=pd.Timestamp(
                start_date
            ).strftime("%Y-%m-%d"),

            end=(
                pd.Timestamp(end_date)
                + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d"),

            progress=False,
            auto_adjust=False,
            threads=False
        )

    except Exception as e:

        st.error(
            f"❌ Download failed: {e}"
        )

        return pd.DataFrame()

    if data is None or data.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Handle yfinance MultiIndex
    # --------------------------------------------------------

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        extracted = {}

        for column_name in [
            "Open",
            "High",
            "Low",
            "Close"
        ]:

            found = False

            for col in data.columns:

                col_values = [
                    str(x).strip().lower()
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

    else:

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        if not all(
            col in data.columns
            for col in required
        ):

            return pd.DataFrame()

        data = data[required]

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for col in [
        "Open",
        "High",
        "Low",
        "Close"
    ]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

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

    # --------------------------------------------------------
    # Check symbol
    # --------------------------------------------------------

    if not symbol:

        st.warning(
            "⚠️ Please enter an NSE stock symbol."
        )

        st.stop()

    # --------------------------------------------------------
    # Check date
    # --------------------------------------------------------

    if pd.Timestamp(
        start_date
    ) >= pd.Timestamp(
        end_date
    ):

        st.error(
            "❌ START DATE must be before END DATE."
        )

        st.stop()

    # --------------------------------------------------------
    # Download message
    # --------------------------------------------------------

    with st.spinner(
        f"📥 Downloading data for {symbol}..."
    ):

        df = download_stock_data(
            symbol,
            start_date,
            end_date
        )

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if df.empty:

        st.error(
            f"""
            ❌ No data found for **{symbol}**.

            Please check the NSE symbol and date range.
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

    latest_date = df.index.max()

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
        + pd.offsets.MonthEnd(1)
    )

    # ========================================================
    # FIND LEVELS
    # ========================================================

    levels = find_resistance_support(
        df,
        order=order,
        tolerance=tolerance
    )

    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔁 Resistance → Support Levels"
    )

    if not levels:

        st.info(
            "No Resistance → Support conversion "
            "was detected."
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

        levels_df["Resistance Date"] = (
            pd.to_datetime(
                levels_df[
                    "Resistance Date"
                ]
            )
        )

        levels_df["Support Date"] = (
            pd.to_datetime(
                levels_df[
                    "Support Date"
                ]
            )
        )

        # ----------------------------------------------------
        # Recent month levels
        # ----------------------------------------------------

        recent_levels_df = (
            levels_df[
                (
                    levels_df[
                        "Support Date"
                    ] >= recent_month_start
                )
                &
                (
                    levels_df[
                        "Support Date"
                    ] <= latest_date
                )
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Recent result
        # ----------------------------------------------------

        if not recent_levels_df.empty:

            st.success(
                f"🔁 {len(recent_levels_df)} "
                f"Resistance → Support conversion(s) "
                f"detected in the recent month."
            )

        else:

            st.info(
                "No new Resistance → Support conversion "
                "detected in the recent month."
            )

        # ----------------------------------------------------
        # Display all levels
        # ----------------------------------------------------

        display_levels = levels_df.copy()

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
            lambda x: f"₹{x:,.2f}"
        )

        st.dataframe(
            display_levels,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📈 Price Chart — Resistance → Support"
    )

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    # --------------------------------------------------------
    # Close price
    # --------------------------------------------------------

    ax.plot(
        df.index,
        df["Close"],
        linewidth=1.8,
        label="Close Price"
    )

    # --------------------------------------------------------
    # Highlight recent month
    # --------------------------------------------------------

    ax.axvspan(
        recent_month_start,
        recent_month_end,
        alpha=0.20,
        label="Recent Month"
    )

    # --------------------------------------------------------
    # Resistance → Support levels
    # --------------------------------------------------------

    for (
        res_date,
        sup_date,
        price
    ) in levels:

        res_timestamp = pd.Timestamp(
            res_date
        )

        sup_timestamp = pd.Timestamp(
            sup_date
        )

        # Horizontal resistance/support
        ax.axhline(
            price,
            linestyle="--",
            alpha=0.45
        )

        # Resistance marker
        if (
            df.index.min()
            <= res_timestamp
            <= df.index.max()
        ):

            ax.scatter(
                res_timestamp,
                price,
                marker="^",
                s=90,
                zorder=5
            )

            ax.annotate(
                f"R ₹{price:.2f}",
                xy=(
                    res_timestamp,
                    price
                ),
                xytext=(
                    0,
                    12
                ),
                textcoords="offset points",
                fontsize=9
            )

        # Support marker
        if (
            df.index.min()
            <= sup_timestamp
            <= df.index.max()
        ):

            ax.scatter(
                sup_timestamp,
                price,
                marker="v",
                s=90,
                zorder=5
            )

            ax.annotate(
                f"R→S ₹{price:.2f}",
                xy=(
                    sup_timestamp,
                    price
                ),
                xytext=(
                    0,
                    -18
                ),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

    # --------------------------------------------------------
    # Recent month label
    # --------------------------------------------------------

    ax.axvline(
        recent_month_start,
        linestyle=":",
        linewidth=1.5
    )

    ax.text(
        recent_month_start,
        0.97,
        "RECENT MONTH",
        transform=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="bold",
        verticalalignment="top"
    )

    # --------------------------------------------------------
    # Chart formatting
    # --------------------------------------------------------

    ax.set_title(
        f"{symbol} — Resistance Became Support",
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
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ========================================================
    # RECENT MONTH DETAILS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📅 Recent Month"
    )

    st.write(
        f"**{recent_month_start.strftime('%d-%m-%Y')} "
        f"→ "
        f"{latest_date.strftime('%d-%m-%Y')}**"
    )

    recent_data = df[
        (
            df.index >= recent_month_start
        )
        &
        (
            df.index <= latest_date
        )
    ].copy()

    st.dataframe(
        recent_data[
            [
                "Open",
                "High",
                "Low",
                "Close"
            ]
        ].sort_index(
            ascending=False
        ).style.format(
            {
                "Open": "₹{:.2f}",
                "High": "₹{:.2f}",
                "Low": "₹{:.2f}",
                "Close": "₹{:.2f}"
            }
        ),
        use_container_width=True
    )

    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💾 Download Analysis"
    )

    csv_df = df.copy()

    csv_df.index.name = "Date"

    csv_df = csv_df.reset_index()

    csv = csv_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download Price Data CSV",
        data=csv,
        file_name=(
            f"{symbol}_Resistance_Support.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        """
        👈 Enter an **NSE Stock Symbol** and
        **START DATE** in the sidebar, then click
        **🔍 ANALYZE STOCK**.
        """
    )

    st.markdown("---")

    st.subheader(
        "📚 How this scanner works"
    )

    st.markdown(
        """
        **1️⃣ Resistance Detection**

        Local price highs are identified as potential
        resistance levels.

        **2️⃣ Retest**

        The program checks whether price subsequently
        returns within the selected tolerance of that
        resistance.

        **3️⃣ Support Confirmation**

        The next three closing prices must rise
        consecutively.

        **4️⃣ Resistance → Support**

        When these conditions are satisfied, the old
        resistance is classified as having become support.

        **5️⃣ Recent Month**

        The latest calendar month is highlighted on the
        chart so that recent conversions are easy to identify.
        """
    )

    st.markdown("---")

    st.caption(
        """
        ⚠️ This is a technical-analysis scanner based on
        predefined mathematical rules. It does not guarantee
        future price movement.
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
