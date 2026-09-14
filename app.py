# ============================================================
# 📈 NSE STOCK ANALYZER
# Jackpot / ACC / BR Signals
# ============================================================
# 👤 Developed by SUJOY ROY
# ✅ Updated: 14-09-2026
#
# IMPORTANT:
# 1. Historical data is downloaded in smaller date ranges.
# 2. This avoids depending on one "6M" period request.
# 3. Latest available NSE trading date is automatically detected.
# 4. Current/latest signal is displayed separately.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from nselib import capital_market


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NSE Stock Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NSE Stock Analyzer by SUJOY ROY")

st.markdown("""
### Signal Logic

🟢 **JACKPOT** → Delivery ↑ + ACTION ↑

🟡 **ACC(G-LZ)** → Delivery ↑ + ACTION ↓

🔴 **BR(G-HZ)** → Delivery ↓ + ACTION ↑

⚪ **NA** → Delivery ↓ + ACTION ↓
""")


# ============================================================
# USER INPUT
# ============================================================

NAME = st.text_input(
    "Enter Stock Symbol (e.g. TCS, INFY, RELIANCE):"
).strip().upper()


# ============================================================
# FUNCTION: FETCH DATA IN MONTHLY CHUNKS
# ============================================================

def fetch_nse_data(symbol):

    today = datetime.now().date()

    # --------------------------------------------------------
    # Start date = approximately 6 months back
    # --------------------------------------------------------

    start_date = today - timedelta(days=190)

    all_data = []

    current_start = start_date

    while current_start <= today:

        # Maximum 28 days per request
        current_end = min(
            current_start + timedelta(days=27),
            today
        )

        from_date = current_start.strftime(
            "%d-%m-%Y"
        )

        to_date = current_end.strftime(
            "%d-%m-%Y"
        )

        try:

            temp = (
                capital_market
                .price_volume_and_deliverable_position_data(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date
                )
            )

            if temp is not None and not temp.empty:

                temp = temp.copy()

                all_data.append(temp)

        except Exception as e:

            # Continue with next date range
            st.warning(
                f"⚠️ NSE request failed for "
                f"{from_date} → {to_date}: {e}"
            )

        current_start = (
            current_end +
            timedelta(days=1)
        )


    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if not all_data:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Combine all chunks
    # --------------------------------------------------------

    df = pd.concat(
        all_data,
        ignore_index=True
    )


    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
            dayfirst=True
        )

        df = df.drop_duplicates(
            subset=["Date"],
            keep="last"
        )

    return df


# ============================================================
# MAIN PROGRAM
# ============================================================

if NAME:

    try:

        with st.spinner(
            f"Fetching NSE data for {NAME}..."
        ):

            # =================================================
            # FETCH DATA
            # =================================================

            df = fetch_nse_data(NAME)


        # =====================================================
        # CHECK DATA
        # =====================================================

        if df.empty:

            st.error(
                "❌ No NSE data received."
            )

            st.stop()


        # =====================================================
        # SHOW RAW DATA DATE RANGE
        # =====================================================

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce",
                dayfirst=True
            )

            df = df.dropna(
                subset=["Date"]
            )


        # =====================================================
        # SORT NEWEST FIRST
        # =====================================================

        df = df.sort_values(
            "Date",
            ascending=False
        ).reset_index(drop=True)


        # =====================================================
        # DISPLAY DATA RANGE
        # =====================================================

        latest_nse_date = df["Date"].max()
        oldest_nse_date = df["Date"].min()

        st.info(
            f"📅 NSE data received: "
            f"{oldest_nse_date.strftime('%d-%b-%Y')} "
            f"→ "
            f"{latest_nse_date.strftime('%d-%b-%Y')}"
        )


        # =====================================================
        # COLUMN NORMALIZATION
        # =====================================================

        df.columns = (
            df.columns
            .astype(str)
            .str.replace(" ", "", regex=True)
            .str.strip()
        )


        # =====================================================
        # VOLUME COLUMN DETECTION
        # =====================================================

        volume_candidates = [
            "TotalTradedQuantity",
            "TradedQty",
            "TOTTRDQTY",
            "TotalTradedQty"
        ]

        volume_col = None

        for col in volume_candidates:

            if col in df.columns:

                volume_col = col
                break


        if volume_col is None:

            st.error(
                "❌ Total traded quantity column "
                "was not found in NSE response."
            )

            st.write(
                "Available columns:",
                df.columns.tolist()
            )

            st.stop()


        df.rename(
            columns={
                volume_col:
                "TotalTradedQuantity"
            },
            inplace=True
        )


        # =====================================================
        # REQUIRED COLUMNS
        # =====================================================

        required_cols = [
            "Date",
            "ClosePrice",
            "TotalTradedQuantity",
            "No.ofTrades",
            "%DlyQttoTradedQty"
        ]


        missing_cols = [
            col
            for col in required_cols
            if col not in df.columns
        ]


        if missing_cols:

            st.error(
                f"❌ Missing required NSE columns: "
                f"{missing_cols}"
            )

            st.write(
                "Available columns:",
                df.columns.tolist()
            )

            st.stop()


        df = df[
            required_cols
        ].copy()


        # =====================================================
        # SYMBOL
        # =====================================================

        df["Symbol"] = NAME


        # =====================================================
        # DATA CLEANING
        # =====================================================

        df.replace(
            ["-", "—", ""],
            np.nan,
            inplace=True
        )


        numeric_cols = [
            "ClosePrice",
            "TotalTradedQuantity",
            "No.ofTrades",
            "%DlyQttoTradedQty"
        ]


        for col in numeric_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(
                    ",",
                    "",
                    regex=False
                )
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        # =====================================================
        # REMOVE INVALID ROWS
        # =====================================================

        df = df.dropna(
            subset=[
                "Date",
                "ClosePrice"
            ]
        )


        # =====================================================
        # SORT NEWEST FIRST
        # =====================================================

        df = df.sort_values(
            "Date",
            ascending=False
        ).reset_index(drop=True)


        # =====================================================
        # AVOID DIVISION BY ZERO
        # =====================================================

        df["No.ofTrades"] = (
            df["No.ofTrades"]
            .replace(0, np.nan)
        )


        # =====================================================
        # ACTION
        # =====================================================

        df["ACTION"] = (
            df["TotalTradedQuantity"] /
            df["No.ofTrades"]
        ).round(2)


        # =====================================================
        # HISTORICAL AVERAGES
        # =====================================================

        avgACTION = df[
            "ACTION"
        ].mean(
            skipna=True
        )


        avgDEL = df[
            "%DlyQttoTradedQty"
        ].mean(
            skipna=True
        )


        df["avgACTION"] = round(
            avgACTION,
            2
        )


        df["avg%DEL"] = round(
            avgDEL,
            2
        )


        # =====================================================
        # ACTION % CHANGE
        # =====================================================

        if avgACTION != 0:

            df["%chngACT"] = (
                (
                    df["ACTION"] -
                    avgACTION
                )
                /
                avgACTION
                *
                100
            ).round(2)

        else:

            df["%chngACT"] = np.nan


        # =====================================================
        # DELIVERY % CHANGE
        # =====================================================

        if avgDEL != 0:

            df["%chngDEL"] = (
                (
                    df["%DlyQttoTradedQty"] -
                    avgDEL
                )
                /
                avgDEL
                *
                100
            ).round(2)

        else:

            df["%chngDEL"] = np.nan


        # =====================================================
        # PRICE CHANGE
        # =====================================================

        df["SHIFT"] = (
            df["ClosePrice"]
            .shift(1)
        )


        df["%iCHANGE"] = (
            (
                df["SHIFT"] -
                df["ClosePrice"]
            )
            /
            df["ClosePrice"]
            *
            100
        ).round(2)


        df["%CHANGE"] = (
            df["%iCHANGE"]
            .shift(-1)
            .round(2)
        )


        # =====================================================
        # SIGNAL LOGIC
        # =====================================================

        conditions = [

            # ACC
            (
                (df["%DlyQttoTradedQty"] >
                 df["avg%DEL"])
                &
                (df["ACTION"] <
                 df["avgACTION"])
            ),

            # BR
            (
                (df["%DlyQttoTradedQty"] <
                 df["avg%DEL"])
                &
                (df["ACTION"] >
                 df["avgACTION"])
            ),

            # JACKPOT
            (
                (df["%DlyQttoTradedQty"] >
                 df["avg%DEL"])
                &
                (df["ACTION"] >
                 df["avgACTION"])
            ),

            # NA
            (
                (df["%DlyQttoTradedQty"] <
                 df["avg%DEL"])
                &
                (df["ACTION"] <
                 df["avgACTION"])
            )
        ]


        remarks = [
            "ACC(G-LZ)",
            "BR(G-HZ)",
            "JACKPOT",
            "NA"
        ]


        df["REMARKS"] = np.select(
            conditions,
            remarks,
            default="NA"
        )


        # =====================================================
        # CURRENT / LATEST TRADING DAY
        # =====================================================

        current = df.iloc[0]

        current_date = current["Date"]


        # =====================================================
        # CURRENT DAY HEADER
        # =====================================================

        st.subheader(
            "📌 Latest NSE Trading Day"
        )


        st.success(
            f"📅 {current_date.strftime('%d-%b-%Y')}"
        )


        # =====================================================
        # CURRENT DAY METRICS
        # =====================================================

        c1, c2, c3, c4, c5 = st.columns(5)


        c1.metric(
            "Close Price",
            f"₹{current['ClosePrice']:,.2f}"
        )


        c2.metric(
            "ACTION",
            f"{current['ACTION']:,.2f}"
        )


        c3.metric(
            "Delivery %",
            f"{current['%DlyQttoTradedQty']:.2f}%"
        )


        c4.metric(
            "ACTION vs Avg",
            f"{current['%chngACT']:.2f}%"
        )


        c5.metric(
            "DEL vs Avg",
            f"{current['%chngDEL']:.2f}%"
        )


        # =====================================================
        # CURRENT SIGNAL
        # =====================================================

        signal = current["REMARKS"]


        if signal == "JACKPOT":

            st.success(
                f"🟢 JACKPOT — {NAME}"
            )


        elif signal == "ACC(G-LZ)":

            st.warning(
                f"🟡 ACCUMULATION — {NAME}"
            )


        elif signal == "BR(G-HZ)":

            st.error(
                f"🔴 BR / DISTRIBUTION — {NAME}"
            )


        else:

            st.info(
                f"⚪ SIGNAL: {signal}"
            )


        # =====================================================
        # CURRENT DAY TABLE
        # =====================================================

        st.subheader(
            "📊 Latest Trading Day Detailed Analysis"
        )


        current_display = pd.DataFrame({

            "Date": [
                current_date.strftime(
                    "%d-%b-%Y"
                )
            ],

            "Close Price": [
                current["ClosePrice"]
            ],

            "Total Traded Qty": [
                current["TotalTradedQuantity"]
            ],

            "No. of Trades": [
                current["No.ofTrades"]
            ],

            "ACTION": [
                current["ACTION"]
            ],

            "Avg ACTION": [
                current["avgACTION"]
            ],

            "% ACTION vs Avg": [
                current["%chngACT"]
            ],

            "Delivery %": [
                current[
                    "%DlyQttoTradedQty"
                ]
            ],

            "Avg Delivery %": [
                current["avg%DEL"]
            ],

            "% Delivery vs Avg": [
                current["%chngDEL"]
            ],

            "SIGNAL": [
                current["REMARKS"]
            ]
        })


        st.dataframe(
            current_display,
            use_container_width=True
        )


        # =====================================================
        # PLOT
        # =====================================================

        df_plot = df.sort_values(
            "Date"
        ).copy()


        df_plot["30EMA"] = (
            df_plot["ClosePrice"]
            .ewm(
                span=30,
                adjust=False
            )
            .mean()
        )


        fig, ax = plt.subplots(
            figsize=(12, 6)
        )


        ax.plot(
            df_plot["Date"],
            df_plot["ClosePrice"],
            label="Close Price"
        )


        ax.plot(
            df_plot["Date"],
            df_plot["30EMA"],
            linestyle="--",
            label="30 EMA"
        )


        signal_colors = {

            "JACKPOT": "green",

            "ACC(G-LZ)": "orange",

            "BR(G-HZ)": "red"
        }


        for label, color in signal_colors.items():

            temp = df_plot[
                df_plot["REMARKS"] ==
                label
            ]


            ax.scatter(
                temp["Date"],
                temp["ClosePrice"],
                color=color,
                s=60,
                label=label,
                zorder=3
            )


        # Highlight latest trading day

        ax.scatter(
            current_date,
            current["ClosePrice"],
            color="black",
            marker="*",
            s=180,
            zorder=5,
            label="Latest Trading Day"
        )


        ax.set_title(
            f"{NAME} – Close Price "
            f"with 30 EMA & Signals"
        )


        ax.set_xlabel(
            "Date"
        )


        ax.set_ylabel(
            "Close Price"
        )


        ax.grid(True)

        ax.legend()


        st.pyplot(
            fig,
            use_container_width=True
        )


        # =====================================================
        # LAST 50 DAYS
        # =====================================================

        st.subheader(
            "📋 Last 50 Trading Days Analysis"
        )


        df_display = df[
            [
                "Date",
                "%chngACT",
                "%chngDEL",
                "REMARKS",
                "ClosePrice",
                "%CHANGE",
                "%DlyQttoTradedQty"
            ]
        ].head(50).copy()


        df_display["Date"] = (
            df_display["Date"]
            .dt.strftime(
                "%d-%b-%Y"
            )
        )


        st.dataframe(
            df_display.style.background_gradient(
                cmap="coolwarm",
                subset=[
                    "%chngACT",
                    "%chngDEL",
                    "%DlyQttoTradedQty"
                ]
            ),
            use_container_width=True
        )


        # =====================================================
        # DEBUG / DATA STATUS
        # =====================================================

        with st.expander(
            "🔧 NSE Data Debug Information"
        ):

            st.write(
                "nselib version: 2.5.1"
            )

            st.write(
                "Rows received:",
                len(df)
            )

            st.write(
                "Oldest NSE date:",
                oldest_nse_date.strftime(
                    "%d-%b-%Y"
                )
            )

            st.write(
                "Latest NSE date:",
                latest_nse_date.strftime(
                    "%d-%b-%Y"
                )
            )

            st.write(
                "Available columns:",
                df.columns.tolist()
            )


    except Exception as e:

        st.error(
            f"⚠️ Error fetching NSE data: {e}"
        )

        st.exception(e)


else:

    st.info(
        "Please enter a valid NSE stock symbol to begin."
    )
