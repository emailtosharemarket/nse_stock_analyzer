# ============================================================
# 📈 NSE STOCK ANALYZER
# Jackpot / ACC / BR Signals
# ============================================================
# 👤 Developed by SUJOY ROY
# ✅ Updated: 14-09-2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from nselib import capital_market

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

NAME = st.text_input(
    "Enter Stock Symbol (e.g. TCS, INFY, RELIANCE):"
).strip().upper()


def fetch_nse_data(symbol):
    today = datetime.now().date()
    start_date = today - timedelta(days=190)
    all_data = []
    current_start = start_date

    while current_start <= today:
        current_end = min(current_start + timedelta(days=27), today)
        from_date = current_start.strftime("%d-%m-%Y")
        to_date = current_end.strftime("%d-%m-%Y")

        try:
            temp = capital_market.price_volume_and_deliverable_position_data(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date
            )
            if temp is not None and not temp.empty:
                all_data.append(temp.copy())
        except Exception as e:
            st.warning(
                f"⚠️ NSE request failed for {from_date} → {to_date}: {e}"
            )

        current_start = current_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.drop_duplicates(subset=["Date"], keep="last")

    return df


if NAME:
    try:
        with st.spinner(f"Fetching NSE data for {NAME}..."):
            df = fetch_nse_data(NAME)

        if df.empty:
            st.error("❌ No NSE data received.")
            st.stop()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["Date"])

        df = df.sort_values("Date", ascending=False).reset_index(drop=True)

        latest_nse_date = df["Date"].max()
        oldest_nse_date = df["Date"].min()

        st.info(
            f"📅 NSE data received: "
            f"{oldest_nse_date.strftime('%d-%b-%Y')} → "
            f"{latest_nse_date.strftime('%d-%b-%Y')}"
        )

        df.columns = (
            df.columns.astype(str)
            .str.replace(" ", "", regex=True)
            .str.strip()
        )

        volume_candidates = [
            "TotalTradedQuantity", "TradedQty", "TOTTRDQTY", "TotalTradedQty"
        ]

        volume_col = next((col for col in volume_candidates if col in df.columns), None)

        if volume_col is None:
            st.error("❌ Total traded quantity column was not found in NSE response.")
            st.write("Available columns:", df.columns.tolist())
            st.stop()

        df.rename(columns={volume_col: "TotalTradedQuantity"}, inplace=True)

        required_cols = [
            "Date", "ClosePrice", "TotalTradedQuantity",
            "No.ofTrades", "%DlyQttoTradedQty"
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required NSE columns: {missing_cols}")
            st.write("Available columns:", df.columns.tolist())
            st.stop()

        df = df[required_cols].copy()
        df["Symbol"] = NAME

        df.replace(["-", "—", ""], np.nan, inplace=True)

        numeric_cols = [
            "ClosePrice", "TotalTradedQuantity",
            "No.ofTrades", "%DlyQttoTradedQty"
        ]

        for col in numeric_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Date", "ClosePrice"])
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
        df["No.ofTrades"] = df["No.ofTrades"].replace(0, np.nan)

        df["ACTION"] = (
            df["TotalTradedQuantity"] / df["No.ofTrades"]
        ).round(2)

        avgACTION = df["ACTION"].mean(skipna=True)
        avgDEL = df["%DlyQttoTradedQty"].mean(skipna=True)

        df["avgACTION"] = round(avgACTION, 2)
        df["avg%DEL"] = round(avgDEL, 2)

        df["%chngACT"] = (
            ((df["ACTION"] - avgACTION) / avgACTION * 100).round(2)
            if avgACTION != 0 else np.nan
        )

        df["%chngDEL"] = (
            ((df["%DlyQttoTradedQty"] - avgDEL) / avgDEL * 100).round(2)
            if avgDEL != 0 else np.nan
        )

        df["SHIFT"] = df["ClosePrice"].shift(1)
        df["%iCHANGE"] = (
            ((df["SHIFT"] - df["ClosePrice"]) / df["ClosePrice"] * 100).round(2)
        )
        df["%CHANGE"] = df["%iCHANGE"].shift(-1).round(2)

        conditions = [
            (df["%DlyQttoTradedQty"] > df["avg%DEL"]) & (df["ACTION"] < df["avgACTION"]),
            (df["%DlyQttoTradedQty"] < df["avg%DEL"]) & (df["ACTION"] > df["avgACTION"]),
            (df["%DlyQttoTradedQty"] > df["avg%DEL"]) & (df["ACTION"] > df["avgACTION"]),
            (df["%DlyQttoTradedQty"] < df["avg%DEL"]) & (df["ACTION"] < df["avgACTION"])
        ]

        df["REMARKS"] = np.select(
            conditions,
            ["ACC(G-LZ)", "BR(G-HZ)", "JACKPOT", "NA"],
            default="NA"
        )

        current = df.iloc[0]
        current_date = current["Date"]

        st.subheader("📌 Latest NSE Trading Day")
        st.success(f"📅 {current_date.strftime('%d-%b-%Y')}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close Price", f"₹{current['ClosePrice']:,.2f}")
        c2.metric("ACTION", f"{current['ACTION']:,.2f}")
        c3.metric("Delivery %", f"{current['%DlyQttoTradedQty']:.2f}%")
        c4.metric("ACTION vs Avg", f"{current['%chngACT']:.2f}%")
        c5.metric("DEL vs Avg", f"{current['%chngDEL']:.2f}%")

        signal = current["REMARKS"]

        if signal == "JACKPOT":
            st.success(f"🟢 JACKPOT — {NAME}")
        elif signal == "ACC(G-LZ)":
            st.warning(f"🟡 ACCUMULATION — {NAME}")
        elif signal == "BR(G-HZ)":
            st.error(f"🔴 BR / DISTRIBUTION — {NAME}")
        else:
            st.info(f"⚪ SIGNAL: {signal}")

        st.subheader("📊 Latest Trading Day Detailed Analysis")

        current_display = pd.DataFrame({
            "Date": [current_date.strftime("%d-%b-%Y")],
            "Close Price": [current["ClosePrice"]],
            "Total Traded Qty": [current["TotalTradedQuantity"]],
            "No. of Trades": [current["No.ofTrades"]],
            "ACTION": [current["ACTION"]],
            "Avg ACTION": [current["avgACTION"]],
            "% ACTION vs Avg": [current["%chngACT"]],
            "Delivery %": [current["%DlyQttoTradedQty"]],
            "Avg Delivery %": [current["avg%DEL"]],
            "% Delivery vs Avg": [current["%chngDEL"]],
            "SIGNAL": [current["REMARKS"]]
        })

        st.dataframe(current_display, use_container_width=True)

        st.subheader("📈 Price, 30 EMA & Signals")

        df_plot = df.sort_values("Date").copy()
        df_plot["30EMA"] = df_plot["ClosePrice"].ewm(span=30, adjust=False).mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_plot["Date"], df_plot["ClosePrice"], label="Close Price")
        ax.plot(df_plot["Date"], df_plot["30EMA"], linestyle="--", label="30 EMA")

        signal_colors = {
            "JACKPOT": "green",
            "ACC(G-LZ)": "orange",
            "BR(G-HZ)": "red"
        }

        for label, color in signal_colors.items():
            temp = df_plot[df_plot["REMARKS"] == label]
            ax.scatter(
                temp["Date"], temp["ClosePrice"],
                color=color, s=60, label=label, zorder=3
            )

        ax.scatter(
            current_date, current["ClosePrice"],
            color="black", marker="*", s=180,
            zorder=5, label="Latest Trading Day"
        )

        ax.set_title(f"{NAME} – Close Price with 30 EMA & Signals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("📋 Last 50 Trading Days Analysis")

        df_display = df[
            ["Date", "%chngACT", "%chngDEL", "REMARKS",
             "ClosePrice", "%CHANGE", "%DlyQttoTradedQty"]
        ].head(50).copy()

        df_display["Date"] = df_display["Date"].dt.strftime("%d-%b-%Y")

        st.dataframe(
            df_display.style.background_gradient(
                cmap="coolwarm",
                subset=["%chngACT", "%chngDEL", "%DlyQttoTradedQty"]
            ),
            use_container_width=True
        )

        with st.expander("🔧 NSE Data Debug Information"):
            st.write("nselib version: 2.5.1")
            st.write("Rows received:", len(df))
            st.write("Oldest NSE date:", oldest_nse_date.strftime("%d-%b-%Y"))
            st.write("Latest NSE date:", latest_nse_date.strftime("%d-%b-%Y"))
            st.write("Available columns:", df.columns.tolist())

    except Exception as e:
        st.error(f"⚠️ Error fetching NSE data: {e}")
        st.exception(e)
else:
    st.info("Please enter a valid NSE stock symbol to begin.")
