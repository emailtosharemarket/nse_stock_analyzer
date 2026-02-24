# 📈 NSE Stock Analyzer – Jackpot, ACC, BR Signals
# 👤 Developed by SUJOY ROY
# ✅ Updated: 24.02.2026 (Fully Error-Safe Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nselib import capital_market

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="NSE Stock Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NSE Stock Analyzer by SUJOY ROY")

st.markdown("""
This tool analyzes **NSE stock data (last 6 months)** using  
**Volume + Delivery + Trade behavior** and highlights:

- 🟢 **JACKPOT** – Strong convergence  
- 🟡 **ACC (G-LZ)** – Accumulation  
- 🔴 **BR (G-HZ)** – Breakout / Distribution  
""")

# ---------------- USER INPUT ---------------- #
NAME = st.text_input("Enter Stock Symbol (e.g. TCS, INFY, RELIANCE):")

# ---------------- MAIN LOGIC ---------------- #
if NAME:
    try:
        with st.spinner("Fetching data from NSE..."):

            # Fetch NSE data
            df = capital_market.price_volume_and_deliverable_position_data(
                NAME.upper(),
                period="6M"
            )

            if df.empty:
                st.error("No data returned from NSE.")
                st.stop()

            df = df.sort_index(ascending=False)

            # ---------------- COLUMN NORMALIZATION ---------------- #
            df.columns = df.columns.str.replace(" ", "", regex=True)

            # ---------------- SAFE VOLUME COLUMN DETECTION ---------------- #
            volume_candidates = [
                'TotalTradedQuantity',
                'TradedQty',
                'TOTTRDQTY'
            ]

            volume_col = None
            for col in volume_candidates:
                if col in df.columns:
                    volume_col = col
                    break

            if volume_col is None:
                st.error("Volume column not found in NSE response.")
                st.stop()

            df.rename(columns={volume_col: 'TotalTradedQuantity'}, inplace=True)

            # ---------------- REQUIRED COLUMNS (SAFE VERSION) ---------------- #
            required_cols = [
                'Date',
                'ClosePrice',
                'TotalTradedQuantity',
                'No.ofTrades',
                '%DlyQttoTradedQty'
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns from NSE: {missing_cols}")
                st.stop()

            df = df[required_cols].copy()

            # Add Symbol manually (NSE sometimes omits it)
            df['Symbol'] = NAME.upper()

            # ---------------- DATA CLEANING ---------------- #
            df.replace('-', np.nan, inplace=True)

            numeric_cols = [
                'TotalTradedQuantity',
                'No.ofTrades',
                'ClosePrice',
                '%DlyQttoTradedQty'
            ]

            df[numeric_cols] = (
                df[numeric_cols]
                .replace(",", "", regex=True)
                .apply(pd.to_numeric, errors='coerce')
            )

            df = df.dropna(subset=['ClosePrice'])

            # Avoid division by zero
            df['No.ofTrades'] = df['No.ofTrades'].replace(0, np.nan)

            # ---------------- DERIVED METRICS ---------------- #
            df['ACTION'] = (df['TotalTradedQuantity'] / df['No.ofTrades']).round(2)

            df['avgACTION'] = df['ACTION'].mean(skipna=True)
            df['avg%DEL'] = df['%DlyQttoTradedQty'].mean(skipna=True)

            df['%chngACT'] = (
                (df['ACTION'] - df['avgACTION']) / df['avgACTION'] * 100
            ).round(2)

            df['%chngDEL'] = (
                (df['%DlyQttoTradedQty'] - df['avg%DEL']) / df['avg%DEL'] * 100
            ).round(2)

            df['SHIFT'] = df['ClosePrice'].shift(1)

            df['%iCHANGE'] = (
                (df['SHIFT'] - df['ClosePrice']) / df['ClosePrice'] * 100
            ).round(2)

            df['%CHANGE'] = df['%iCHANGE'].shift(-1).round(2)

            # ---------------- REMARKS LOGIC ---------------- #
            conditions = [
                (df['%DlyQttoTradedQty'] > df['avg%DEL']) & (df['ACTION'] < df['avgACTION']),
                (df['%DlyQttoTradedQty'] < df['avg%DEL']) & (df['ACTION'] > df['avgACTION']),
                (df['%DlyQttoTradedQty'] > df['avg%DEL']) & (df['ACTION'] > df['avgACTION']),
                (df['%DlyQttoTradedQty'] < df['avg%DEL']) & (df['ACTION'] < df['avgACTION'])
            ]

            remarks = ['ACC(G-LZ)', 'BR(G-HZ)', 'JACKPOT', 'NA']
            df['REMARKS'] = np.select(conditions, remarks, default='NA')

            # ---------------- PLOT SECTION ---------------- #
            df_plot = df.copy()
            df_plot['Date'] = pd.to_datetime(
                df_plot['Date'],
                format='%d-%b-%Y',
                errors='coerce'
            )

            df_plot = df_plot.dropna(subset=['Date'])

            df_plot = df_plot.sort_values("Date")

            df_plot['30EMA'] = df_plot['ClosePrice'].ewm(
                span=30,
                adjust=False
            ).mean()

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(
                df_plot['Date'],
                df_plot['ClosePrice'],
                label='Close Price'
            )

            ax.plot(
                df_plot['Date'],
                df_plot['30EMA'],
                linestyle='--',
                label='30 EMA'
            )

            colors = {
                'JACKPOT': 'green',
                'ACC(G-LZ)': 'orange',
                'BR(G-HZ)': 'red'
            }

            for label, color in colors.items():
                temp = df_plot[df_plot['REMARKS'] == label]
                ax.scatter(
                    temp['Date'],
                    temp['ClosePrice'],
                    color=color,
                    label=label,
                    s=60,
                    zorder=3
                )

            ax.set_title(f"{NAME.upper()} – Close Price with 30 EMA & Signals")
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.grid(True)
            ax.legend()

            st.pyplot(fig)

            # ---------------- DATA TABLE ---------------- #
            st.subheader("📋 Last 50 Days Analysis")

            df_display = df[
                [
                    'Date',
                    '%chngACT',
                    '%chngDEL',
                    'REMARKS',
                    'ClosePrice',
                    '%CHANGE',
                    '%DlyQttoTradedQty'
                ]
            ].head(50)

            st.dataframe(
                df_display.style.background_gradient(
                    cmap="coolwarm",
                    subset=['%chngACT', '%chngDEL', '%DlyQttoTradedQty']
                )
            )

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")

else:
    st.info("Please enter a valid NSE stock symbol to begin.")
