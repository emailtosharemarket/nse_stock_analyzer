# 📈 NSE Stock Analyzer – Jackpot, ACC, BR Signals
# ✅ UPDATED ON 29.04.2025

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nselib import capital_market

st.set_page_config(page_title="NSE Stock Analyzer", page_icon="📊", layout="wide")

st.title("📊 NSE Stock Analyzer – Jackpot, ACC & BR Signals")

st.markdown(
    """
    This tool analyzes **NSE stock data** (last 6 months) using **volume–trade patterns**  
    and highlights signals such as:
    - 🟢 **JACKPOT** – strong convergence of delivery & action  
    - 🟡 **ACC(G-LZ)** – accumulation phase  
    - 🔴 **BR(G-HZ)** – distribution or breakout  
    """
)

# ---------------- USER INPUT ---------------- #
NAME = st.text_input("Enter Stock Symbol (e.g. TCS, INFY, RELIANCE):")

if NAME:
    try:
        with st.spinner("Fetching data from NSE..."):
            df = capital_market.price_volume_and_deliverable_position_data(NAME, period='6M')
            df = df.sort_index(ascending=False)

            # Clean & convert
            df = df[['Symbol', 'Series', 'Date', 'ClosePrice', 'TotalTradedQuantity', 'No.ofTrades', '%DlyQttoTradedQty']]
            df.replace('-', np.nan, inplace=True)
            numeric_cols = ['TotalTradedQuantity', 'No.ofTrades', 'ClosePrice', '%DlyQttoTradedQty']
            df[numeric_cols] = df[numeric_cols].replace(",", "", regex=True).apply(pd.to_numeric, errors='coerce').round(2)

            # Calculate derived columns
            df['ACTION'] = df['TotalTradedQuantity'].div(df['No.ofTrades']).round(2)
            df['avgACTION'] = df['ACTION'].mean(skipna=True).round(2)
            df['avg%DEL'] = df['%DlyQttoTradedQty'].mean(skipna=True).round(2)
            df['%chngACT'] = ((df['ACTION'] - df['avgACTION']) / df['avgACTION'] * 100).round(2)
            df['%chngDEL'] = ((df['%DlyQttoTradedQty'] - df['avg%DEL']) / df['avg%DEL'] * 100).round(2)
            df['SHIFT'] = df['ClosePrice'].shift(1)
            df['%iCHANGE'] = ((df['SHIFT'] - df['ClosePrice']) / df['ClosePrice'] * 100).round(2)
            df['%CHANGE'] = df['%iCHANGE'].shift(-1).round(2)

            # Remarks logic
            conditions = [
                (df['%DlyQttoTradedQty'] > df['avg%DEL']) & (df['ACTION'] < df['avgACTION']),
                (df['%DlyQttoTradedQty'] < df['avg%DEL']) & (df['ACTION'] > df['avgACTION']),
                (df['%DlyQttoTradedQty'] > df['avg%DEL']) & (df['ACTION'] > df['avgACTION']),
                (df['%DlyQttoTradedQty'] < df['avg%DEL']) & (df['ACTION'] < df['avgACTION'])
            ]
            remarks = ['ACC(G-LZ)', 'BR(G-HZ)', 'JACKPOT', 'NA']
            df['REMARKS'] = np.select(conditions, remarks, default='NA')

            # ------------- PLOT SECTION -------------
            df_plot = df.dropna(subset=['ClosePrice']).copy()
            df_plot['Date'] = pd.to_datetime(df_plot['Date'], format='%d-%b-%Y', errors='coerce')
            df_plot['30EMA'] = df_plot['ClosePrice'].ewm(span=30, adjust=False).mean()

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_plot['Date'], df_plot['ClosePrice'], label='Close Price', color='blue')
            ax.plot(df_plot['Date'], df_plot['30EMA'], label='30 EMA', color='orange', linestyle='--')

            colors = {'JACKPOT': 'green', 'ACC(G-LZ)': 'yellow', 'BR(G-HZ)': 'red'}
            for label, color in colors.items():
                temp = df_plot[df_plot['REMARKS'] == label]
                ax.scatter(temp['Date'], temp['ClosePrice'], color=color, label=label, s=50, zorder=3)

            ax.set_title(f'{NAME.upper()} – Close Price with 30EMA & Signals')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # ------------- DATA TABLE -------------
            df_display = df[['Symbol', 'Date', '%chngACT', '%chngDEL', 'REMARKS', 'ClosePrice', '%CHANGE', '%DlyQttoTradedQty']].head(50)
            st.subheader("📋 Last 50 Days Analysis")
            st.dataframe(df_display.style.background_gradient(cmap="coolwarm", subset=['%chngACT', '%chngDEL', '%DlyQttoTradedQty']))

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
else:
    st.info("Please enter a valid NSE stock symbol to begin.")
