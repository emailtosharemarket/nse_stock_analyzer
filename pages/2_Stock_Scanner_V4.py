# ============================================================
# 🚀 STOCK SCANNER V4
# Technical + Fundamental + Shareholding Scanner
# 👤 Developed by SUJOY ROY
# ============================================================

import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Stock Scanner",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Stock Scanner V4")
st.caption("Technical + Fundamental + Shareholding screening model")

st.info(
    "⚠️ Screening/ranking model only. It is not a guarantee of returns. "
    "Fundamental/shareholding data is sourced from Screener.in and price data from Yahoo Finance."
)

with st.expander("📖 Scanner Methodology", expanded=False):
    st.markdown("""
**Technical — 50 points**
- RSI(14) > 50 → +15
- RSI(14) > 60 → +10 additional
- MACD bullish → +15
- RSI CROSS50 exists → +10

**Fundamental — 30 points**
- Price ≤ Intrinsic Value → +15
- Profit growth > 0% → +10
- Profit growth ≥ 15% → +5 additional

**Shareholding — 20 points**
- Promoters increased → +5
- FIIs increased → +5
- DIIs increased → +5
- Public decreased → +5

**ACTION**
- BUY → score ≥ 70 and Price ≤ IV
- WATCH → score ≥ 60
- WAIT → otherwise

**Intrinsic Value**
`EPS_TTM × (8.5 + 2 × ProfitGrowth_TTM) × 4.4 / 7`

**Trade levels**
- Entry = current price
- Stop = technical/volatility based
- T1 = Entry + 2R
- T2 = Entry + 3R
""")

stock_text = st.text_area(
    "Enter NSE symbols (comma, space, or newline separated)",
    value="SETL, BERGEPAINT, AARTIND, RAMRAT, MANINDS",
    height=100,
    help="Example: TCS, INFY, RELIANCE"
)

col1, col2 = st.columns([1, 4])
with col1:
    scan_clicked = st.button("🔍 ANALYSE STOCKS", type="primary", use_container_width=True)
with col2:
    st.caption("Tip: Start with a small number of stocks because Screener and Yahoo Finance are external services.")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

SCREENER_URLS = [
    "https://www.screener.in/company/{symbol}/consolidated/",
    "https://www.screener.in/company/{symbol}/",
]


def clean_symbol(symbol):
    return str(symbol).strip().upper().replace(".NS", "")


def parse_symbols(text):
    parts = re.split(r"[\s,;]+", text or "")
    return list(dict.fromkeys(clean_symbol(x) for x in parts if clean_symbol(x)))


def num(value):
    if value is None:
        return np.nan
    s = str(value).strip()
    if not s or s in {"-", "--", "nan", "None"}:
        return np.nan
    s = s.replace(",", "").replace("₹", "").replace("%", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


@st.cache_data(ttl=1800, show_spinner=False)
def get_screener_html(symbol):
    session = requests.Session()
    session.headers.update(HEADERS)
    last_error = None

    for template in SCREENER_URLS:
        url = template.format(symbol=symbol)
        try:
            response = session.get(url, timeout=30)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                continue

            html = response.text
            if (
                "Profit & Loss" in html
                or "Shareholding Pattern" in html
                or "Compounded Profit Growth" in html
            ):
                return html

            last_error = "Company data not found in response"
        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(f"Screener access failed: {last_error}")


def get_tables(html):
    try:
        return pd.read_html(StringIO(html))
    except Exception as exc:
        raise RuntimeError(f"Could not read Screener tables: {exc}")


def row_contains(row, text):
    return text.lower() in " ".join(str(x) for x in row.tolist()).lower()


def extract_eps_ttm(tables):
    for table in tables:
        for _, row in table.iterrows():
            if not row_contains(row, "EPS in Rs"):
                continue
            columns = [str(c).strip().lower() for c in table.columns]
            for i, col in enumerate(columns):
                if "ttm" in col and i < len(row):
                    value = num(row.iloc[i])
                    if not np.isnan(value):
                        return value

    for table in tables:
        for _, row in table.iterrows():
            if not row_contains(row, "EPS in Rs"):
                continue
            values = [num(x) for x in row.iloc[1:]]
            values = [x for x in values if not np.isnan(x)]
            if len(values) >= 4:
                return round(sum(values[-4:]), 4)

    raise RuntimeError("Could not determine EPS TTM")


def extract_growth_ttm(html, tables):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    for match in re.finditer(r"Compounded\s+Profit\s+Growth", text, re.IGNORECASE):
        section = text[match.start():match.start() + 1800]
        hit = re.search(
            r"TTM\s*:?\s*([+-]?\d+(?:\.\d+)?)\s*%?",
            section,
            re.IGNORECASE
        )
        if hit:
            return float(hit.group(1))

    for table in tables:
        for _, row in table.iterrows():
            text_row = " ".join(str(x) for x in row.tolist())
            if "Compounded Profit Growth" in text_row and "TTM" in text_row:
                values = [num(x) for x in row.tolist()]
                values = [x for x in values if not np.isnan(x)]
                if values:
                    return values[-1]

    for table in tables:
        for _, row in table.iterrows():
            if not row_contains(row, "Net Profit"):
                continue
            values = [num(x) for x in row.iloc[1:]]
            values = [x for x in values if not np.isnan(x)]
            if len(values) >= 8:
                old = sum(values[-8:-4])
                new = sum(values[-4:])
                if old != 0:
                    return (new / old - 1) * 100

    raise RuntimeError("Could not determine TTM profit growth")


def extract_shareholding(tables):
    categories = {
        "Promoters": re.compile(r"^\s*promoters", re.IGNORECASE),
        "FIIs": re.compile(r"^\s*fii?s?", re.IGNORECASE),
        "DIIs": re.compile(r"^\s*dii?s?", re.IGNORECASE),
        "Public": re.compile(r"^\s*public", re.IGNORECASE),
    }

    found = {}

    for table in tables:
        local = {}

        for _, row in table.iterrows():
            values = [str(x).strip() for x in row.tolist()]
            if not values:
                continue

            label = re.sub(r"\s*\+\s*$", "", values[0]).strip()

            for category, pattern in categories.items():
                if pattern.match(label):
                    numbers = [num(x) for x in values[1:]]
                    numbers = [x for x in numbers if not np.isnan(x)]
                    if len(numbers) >= 2:
                        local[category] = numbers[-2:]

        if len(local) == 4:
            found = local
            break

    result = {}
    for category in categories:
        if category not in found:
            result[category] = "N/A"
            continue

        previous, latest = found[category]

        if latest > previous + 1e-9:
            result[category] = "+"
        elif latest < previous - 1e-9:
            result[category] = "-"
        else:
            result[category] = "="

    return result


@st.cache_data(ttl=900, show_spinner=False)
def get_ohlc(symbol):
    data = yf.download(
        f"{symbol}.NS",
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    needed = [c for c in ["Open", "High", "Low", "Close"] if c in data.columns]
    data = data[needed].dropna()

    if data.empty:
        raise RuntimeError(f"OHLC data unavailable for {symbol}")

    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.where(
        ~((avg_loss == 0) & (avg_gain > 0)),
        100
    )

    return rsi


def calculate_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def atr(data, period=14):
    previous_close = data["Close"].shift(1)
    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - previous_close).abs()
    tr3 = (data["Low"] - previous_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def cross50(rsi):
    previous = rsi.shift(1)
    crossing = (previous <= 50) & (rsi > 50)
    dates = rsi.index[crossing.fillna(False)]

    if len(dates) == 0:
        return ""

    return dates[-1].strftime("%d-%m-%Y")


def calculate_trade_levels(data):
    entry = float(data["Close"].iloc[-1])

    atr_series = atr(data)
    current_atr = float(atr_series.iloc[-1])

    recent_low = float(data["Low"].tail(20).min())
    atr_stop = entry - 2 * current_atr
    minimum_stop = entry - current_atr

    stop = max(recent_low, atr_stop)

    if stop >= entry:
        stop = atr_stop

    if stop > minimum_stop:
        stop = minimum_stop

    if stop <= 0 or stop >= entry:
        stop = entry - 2 * current_atr

    risk_amount = entry - stop
    target1 = entry + 2 * risk_amount
    target2 = entry + 3 * risk_amount
    risk_percent = risk_amount / entry * 100

    return (
        entry, stop, target1, target2,
        risk_percent, 2.0, current_atr, recent_low
    )


def calculate_score(
    rsi_value, macd_bullish, has_cross50,
    price, iv, growth, holdings
):
    score = 0

    if rsi_value > 50:
        score += 15
    if rsi_value > 60:
        score += 10
    if macd_bullish:
        score += 15
    if has_cross50:
        score += 10

    if price <= iv:
        score += 15
    if growth > 0:
        score += 10
    if growth >= 15:
        score += 5

    if holdings["Promoters"] == "+":
        score += 5
    if holdings["FIIs"] == "+":
        score += 5
    if holdings["DIIs"] == "+":
        score += 5
    if holdings["Public"] == "-":
        score += 5

    return score


def action(score, price, iv):
    if score >= 70 and price <= iv:
        return "BUY"
    if score >= 60:
        return "WATCH"
    return "WAIT"


def analyze(symbol):
    symbol = clean_symbol(symbol)

    data = get_ohlc(symbol)
    close = data["Close"]

    rsi_series = rsi_wilder(close)
    macd_line, signal_line = calculate_macd(close)

    current_rsi = float(rsi_series.iloc[-1])
    macd_bullish = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])
    cross_date = cross50(rsi_series)

    html = get_screener_html(symbol)
    tables = get_tables(html)

    eps_ttm = extract_eps_ttm(tables)
    growth_ttm = extract_growth_ttm(html, tables)

    iv = round(
        eps_ttm * (8.5 + 2 * growth_ttm) * 4.4 / 7,
        2
    )

    holdings = extract_shareholding(tables)

    (
        entry, stop, target1, target2,
        risk_percent, rr, current_atr, recent_low
    ) = calculate_trade_levels(data)

    score = calculate_score(
        current_rsi,
        macd_bullish,
        bool(cross_date),
        entry,
        iv,
        growth_ttm,
        holdings
    )

    decision = action(score, entry, iv)

    return {
        "STOCKS": symbol,
        "MACD": "+" if macd_bullish else "-",
        "RSI": round(current_rsi, 2),
        "CROSS50": cross_date,
        "IV": iv,
        "CLOSE PRICE": round(entry, 2),
        "PY_SCORE": score,
        "Promoters": holdings["Promoters"],
        "FIIs": holdings["FIIs"],
        "DIIs": holdings["DIIs"],
        "Public": holdings["Public"],
        "ENTRY": round(entry, 2),
        "STOP LOSS": round(stop, 2),
        "TARGET 1": round(target1, 2),
        "TARGET 2": round(target2, 2),
        "RISK %": round(risk_percent, 2),
        "R:R": rr,
        "ACTION": decision,
        "_EPS_TTM": round(eps_ttm, 4),
        "_Growth_TTM": round(growth_ttm, 2),
        "_ATR": round(current_atr, 2),
        "_SwingLow20": round(recent_low, 2),
    }


def style_action(val):
    if val == "BUY":
        return "font-weight: bold; background-color: #90EE90;"
    if val == "WATCH":
        return "font-weight: bold; background-color: #FFD700;"
    if val == "WAIT":
        return "font-weight: bold; background-color: #FFB6C1;"
    return ""


if scan_clicked:
    symbols = parse_symbols(stock_text)

    if not symbols:
        st.error("Please enter at least one valid NSE symbol.")
        st.stop()

    st.subheader(f"🔎 Analysing {len(symbols)} stock(s)")

    results = []
    errors = []

    progress = st.progress(0)

    for i, symbol in enumerate(symbols, 1):
        with st.status(f"Analysing {symbol}...", expanded=False) as status:
            try:
                result = analyze(symbol)
                results.append(result)
                status.update(label=f"✅ {symbol} completed", state="complete")
            except Exception as exc:
                errors.append({"STOCKS": symbol, "ERROR": str(exc)})
                status.update(label=f"❌ {symbol} failed", state="error")

        progress.progress(i / len(symbols))

    if results:
        columns = [
            "STOCKS", "MACD", "RSI", "CROSS50", "IV",
            "CLOSE PRICE", "PY_SCORE", "Promoters", "FIIs",
            "DIIs", "Public", "ENTRY", "STOP LOSS",
            "TARGET 1", "TARGET 2", "RISK %", "R:R", "ACTION"
        ]

        output = pd.DataFrame(results)[columns]

        # Highest score first; BUY before WATCH before WAIT at equal score.
        action_order = {"BUY": 0, "WATCH": 1, "WAIT": 2}
        output["_ACTION_ORDER"] = output["ACTION"].map(action_order).fillna(9)
        output = (
            output.sort_values(
                ["PY_SCORE", "_ACTION_ORDER"],
                ascending=[False, True]
            )
            .drop(columns="_ACTION_ORDER")
            .reset_index(drop=True)
        )

        st.success(f"✅ Successfully analysed {len(results)} of {len(symbols)} stock(s).")

        st.subheader("📊 Scanner Results")

        st.dataframe(
            output.style
            .map(style_action, subset=["ACTION"])
            .background_gradient(
                cmap="YlGn",
                subset=["PY_SCORE"]
            ),
            use_container_width=True,
            hide_index=True
        )

        csv = output.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Results CSV",
            data=csv,
            file_name="stock_analysis_v4.csv",
            mime="text/csv"
        )

        buys = output[output["ACTION"] == "BUY"]
        watches = output[output["ACTION"] == "WATCH"]

        a, b, c = st.columns(3)
        a.metric("🟢 BUY", len(buys))
        b.metric("🟡 WATCH", len(watches))
        c.metric("⚪ WAIT", len(output) - len(buys) - len(watches))

        if not buys.empty:
            st.subheader("🟢 BUY Candidates")
            st.dataframe(buys, use_container_width=True, hide_index=True)

        if errors:
            st.subheader("⚠️ Stocks with Errors")
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
else:
    st.info("Enter one or more NSE symbols above and click **🔍 ANALYSE STOCKS**.")
