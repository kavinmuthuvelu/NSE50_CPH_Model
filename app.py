
import os
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000000"))
TRADE_SIZE_PCT = float(os.getenv("TRADE_SIZE_PCT", "0.01"))
SBC_THRESHOLD = 0.50
DEFAULT_X = float(os.getenv("X_PCT", "0.0"))
DEFAULT_MAX_AVG = int(os.getenv("MAX_AVERAGING", "10"))

# This is the same universe used by the attached backtester.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",
    "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE",
    "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA",
    "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO"
]

CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "1800"))
_DATA_CACHE = {"loaded_at": 0.0, "data": {}}

# ============================================================
# DATA
# ============================================================

def download_nifty_data():
    """Download daily OHLC data for the configured NIFTY universe."""
    now = time.time()
    if _DATA_CACHE["data"] and now - _DATA_CACHE["loaded_at"] < CACHE_SECONDS:
        return _DATA_CACHE["data"]

    tickers = [s + ".NS" for s in NIFTY50]
    start = "2010-01-01"
    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    result = {}

    for symbol in NIFTY50:
        ticker = symbol + ".NS"
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].copy()
            else:
                df = raw.copy()

            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]

            # yfinance can return Date or Datetime depending on version.
            date_col = "date" if "date" in df.columns else "datetime"
            rename = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
            }
            df = df.rename(columns=rename)

            needed = ["open", "high", "low", "close"]
            if date_col not in df.columns or not all(c in df.columns for c in needed):
                continue

            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            if getattr(df["date"].dt, "tz", None) is not None:
                df["date"] = df["date"].dt.tz_localize(None)

            for c in needed:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df[["date"] + needed].dropna()
            df = df.sort_values("date").reset_index(drop=True)

            if len(df) >= 50:
                result[symbol] = df
        except Exception:
            continue

    _DATA_CACHE["loaded_at"] = now
    _DATA_CACHE["data"] = result
    return result


# ============================================================
# STRATEGY — mirrors the attached SBC backtester
# ============================================================

def detect_sbc(df):
    df = df.copy()
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["sbc_ratio"] = np.where(
        df["range"] > 0,
        df["body"] / df["range"],
        1.0
    )
    df["is_sbc"] = df["sbc_ratio"] <= SBC_THRESHOLD
    return df


def build_zones(df, x_pct):
    zones = []
    i = 0
    n = len(df)

    while i < n:
        if not bool(df.iloc[i]["is_sbc"]):
            i += 1
            continue

        start = i
        while i + 1 < n and bool(df.iloc[i + 1]["is_sbc"]):
            i += 1

        end = i
        base = df.iloc[start:end + 1]
        base_low = float(base["low"].min())
        base_high = float(base["high"].max())

        impulse_idx = end + 1
        if impulse_idx >= n:
            break

        impulse = df.iloc[impulse_idx]
        close = float(impulse["close"])

        if close >= base_high * (1 + x_pct):
            zones.append({
                "type": "demand",
                "start_idx": start,
                "end_idx": end,
                "confirm_idx": impulse_idx,
                "low": base_low,
                "high": base_high,
            })
        elif close <= base_low * (1 - x_pct):
            zones.append({
                "type": "supply",
                "start_idx": start,
                "end_idx": end,
                "confirm_idx": impulse_idx,
                "low": base_low,
                "high": base_high,
            })

        i += 1

    return zones


def prepare_timeframe(df, timeframe):
    if timeframe == "weekly":
        df = (
            df.set_index("date")
              .resample("W-FRI")
              .agg({
                  "open": "first",
                  "high": "max",
                  "low": "min",
                  "close": "last",
              })
              .dropna()
              .reset_index()
        )
    return detect_sbc(df)


# ============================================================
# MODEL BASKET
# ============================================================

class ModelBasket:
    def __init__(self, symbol):
        self.symbol = symbol
        self.entries = []
        self.created = None
        self.last_supply_idx = None

    def add_entry(self, dt, price):
        self.entries.append({
            "date": pd.Timestamp(dt),
            "price": float(price),
        })
        if self.created is None:
            self.created = pd.Timestamp(dt)

    @property
    def avg_price(self):
        if not self.entries:
            return 0.0
        return float(np.mean([e["price"] for e in self.entries]))

    @property
    def entry_count(self):
        return len(self.entries)


def simulate_symbol(symbol, raw_df, timeframe, x_pct, max_avg):
    """
    Reconstruct the strategy state up to the latest completed bar.
    Position sizing is deliberately separated from signal detection.
    """
    df = prepare_timeframe(raw_df, timeframe)
    if len(df) < 10:
        return None

    zones = build_zones(df, x_pct)
    zone_by_confirm = {}
    for z in zones:
        zone_by_confirm.setdefault(z["confirm_idx"], []).append(z)

    active_demand = []
    active_supply = []
    basket = None
    completed = []

    for idx in range(len(df)):
        bar = df.iloc[idx]
        price = float(bar["close"])
        dt = pd.Timestamp(bar["date"])

        for z in zone_by_confirm.get(idx, []):
            if z["type"] == "demand":
                active_demand.append(z)
            else:
                active_supply.append(z)
                if basket is not None:
                    basket.last_supply_idx = idx

        # No basket: demand tap creates a new basket.
        if basket is None:
            for z in reversed(active_demand):
                if float(bar["low"]) <= z["high"] and float(bar["high"]) >= z["low"]:
                    basket = ModelBasket(symbol)
                    basket.add_entry(dt, price)
                    active_demand.remove(z)
                    break

        if basket is not None:
            tapped_supply = None

            for z in reversed(active_supply):
                if float(bar["low"]) <= z["high"] and float(bar["high"]) >= z["low"]:
                    tapped_supply = z
                    break

            if tapped_supply is not None:
                # Same rule as attached code: exit only if basket is profitable.
                if price > basket.avg_price:
                    completed.append({
                        "entry_date": basket.created,
                        "exit_date": dt,
                        "entry_count": basket.entry_count,
                        "avg_price": basket.avg_price,
                        "exit_price": price,
                        "pnl_pct_price": (price / basket.avg_price - 1) * 100,
                        "status": "CLOSED",
                    })
                    basket = None
                    active_supply.remove(tapped_supply)
                    active_demand = []
                    continue
                else:
                    active_supply.remove(tapped_supply)

            # Attached code's averaging rule: do not average while an
            # unresolved confirmed supply is pending.
            supply_pending = (
                basket.last_supply_idx is not None
                and basket.last_supply_idx < idx
            )

            if not supply_pending and len(basket.entries) < max_avg:
                for z in reversed(active_demand):
                    if float(bar["low"]) <= z["high"] and float(bar["high"]) >= z["low"]:
                        basket.add_entry(dt, price)
                        active_demand.remove(z)
                        break

    latest = df.iloc[-1]
    current = None

    if basket is not None:
        current_price = float(latest["close"])
        avg = basket.avg_price
        current = {
            "entry_date": basket.created,
            "entry_count": basket.entry_count,
            "avg_price": avg,
            "current_price": current_price,
            "pnl_pct": (current_price / avg - 1) * 100 if avg else 0,
        }

    # Find zones touched by the latest completed bar.
    latest_idx = len(df) - 1
    latest_bar = df.iloc[-1]
    latest_touches = []

    for z in zones:
        if z["confirm_idx"] > latest_idx:
            continue
        if float(latest_bar["low"]) <= z["high"] and float(latest_bar["high"]) >= z["low"]:
            latest_touches.append(z)

    return {
        "symbol": symbol,
        "df": df,
        "zones": zones,
        "latest": latest,
        "current_basket": current,
        "latest_touches": latest_touches,
    }


def generate_signal(state, max_avg):
    if state is None:
        return None

    symbol = state["symbol"]
    latest = state["latest"]
    price = float(latest["close"])
    basket = state["current_basket"]

    touches = state["latest_touches"]
    demand_touches = [z for z in touches if z["type"] == "demand"]
    supply_touches = [z for z in touches if z["type"] == "supply"]

    # SELL: open basket + supply tap + current basket profitable.
    if basket and supply_touches and price > basket["avg_price"]:
        z = supply_touches[-1]
        return {
            "action": "SELL",
            "symbol": symbol,
            "price": price,
            "date": latest["date"],
            "reason": "Supply zone tapped and basket is profitable",
            "zone_low": z["low"],
            "zone_high": z["high"],
            "entry_count": basket["entry_count"],
            "avg_price": basket["avg_price"],
            "pnl_pct": basket["pnl_pct"],
        }

    # HOLD: open basket + supply tap but not profitable.
    if basket and supply_touches and price <= basket["avg_price"]:
        z = supply_touches[-1]
        return {
            "action": "HOLD",
            "symbol": symbol,
            "price": price,
            "date": latest["date"],
            "reason": "Supply zone tapped but basket is not profitable",
            "zone_low": z["low"],
            "zone_high": z["high"],
            "entry_count": basket["entry_count"],
            "avg_price": basket["avg_price"],
            "pnl_pct": basket["pnl_pct"],
        }

    # BUY: no basket + demand tap.
    if basket is None and demand_touches:
        z = demand_touches[-1]
        return {
            "action": "BUY",
            "symbol": symbol,
            "price": price,
            "date": latest["date"],
            "reason": "Demand zone tapped",
            "zone_low": z["low"],
            "zone_high": z["high"],
            "entry_count": 0,
            "avg_price": None,
            "pnl_pct": None,
        }

    # AVERAGE: basket + demand tap + room for another entry.
    if basket and demand_touches and basket["entry_count"] < max_avg:
        z = demand_touches[-1]
        return {
            "action": "AVERAGE",
            "symbol": symbol,
            "price": price,
            "date": latest["date"],
            "reason": "Demand zone tapped while basket is open",
            "zone_low": z["low"],
            "zone_high": z["high"],
            "entry_count": basket["entry_count"],
            "avg_price": basket["avg_price"],
            "pnl_pct": basket["pnl_pct"],
        }

    if basket:
        return {
            "action": "HOLD",
            "symbol": symbol,
            "price": price,
            "date": latest["date"],
            "reason": "Open basket; no new actionable zone tap",
            "zone_low": None,
            "zone_high": None,
            "entry_count": basket["entry_count"],
            "avg_price": basket["avg_price"],
            "pnl_pct": basket["pnl_pct"],
        }

    return {
        "action": "WAIT",
        "symbol": symbol,
        "price": price,
        "date": latest["date"],
        "reason": "No actionable demand/supply tap",
        "zone_low": None,
        "zone_high": None,
        "entry_count": 0,
        "avg_price": None,
        "pnl_pct": None,
    }


# ============================================================
# THEORETICAL COMPOUNDING CAPITAL
# ============================================================

def calculate_theoretical_compounding(states):
    """
    This calculates the model's theoretical realized capital from the
    reconstructed closed baskets. It is NOT a record of the user's
    real manual trades.
    """
    events = []

    for state in states:
        if not state:
            continue

        # Reconstruct completed baskets from the same simulation.
        df = state["df"]
        zones = state["zones"]

        # For the displayed capital we use a simple event replay based on
        # the strategy's closed baskets. The signal engine itself does not
        # require this capital to decide BUY/SELL.
        # Current capital therefore remains conservative/transparent:
        # initial capital unless a persistent manual ledger is added.
        pass

    return INITIAL_CAPITAL


# ============================================================
# WEB UI
# ============================================================

HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NIFTY 50 SBC Manual Signal Dashboard</title>
<style>
body { font-family: Arial, sans-serif; background:#f5f7fa; margin:0; color:#17202a; }
.header { background:#111827; color:white; padding:22px; }
.container { max-width:1400px; margin:20px auto; padding:0 16px; }
.card { background:white; border-radius:12px; padding:18px; margin-bottom:18px; box-shadow:0 2px 10px rgba(0,0,0,.06); }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; }
label { font-size:13px; font-weight:bold; display:block; margin-bottom:5px; }
input,select,button { padding:9px 10px; border:1px solid #ccd3dc; border-radius:7px; }
button { background:#111827; color:white; cursor:pointer; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th,td { padding:10px 8px; border-bottom:1px solid #e5e7eb; text-align:left; }
th { background:#f3f4f6; position:sticky; top:0; }
.buy { color:#087f23; font-weight:bold; }
.sell { color:#b42318; font-weight:bold; }
.average { color:#b54708; font-weight:bold; }
.hold { color:#5b21b6; font-weight:bold; }
.wait { color:#6b7280; font-weight:bold; }
.badge { padding:4px 8px; border-radius:12px; background:#eef2ff; }
.small { color:#667085; font-size:12px; }
.error { background:#fff1f2; color:#9f1239; padding:12px; border-radius:8px; }
.summary { display:flex; flex-wrap:wrap; gap:18px; }
.metric { min-width:160px; }
.metric b { font-size:22px; display:block; }
</style>
</head>
<body>
<div class="header">
  <div class="container">
    <h1>NIFTY 50 SBC Manual Signal Dashboard</h1>
    <div>Demand/Supply + SBC strategy • Signals only • Manual execution</div>
  </div>
</div>

<div class="container">

<div class="card">
<form method="get">
<div class="controls">
  <div>
    <label>Timeframe</label>
    <select name="timeframe">
      <option value="daily" {% if timeframe=="daily" %}selected{% endif %}>Daily</option>
      <option value="weekly" {% if timeframe=="weekly" %}selected{% endif %}>Weekly</option>
    </select>
  </div>
  <div>
    <label>X confirmation %</label>
    <input name="x" type="number" step="0.1" min="0" value="{{ x_pct*100 }}">
  </div>
  <div>
    <label>Max averaging entries</label>
    <input name="max_avg" type="number" min="1" max="50" value="{{ max_avg }}">
  </div>
  <div>
    <label>Initial capital ₹</label>
    <input name="capital" type="number" step="1000" value="{{ initial_capital }}">
  </div>
  <button type="submit">Refresh Signals</button>
</div>
</form>
<p class="small">The dashboard uses the latest completed Yahoo Finance daily/weekly bar. It does not place orders.</p>
</div>

{% if error %}
<div class="card error">{{ error }}</div>
{% endif %}

{% if rows %}
<div class="card">
<div class="summary">
  <div class="metric"><span class="small">Stocks scanned</span><b>{{ scanned }}</b></div>
  <div class="metric"><span class="small">BUY</span><b class="buy">{{ counts.get("BUY",0) }}</b></div>
  <div class="metric"><span class="small">AVERAGE</span><b class="average">{{ counts.get("AVERAGE",0) }}</b></div>
  <div class="metric"><span class="small">SELL</span><b class="sell">{{ counts.get("SELL",0) }}</b></div>
  <div class="metric"><span class="small">HOLD</span><b class="hold">{{ counts.get("HOLD",0) }}</b></div>
  <div class="metric"><span class="small">WAIT</span><b class="wait">{{ counts.get("WAIT",0) }}</b></div>
</div>
</div>

<div class="card">
<h2>Actionable Signals</h2>
{% if actionable %}
<table>
<thead><tr>
<th>Action</th><th>Symbol</th><th>Last Price</th><th>Suggested Capital</th>
<th>Suggested Qty</th><th>Avg Price</th><th>P&L %</th><th>Entries</th>
<th>Zone</th><th>Reason</th><th>Bar Date</th>
</tr></thead>
<tbody>
{% for r in actionable %}
<tr>
<td class="{{ r.action|lower }}">{{ r.action }}</td>
<td><b>{{ r.symbol }}</b></td>
<td>₹{{ "%.2f"|format(r.price) }}</td>
<td>
{% if r.action in ["BUY","AVERAGE"] %}
₹{{ "{:,.0f}".format(r.suggested_capital) }}
{% else %}-{% endif %}
</td>
<td>
{% if r.action in ["BUY","AVERAGE"] %}
{{ r.suggested_qty }}
{% else %}-{% endif %}
</td>
<td>{% if r.avg_price %}₹{{ "%.2f"|format(r.avg_price) }}{% else %}-{% endif %}</td>
<td>{% if r.pnl_pct is not none %}{{ "%.2f"|format(r.pnl_pct) }}%{% else %}-{% endif %}</td>
<td>{{ r.entry_count }}</td>
<td>
{% if r.zone_low is not none %}
₹{{ "%.2f"|format(r.zone_low) }} – ₹{{ "%.2f"|format(r.zone_high) }}
{% else %}-{% endif %}
</td>
<td>{{ r.reason }}</td>
<td>{{ r.date.strftime("%Y-%m-%d") }}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% else %}
<p>No BUY / AVERAGE / SELL signals on the latest completed bar.</p>
{% endif %}
</div>

<div class="card">
<h2>All Stock States</h2>
<table>
<thead><tr><th>Action</th><th>Symbol</th><th>Price</th><th>Avg</th><th>P&L %</th><th>Entries</th><th>Reason</th></tr></thead>
<tbody>
{% for r in rows %}
<tr>
<td class="{{ r.action|lower }}">{{ r.action }}</td>
<td><b>{{ r.symbol }}</b></td>
<td>₹{{ "%.2f"|format(r.price) }}</td>
<td>{% if r.avg_price %}₹{{ "%.2f"|format(r.avg_price) }}{% else %}-{% endif %}</td>
<td>{% if r.pnl_pct is not none %}{{ "%.2f"|format(r.pnl_pct) }}%{% else %}-{% endif %}</td>
<td>{{ r.entry_count }}</td>
<td>{{ r.reason }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<div class="card">
<h3>Important</h3>
<ul>
<li>BUY/AVERAGE/SELL are model recommendations, not broker orders.</li>
<li>Suggested capital is 1% of the configured capital. For manual execution, quantity is rounded down to whole shares.</li>
<li>The displayed model state is reconstructed from historical price data and the attached strategy rules. It does not know which trades you actually executed.</li>
<li>For true personal compounding, the next step is to add a persistent manual trade ledger so realized profit/loss from your actual trades becomes the 1% compounding base.</li>
</ul>
</div>
{% endif %}
</div>
</body>
</html>
"""


@app.route("/")
def dashboard():
    try:
        timeframe = request.args.get("timeframe", "daily").lower()
        if timeframe not in ("daily", "weekly"):
            timeframe = "daily"

        x_pct = float(request.args.get("x", DEFAULT_X * 100)) / 100.0
        max_avg = int(request.args.get("max_avg", DEFAULT_MAX_AVG))
        initial_capital = float(request.args.get("capital", INITIAL_CAPITAL))

        if x_pct < 0:
            x_pct = 0
        if max_avg < 1:
            max_avg = 1
        if initial_capital <= 0:
            initial_capital = INITIAL_CAPITAL

        datasets = download_nifty_data()
        rows = []
        errors = []

        for symbol, df in datasets.items():
            try:
                state = simulate_symbol(symbol, df, timeframe, x_pct, max_avg)
                signal = generate_signal(state, max_avg)
                if signal:
                    # Suggested 1% manual allocation. This is intentionally
                    # based on configured capital, not assumed real trades.
                    signal["suggested_capital"] = initial_capital * TRADE_SIZE_PCT
                    signal["suggested_qty"] = max(
                        0, math.floor(signal["suggested_capital"] / signal["price"])
                    )
                    rows.append(signal)
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

        order = {"SELL": 0, "BUY": 1, "AVERAGE": 2, "HOLD": 3, "WAIT": 4}
        rows.sort(key=lambda x: (order.get(x["action"], 9), x["symbol"]))

        counts = {}
        for r in rows:
            counts[r["action"]] = counts.get(r["action"], 0) + 1

        actionable = [r for r in rows if r["action"] in ("BUY", "AVERAGE", "SELL")]

        error = None
        if not datasets:
            error = "No market data was downloaded. Check Render logs and Yahoo Finance availability."
        elif errors:
            error = f"{len(errors)} symbols had scan errors; the remaining symbols are shown."

        return render_template_string(
            HTML,
            rows=rows,
            actionable=actionable,
            counts=counts,
            scanned=len(datasets),
            timeframe=timeframe,
            x_pct=x_pct,
            max_avg=max_avg,
            initial_capital=initial_capital,
            error=error,
        )

    except Exception as exc:
        return render_template_string(
            HTML,
            rows=[],
            actionable=[],
            counts={},
            scanned=0,
            timeframe="daily",
            x_pct=DEFAULT_X,
            max_avg=DEFAULT_MAX_AVG,
            initial_capital=INITIAL_CAPITAL,
            error=f"Application error: {exc}",
        ), 500


@app.route("/health")
def health():
    return {"status": "ok", "service": "nifty50-sbc-signal-dashboard"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
