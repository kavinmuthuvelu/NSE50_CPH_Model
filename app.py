
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, redirect, url_for, render_template_string, jsonify

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================
INITIAL_CAPITAL = 1_000_000
TRADE_SIZE_PCT = 0.01
SBC_THRESHOLD = 0.50
DEFAULT_X_PCT = 0.00
DEFAULT_MAX_AVG = 10

# Current NIFTY 50 universe used by the dashboard.
# Yahoo Finance symbols.
NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SHRIRAMFIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATASTEEL.NS",
    "TATAMOTORS.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS"
]

CACHE_DIR = Path(os.environ.get("RENDER_DISK_PATH", "/tmp")) / "sbc_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "market_cache.json"

DATA_LOCK = threading.Lock()
REFRESH_THREAD = None
REFRESH_STATUS = {
    "running": False,
    "started": None,
    "finished": None,
    "message": "No refresh has been run yet.",
    "success": 0,
    "failed": 0,
}

# In-memory raw data cache.
DATA = {}


# ============================================================
# CACHE HELPERS
# ============================================================
def save_cache():
    payload = {}
    for symbol, df in DATA.items():
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        payload[symbol] = tmp.to_dict(orient="records")

    try:
        CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:
        print(f"Cache save failed: {exc}")


def load_cache():
    global DATA
    if not CACHE_FILE.exists():
        return

    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        loaded = {}
        for symbol, records in payload.items():
            df = pd.DataFrame(records)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["date", "open", "high", "low", "close"])
            if len(df) >= 50:
                loaded[symbol] = df.sort_values("date").reset_index(drop=True)

        DATA = loaded
        print(f"Loaded {len(DATA)} symbols from cache.")
    except Exception as exc:
        print(f"Cache load failed: {exc}")


# ============================================================
# DATA DOWNLOAD
# ============================================================
def normalize_download(raw, symbol):
    if raw is None or raw.empty:
        return None

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return MultiIndex columns even for one ticker.
        if symbol in df.columns.get_level_values(-1):
            try:
                df = df.xs(symbol, axis=1, level=-1)
            except Exception:
                df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "date":
            rename[c] = "date"
        elif lc == "open":
            rename[c] = "open"
        elif lc == "high":
            rename[c] = "high"
        elif lc == "low":
            rename[c] = "low"
        elif lc == "close":
            rename[c] = "close"
        elif lc == "volume":
            rename[c] = "volume"

    df = df.rename(columns=rename)

    required = ["date", "open", "high", "low", "close"]
    if any(c not in df.columns for c in required):
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    return df if len(df) >= 50 else None


def download_symbol(symbol, period="5y"):
    # One ticker per request is slower but isolates Yahoo failures.
    # It is deliberately NOT called during Flask startup.
    for attempt in range(2):
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20,
            )
            df = normalize_download(raw, symbol)
            if df is not None:
                return df
        except Exception as exc:
            print(f"{symbol}: attempt {attempt + 1} failed: {exc}")
        time.sleep(1.5 * (attempt + 1))
    return None


def refresh_worker(period="5y"):
    global REFRESH_STATUS, DATA

    with DATA_LOCK:
        REFRESH_STATUS = {
            "running": True,
            "started": datetime.utcnow().isoformat(),
            "finished": None,
            "message": "Refreshing market data in the background...",
            "success": 0,
            "failed": 0,
        }

    new_data = {}
    failed = []

    # Small pauses help reduce Yahoo burst/rate-limit problems.
    for idx, symbol in enumerate(NIFTY50):
        df = download_symbol(symbol, period=period)

        if df is not None:
            new_data[symbol] = df
        else:
            failed.append(symbol)

        if idx < len(NIFTY50) - 1:
            time.sleep(0.35)

        with DATA_LOCK:
            REFRESH_STATUS["success"] = len(new_data)
            REFRESH_STATUS["failed"] = len(failed)
            REFRESH_STATUS["message"] = (
                f"Downloaded {len(new_data)}/{len(NIFTY50)} stocks. "
                f"Failed: {len(failed)}."
            )

    # Preserve previously cached symbols when a refresh temporarily fails.
    with DATA_LOCK:
        merged = dict(DATA)
        merged.update(new_data)
        DATA = merged
        save_cache()

        REFRESH_STATUS["running"] = False
        REFRESH_STATUS["finished"] = datetime.utcnow().isoformat()
        REFRESH_STATUS["message"] = (
            f"Refresh complete. Available stocks: {len(DATA)}/{len(NIFTY50)}. "
            f"New failures: {len(failed)}."
        )

    print(
        f"Refresh complete: success={len(new_data)}, "
        f"failed={len(failed)}, cached_total={len(DATA)}"
    )


def start_refresh(period="5y"):
    global REFRESH_THREAD

    with DATA_LOCK:
        if REFRESH_STATUS["running"]:
            return False

    REFRESH_THREAD = threading.Thread(
        target=refresh_worker,
        args=(period,),
        daemon=True,
    )
    REFRESH_THREAD.start()
    return True


# IMPORTANT: load cache only. NEVER download data during import/startup.
load_cache()


# ============================================================
# STRATEGY ENGINE
# ============================================================
def make_timeframe(df, timeframe):
    base = df.set_index("date").sort_index()

    if timeframe == "daily":
        return base.reset_index()

    if timeframe == "weekly":
        return (
            base.resample("W-FRI")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            })
            .dropna()
            .reset_index()
        )

    raise ValueError("Unsupported timeframe")


def detect_sbc(df):
    out = df.copy()
    rng = out["high"] - out["low"]
    body = (out["close"] - out["open"]).abs()

    out["range"] = rng
    out["body"] = body
    out["sbc_ratio"] = np.where(rng > 0, body / rng, np.nan)
    out["is_sbc"] = (out["sbc_ratio"] <= SBC_THRESHOLD).fillna(False)

    return out


def build_zones(df, x_pct):
    """
    Unlimited consecutive SBC block -> one base.
    Immediately following candle confirms demand/supply.
    Confirmation candle cannot trade the new zone.
    """
    zones = []
    n = len(df)
    i = 0

    while i < n:
        # Defensive bounds check. This is the exact class of failure
        # seen in the previous Render deployment.
        if i < 0 or i >= n:
            break

        if not bool(df.iloc[i]["is_sbc"]):
            i += 1
            continue

        start = i

        while i + 1 < n and bool(df.iloc[i + 1]["is_sbc"]):
            i += 1

        end = i
        impulse_idx = end + 1

        if impulse_idx >= n:
            break

        base = df.iloc[start:end + 1]
        base_low = float(base["low"].min())
        base_high = float(base["high"].max())

        impulse = df.iloc[impulse_idx]
        close = float(impulse["close"])

        zone_type = None

        if close >= base_high * (1.0 + x_pct):
            zone_type = "demand"
        elif close <= base_low * (1.0 - x_pct):
            zone_type = "supply"

        if zone_type:
            zones.append({
                "id": f"{zone_type}_{start}_{end}_{impulse_idx}",
                "type": zone_type,
                "start_idx": start,
                "end_idx": end,
                "confirm_idx": impulse_idx,
                "tradable_from": impulse_idx + 1,
                "low": base_low,
                "high": base_high,
            })

        i += 1

    return zones


def candle_touches_zone(row, zone):
    return (
        float(row["high"]) >= zone["low"]
        and float(row["low"]) <= zone["high"]
    )


def simulate_symbol(symbol, raw_df, timeframe, x_pct, max_avg):
    df = make_timeframe(raw_df, timeframe)
    df = detect_sbc(df)

    if len(df) < 10:
        return {
            "symbol": symbol,
            "signal": "WAIT",
            "reason": "Insufficient data",
        }

    zones = build_zones(df, x_pct)
    if not zones:
        return {
            "symbol": symbol,
            "signal": "WAIT",
            "reason": "No confirmed SBC zones",
        }

    # Simulate from the beginning to determine current basket state.
    active_demand = []
    active_supply = []
    entries = []
    basket_open = False
    entry_count = 0
    last_entry_price = None
    last_entry_date = None
    last_signal_zone = None

    for i in range(len(df)):
        row = df.iloc[i]

        # Add zones confirmed before this candle.
        for z in zones:
            if z["tradable_from"] == i:
                if z["type"] == "demand":
                    active_demand.append(z)
                else:
                    active_supply.append(z)

        # Supply can close a profitable basket.
        if basket_open and active_supply:
            touched = [
                z for z in active_supply
                if i >= z["tradable_from"] and candle_touches_zone(row, z)
            ]

            if touched:
                # We do not know exact manual fill/realized P&L here,
                # so only use supply as an exit/hold event marker.
                # If current close is above last entry, treat as profitable.
                current_close = float(row["close"])
                if last_entry_price is not None and current_close > last_entry_price:
                    basket_open = False
                    entry_count = 0
                    entries = []
                    last_entry_price = None
                    last_entry_date = None
                    active_supply = [
                        z for z in active_supply if z not in touched
                    ]
                else:
                    # Consume supply while retaining the basket.
                    active_supply = [
                        z for z in active_supply if z not in touched
                    ]

        # Demand entry/averaging.
        touched_demand = [
            z for z in active_demand
            if i >= z["tradable_from"] and candle_touches_zone(row, z)
        ]

        if touched_demand:
            z = touched_demand[-1]
            price = float(row["close"])

            if not basket_open:
                basket_open = True
                entry_count = 1
                last_entry_price = price
                last_entry_date = row["date"]
                last_signal_zone = z
            elif entry_count < max_avg:
                entry_count += 1
                last_entry_price = price
                last_entry_date = row["date"]
                last_signal_zone = z

            # A zone is not repeatedly consumed on every candle.
            active_demand = [x for x in active_demand if x["id"] != z["id"]]

    latest = df.iloc[-1]
    current_price = float(latest["close"])
    latest_date = pd.to_datetime(latest["date"]).strftime("%Y-%m-%d")

    # Current-day/week signal has priority.
    current_demand = [
        z for z in zones
        if z["type"] == "demand"
        and z["tradable_from"] <= len(df) - 1
        and candle_touches_zone(latest, z)
    ]
    current_supply = [
        z for z in zones
        if z["type"] == "supply"
        and z["tradable_from"] <= len(df) - 1
        and candle_touches_zone(latest, z)
    ]

    if current_supply and basket_open:
        signal = "SELL / HOLD CHECK"
        reason = "Supply zone touched while a basket is open; check realized P&L manually."
    elif current_demand:
        signal = "AVERAGE" if basket_open else "BUY"
        reason = (
            "Demand zone touched."
            if not basket_open
            else f"Demand zone touched; existing basket has {entry_count}/{max_avg} entries."
        )
    elif basket_open:
        signal = "HOLD"
        reason = f"Basket open; waiting for next demand/supply event. Entries: {entry_count}/{max_avg}."
    else:
        signal = "WAIT"
        reason = "No current tradable demand-zone touch."

    suggested_capital = INITIAL_CAPITAL * TRADE_SIZE_PCT
    qty = int(suggested_capital // current_price) if current_price > 0 else 0

    return {
        "symbol": symbol.replace(".NS", ""),
        "signal": signal,
        "price": current_price,
        "quantity": qty,
        "capital": suggested_capital,
        "entries": entry_count,
        "max_avg": max_avg,
        "date": latest_date,
        "reason": reason,
    }


def calculate_signals(timeframe, x_pct, max_avg):
    results = []
    errors = []

    with DATA_LOCK:
        snapshot = dict(DATA)

    if not snapshot:
        return results, ["No market data is cached yet. Click Refresh Data."]

    for symbol, raw_df in snapshot.items():
        try:
            results.append(
                simulate_symbol(symbol, raw_df, timeframe, x_pct, max_avg)
            )
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")

    order = {
        "BUY": 0,
        "AVERAGE": 1,
        "SELL / HOLD CHECK": 2,
        "HOLD": 3,
        "WAIT": 4,
    }
    results.sort(key=lambda x: (order.get(x.get("signal"), 9), x["symbol"]))
    return results, errors


# ============================================================
# HTML
# ============================================================
HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIFTY50 SBC Supply/Demand Dashboard</title>
<style>
body { font-family: Arial, sans-serif; margin:0; background:#f5f7fa; color:#172033; }
.container { max-width:1400px; margin:auto; padding:22px; }
h1 { margin:0 0 6px; }
.small { color:#667085; font-size:13px; }
.card { background:white; border-radius:12px; padding:18px; margin:14px 0; box-shadow:0 2px 10px rgba(0,0,0,.06); }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; }
label { font-size:13px; font-weight:600; display:block; margin-bottom:5px; }
input,select,button { padding:9px 10px; border:1px solid #ccd3df; border-radius:7px; }
button { cursor:pointer; font-weight:700; }
.primary { background:#172033; color:white; }
table { width:100%; border-collapse:collapse; background:white; }
th,td { padding:9px 8px; border-bottom:1px solid #e8ebf0; text-align:left; font-size:13px; }
th { background:#f1f4f8; position:sticky; top:0; }
.badge { display:inline-block; padding:4px 8px; border-radius:999px; font-weight:700; font-size:12px; }
.buy { background:#d1fadf; color:#05603a; }
.avg { background:#dbeafe; color:#1e40af; }
.sell { background:#fee4e2; color:#b42318; }
.hold { background:#fef0c7; color:#92400e; }
.wait { background:#eef2f6; color:#475467; }
.error { color:#b42318; }
#refreshStatus { white-space:pre-wrap; }
</style>
</head>
<body>
<div class="container">
<div class="card">
<h1>NIFTY 50 — SBC Supply / Demand Signal Dashboard</h1>
<div class="small">Manual execution only. No broker orders are placed by this dashboard.</div>
</div>

<div class="card">
<form method="get" action="/">
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
<input name="x" type="number" step="0.1" value="{{ x_pct*100 }}">
</div>
<div>
<label>Max averaging entries</label>
<input name="max_avg" type="number" min="1" max="50" value="{{ max_avg }}">
</div>
<div>
<label>Capital</label>
<input name="capital" type="number" step="10000" value="{{ capital }}">
</div>
<div>
<button class="primary" type="submit">Calculate Signals</button>
</div>
</div>
</form>
</div>

<div class="card">
<div class="controls">
<div>
<button class="primary" onclick="refreshData()">Refresh Yahoo Data</button>
</div>
<div>
<b>Cached stocks:</b> {{ cached_count }}/50
</div>
<div>
<b>Last status:</b> <span id="refreshStatus">{{ refresh.message }}</span>
</div>
</div>
<div class="small" style="margin-top:8px">
Refresh runs in the background, so the webpage does not wait for Yahoo Finance and should not produce a 502 merely because Yahoo is slow or rate-limits a stock.
</div>
</div>

<div class="card">
<h2>Recommendations</h2>
<div class="small">
{{ timeframe|capitalize }} · X={{ "%.2f"|format(x_pct*100) }}% ·
Max averaging={{ max_avg }} · Suggested individual entry =
{{ "{:,.0f}".format(capital*0.01) }} (1% of capital)
</div>
<br>
<table>
<thead><tr>
<th>Stock</th><th>Signal</th><th>Price</th><th>Qty</th>
<th>Suggested Capital</th><th>Entries</th><th>Date</th><th>Reason</th>
</tr></thead>
<tbody>
{% for r in results %}
<tr>
<td><b>{{ r.symbol }}</b></td>
<td>
<span class="badge
{% if r.signal=='BUY' %}buy
{% elif r.signal=='AVERAGE' %}avg
{% elif 'SELL' in r.signal %}sell
{% elif r.signal=='HOLD' %}hold
{% else %}wait{% endif %}">
{{ r.signal }}
</span>
</td>
<td>{% if r.price is defined %}₹{{ "{:,.2f}".format(r.price) }}{% else %}-{% endif %}</td>
<td>{% if r.quantity is defined %}{{ r.quantity }}{% else %}-{% endif %}</td>
<td>{% if r.capital is defined %}₹{{ "{:,.0f}".format(r.capital) }}{% else %}-{% endif %}</td>
<td>{% if r.entries is defined %}{{ r.entries }}/{{ r.max_avg }}{% else %}-{% endif %}</td>
<td>{{ r.date|default("-") }}</td>
<td>{{ r.reason }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

{% if errors %}
<div class="card">
<h3>Non-fatal processing messages</h3>
<div class="small error">
{% for e in errors %}{{ e }}<br>{% endfor %}
</div>
</div>
{% endif %}
</div>

<script>
async function refreshData() {
    const status = document.getElementById("refreshStatus");
    status.textContent = "Starting background refresh...";

    try {
        const r = await fetch("/refresh", {method:"POST"});
        const data = await r.json();
        status.textContent = data.message;

        if (data.started) {
            pollRefresh();
        }
    } catch(e) {
        status.textContent = "Could not start refresh: " + e;
    }
}

async function pollRefresh() {
    try {
        const r = await fetch("/refresh_status");
        const data = await r.json();
        const status = document.getElementById("refreshStatus");
        status.textContent = data.message +
            " Success: " + data.success +
            " Failed: " + data.failed;

        if (data.running) {
            setTimeout(pollRefresh, 3000);
        } else {
            setTimeout(() => location.reload(), 1000);
        }
    } catch(e) {
        setTimeout(pollRefresh, 5000);
    }
}
</script>
</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def dashboard():
    try:
        timeframe = request.args.get("timeframe", "daily").lower()
        if timeframe not in ("daily", "weekly"):
            timeframe = "daily"

        x_pct = float(request.args.get("x", DEFAULT_X_PCT * 100)) / 100.0
        x_pct = max(0.0, min(x_pct, 10.0))

        max_avg = int(request.args.get("max_avg", DEFAULT_MAX_AVG))
        max_avg = max(1, min(max_avg, 50))

        capital = float(request.args.get("capital", INITIAL_CAPITAL))
        capital = max(1000.0, capital)

        # IMPORTANT: this is pure calculation from cache.
        # No Yahoo download occurs here.
        results, errors = calculate_signals(timeframe, x_pct, max_avg)

        # Capital is applied to display sizing.
        for r in results:
            if "price" in r:
                r["capital"] = capital * TRADE_SIZE_PCT
                r["quantity"] = int(r["capital"] // r["price"]) if r["price"] > 0 else 0

        with DATA_LOCK:
            status = dict(REFRESH_STATUS)
            cached_count = len(DATA)

        return render_template_string(
            HTML,
            results=results,
            errors=errors,
            timeframe=timeframe,
            x_pct=x_pct,
            max_avg=max_avg,
            capital=capital,
            refresh=status,
            cached_count=cached_count,
        )
    except Exception as exc:
        # Never let a strategy calculation exception kill the web worker.
        return (
            "<h2>Dashboard calculation error</h2>"
            f"<pre>{type(exc).__name__}: {exc}</pre>"
            "<p>Use the Refresh button after returning to the dashboard.</p>",
            200,
        )


@app.route("/health")
def health():
    with DATA_LOCK:
        count = len(DATA)
        running = REFRESH_STATUS["running"]
    return jsonify({
        "status": "ok",
        "cached_stocks": count,
        "refresh_running": running,
    })


@app.route("/refresh", methods=["POST"])
def refresh():
    started = start_refresh(period="5y")
    with DATA_LOCK:
        status = dict(REFRESH_STATUS)

    if started:
        return jsonify({
            "started": True,
            "message": "Background refresh started. You can stay on this page.",
        })

    return jsonify({
        "started": False,
        "message": "A refresh is already running.",
    })


@app.route("/refresh_status")
def refresh_status():
    with DATA_LOCK:
        return jsonify(dict(REFRESH_STATUS))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
