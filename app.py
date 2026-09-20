
import os
import json
import time
import threading
import urllib.request
import urllib.error
import re
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

# DhanHQ configuration. Keep the access token in Render Environment Variables;
# never put the token in source code or the browser. Dhan access tokens are
# time-limited (typically 24 hours when generated from Dhan Web).
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "").strip()
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "").strip()
WEB_DHAN_TOKEN = ""
WEB_DHAN_CLIENT_ID = ""

# Dhan trading symbols do not always exactly match Yahoo symbols. These aliases
# cover common NIFTY 50 naming differences and can be extended if required.
DHAN_TO_YAHOO = {
    "M&M": "M&M.NS",
    "MM": "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "ETERNAL": "ETERNAL.NS",
}

DHAN_HOLDINGS = []
DHAN_STATUS = {
    "connected": bool(DHAN_ACCESS_TOKEN),
    "last_sync": None,
    "message": "Dhan is not configured. Add DHAN_ACCESS_TOKEN in Render Environment Variables (DHAN_CLIENT_ID is optional).",
    "count": 0,
}
DHAN_LOCK = threading.Lock()


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
# DHAN HOLDINGS
# ============================================================
def dhan_get_holdings():
    """Fetch current demat holdings from DhanHQ without exposing the token."""
    token = WEB_DHAN_TOKEN or DHAN_ACCESS_TOKEN
    client_id = WEB_DHAN_CLIENT_ID or DHAN_CLIENT_ID
    if not token:
        raise RuntimeError("Enter the Dhan API access token on the webpage first.")

    req = urllib.request.Request(
        "https://api.dhan.co/v2/holdings",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "access-token": token,
            **({"dhanClientId": client_id} if client_id else {}),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Dhan API HTTP {exc.code}: {body[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Dhan API connection failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError("Dhan API returned an invalid JSON response.") from exc

    if not isinstance(data, list):
        # Dhan may return an error object rather than the holdings array.
        raise RuntimeError(f"Dhan API returned an unexpected response: {str(data)[:500]}")

    return data


def normalize_dhan_symbol(symbol):
    """Normalize Dhan trading symbols for matching against the NIFTY50 universe."""
    s = str(symbol or "").strip().upper()
    # Common exchange/security suffixes returned by broker APIs.
    s = re.sub(r"[-_ ]?(EQ|BE|ETF|N1|N2|N3|N4|N5)$", "", s).strip()
    aliases = {
        "MM": "M&M",
        "M&M": "M&M",
        "BAJAJ-AUTO": "BAJAJ-AUTO",
        "BAJAJFINSV": "BAJAJFINSV",
        "BAJFINANCE": "BAJFINANCE",
        "TATAMOTORS": "TATAMOTORS",
        "ETERNAL": "ETERNAL",
    }
    return aliases.get(s, s)


def dhan_symbol_to_yahoo(trading_symbol):
    symbol = normalize_dhan_symbol(trading_symbol)
    if symbol in DHAN_TO_YAHOO:
        return DHAN_TO_YAHOO[symbol]
    return symbol + ".NS"


def sync_dhan_holdings():
    global DHAN_HOLDINGS, DHAN_STATUS
    holdings = dhan_get_holdings()

    normalized = []
    for h in holdings:
        symbol = str(h.get("tradingSymbol", "")).strip()
        if not symbol:
            continue
        normalized.append({
            "symbol": symbol,
            "normalized_symbol": normalize_dhan_symbol(symbol),
            "yahoo_symbol": dhan_symbol_to_yahoo(symbol),
            "security_id": str(h.get("securityId", "")),
            "isin": str(h.get("isin", "")),
            "total_qty": int(h.get("totalQty", 0) or 0),
            "available_qty": int(h.get("availableQty", 0) or 0),
            "dp_qty": int(h.get("dpQty", 0) or 0),
            "t1_qty": int(h.get("t1Qty", 0) or 0),
            "avg_cost": float(h.get("avgCostPrice", 0) or 0),
        })

    with DHAN_LOCK:
        DHAN_HOLDINGS = normalized
        DHAN_STATUS = {
            "connected": True,
            "last_sync": datetime.utcnow().isoformat(),
            "message": f"Dhan holdings synced successfully: {len(normalized)} holdings.",
            "count": len(normalized),
        }

    return normalized


def match_holdings_to_signals(holdings, signals):
    """Match live Dhan holdings to the dashboard's current SBC signals."""
    signal_map = {}
    for r in signals:
        sym = normalize_dhan_symbol(r.get("symbol", ""))
        signal_map[sym] = r

    rows = []
    for h in holdings:
        symbol = normalize_dhan_symbol(h.get("symbol", ""))
        sig = signal_map.get(symbol)

        signal = sig.get("signal", "NO SIGNAL") if sig else "NO SIGNAL"
        price = float(sig.get("price", 0) or 0) if sig else 0.0
        qty = int(h.get("total_qty", 0) or 0)
        avg_cost = float(h.get("avg_cost", 0) or 0)

        pnl = (price - avg_cost) * qty if price > 0 and avg_cost > 0 else None
        pnl_pct = ((price / avg_cost) - 1.0) * 100 if price > 0 and avg_cost > 0 else None

        if sig is None:
            match = "NOT IN NIFTY50"
        elif "SELL" in signal:
            match = "SELL SIGNAL"
        elif signal in ("BUY", "AVERAGE"):
            match = "BUY / ADD SIGNAL"
        elif signal == "HOLD":
            match = "HOLD"
        else:
            match = "WAIT"

        rows.append({
            **h,
            "signal": signal,
            "price": price,
            "match": match,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    return rows


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
<h2>🏦 Dhan Demat Holdings Sync</h2>
<div class="small">
Dhan API tokens expire daily. Enter the fresh token here each day — no Render redeploy is required.
The token is sent to the backend over HTTPS and kept only in server memory.
This feature is <b>read-only</b> and does not place or modify orders.
</div>
<div class="controls" style="margin-top:12px">
<div style="min-width:280px;flex:1">
<label style="display:block;font-size:12px;margin-bottom:5px">Dhan Access Token</label>
<input id="dhanToken" type="password" placeholder="Paste today's Dhan API access token"
style="width:100%;padding:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;color:#fff">
</div>
<div style="min-width:200px">
<label style="display:block;font-size:12px;margin-bottom:5px">Client ID (optional)</label>
<input id="dhanClientId" type="text" placeholder="Dhan Client ID"
style="width:100%;padding:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;color:#fff">
</div>
<div style="display:flex;gap:8px;align-items:end">
<button class="primary" onclick="saveDhanToken()">Save Token</button>
<button class="secondary" onclick="clearDhanToken()">Clear</button>
<button class="primary" onclick="syncDhan()">↻ Sync Dhan Holdings</button>
</div>
</div>
<div id="dhanStatus" class="small" style="margin-top:10px">{{ dhan_status.message }}</div>
<div class="small" style="margin-top:8px">
<b>Daily workflow:</b> Generate today's token in Dhan → paste → Save Token → Sync Dhan Holdings.
If Render restarts or sleeps, enter the token again.
</div>
<div class="controls" style="margin-top:10px">
<div><b>Configured:</b> {{ "Yes" if dhan_status.connected else "No" }}</div>
<div><b>Holdings:</b> {{ dhan_status.count }}</div>
<div><b>Last sync:</b> {{ dhan_status.last_sync|default("-") }}</div>
</div>
</div>

{% if matched_holdings %}
<div class="card">
<h2>📊 My Dhan Holdings vs SBC Signal</h2>
<div class="small" style="margin-bottom:10px">
The table includes all Dhan holdings. NIFTY50 holdings are matched to the current signal; other holdings are marked <b>NOT IN NIFTY50</b>.
</div>
<table>
<thead><tr>
<th>Stock</th><th>Holding Qty</th><th>Avg Cost</th><th>Signal Price</th>
<th>Holding P&L</th><th>SBC Signal</th><th>Match</th>
</tr></thead>
<tbody>
{% for h in matched_holdings %}
<tr>
<td><b>{{ h.symbol }}</b></td>
<td>{{ h.total_qty }}</td>
<td>₹{{ "{:,.2f}".format(h.avg_cost) }}</td>
<td>{% if h.price %}₹{{ "{:,.2f}".format(h.price) }}{% else %}-{% endif %}</td>
<td>{% if h.pnl is not none %}₹{{ "{:,.2f}".format(h.pnl) }} ({{ "%.2f"|format(h.pnl_pct) }}%){% else %}-{% endif %}</td>
<td>
<span class="badge
{% if h.signal=='BUY' %}buy
{% elif h.signal=='AVERAGE' %}avg
{% elif 'SELL' in h.signal %}sell
{% elif h.signal=='HOLD' %}hold
{% else %}wait{% endif %}">{{ h.signal }}</span>
</td>
<td><b>{{ h.match }}</b></td>
</tr>
{% endfor %}
</tbody>
</table>
</div>
{% endif %}

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

async function saveDhanToken(){
  const token=(document.getElementById("dhanToken")?.value||"").trim();
  const clientId=(document.getElementById("dhanClientId")?.value||"").trim();
  const status=document.getElementById("dhanStatus");
  if(!token){if(status)status.textContent="Please paste today's Dhan access token.";return;}
  if(status)status.textContent="Saving token…";
  try{
    const r=await fetch("/dhan/credentials",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({token:token,client_id:clientId})});
    const j=await r.json();
    if(!r.ok||!j.ok)throw new Error(j.error||"Unable to save token");
    document.getElementById("dhanToken").value="";
    if(status)status.textContent="✓ Token saved. Now click Sync Dhan Holdings.";
  }catch(e){if(status)status.textContent="Error: "+e.message;}
}
async function clearDhanToken(){
  try{
    await fetch("/dhan/credentials/clear",{method:"POST"});
    const status=document.getElementById("dhanStatus");
    if(status)status.textContent="Dhan token cleared.";
  }catch(e){}
}

async function syncDhan() {
    const status = document.getElementById("dhanStatus");
    status.textContent = "Syncing Dhan holdings...";
    try {
        const r = await fetch("/dhan/sync", {method:"POST"});
        const data = await r.json();
        status.textContent = data.message || "Dhan sync completed.";
        setTimeout(() => location.reload(), 800);
    } catch(e) {
        status.textContent = "Could not sync Dhan holdings: " + e;
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
        with DHAN_LOCK:
            dhan_status = dict(DHAN_STATUS)
            dhan_holdings = list(DHAN_HOLDINGS)

        matched_holdings = match_holdings_to_signals(dhan_holdings, results) if dhan_holdings else []

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
            dhan_status=dhan_status,
            matched_holdings=matched_holdings,
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
    with DHAN_LOCK:
        dstatus = dict(DHAN_STATUS)
    return jsonify({
        "status": "ok",
        "cached_stocks": count,
        "refresh_running": running,
        "dhan_connected": dstatus["connected"],
        "dhan_holdings": dstatus["count"],
        "dhan_last_sync": dstatus["last_sync"],
    })


@app.route("/dhan/sync", methods=["POST"])
def dhan_sync():
    try:
        holdings = sync_dhan_holdings()
        return jsonify({
            "success": True,
            "count": len(holdings),
            "message": f"Dhan holdings synced successfully: {len(holdings)} holdings loaded.",
        })
    except Exception as exc:
        with DHAN_LOCK:
            DHAN_STATUS["connected"] = bool(os.environ.get("DHAN_ACCESS_TOKEN", "").strip())
            DHAN_STATUS["message"] = str(exc)
        return jsonify({
            "success": False,
            "count": 0,
            "message": f"Dhan sync failed: {exc}",
        }), 200


@app.route("/dhan/credentials", methods=["POST"])
def dhan_credentials():
    global WEB_DHAN_TOKEN, WEB_DHAN_CLIENT_ID
    data = request.get_json(silent=True) or {}
    token = str(data.get("token", "")).strip()
    client_id = str(data.get("client_id", "")).strip()
    if not token:
        return jsonify({"ok": False, "error": "Please enter the Dhan access token."}), 400
    WEB_DHAN_TOKEN = token
    WEB_DHAN_CLIENT_ID = client_id
    with DHAN_LOCK:
        DHAN_STATUS["connected"] = False
        DHAN_STATUS["message"] = "Token saved. Click Sync Dhan Holdings."
    return jsonify({"ok": True})


@app.route("/dhan/credentials/clear", methods=["POST"])
def dhan_credentials_clear():
    global WEB_DHAN_TOKEN, WEB_DHAN_CLIENT_ID
    WEB_DHAN_TOKEN = ""
    WEB_DHAN_CLIENT_ID = ""
    with DHAN_LOCK:
        DHAN_HOLDINGS.clear()
        DHAN_STATUS["connected"] = False
        DHAN_STATUS["count"] = 0
        DHAN_STATUS["message"] = "Dhan token cleared."
    return jsonify({"ok": True})


@app.route("/dhan/status")
def dhan_status_api():
    with DHAN_LOCK:
        holdings = list(DHAN_HOLDINGS)
        status = dict(DHAN_STATUS)
    return jsonify({"status": status, "holdings": holdings})


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
