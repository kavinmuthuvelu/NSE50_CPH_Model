# NIFTY 50 SBC Manual Signal Dashboard

## Files
- `app.py` — Flask dashboard and signal engine
- `requirements.txt` — Render dependencies
- `.python-version` — Python 3.13

## Render
Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn app:app`

The app binds to `0.0.0.0:$PORT` and also exposes `/health`.

## Strategy
The signal engine mirrors the attached SBC Supply/Demand backtester:
- SBC = Body / (High-Low) <= 0.50
- Unlimited consecutive SBC candles form one base
- X confirmation is configurable
- Demand zone tap -> BUY
- Demand zone tap with open basket -> AVERAGE
- Supply zone tap with profitable basket -> SELL
- Supply zone tap while losing -> HOLD
- Maximum averaging is configurable
- Suggested manual allocation = 1% of configured capital

## Important
The dashboard is signal-only. It does not place broker orders.
The current displayed capital is a configured/manual planning value; it is not automatically synchronized with your real brokerage account.
