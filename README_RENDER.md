# NIFTY 50 SBC Manual Signal Dashboard

## Render deployment

Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn --workers 1 --timeout 180 app:app`

The app binds to `0.0.0.0:$PORT`.

## What was fixed for Render
- Yahoo Finance downloads are split into small batches to reduce rate limiting.
- Downloads use one thread and retry logic.
- Zone confirmation checks the impulse candle only when it exists.
- One Gunicorn worker is used to reduce memory usage on Render Free.
- Request timeout is extended for the first cold calculation.

## Strategy
- SBC = Body / (High-Low) <= 0.50
- Consecutive SBC candles form one base.
- Demand confirmation = close above base high by X%.
- Supply confirmation = close below base low by X%.
- Demand tap -> BUY.
- Demand tap with open basket -> AVERAGE.
- Supply tap with profitable basket -> SELL.
- Supply tap while losing -> HOLD.
- Signals are recommendations only; trades are executed manually.

## Important
The dashboard does not place broker orders.
The displayed capital is a manual planning input and is not connected to a brokerage account.
