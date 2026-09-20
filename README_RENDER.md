# NIFTY50 SBC Render Dashboard — Dhan Sync

## Dhan configuration in Render

In **Render → Service → Environment**, add:

- `DHAN_ACCESS_TOKEN` = your current Dhan API access token
- `DHAN_CLIENT_ID` = your Dhan client ID (optional, but recommended)

Redeploy after saving environment variables.

## Dhan sync

The dashboard has a **Sync Dhan Holdings** button.

It:
1. Calls Dhan `/v2/holdings` from the Render backend.
2. Normalizes broker trading symbols.
3. Matches holdings against the dashboard's NIFTY50 SBC signals.
4. Shows holding quantity, average cost, signal price and calculated holding P&L.
5. Marks each holding as `BUY / ADD SIGNAL`, `SELL SIGNAL`, `HOLD`, `WAIT`, or `NOT IN NIFTY50`.

This is **read-only**. No Dhan orders are placed by the sync feature.

The Dhan access token remains server-side in Render environment variables and is not exposed in the browser.

## Start command

If Render's Start Command field is used directly:

`gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:$PORT --access-logfile - app:app`

Do not prefix that command with `web:` in Render's Start Command field.
