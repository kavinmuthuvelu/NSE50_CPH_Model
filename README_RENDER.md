# NIFTY50 SBC Dashboard — Dhan + Server Yahoo Cache

This version keeps the existing Dhan web-token workflow and adds a server-side Yahoo Finance history cache.

## Yahoo cache behavior
- Dashboard page loads **never download Yahoo data**. Signals are calculated from cached data.
- Click **Smart Refresh Yahoo Data** only when you want to update market data.
- If a stock already has cached history, the app requests only the recent 10-calendar-day overlap and merges it into the existing history. It does **not** download the full 5-year history again.
- Stocks with no cache receive the initial 5-year download.
- Existing cached history is preserved if Yahoo temporarily fails for a symbol.
- Cache is stored under `RENDER_DISK_PATH/sbc_cache` when Render persistent disk is configured; otherwise it uses `/tmp/sbc_cache`, which can be cleared when the service restarts/redeploys.

## Recommended Render setup
For cache persistence across restarts/redeploys, attach a Render persistent disk and set/use its mount path as `RENDER_DISK_PATH`.

## Dhan
1. Generate today's Dhan API access token.
2. Paste it into **Dhan Access Token** on the webpage.
3. Optionally enter Client ID.
4. Save Token.
5. Click **Sync Dhan Holdings**.

The Dhan sync is read-only. The web-entered token is kept in server memory and is not written to GitHub or browser local storage.

## Start command
`gunicorn --workers 1 --timeout 120 --access-logfile - app:app`
