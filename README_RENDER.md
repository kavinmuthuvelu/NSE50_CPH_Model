# NIFTY50 SBC Supply/Demand Render Dashboard

## Render settings

Build Command:
`pip install -r requirements.txt`

Start Command:
`gunicorn --workers 1 --timeout 120 --access-logfile - app:app`

The application deliberately does NOT download Yahoo Finance data during Flask/Gunicorn startup.

Open the dashboard and click **Refresh Yahoo Data**. The refresh runs in a background thread and the dashboard remains available even if Yahoo temporarily rate-limits individual stocks.

The dashboard uses cached data when available and preserves previously cached stocks if a refresh fails for a symbol.

Manual trading only. No broker orders are placed.
