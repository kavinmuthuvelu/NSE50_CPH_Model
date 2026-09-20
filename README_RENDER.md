# NIFTY50 SBC Render Dashboard — Dhan Web Token

Dhan tokens expire daily. Enter a fresh token directly on the dashboard.

Daily workflow:
1. Generate today's Dhan API access token.
2. Paste it into **Dhan Access Token**.
3. Optionally enter Client ID.
4. Click **Save Token**.
5. Click **Sync Dhan Holdings**.

The webpage token takes priority over the optional `DHAN_ACCESS_TOKEN` Render environment variable. The entered token is held only in server memory; if Render restarts/sleeps/redeploys, enter it again.

The Dhan sync is read-only.
