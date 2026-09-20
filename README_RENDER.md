# NIFTY50 SBC Render V8 — GitHub Persistent Cache

## V8 architecture

GitHub is the persistent market-data cache. Render Persistent Disk is NOT required.

```text
GitHub data/*.parquet
        |
        v
Render temporary cache
        |
        +----> SBC dashboard
        |
        +----> Yahoo incremental refresh
                    |
                    v
              updated data
                    |
                    v
              ONE GitHub commit
```

## Render environment variables

Set these in Render:

```text
GITHUB_CACHE_REPO=kavinmuthuvelu/nifty50-sbc-data
GITHUB_CACHE_BRANCH=main
GITHUB_CACHE_DIR=data
GITHUB_CACHE_TOKEN=<your fine-grained GitHub token>
```

The token must have repository **Contents: Read and write** permission for `nifty50-sbc-data`.

Do not put the token in source code or the browser.

## Initial GitHub data

The repository should contain:

```text
data/
  ADANIENT.NS.parquet
  ADANIPORTS.NS.parquet
  ...
  RELIANCE.NS.parquet
  ...
  WIPRO.NS.parquet
```

## Runtime behavior

### Render restart

The app loads any local temporary cache and then pulls missing symbols from GitHub in the background.

### Yahoo refresh

The dashboard's **Smart Refresh Yahoo Data** button:

1. Uses existing cached history.
2. Requests only a small recent overlap from Yahoo for symbols already cached.
3. Downloads the full initial period only for a symbol that has no cache.
4. Merges/deduplicates candles.
5. Recalculates signals.
6. Saves a temporary local cache.
7. Writes all changed symbols to GitHub as a **single Git commit**.

This avoids 50 separate GitHub commits per refresh.

## Important

GitHub is intended for daily market-data persistence, not high-frequency writes.

## Dhan

The Dhan web Access Token entry remains available. The Dhan integration is read-only and does not place orders.
