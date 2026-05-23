# Local TickerTape Repository

This folder is the local data repository that drives future development.

Tracked files:

- `README.md`: this contract.
- `.gitignore`: keeps bulk local data out of git.

Generated files:

- `tickertape.sqlite`: local SQLite database with company metadata, latest stock records, snapshot metadata, financial sections, event sections, and sync history.
- `raw/YYYY-MM-DD/*.page_props.json.gz`: compressed raw `pageProps` payloads extracted from TickerTape pages.
- `logs/*.log`: local sync logs.

The database is local by design. Do not commit scraped datasets or raw payloads unless there is a deliberate licensing and storage decision.

Run:

```bash
bash scripts/run_tickertape_daily_sync.sh
```

Smoke test:

```bash
bash scripts/run_tickertape_daily_sync.sh --limit 3
```

Status:

```bash
python3 tools/tickertape_status.py
```
