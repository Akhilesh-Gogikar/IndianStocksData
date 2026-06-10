# IndianStocksData

## Market intelligence scaffold

A production-oriented market-intelligence + DBaaS scaffold is available in [`system/`](system/README.md).

It now includes:

- Daily ingestion orchestration.
- Queryable historical snapshots.
- Data-quality and freshness audits.
- Composable FastAPI profiles for market-data and agent-runtime APIs.
- Agent discovery endpoints and workflow manifests for AI-native integrations.
- Dockerized deployment path for easy hosting.
- Make targets that let you run the right API surface quickly.

## Quick commands

```bash
make api-profiles
make api-market-data
make api-agent-runtime
```

Use `PROFILE=full make api` to expose all API routes in one process for local smoke testing.
If you need the same profile split across the dedicated entrypoints, use `PROFILE=full make api-profiles`.
You can also run `PROFILE=full make api-profiles` to validate the available profiles without starting the server.
