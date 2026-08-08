# Repository conventions

## Never commit bulk data artifacts

These paths are listed in `.gitignore` and must never be committed — not even with
`git add -f` or an explicit pathspec:

- `local_repository/raw/**` — daily TickerTape payloads (~4,000 files per refresh)
- `local_repository/tickertape.sqlite` — 5.5 GB
- `system/market_intel.db` — 1.3 GB

Do not run `git lfs track` on these paths, and do not recreate `.gitattributes`
entries for them.

**Why.** Committing them pushed hundreds of MB per daily refresh, and tracking the two
databases in Git LFS stored a full new copy of each on every revision — roughly 5.5 GB
of new LFS objects per run. The repository reached 5.5 GB with 205,068 tracked files
before these paths were untracked in `6a93158`, which brought the tip to 321 files.

**The data does not travel through git.** The serving path is the direct upload in
`scripts/run_tickertape_publish_pipeline.sh` (`run_upload` →
`cerebral-insights-platform/scripts/load_tickertape_to_server.sh`), and raw payloads
are excluded from it by default (`INCLUDE_RAW=0`). Nothing downstream reads these
artifacts out of git.

A refresh commit should contain logs and metadata only — on the order of a few hundred
files, not thousands.

## Cloning

History still contains the previously committed blobs, so a full clone downloads
several GB and may take more than ten minutes. Clone shallow and blobless:

```bash
git clone --depth=1 --filter=blob:none --single-branch --branch latest-data <url>
```

That yields 53 MB in about 8 seconds.
