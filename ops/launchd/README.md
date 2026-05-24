# TickerTape Launchd Jobs

These repo-managed launchd plists schedule the two-window laptop pipeline:

- `com.akhilesh.indianstocksdata.tickertape-primary.plist`: 06:15 local system time, primary local scrape only.
- `com.akhilesh.indianstocksdata.tickertape-repair-publish.plist`: 08:15 local system time, repair pass, publish gate, canonicalization, and server upload.

Install with:

```bash
mkdir -p "$HOME/Library/LaunchAgents"
cp ops/launchd/com.akhilesh.indianstocksdata.tickertape-*.plist "$HOME/Library/LaunchAgents/"
launchctl load "$HOME/Library/LaunchAgents/com.akhilesh.indianstocksdata.tickertape-primary.plist"
launchctl load "$HOME/Library/LaunchAgents/com.akhilesh.indianstocksdata.tickertape-repair-publish.plist"
```

The repair/publish job waits up to 3 hours for the primary job lock so it can still run if the scrape takes longer than the two-hour calendar gap.
