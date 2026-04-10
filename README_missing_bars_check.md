# Missing Bars Check (AUD / Gold / Oil)

This script checks MySQL price tables for missing 1-minute bars larger than a threshold, while ignoring gaps that span weekends or full no-data weekdays (treated as holidays).

## Script

- `check_missing_bars.py`

## Default table mapping

- `aud` -> `aud_prices`
- `gold` -> `gold_prices`
- `oil` -> `prices`

## Quick run

```bash
python3 check_missing_bars.py --host 192.168.0.4
```

The script uses DB credentials from `.env`:

- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`

You can override by args:

```bash
python3 check_missing_bars.py --host 192.168.0.4 --user YOUR_USER --password YOUR_PASS --database YOUR_DB
```

## Change threshold

```bash
python3 check_missing_bars.py --host 192.168.0.4 --max-missing-bars 2
```

## Sync only latest rows from remote MySQL

Use `sync_latest_from_remote_mysql.py` to copy only rows with `timestamp > local MAX(timestamp)` for:

- `aud_prices`
- `gold_prices`
- `prices` (oil)

Dry run (no local writes):

```bash
python3 sync_latest_from_remote_mysql.py --remote-host 192.168.0.4 --dry-run
```

Run sync:

```bash
python3 sync_latest_from_remote_mysql.py --remote-host 192.168.0.4
```

You can set explicit local target if needed:

```bash
python3 sync_latest_from_remote_mysql.py \
  --remote-host 192.168.0.4 \
  --local-host 127.0.0.1 \
  --local-port 3306
```
