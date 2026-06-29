# Legacy pre-v16 production (archived 2026-06-29)

Moved here to keep repo root clean. **Do not import from production v16 code.**

## Contents

| Path | Was |
|------|-----|
| `bots/` | Root-level trading bots (v14, v15, old oil) |
| `launchers/` | Old backtest runners, parity scripts, stats helpers |
| `V13/` | v13 legacy tree |
| `v15/` | v15 hybrid system (backtest + live reference) |
| `plists/` | Old launchd plist (v15 hybrid) |
| `misc/` | Stray root CSVs and notes |

## Active production (repo root)

- `trading_bot_gold_v16.py` / `trading_bot_oil_v16.py`
- `run_gold_v16_backtest.py` / `run_oil_v16_backtest.py`
- `v16/` package
- `watchdog_bots.sh` + `scripts/install_*`

Restore a file by copying back to root only if you need to compare old behaviour.
