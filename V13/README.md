# V13 (legacy)

v13 energetic bot + backtest, kept for reference.

## Run v13 bot
```bash
.venv/bin/python3 V13/trading_bot_v13.py
```

## Run v13 backtest
```bash
.venv/bin/python3 V13/backtest.py
```

## Layout
| Path | Contents |
|------|----------|
| `V13/trading_bot_v13.py` | Live bot |
| `V13/backtest.py` | Walk-forward backtest |
| `V13/config/v13_config.py` | Config (shim at `config/v13_config.py`) |
| `V13/xgboost/` | Train scripts + v13 model joblibs |
| `V13/runtime/bot_assets/` | `wf_models_v13`, prod joblibs |
| `V13/runtime/logs/` | Bot logs |
| `V13/legacy/` | Old v10 bots |
