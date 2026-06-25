# v14 legacy archive

Archived **2026-06-25**. The entire `v14/` source tree is at `source_tree/`.

## Production layout (current)

See **`AGENTS.md`** at repo root.

## Contents

```
archive/v14_legacy/
├── source_tree/          # full former v14/ directory
├── bots/                 # trading_bot_hybrid_v14.py
├── research/             # v14 research sweeps
├── docs/                 # baseline docs
├── dev_scripts/
├── launchd/              # original v14 bot launchd
└── runtime_artifacts/    # old CSVs and logs
```

## Config shims (repo root)

`config/v14_config.py` and `config/v14_patterns.py` re-export from `hybrid_config` / `pattern_registry` for one release cycle.

## Model files

On-disk dirs `wf_models_v14` and `wf_models_v14_patterns` are symlinked as `wf_models` and `pattern_models`. Joblib filenames still contain `v14` until a future rename migration.
