from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG_PATH = PROJECT_ROOT / "config" / "active_paths.json"

# Centralized deployment defaults: replace files under config/artifacts only.
DEFAULT_SIGNAL_MODEL_PATH = "config/artifacts/model_bundle.joblib"
DEFAULT_WEAK_PERIODS_JSON = "config/artifacts/weak-filter.json"

# Compatibility fallback while migrating existing setups.
LEGACY_SIGNAL_MODEL_PATH = "runtime/bot_assets/backtest_model_best_base_weak_nostate.joblib"
LEGACY_WEAK_PERIODS_JSON = "runtime/bot_assets/weak-filter.json"


def _exists(path_like: str) -> bool:
    p = Path(path_like)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.exists()


def load_runtime_paths() -> dict[str, str]:
    data = {
        "signal_model_path": DEFAULT_SIGNAL_MODEL_PATH,
        "weak_periods_json": DEFAULT_WEAK_PERIODS_JSON,
    }
    if not ACTIVE_CONFIG_PATH.exists():
        return data
    try:
        payload = json.loads(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return data
    if isinstance(payload, dict):
        signal_model_path = payload.get("signal_model_path")
        weak_periods_json = payload.get("weak_periods_json")
        if isinstance(signal_model_path, str) and signal_model_path.strip():
            data["signal_model_path"] = signal_model_path.strip()
        if isinstance(weak_periods_json, str) and weak_periods_json.strip():
            data["weak_periods_json"] = weak_periods_json.strip()

    if not _exists(data["signal_model_path"]) and _exists(LEGACY_SIGNAL_MODEL_PATH):
        data["signal_model_path"] = LEGACY_SIGNAL_MODEL_PATH
    if not _exists(data["weak_periods_json"]) and _exists(LEGACY_WEAK_PERIODS_JSON):
        data["weak_periods_json"] = LEGACY_WEAK_PERIODS_JSON
    return data
