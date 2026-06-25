"""Oil walk-forward XGBoost model loading — shared by live bot and backtest."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
OIL_MODEL_DIR = _REPO_ROOT / "v15" / "oil" / "wf_models"


def model_path(stype: str, month: str) -> Path:
    """Path to saved model: v15/oil/wf_models/{wr90|ret|si}/{YYYY-MM}.joblib"""
    p = OIL_MODEL_DIR / stype / f"{month}.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=32)
def _load_model(path: str):
    return joblib.load(path)


def score_wf_model(
    stype: str,
    bar_ts: pd.Timestamp,
    row,
    feat_names: list[str],
) -> float | None:
    """Score one bar with the monthly WF model (same as backtest train_ml)."""
    ts = pd.Timestamp(bar_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    month_str = str(pd.Period(ts, "M"))
    path = model_path(stype, month_str)
    if not path.exists():
        return None
    model = _load_model(str(path))
    X = np.array([[float(row.get(f, 0) if hasattr(row, "get") else row[f]) for f in feat_names]])
    return float(model.predict_proba(X)[0, 1])
