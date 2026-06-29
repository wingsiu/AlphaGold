"""
v15 Pattern Registry — ATR-Based TP/SL (target_mode="atr")
===========================================================
All patterns use ATR(14)-scaled TP/SL multipliers instead of fixed absolute values.
This makes entries adapt to current market volatility automatically.

Key difference from v14:  execution blocks use `target_mode: "atr"` with
`tp_atr` and `sl_atr` as ATR multipliers, and provide approximate absolute
values for live scorer fallback via `tp`/`sl`.

Each pattern gets its own model directory under `v15_pattern_models/`.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_HYGIENE = {
    "min_warmup_bars": 240,
}

PATTERN_MODEL_DIR = PROJECT_ROOT / "runtime" / "bot_assets" / "v15_pattern_models"

# ATR multipliers are calibrated for gold 1-min ATR(14) ~ $1.2-1.8.
# tp/sl fields remain as approximate absolute fallbacks for live scorer
# when ATR column is unexpectedly missing.

V15_PATTERN_REGISTRY = {
    # =========================================================================
    # 1. Uptrend Retrace  (long: buy pullbacks in uptrend)
    # =========================================================================
    "uptrend_retrace_v15": {
        "direction_bias": "long",
        "priority": 1,
        "feature_set": "v2398",
        "thresholds": {"prob": 0.45},
        "context": [],
        "pattern": [
            {"feat": "rise_from_low_240", "op": ">=", "val": 30.0},
            {"feat": "drop_from_high_240", "op": ">=", "val": 10.0},
        ],
        "exclude": [
            {"feat": "near_high_zone", "op": "==", "val": 1.0},
        ],
        "execution": {
            "horizon": 15,
            "target_mode": "atr",
            "tp_atr": 13.0,
            "sl_atr": 10.0,
            "tp": 20.0,
            "sl": 15.0,
        },
    },
    # =========================================================================
    # 2. Downtrend Retrace  (short: sell retracements in downtrend)
    #    V14-style price-based: fell from high ≥25pts, bounced from low ≥5pts
    #    TP/SL from V38 best: H=480, TP=0.65×ATR5, SL=0.30×ATR5, prob=0.40
    #    V38 WF score: +1,368 PnL, 50.4% WR, 133 trades
    # =========================================================================
    "downtrend_retrace_v15": {
        "direction_bias": "short",
        "priority": 2,
        "feature_set": "v2398",
        "thresholds": {"prob": 0.40},
        "context": [],
        "pattern": [
            {"feat": "drop_from_high_240", "op": ">=", "val": 25.0},
            {"feat": "rise_from_low_240", "op": ">=", "val": 5.0},
        ],
        "exclude": [
            {"feat": "near_low_zone", "op": "==", "val": 1.0},
        ],
        # H=480, TP=10.0×ATR(~$20), SL=4.5×ATR(~$9)  (V38 best: +1368 PnL, 50.4% WR)
        "execution": {
            "horizon": 480,
            "target_mode": "atr",
            "tp_atr": 10.0,
            "sl_atr": 4.5,
            "tp": 20.0,
            "sl": 9.0,
        },
    },
}

V15_PRODUCTION_PATTERNS: tuple[str, ...] = (
    "uptrend_retrace_v15",
    "downtrend_retrace_v15",
)

V15_MIGRATED_PATTERNS: dict[str, str] = {
    "uptrend_retrace": "uptrend_retrace_v15",
}


def v15_pattern_prob_override() -> float | None:
    """Env override for pattern probability threshold."""
    raw = os.environ.get("V15_PATTERN_PROB_BASE", "").strip()
    return float(raw) if raw else None


def v15_pattern_feature_set() -> str:
    """Feature set for v15 training/inference."""
    return os.environ.get("V15_PATTERN_FEATURE_SET", "current").strip().lower()


def collect_v15_pa_groups(pattern_names: list[str] | None = None) -> tuple[str, ...]:
    """Union of pa_groups across active v15 patterns."""
    names = pattern_names if pattern_names is not None else list(V15_PATTERN_REGISTRY.keys())
    out: set[str] = set()
    for name in names:
        spec = V15_PATTERN_REGISTRY.get(name, {})
        for g in spec.get("pa_groups", ()):
            out.add(g)
    return tuple(sorted(out))
