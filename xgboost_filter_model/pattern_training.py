"""Shared helpers for pattern specialist training and backtest (single-stage)."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

from config.v14_config import MODEL_CONFIG, WF_CONFIG
from config.v14_patterns import EXCLUDE_COLS, PATTERN_MODEL_DIR, PATTERN_REGISTRY, PROJECT_ROOT


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def execution_target_mode(execution: dict) -> str:
    return str(execution.get("target_mode", "fixed")).strip().lower()


def execution_tp_sl(execution: dict) -> tuple[float, float]:
    """Return TP/SL — fixed DB units or ATR multipliers when target_mode=='atr'."""
    if execution_target_mode(execution) == "atr":
        tp = execution.get("tp_atr", execution.get("tp"))
        sl = execution.get("sl_atr", execution.get("sl"))
    else:
        tp, sl = execution["tp"], execution["sl"]
    return float(tp), float(sl)


def dynamic_tp_sl_series(df: pd.DataFrame, execution: dict) -> tuple[pd.Series, pd.Series]:
    """Per-bar TP/SL for labeling or backtest (ATR × mult or fixed)."""
    tp, sl = execution_tp_sl(execution)
    if execution_target_mode(execution) == "atr":
        if "atr" not in df.columns:
            raise ValueError("atr column required for ATR-based TP/SL")
        atr = df["atr"].astype(float)
        return atr * tp, atr * sl
    return pd.Series(tp, index=df.index), pd.Series(sl, index=df.index)


def assign_exec_tp_sl(df: pd.DataFrame, index: pd.Index, execution: dict) -> None:
    """Write exec_tp / exec_sl for signal rows."""
    tp_s, sl_s = dynamic_tp_sl_series(df.loc[index], execution)
    df.loc[index, "exec_tp"] = tp_s.values
    df.loc[index, "exec_sl"] = sl_s.values


def add_pattern_entry_target(df: pd.DataFrame, direction_bias: str) -> pd.DataFrame:
    """Label: 1 if trade in pattern direction hits TP before SL within horizon."""
    df = df.copy()
    if direction_bias == "long":
        df["target_pattern"] = (
            (df["future_max_move"] >= df["dynamic_tp"])
            & (df["future_min_move"] <= df["dynamic_sl"])
        ).astype(int)
    else:
        df["target_pattern"] = (
            (df["future_min_move"] >= df["dynamic_tp"])
            & (df["future_max_move"] <= df["dynamic_sl"])
        ).astype(int)
    return df


def apply_exec_labels(
    df: pd.DataFrame,
    horizon: int,
    tp: float,
    sl: float,
    *,
    future_moves: pd.DataFrame | None = None,
    target_mode: str = "fixed",
) -> pd.DataFrame:
    """Attach future moves + dynamic TP/SL for pattern training (no feature recompute)."""
    out = df.copy()
    if future_moves is not None:
        out["future_max_move"] = future_moves["future_max_move"]
        out["future_min_move"] = future_moves["future_min_move"]
    else:
        from xgboost_filter_model.train_filter_v14 import build_target

        moves = build_target(out[["open", "high", "low", "close"]], horizon, tp, sl)
        out["future_max_move"] = moves["future_max_move"]
        out["future_min_move"] = moves["future_min_move"]
    tp_s, sl_s = dynamic_tp_sl_series(out, {"target_mode": target_mode, "tp": tp, "sl": sl})
    out["dynamic_tp"] = tp_s
    out["dynamic_sl"] = sl_s
    return out


def precompute_future_moves(df: pd.DataFrame, horizons: list[int]) -> dict[int, pd.DataFrame]:
    """Cache future_max/min_move per horizon (TP/SL do not affect these)."""
    from xgboost_filter_model.train_filter_v14 import build_target

    ohlc = df[["open", "high", "low", "close"]]
    return {
        h: build_target(ohlc, h, tp=1.0, sl=1.0)[["future_max_move", "future_min_move"]]
        for h in horizons
    }


def pattern_execution(pattern_name: str) -> dict:
    """Return execution block from PATTERN_REGISTRY."""
    return dict(PATTERN_REGISTRY[pattern_name]["execution"])


def pattern_horizons(pattern_names: list[str]) -> list[int]:
    return sorted({pattern_execution(n)["horizon"] for n in pattern_names})


def label_df_for_pattern(
    df_feat: pd.DataFrame,
    pattern_name: str,
    future_by_h: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Apply pattern-specific H/TP/SL labels (not global EXECUTION_CONFIG)."""
    ex = pattern_execution(pattern_name)
    h = int(ex["horizon"])
    tp, sl = execution_tp_sl(ex)
    mode = execution_target_mode(ex)
    return apply_exec_labels(
        df_feat, h, tp, sl, future_moves=future_by_h[h], target_mode=mode
    )


def fit_pattern_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    min_samples: int = 20,
) -> xgb.XGBClassifier | None:
    if len(X) < min_samples or y.sum() == 0 or y.sum() == len(y):
        return None
    cfg = MODEL_CONFIG["s1"].copy()
    cfg["scale_pos_weight"] = (len(y) - y.sum()) / y.sum()
    model = xgb.XGBClassifier(**cfg)
    model.fit(X, y)
    return model


def pattern_model_dir(base: Path, pattern_name: str, *, variant: str | None = None) -> Path:
    d = base / pattern_name
    if variant:
        d = d / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def pattern_variant_tag(
    horizon: int, tp: float, sl: float, *, target_mode: str = "fixed"
) -> str:
    if execution_target_mode({"target_mode": target_mode}) == "atr":
        tp_s = str(tp).replace(".", "p")
        sl_s = str(sl).replace(".", "p")
        return f"h{int(horizon)}_atr{tp_s}x_{sl_s}x"
    return f"h{int(horizon)}_tp{int(tp)}_sl{int(sl)}"


def prod_model_path(out_dir: Path) -> Path:
    """Single-stage model (filter_prod.joblib kept for backward compatibility)."""
    return out_dir / "filter_prod.joblib"


def cycle_model_path(out_dir: Path, cycle: int, start_date) -> Path:
    return out_dir / f"filter_cycle_{cycle}_{start_date}.joblib"


def wf_anchor_ts() -> pd.Timestamp:
    """Canonical WF anchor — Friday 17:00 NY (2025-01-03 22:00 UTC), same as v13/v14."""
    wf_start = pd.to_datetime(WF_CONFIG["wf_start"])
    if wf_start.tzinfo is None:
        return wf_start.tz_localize("UTC")
    return wf_start.tz_convert("UTC")


def wf_timestamps():
    return wf_anchor_ts(), WF_CONFIG["retrain_days"]


def prod_train_slice(df_pat: pd.DataFrame, wf_start: pd.Timestamp) -> pd.DataFrame:
    """Bars before first walk-forward cycle (or fallback slice) for prod model."""
    data_start = df_pat.index.min()
    retrain_days = WF_CONFIG["retrain_days"]
    first_cycle_end = wf_start + pd.Timedelta(days=retrain_days)
    df_pre = df_pat[df_pat.index < first_cycle_end]
    if len(df_pre) < 50:
        mid = data_start + (df_pat.index.max() - data_start) / 2
        df_pre = df_pat[df_pat.index < mid]
    if len(df_pre) < 30:
        df_pre = df_pat.iloc[: max(30, len(df_pat) // 2)]
    return df_pre


def backup_variant_models(out_dir: Path, *, tag: str = "") -> None:
    """Copy existing prod + cycle models before overwrite."""
    prod = prod_model_path(out_dir)
    if not prod.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap_root = PROJECT_ROOT / "v14" / "runtime" / "model_snapshots"
    snap = snap_root / out_dir.relative_to(PATTERN_MODEL_DIR) / f"{stamp}{('_' + tag) if tag else ''}"
    snap.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prod, snap / prod.name)
    for cp in sorted(out_dir.glob("filter_cycle_*.joblib")):
        shutil.copy2(cp, snap / cp.name)


def wf_cycle_at(
    dt: pd.Timestamp,
    wf_anchor: pd.Timestamp | None = None,
    retrain_days: int | None = None,
) -> tuple[pd.Timestamp, int]:
    """Return (cycle_start, cycle_num) on the canonical bi-weekly Friday grid."""
    anchor = wf_anchor if wf_anchor is not None else wf_anchor_ts()
    rd = retrain_days if retrain_days is not None else WF_CONFIG["retrain_days"]
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    elapsed = max(0, (dt - anchor).days)
    skip = elapsed // rd
    return anchor + pd.Timedelta(days=skip * rd), 1 + skip


def wf_cycle_window(
    cycle_num: int,
    wf_anchor: pd.Timestamp | None = None,
    retrain_days: int | None = None,
) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """Return (cycle_num, cycle_start, cycle_end) for a 1-based WF cycle on the anchor grid."""
    anchor = wf_anchor if wf_anchor is not None else wf_anchor_ts()
    rd = retrain_days if retrain_days is not None else WF_CONFIG["retrain_days"]
    cur = anchor
    cycle = 1
    while cycle < cycle_num:
        cur = cur + pd.Timedelta(days=rd)
        cycle += 1
    return cycle, cur, cur + pd.Timedelta(days=rd)


def wf_train_mode() -> str:
    """incremental (default): only latest cycle. full: rewrite every cycle file."""
    mode = os.environ.get("V14_WF_TRAIN_MODE", "incremental").strip().lower()
    return mode if mode in ("full", "incremental") else "incremental"


def wf_train_as_of() -> pd.Timestamp:
    raw = os.environ.get("V14_WF_TRAIN_AS_OF", "").strip()
    if raw:
        ts = pd.Timestamp(raw)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return pd.Timestamp.now(tz="UTC")


def wf_force_latest_cycle() -> bool:
    return os.environ.get("V14_WF_FORCE_LATEST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def wf_train_grace_days() -> int:
    return int(WF_CONFIG.get("wf_train_grace_days", 1))


def wf_incremental_train_target(
    as_of: pd.Timestamp | None = None,
) -> tuple[int, pd.Timestamp] | None:
    """
    Cycle model to train for the active WF window, only after it has started.

    Example: cycle 38 starts 2026-06-05 → training allowed from 2026-06-06
    (wf_train_grace_days after start). Uses data strictly before 2026-06-05.

    Before that (e.g. 2026-05-30 still in cycle 37): returns None so we do not
    re-run mid-cycle. Pair with last_trained_wf_cycle + no V14_WF_FORCE_LATEST.
    """
    as_of = as_of or wf_train_as_of()
    anchor = wf_anchor_ts()
    rd = int(WF_CONFIG["retrain_days"])
    grace = wf_train_grace_days()

    start, cycle = wf_cycle_at(as_of, anchor, rd)
    # Calendar-day grace (anchor cycles start 22:00 UTC — compare dates, not timestamps)
    train_after = start.date() + pd.Timedelta(days=grace)
    if as_of.date() < train_after:
        return None
    return cycle, start


def iter_wf_train_targets(
    cycle_model_path_fn,
    *,
    as_of: pd.Timestamp | None = None,
) -> list[tuple[int, pd.Timestamp]]:
    """
    Which WF cycles to train this run.

    incremental: one model for the current cycle only (train data strictly before
    cycle_start). Older cycle_*.joblib files are not touched.
    full: every cycle from wf_start through as_of (legacy behaviour).
    """
    as_of = as_of or wf_train_as_of()
    anchor = wf_anchor_ts()
    rd = int(WF_CONFIG["retrain_days"])

    if wf_train_mode() == "full":
        out: list[tuple[int, pd.Timestamp]] = []
        cur = anchor
        cycle = 1
        while cur < as_of:
            out.append((cycle, cur))
            cur = cur + pd.Timedelta(days=rd)
            cycle += 1
        return out

    grace = wf_train_grace_days()
    pending = wf_incremental_train_target(as_of)
    if pending is None:
        cur_start, cur_cycle = wf_cycle_at(as_of, anchor, rd)
        train_after = cur_start.date() + pd.Timedelta(days=grace)
        print(
            f"  WF incremental: skip — wait until {train_after} to train "
            f"cycle_{cur_cycle} (starts {cur_start.date()})"
        )
        return []

    cycle, start = pending
    path = cycle_model_path_fn(cycle, start.date())
    if wf_force_latest_cycle() or not path.exists():
        return [(cycle, start)]
    print(
        f"  WF incremental: keep cycle_{cycle} ({start.date()}) — "
        f"exists at {path.name}"
    )
    return []


def fixed_wf_cycle_from_env() -> tuple[int, pd.Timestamp] | None:
    """V14_FIXED_WF_CYCLE=21 pins scoring to that cycle's trained models."""
    raw = os.environ.get("V14_FIXED_WF_CYCLE", "").strip()
    if not raw:
        return None
    cycle_num = int(raw)
    _, cycle_start, _ = wf_cycle_window(cycle_num)
    return cycle_num, cycle_start


def iter_wf_cycles(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    wf_anchor: pd.Timestamp | None = None,
    retrain_days: int | None = None,
):
    """Yield (cycle_num, cycle_start, cycle_end) aligned to WF_CONFIG wf_start."""
    anchor = wf_anchor if wf_anchor is not None else wf_anchor_ts()
    rd = retrain_days if retrain_days is not None else WF_CONFIG["retrain_days"]
    if start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize("UTC")
    else:
        start_dt = start_dt.tz_convert("UTC")
    if end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize("UTC")
    else:
        end_dt = end_dt.tz_convert("UTC")
    cur, cycle = wf_cycle_at(start_dt, anchor, rd)
    while cur < end_dt:
        ce = min(cur + pd.Timedelta(days=rd), end_dt)
        yield cycle, cur, ce
        cur = ce
        cycle += 1


def simulator_s2_prob(prob: float, direction_bias: str) -> float:
    """Map P(direction win) to s2_prob expected by simulate_v13_core."""
    return float(prob) if direction_bias == "long" else float(1.0 - prob)
