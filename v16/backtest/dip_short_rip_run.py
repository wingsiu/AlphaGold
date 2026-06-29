"""Run dip_short_rip backtest with v15-style single-position execution."""
from __future__ import annotations

import pandas as pd

from v16.backtest.ml import walk_forward_short_scores
from v16.backtest.position_sim import simulate_single_position
from v16.patterns.dip_short_rip import (
    build_labeled_set,
    feature_columns,
    resolve_execution,
    router_mask,
)


def run_dip_short_rip(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    cfg: dict,
    *,
    mechanical: bool,
    ml_prob: float | None = None,
) -> pd.DataFrame:
    """Mechanical or ML-filtered dip_short_rip with no overlapping trades."""
    ex = resolve_execution(cfg, mechanical=mechanical)
    tp, sl, h = float(ex["tp"]), float(ex["sl"]), int(ex["horizon"])
    router = router_mask(feats, df.index, cfg=cfg)

    if mechanical:
        signals = router
    else:
        prob = float(ml_prob if ml_prob is not None else cfg["ml_prob"])
        labeled = build_labeled_set(df, feats, cfg=cfg)
        scores = walk_forward_short_scores(
            labeled, feats, feature_columns(feats), prob_threshold=prob
        )
        ml_ok = pd.Series(False, index=df.index)
        if not scores.empty:
            ts = pd.to_datetime(scores["signal_ts"], utc=True)
            hit = df.index.intersection(ts)
            ml_ok.loc[hit] = True
        signals = router & ml_ok

    refresh = cfg.get("same_dir_refresh", "entry")
    upgrade = bool(cfg.get("upgrade_stop", False))

    return simulate_single_position(
        df,
        signals,
        side=-1,
        tp=tp,
        sl=sl,
        horizon=h,
        same_dir_refresh=refresh,
        upgrade_stop=upgrade,
    )
