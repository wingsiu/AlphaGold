"""v16 gold momentum + dip short legs."""
from __future__ import annotations

import copy

import pandas as pd

from v16.backtest.dip_short_rip_run import run_dip_short_rip
from v16.backtest.features import build_features
from v16.backtest.impulse_features import attach_structure_features, structure_kwargs
from v16.backtest.impulse_ml import filter_signal_table, walk_forward_model_scores
from v16.backtest.position_sim import simulate_position_impulse_stop
from v16.config import v16_config
from v16.config.gold_config import DIP_SHORT, MOMENTUM
from v16.gold.merge import df_to_trades
from v16.patterns.momentum_15m_hold import build_labeled_set, build_signal_table


def _utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run_momentum_leg(df: pd.DataFrame, oos: pd.Timestamp) -> pd.DataFrame:
    cfg = copy.deepcopy(v16_config.MOMENTUM_V16_WINNER_PRECLOSE)
    df_oos = df[df.index >= oos]
    signals = build_signal_table(df_oos, cfg=cfg)
    labeled = build_labeled_set(df, cfg=cfg)
    feats = build_features(df)
    skw = structure_kwargs(cfg)
    if skw:
        feats = attach_structure_features(df, feats, **skw)
    scores = walk_forward_model_scores(
        df, feats, labeled, MOMENTUM["model"], prob_threshold=0.0,
        retrain_freq=MOMENTUM["retrain_freq"], cfg=cfg,
    )
    scores_oos = scores[pd.to_datetime(scores["signal_ts"], utc=True) >= oos]
    ml_filt = filter_signal_table(signals, scores_oos[scores_oos["p_win"] >= MOMENTUM["ml_prob"]])
    cfg["impulse_stop"] = {
        **cfg.get("impulse_stop", {}),
        "tp_enabled": False,
        "horizon": MOMENTUM["horizon"],
        "exit_on_structure_change": True,
        "exit_on_structure_change_min_pnl": -1e9,
    }
    return simulate_position_impulse_stop(df_oos, ml_filt, cfg=cfg)


def run_dip_leg(df: pd.DataFrame, oos: pd.Timestamp) -> pd.DataFrame:
    cfg = copy.deepcopy(v16_config.DIP_SHORT_RIP)
    df_oos = df[df.index >= oos]
    feats = build_features(df_oos)
    return run_dip_short_rip(df_oos, feats, cfg, mechanical=False, ml_prob=float(DIP_SHORT["ml_prob"]))


def run_v16_legs(df: pd.DataFrame, oos_start: str | pd.Timestamp) -> tuple[list[dict], list[dict]]:
    oos = _utc_ts(oos_start)
    mom = run_momentum_leg(df, oos)
    dip = run_dip_leg(df, oos)
    return (
        df_to_trades(mom, "v16_momentum", "v16_momentum"),
        df_to_trades(dip, "v16_dip_short", "v16_dip_short"),
    )
