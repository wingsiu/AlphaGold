"""Walk-forward dual-side XGBoost for v16 scalp."""
from __future__ import annotations

import pandas as pd
import xgboost as xgb

from v16.backtest.scaleout_sim import simulate_scaleout_trade
from v16.config.v16_config import EXIT_CONFIG, ML_CONFIG


def _exit_kwargs() -> dict:
    return {
        "first_scale_pnl": EXIT_CONFIG["first_scale_pnl"],
        "first_scale_frac": EXIT_CONFIG["first_scale_frac"],
        "final_scale_pnl": EXIT_CONFIG["final_scale_pnl"],
        "initial_sl": EXIT_CONFIG["initial_sl"],
        "runner_lock_pnl": EXIT_CONFIG["runner_lock_pnl"],
        "horizon": EXIT_CONFIG["horizon_minutes"],
    }


def _fit_binary(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier | None:
    if len(y) < ML_CONFIG["min_train_rows"] or y.nunique() < 2:
        return None
    pos = max(1, int((y == 0).sum()))
    neg = max(1, int((y == 1).sum()))
    m = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos / neg,
        random_state=42,
        eval_metric="logloss",
    )
    m.fit(X, y)
    return m


def _triclass_label_margin(row: pd.Series, margin: float = 2.0) -> int:
    lp, sp = float(row["long_pnl"]), float(row["short_pnl"])
    if lp > sp + margin and lp > 0:
        return 1
    if sp > lp + margin and sp > 0:
        return 2
    return 0


def _triclass_label(row: pd.Series) -> int:
    """0=skip, 1=long, 2=short from scale-out outcomes."""
    lw, sw = int(row["long_win"]), int(row["short_win"])
    lp, sp = float(row["long_pnl"]), float(row["short_pnl"])
    if lw and sw:
        return 1 if lp >= sp else 2
    if lw:
        return 1
    if sw:
        return 2
    return 0


def _fit_multiclass(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier | None:
    if len(y) < ML_CONFIG["min_train_rows"] or y.nunique() < 2:
        return None
    m = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
    )
    m.fit(X, y)
    return m


def _trend_side(feats: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Simple MA trend: +1 up, -1 down, 0 mixed."""
    r = feats.loc[ts]
    up = r["dist_ema_20"] > 0 and r["dist_ema_50"] > 0 and r["ema_20_slope"] > 0
    dn = r["dist_ema_20"] < 0 and r["dist_ema_50"] < 0 and r["ema_20_slope"] < 0
    if up:
        return 1
    if dn:
        return -1
    return 0


def _profitable_lon_hours(
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    min_trades: int = 8,
    min_avg_pnl: float = 1.5,
) -> set[int]:
    tmp = labeled.copy()
    tmp["lon_h"] = feats.loc[labeled.index, "lon_hour"].astype(int)
    tmp["oracle_pnl"] = labeled["best_pnl"]
    good: set[int] = set()
    for h, grp in tmp.groupby("lon_h"):
        if len(grp) >= min_trades and grp["oracle_pnl"].mean() >= min_avg_pnl:
            good.add(int(h))
    return good


def walk_forward_triclass(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
    *,
    exit_overrides: dict | None = None,
    prob_threshold: float | None = None,
    require_trend_align: bool = False,
    slot_hours: set[int] | None = None,
    slot_feats: pd.DataFrame | None = None,
    label_margin: float | None = None,
) -> pd.DataFrame:
    if labeled.empty:
        return pd.DataFrame()

    kw = _exit_kwargs()
    if exit_overrides:
        kw.update(exit_overrides)

    if label_margin is not None:
        y_all = labeled.apply(lambda r: _triclass_label_margin(r, label_margin), axis=1)
    else:
        y_all = labeled.apply(_triclass_label, axis=1)
    X_all = feats.loc[labeled.index, feat_cols]
    sf = slot_feats if slot_feats is not None else feats
    use_slots = "lon_hour" in sf.columns
    trades: list[dict] = []
    train_days = int(ML_CONFIG["train_days"])
    thresh = float(prob_threshold if prob_threshold is not None else ML_CONFIG["prob_threshold"])

    test_start = labeled.index.min() + pd.Timedelta(days=train_days)
    for period in pd.period_range(test_start, labeled.index.max(), freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        train_ix = labeled.index[labeled.index < m_start]
        test_ix = labeled.index[(labeled.index >= m_start) & (labeled.index < m_end)]
        if len(train_ix) < ML_CONFIG["min_train_rows"] or len(test_ix) == 0:
            continue

        hours = slot_hours
        if hours is None and use_slots:
            hours = _profitable_lon_hours(labeled.loc[train_ix], sf)

        model = _fit_multiclass(X_all.loc[train_ix], y_all.loc[train_ix])
        if model is None:
            continue

        for ts in test_ix:
            if hours and use_slots and int(sf.loc[ts, "lon_hour"]) not in hours:
                continue
            probs = model.predict_proba(X_all.loc[[ts]])[0]
            cls = int(probs.argmax())
            prob = float(probs[cls])
            if cls == 0 or prob < thresh:
                continue
            side = 1 if cls == 1 else -1
            if require_trend_align:
                tr = _trend_side(feats, ts)
                if tr != 0 and side != tr:
                    continue

            row = labeled.loc[ts]
            entry_idx = int(row["entry_idx"])
            nxt = df.iloc[entry_idx]
            ep = float(nxt["open_ask"] if side == 1 else nxt["open_bid"])
            res = simulate_scaleout_trade(df, entry_idx, side, ep, **kw)
            trades.append(
                {
                    "signal_ts": ts,
                    "side": side,
                    "p_pick": prob,
                    "pnl": res.pnl,
                    "exit_reason": res.exit_reason,
                    "scaled_half": res.scaled_half,
                    "win": res.pnl > 0,
                }
            )

    return pd.DataFrame(trades)


def walk_forward_trend_gate(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
    *,
    exit_overrides: dict | None = None,
) -> pd.DataFrame:
    """Dual-side ML but only when pick aligns with EMA trend."""
    if labeled.empty:
        return pd.DataFrame()

    kw = _exit_kwargs()
    if exit_overrides:
        kw.update(exit_overrides)

    X_all = feats.loc[labeled.index, feat_cols]
    trades: list[dict] = []
    train_days = int(ML_CONFIG["train_days"])
    thresh = float(ML_CONFIG["prob_threshold"])
    min_edge = float(ML_CONFIG["min_edge"])

    test_start = labeled.index.min() + pd.Timedelta(days=train_days)
    for period in pd.period_range(test_start, labeled.index.max(), freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        train_ix = labeled.index[labeled.index < m_start]
        test_ix = labeled.index[(labeled.index >= m_start) & (labeled.index < m_end)]
        if len(train_ix) < ML_CONFIG["min_train_rows"] or len(test_ix) == 0:
            continue

        m_long = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "long_win"])
        m_short = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "short_win"])
        if m_long is None or m_short is None:
            continue

        for ts in test_ix:
            tr = _trend_side(feats, ts)
            if tr == 0:
                continue
            x = X_all.loc[[ts]]
            p_long = float(m_long.predict_proba(x)[0, 1])
            p_short = float(m_short.predict_proba(x)[0, 1])
            if tr == 1:
                side, prob, opp = 1, p_long, p_short
            else:
                side, prob, opp = -1, p_short, p_long
            if prob < thresh or (prob - opp) < min_edge:
                continue

            row = labeled.loc[ts]
            entry_idx = int(row["entry_idx"])
            nxt = df.iloc[entry_idx]
            ep = float(nxt["open_ask"] if side == 1 else nxt["open_bid"])
            res = simulate_scaleout_trade(df, entry_idx, side, ep, **kw)
            trades.append(
                {
                    "signal_ts": ts,
                    "side": side,
                    "p_pick": prob,
                    "pnl": res.pnl,
                    "exit_reason": res.exit_reason,
                    "scaled_half": res.scaled_half,
                    "win": res.pnl > 0,
                }
            )

    return pd.DataFrame(trades)


def walk_forward_dual_v15_exit(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
    *,
    tp: float = 30.0,
    sl: float = 25.0,
    horizon: int = 30,
) -> pd.DataFrame:
    """Dual-side ML with v15-style fixed TP/SL exit."""
    from v16.backtest.fixed_tpsl_sim import simulate_fixed_tpsl

    if labeled.empty:
        return pd.DataFrame()

    X_all = feats.loc[labeled.index, feat_cols]
    trades: list[dict] = []
    train_days = int(ML_CONFIG["train_days"])
    thresh = float(ML_CONFIG["prob_threshold"])
    min_edge = float(ML_CONFIG["min_edge"])

    test_start = labeled.index.min() + pd.Timedelta(days=train_days)
    for period in pd.period_range(test_start, labeled.index.max(), freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        train_ix = labeled.index[labeled.index < m_start]
        test_ix = labeled.index[(labeled.index >= m_start) & (labeled.index < m_end)]
        if len(train_ix) < ML_CONFIG["min_train_rows"] or len(test_ix) == 0:
            continue

        m_long = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "long_win"])
        m_short = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "short_win"])
        if m_long is None or m_short is None:
            continue

        for ts in test_ix:
            x = X_all.loc[[ts]]
            p_long = float(m_long.predict_proba(x)[0, 1])
            p_short = float(m_short.predict_proba(x)[0, 1])
            if p_long >= p_short:
                side, prob, opp = 1, p_long, p_short
            else:
                side, prob, opp = -1, p_short, p_long
            if prob < thresh or (prob - opp) < min_edge:
                continue

            row = labeled.loc[ts]
            entry_idx = int(row["entry_idx"])
            nxt = df.iloc[entry_idx]
            ep = float(nxt["open_ask"] if side == 1 else nxt["open_bid"])
            res = simulate_fixed_tpsl(df, entry_idx, side, ep, tp=tp, sl=sl, horizon=horizon)
            trades.append(
                {
                    "signal_ts": ts,
                    "side": side,
                    "p_pick": prob,
                    "pnl": res.pnl,
                    "exit_reason": res.exit_reason,
                    "win": res.pnl > 0,
                }
            )

    return pd.DataFrame(trades)


def walk_forward_dual(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
    *,
    exit_overrides: dict | None = None,
) -> pd.DataFrame:
    if labeled.empty:
        return pd.DataFrame()

    kw = _exit_kwargs()
    if exit_overrides:
        kw.update(exit_overrides)

    X_all = feats.loc[labeled.index, feat_cols]
    trades: list[dict] = []
    train_days = int(ML_CONFIG["train_days"])
    thresh = float(ML_CONFIG["prob_threshold"])
    min_edge = float(ML_CONFIG["min_edge"])

    test_start = labeled.index.min() + pd.Timedelta(days=train_days)
    for period in pd.period_range(test_start, labeled.index.max(), freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        train_ix = labeled.index[labeled.index < m_start]
        test_ix = labeled.index[(labeled.index >= m_start) & (labeled.index < m_end)]
        if len(train_ix) < ML_CONFIG["min_train_rows"] or len(test_ix) == 0:
            continue

        m_long = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "long_win"])
        m_short = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "short_win"])
        if m_long is None or m_short is None:
            continue

        for ts in test_ix:
            x = X_all.loc[[ts]]
            p_long = float(m_long.predict_proba(x)[0, 1])
            p_short = float(m_short.predict_proba(x)[0, 1])
            if p_long >= p_short:
                side, prob = 1, p_long
                opp = p_short
            else:
                side, prob = -1, p_short
                opp = p_long
            if prob < thresh or (prob - opp) < min_edge:
                continue

            row = labeled.loc[ts]
            entry_idx = int(row["entry_idx"])
            nxt = df.iloc[entry_idx]
            ep = float(nxt["open_ask"] if side == 1 else nxt["open_bid"])
            res = simulate_scaleout_trade(df, entry_idx, side, ep, **kw)
            trades.append(
                {
                    "signal_ts": ts,
                    "side": side,
                    "p_long": p_long,
                    "p_short": p_short,
                    "p_pick": prob,
                    "pnl": res.pnl,
                    "exit_reason": res.exit_reason,
                    "scaled_half": res.scaled_half,
                    "bars_held": res.bars_held,
                    "win": res.pnl > 0,
                }
            )

    return pd.DataFrame(trades)


def walk_forward_long(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
    *,
    exit_overrides: dict | None = None,
    prob_threshold: float | None = None,
) -> pd.DataFrame:
    """Single-side long classifier (dip fade / bounce setups)."""
    if labeled.empty:
        return pd.DataFrame()

    kw = _exit_kwargs()
    if exit_overrides:
        kw.update(exit_overrides)

    X_all = feats.loc[labeled.index, feat_cols]
    trades: list[dict] = []
    train_days = int(ML_CONFIG["train_days"])
    thresh = float(prob_threshold if prob_threshold is not None else ML_CONFIG["prob_threshold"])

    test_start = labeled.index.min() + pd.Timedelta(days=train_days)
    for period in pd.period_range(test_start, labeled.index.max(), freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        train_ix = labeled.index[labeled.index < m_start]
        test_ix = labeled.index[(labeled.index >= m_start) & (labeled.index < m_end)]
        if len(train_ix) < ML_CONFIG["min_train_rows"] or len(test_ix) == 0:
            continue

        model = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "long_win"])
        if model is None:
            continue

        for ts in test_ix:
            x = X_all.loc[[ts]]
            prob = float(model.predict_proba(x)[0, 1])
            if prob < thresh:
                continue
            row = labeled.loc[ts]
            entry_idx = int(row["entry_idx"])
            nxt = df.iloc[entry_idx]
            ep = float(nxt["open_ask"])
            res = simulate_scaleout_trade(df, entry_idx, 1, ep, **kw)
            trades.append(
                {
                    "signal_ts": ts,
                    "side": 1,
                    "p_long": prob,
                    "pnl": res.pnl,
                    "exit_reason": res.exit_reason,
                    "scaled_half": res.scaled_half,
                    "win": res.pnl > 0,
                }
            )

    return pd.DataFrame(trades)


def walk_forward_short_probs(
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
) -> pd.DataFrame:
    """Walk-forward short classifier — all OOS probabilities (no threshold filter)."""
    return walk_forward_short_scores(
        labeled, feats, feat_cols, prob_threshold=0.0
    )


def walk_forward_short_scores(
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
    *,
    prob_threshold: float | None = None,
) -> pd.DataFrame:
    """Walk-forward short classifier — scores only (no trade simulation)."""
    if labeled.empty:
        return pd.DataFrame()

    X_all = feats.loc[labeled.index, feat_cols]
    rows: list[dict] = []
    train_days = int(ML_CONFIG["train_days"])
    thresh = float(prob_threshold if prob_threshold is not None else ML_CONFIG["prob_threshold"])

    test_start = labeled.index.min() + pd.Timedelta(days=train_days)
    for period in pd.period_range(test_start, labeled.index.max(), freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        train_ix = labeled.index[labeled.index < m_start]
        test_ix = labeled.index[(labeled.index >= m_start) & (labeled.index < m_end)]
        if len(train_ix) < ML_CONFIG["min_train_rows"] or len(test_ix) == 0:
            continue

        model = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "short_win"])
        if model is None:
            continue

        for ts in test_ix:
            prob = float(model.predict_proba(X_all.loc[[ts]])[0, 1])
            if prob < thresh:
                continue
            row = labeled.loc[ts]
            rows.append(
                {
                    "signal_ts": ts,
                    "entry_idx": int(row["entry_idx"]),
                    "p_short": prob,
                }
            )

    return pd.DataFrame(rows)


def walk_forward_short(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    feats: pd.DataFrame,
    feat_cols: list[str],
    *,
    exit_overrides: dict | None = None,
    prob_threshold: float | None = None,
) -> pd.DataFrame:
    """Single-side short classifier."""
    if labeled.empty:
        return pd.DataFrame()

    kw = _exit_kwargs()
    if exit_overrides:
        kw.update(exit_overrides)

    X_all = feats.loc[labeled.index, feat_cols]
    trades: list[dict] = []
    train_days = int(ML_CONFIG["train_days"])
    thresh = float(prob_threshold if prob_threshold is not None else ML_CONFIG["prob_threshold"])

    test_start = labeled.index.min() + pd.Timedelta(days=train_days)
    for period in pd.period_range(test_start, labeled.index.max(), freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        train_ix = labeled.index[labeled.index < m_start]
        test_ix = labeled.index[(labeled.index >= m_start) & (labeled.index < m_end)]
        if len(train_ix) < ML_CONFIG["min_train_rows"] or len(test_ix) == 0:
            continue

        model = _fit_binary(X_all.loc[train_ix], labeled.loc[train_ix, "short_win"])
        if model is None:
            continue

        for ts in test_ix:
            x = X_all.loc[[ts]]
            prob = float(model.predict_proba(x)[0, 1])
            if prob < thresh:
                continue
            row = labeled.loc[ts]
            entry_idx = int(row["entry_idx"])
            nxt = df.iloc[entry_idx]
            ep = float(nxt["open_bid"])
            res = simulate_scaleout_trade(df, entry_idx, -1, ep, **kw)
            trades.append(
                {
                    "signal_ts": ts,
                    "side": -1,
                    "p_short": prob,
                    "pnl": res.pnl,
                    "exit_reason": res.exit_reason,
                    "scaled_half": res.scaled_half,
                    "win": res.pnl > 0,
                }
            )

    return pd.DataFrame(trades)
