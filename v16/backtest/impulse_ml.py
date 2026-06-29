"""Walk-forward binary filters for impulse_1m_15m (tabular + LSTM)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from v16.backtest.impulse_features import (
    IMPULSE_EXPLICIT,
    attach_structure_features,
    enrich_impulse_derived,
    impulse_ml_feature_columns,
    structure_kwargs,
)
from v16.backtest.lstm_filter import SEQ_FEATURE_NAMES, SEQ_LEN, build_sequences, walk_forward_lstm_scores
from v16.config.v16_config import ML_CONFIG

LSTM_SEQ_LEN_DEFAULT = SEQ_LEN  # 30
LSTM_SEQ_LEN_LONG = 120

MODEL_NAMES = ("xgb", "rf", "hgb", "lgb", "et", "gbc", "mlp", "ada", "ens", "logreg", "lstm")


def build_tabular_matrix(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    seq_len: int = SEQ_LEN,
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Point features (v16 + v15 dip + structure + impulse) + sequence aggregates."""
    skw = structure_kwargs(cfg)
    if skw:
        feats = attach_structure_features(df, feats, **skw)

    X_seq, y, ts = build_sequences(df, feats, labeled, seq_len=seq_len)
    if len(ts) == 0:
        return np.empty((0, 0)), np.empty(0), pd.DatetimeIndex([]), []

    feat_cols = [c for c in impulse_ml_feature_columns(feats, labeled) if c != "side"]
    frame = feats.loc[ts, [c for c in feat_cols if c in feats.columns]].copy()
    frame["side"] = labeled.loc[ts, "side"].astype(float).values

    for col in IMPULSE_EXPLICIT:
        if col in labeled.columns and col not in frame.columns:
            frame[col] = labeled.loc[ts, col].astype(float).values

    frame = enrich_impulse_derived(frame, feats, ts)

    for j, name in enumerate(SEQ_FEATURE_NAMES):
        ch = X_seq[:, :, j]
        frame[f"{name}_mean"] = ch.mean(axis=1)
        frame[f"{name}_max"] = ch.max(axis=1)
        frame[f"{name}_std"] = ch.std(axis=1)

    names = list(frame.columns)
    return frame.to_numpy(dtype=np.float32), y, ts, names


def _make_model(name: str) -> Any:
    if name == "xgb":
        return "xgb"
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=200,
            min_samples_leaf=20,
            random_state=42,
        )
    if name == "lgb":
        if not _HAS_LGB:
            raise ValueError("lightgbm not installed")
        return "lgb"
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if name == "gbc":
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            min_samples_leaf=15,
            random_state=42,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=400,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "ada":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, class_weight="balanced"),
            n_estimators=120,
            learning_rate=0.06,
            random_state=42,
        )
    if name == "ens":
        return "ens"
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown tabular model: {name}")


def _fit_xgb(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    y_i = y.astype(int)
    pos = max(1, int((y_i == 0).sum()))
    neg = max(1, int((y_i == 1).sum()))
    m = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos / neg,
        random_state=42,
        eval_metric="logloss",
    )
    m.fit(X, y_i)
    return m


def _fit_lgb(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    y_i = y.astype(int)
    pos = max(1, int((y_i == 0).sum()))
    neg = max(1, int((y_i == 1).sum()))
    m = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos / neg,
        random_state=42,
        verbose=-1,
    )
    m.fit(X, y_i)
    return m


def _fit_ensemble(
    X: np.ndarray, y: np.ndarray
) -> tuple[xgb.XGBClassifier, RandomForestClassifier]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y.astype(int))
    return _fit_xgb(X, y), rf


def _predict_proba(model: Any, name: str, X: np.ndarray) -> np.ndarray:
    if name == "xgb":
        return model.predict_proba(X)[:, 1]
    if name == "lgb":
        return model.predict_proba(X)[:, 1]
    if name == "ens":
        xgb_m, rf_m = model
        return 0.5 * xgb_m.predict_proba(X)[:, 1] + 0.5 * rf_m.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


TABULAR_MODELS = tuple(m for m in MODEL_NAMES if m != "lstm")


def _wf_test_windows(
    test_start: pd.Timestamp,
    ts_max: pd.Timestamp,
    *,
    retrain_freq: str,
    retrain_days: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Rolling OOS windows: v15 14d grid (14D), monthly (M), or rolling biweekly (2W)."""
    freq = str(retrain_freq).upper()
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(ts_max)
    if start.tz is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tz is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    if freq in ("14D", "V15"):
        from xgboost_filter_model.pattern_training import iter_wf_cycles

        rd = int(retrain_days if retrain_days is not None else ML_CONFIG.get("retrain_days", 14))
        for _cycle, c_start, c_end in iter_wf_cycles(start, end + pd.Timedelta(days=1), retrain_days=rd):
            w_start = max(c_start, start)
            w_end = min(c_end, end + pd.Timedelta(seconds=1))
            if w_start < w_end:
                windows.append((w_start, w_end))
        return windows

    if freq in ("2W", "BIWEEKLY"):
        step = pd.Timedelta(days=int(retrain_days if retrain_days is not None else 14))
        cursor = start
        while cursor <= end:
            w_end = cursor + step
            windows.append((cursor, w_end))
            cursor = w_end
        return windows

    for period in pd.period_range(start, end, freq="M"):
        m_start = period.start_time.tz_localize("UTC")
        m_end = (period + 1).start_time.tz_localize("UTC")
        windows.append((m_start, m_end))
    return windows


def walk_forward_tabular_scores(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    labeled: pd.DataFrame,
    model_name: str,
    *,
    seq_len: int = SEQ_LEN,
    prob_threshold: float = 0.0,
    train_days: int | None = None,
    min_train_rows: int | None = None,
    retrain_freq: str | None = None,
    retrain_days: int | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    if labeled.empty or model_name not in TABULAR_MODELS:
        return pd.DataFrame()

    X_all, y_all, ts_all, _ = build_tabular_matrix(
        df, feats, labeled, seq_len=seq_len, cfg=cfg
    )
    if len(ts_all) == 0:
        return pd.DataFrame()

    tdays = int(train_days if train_days is not None else ML_CONFIG["train_days"])
    min_rows = int(min_train_rows if min_train_rows is not None else max(80, ML_CONFIG["min_train_rows"]))
    rf = str(retrain_freq if retrain_freq is not None else ML_CONFIG.get("retrain_freq", "14D"))
    rd = retrain_days if retrain_days is not None else ML_CONFIG.get("retrain_days")
    thresh = float(prob_threshold)
    rows: list[dict] = []
    test_start = ts_all.min() + pd.Timedelta(days=tdays)

    for w_start, w_end in _wf_test_windows(test_start, ts_all.max(), retrain_freq=rf, retrain_days=rd):
        train_mask = ts_all < w_start
        test_mask = (ts_all >= w_start) & (ts_all < w_end)
        if train_mask.sum() < min_rows or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        if model_name == "xgb":
            model = _fit_xgb(X_tr, y_tr)
        elif model_name == "lgb":
            model = _fit_lgb(X_tr, y_tr)
        elif model_name == "ens":
            model = _fit_ensemble(X_tr, y_tr)
        else:
            model = _make_model(model_name)
            model.fit(X_tr, y_tr.astype(int))

        probs = _predict_proba(model, model_name, X_all[test_mask])
        for ts, prob in zip(ts_all[test_mask], probs):
            if float(prob) < thresh:
                continue
            row = labeled.loc[ts]
            rows.append(
                {
                    "signal_ts": ts,
                    "entry_idx": int(row["entry_idx"]),
                    "side": int(row["side"]),
                    "p_win": float(prob),
                    "model": model_name,
                }
            )

    return pd.DataFrame(rows)


def walk_forward_model_scores(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    labeled: pd.DataFrame,
    model_name: str,
    *,
    seq_len: int = SEQ_LEN,
    lstm_seq_len: int | None = None,
    prob_threshold: float = 0.0,
    train_days: int | None = None,
    min_train_rows: int | None = None,
    retrain_freq: str | None = None,
    retrain_days: int | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Dispatch to tabular or LSTM walk-forward scorer."""
    name = model_name.lower()
    lstm_len = int(lstm_seq_len if lstm_seq_len is not None else seq_len)
    if name == "lstm":
        out = walk_forward_lstm_scores(
            df,
            feats,
            labeled,
            seq_len=lstm_len,
            prob_threshold=prob_threshold,
            train_days=train_days,
            min_train_rows=min_train_rows,
            retrain_freq=retrain_freq,
            retrain_days=retrain_days,
        )
        if not out.empty:
            out = out.copy()
            out["model"] = "lstm"
        return out
    return walk_forward_tabular_scores(
        df,
        feats,
        labeled,
        name,
        seq_len=seq_len,
        prob_threshold=prob_threshold,
        train_days=train_days,
        min_train_rows=min_train_rows,
        retrain_freq=retrain_freq,
        retrain_days=retrain_days,
        cfg=cfg,
    )


def apply_ml_filter_to_sides(
    df: pd.DataFrame,
    base_sides: pd.Series,
    scores: pd.DataFrame,
) -> pd.Series:
    out = pd.Series(0, index=df.index, dtype=int)
    if scores.empty:
        return out
    ts = pd.to_datetime(scores["signal_ts"], utc=True)
    for t in df.index.intersection(ts):
        out.loc[t] = int(base_sides.loc[t])
    return out


def filter_signal_table(
    signal_table: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    if scores.empty or signal_table.empty:
        return signal_table.iloc[0:0]
    ts = pd.to_datetime(scores["signal_ts"], utc=True)
    hit = signal_table.index.intersection(ts)
    return signal_table.loc[hit]


def evaluate_threshold_sweep(
    df_oos: pd.DataFrame,
    base_sides: pd.Series,
    scores_oos: pd.DataFrame,
    cfg: dict,
    thresholds: tuple[float, ...],
    *,
    exit_mode: str = "scaleout",
    execution: dict | None = None,
    signal_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from v16.backtest.position_sim import (
        simulate_position_impulse_stop,
        simulate_position_sided,
        simulate_position_sided_scaleout,
    )
    from v16.config import v16_config

    so = dict(cfg.get("scaleout", v16_config.EXIT_CONFIG))
    ex = execution or cfg.get("execution", {"tp": 25.0, "sl": 35.0, "horizon": 60})
    is_cfg = cfg.get("impulse_stop", {})
    rows: list[dict] = []
    model = scores_oos["model"].iloc[0] if not scores_oos.empty and "model" in scores_oos.columns else "unknown"

    for p in thresholds:
        sub = scores_oos[scores_oos["p_win"] >= p]
        if exit_mode == "impulse_stop":
            if signal_table is None:
                raise ValueError("signal_table required for impulse_stop exit")
            filt = filter_signal_table(signal_table, sub)
            tdf = simulate_position_impulse_stop(
                df_oos,
                filt,
                tp_multiple=float(is_cfg.get("tp_multiple", 3.0)),
                horizon=int(is_cfg.get("horizon", 120)),
                min_sl_pts=float(is_cfg.get("min_sl_pts", 1.0)),
                max_sl_pts=float(is_cfg.get("max_sl_pts", 80.0)),
                same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
                cfg=cfg,
            )
            kept = len(filt)
        else:
            filtered = apply_ml_filter_to_sides(df_oos, base_sides, sub)
            kept = int((filtered != 0).sum())
            if exit_mode == "execution":
                tdf = simulate_position_sided(
                    df_oos,
                    filtered,
                    tp=float(ex["tp"]),
                    sl=float(ex["sl"]),
                    horizon=int(ex["horizon"]),
                    same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
                    upgrade_stop=bool(cfg.get("upgrade_stop", False)),
                )
            else:
                tdf = simulate_position_sided_scaleout(
                    df_oos,
                    filtered,
                    scaleout_kw=so,
                    same_dir_refresh=cfg.get("same_dir_refresh", "entry"),
                )
        rows.append(
            {
                "model": model,
                "prob": p,
                "signals": kept,
                "trades": len(tdf),
                "wr": round(float(tdf["win"].mean() * 100), 1) if not tdf.empty else 0.0,
                "net": round(float(tdf["pnl"].sum()), 1) if not tdf.empty else 0.0,
                "avg": round(float(tdf["pnl"].mean()), 2) if not tdf.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)
