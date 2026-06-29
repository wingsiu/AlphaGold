"""Walk-forward LSTM binary filter on 1m bar sequences."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from v16.config.v16_config import ML_CONFIG

SEQ_LEN = 30
SEQ_FEATURE_NAMES = (
    "body_n",
    "range_n",
    "ret1_n",
    "close_loc",
    "vol_ratio_n",
    "slot_min",
    "body_signed",
)


def _bar_matrix(df: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
    """Per-bar raw sequence channels aligned to df.index."""
    atr = feats["atr_14"].replace(0, np.nan).to_numpy(dtype=float)
    atr = np.where(np.isfinite(atr) & (atr > 0), atr, 1.0)
    body = feats["body"].to_numpy(dtype=float)
    rng = feats["range_1"].to_numpy(dtype=float)
    ret1 = feats["ret_3"].to_numpy(dtype=float) / 3.0  # ~1m move
    close_loc = feats["close_loc"].to_numpy(dtype=float)
    vol = np.clip(feats["vol_ratio"].to_numpy(dtype=float), 0, 5)
    slot_min = feats["minute_in_15m"].to_numpy(dtype=float) / 14.0
    return np.column_stack(
        [
            body / atr,
            rng / atr,
            ret1 / atr,
            close_loc,
            vol,
            slot_min,
            np.zeros(len(df), dtype=float),  # body_signed filled per sample
        ]
    )


def build_sequences(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """(N, seq_len, n_feat) arrays for labeled impulse signals."""
    mat = _bar_matrix(df, feats)
    n_feat = mat.shape[1]
    valid_ts: list[pd.Timestamp] = []
    chunks: list[np.ndarray] = []

    for ts, row in labeled.iterrows():
        end_idx = int(df.index.get_loc(ts))
        start_idx = end_idx - seq_len + 1
        if start_idx < 0:
            continue
        block = mat[start_idx : end_idx + 1].copy()
        side = float(row["side"])
        block[:, 6] = block[:, 0] * side  # body_signed
        chunks.append(block)
        valid_ts.append(ts)

    if not chunks:
        return np.empty((0, seq_len, n_feat)), np.empty(0), pd.DatetimeIndex([])

    X = np.stack(chunks, axis=0).astype(np.float32)
    y = labeled.loc[valid_ts, "win"].to_numpy(dtype=np.float32)
    return X, y, pd.DatetimeIndex(valid_ts)


def _make_model(seq_len: int, n_feat: int) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_feat), name="seq")
    x = keras.layers.Masking(mask_value=0.0)(inp)
    x = keras.layers.LSTM(48, return_sequences=False)(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(24, activation="relu")(x)
    x = keras.layers.Dropout(0.15)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def _fit_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seq_len: int,
    n_feat: int,
    epochs: int = 30,
    batch_size: int = 64,
) -> tuple[keras.Model, StandardScaler]:
    """Fit scaler + LSTM on flattened train windows."""
    flat = X_train.reshape(-1, n_feat)
    scaler = StandardScaler()
    scaler.fit(flat)
    Xs = scaler.transform(flat).reshape(X_train.shape[0], seq_len, n_feat).astype(np.float32)

    pos = max(1, int((y_train == 0).sum()))
    neg = max(1, int((y_train == 1).sum()))
    class_weight = {0: 1.0, 1: pos / neg}

    # Chronological val tail (last 15%)
    n_val = max(32, int(len(y_train) * 0.15))
    if len(y_train) - n_val < ML_CONFIG["min_train_rows"]:
        n_val = max(0, len(y_train) // 5)

    if n_val > 0:
        X_tr, X_va = Xs[:-n_val], Xs[-n_val:]
        y_tr, y_va = y_train[:-n_val], y_train[-n_val:]
    else:
        X_tr, X_va, y_tr, y_va = Xs, None, y_train, None

    model = _make_model(seq_len, n_feat)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc" if n_val else "auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        )
    ]
    fit_kw: dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "class_weight": class_weight,
        "verbose": 0,
        "callbacks": callbacks,
    }
    if n_val:
        fit_kw["validation_data"] = (X_va, y_va)
    model.fit(X_tr, y_tr, **fit_kw)
    return model, scaler


def _predict_batch(
    model: keras.Model,
    scaler: StandardScaler,
    X: np.ndarray,
    *,
    seq_len: int,
    n_feat: int,
) -> np.ndarray:
    flat = X.reshape(-1, n_feat)
    Xs = scaler.transform(flat).reshape(X.shape[0], seq_len, n_feat).astype(np.float32)
    return model.predict(Xs, verbose=0).ravel()


def walk_forward_lstm_scores(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    seq_len: int = SEQ_LEN,
    prob_threshold: float = 0.0,
    train_days: int | None = None,
    min_train_rows: int | None = None,
    retrain_freq: str | None = None,
    retrain_days: int | None = None,
) -> pd.DataFrame:
    """Walk-forward OOS probabilities for impulse signals."""
    if labeled.empty:
        return pd.DataFrame()

    from v16.backtest.impulse_ml import _wf_test_windows

    X_all, y_all, ts_all = build_sequences(df, feats, labeled, seq_len=seq_len)
    if len(ts_all) == 0:
        return pd.DataFrame()

    n_feat = X_all.shape[2]
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

        model, scaler = _fit_lstm(
            X_all[train_mask],
            y_all[train_mask],
            seq_len=seq_len,
            n_feat=n_feat,
        )
        probs = _predict_batch(model, scaler, X_all[test_mask], seq_len=seq_len, n_feat=n_feat)
        test_ts = ts_all[test_mask]

        for ts, prob in zip(test_ts, probs):
            if float(prob) < thresh:
                continue
            row = labeled.loc[ts]
            rows.append(
                {
                    "signal_ts": ts,
                    "entry_idx": int(row["entry_idx"]),
                    "side": int(row["side"]),
                    "p_win": float(prob),
                }
            )
        keras.backend.clear_session()

    return pd.DataFrame(rows)


def apply_lstm_filter_to_sides(
    df: pd.DataFrame,
    base_sides: pd.Series,
    scores: pd.DataFrame,
) -> pd.Series:
    """Keep only signals that passed LSTM threshold."""
    out = pd.Series(0, index=df.index, dtype=int)
    if scores.empty:
        return out
    ts = pd.to_datetime(scores["signal_ts"], utc=True)
    hit = df.index.intersection(ts)
    for t in hit:
        out.loc[t] = int(base_sides.loc[t])
    return out
