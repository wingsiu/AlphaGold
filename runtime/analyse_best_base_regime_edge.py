#!/usr/bin/env python3
"""Rank reconstructable regime features for the promoted best-base model.

This script uses the promoted corrected trades plus raw 1-minute bars to compute
interpretable regime/context proxies at each signal time, then ranks feature
buckets by trade performance.

Outputs:
- runtime/best_base_regime_feature_report.json
- runtime/best_base_regime_feature_report.md
- runtime/best_base_regime_feature_buckets.csv
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, cast
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.data_loader import DataLoader
from training.image_trend_ml import _prepare_ohlcv

RUNTIME_DIR = PROJECT_ROOT / "runtime"
TRAINING_DIR = PROJECT_ROOT / "training"
HK_TZ = ZoneInfo("Asia/Hong_Kong")

REPORT_PATH = TRAINING_DIR / "backtest_report_best_base_corrected.json"
TRADES_PATH = TRAINING_DIR / "backtest_trades_best_base_corrected.csv"

MIN_BUCKET_TRADES = 25
TOP_N = 12


def _to_hkt_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert(HK_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("ts", "entry_time", "exit_time", "last_target_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def _load_bars(cfg: dict[str, Any]) -> pd.DataFrame:
    raw = DataLoader().load_data(
        cfg.get("table", "gold_prices"),
        start_date=cfg.get("start_date"),
        end_date=cfg.get("end_date"),
    )
    bars = _prepare_ohlcv(raw, str(cfg.get("timeframe", "1min")))
    return bars.sort_index()


def _session_bucket_hkt(ts: pd.Timestamp) -> str:
    h = ts.tz_convert(HK_TZ).hour
    if 8 <= h < 13:
        return "hkt_morning"
    if 13 <= h < 17:
        return "hkt_afternoon"
    if 17 <= h < 21:
        return "europe_overlap"
    if 21 <= h or h < 1:
        return "us_open"
    return "overnight"


def _bucket_wr90(v: float) -> str:
    if pd.isna(v):
        return "missing"
    if v >= -10:
        return "near_high"
    if v >= -30:
        return "upper_zone"
    if v >= -70:
        return "mid_zone"
    if v >= -90:
        return "lower_zone"
    return "near_low"


def _bucket_signed_move(v: float, levels: tuple[float, float]) -> str:
    lo, hi = levels
    if pd.isna(v):
        return "missing"
    if v <= -hi:
        return f"<=-{hi}"
    if v <= -lo:
        return f"[-{hi},-{lo}]"
    if v < lo:
        return f"(-{lo},{lo})"
    if v < hi:
        return f"[{lo},{hi})"
    return f">={hi}"


def _safe_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    non_na = series.dropna()
    if non_na.nunique() < min(q, 4):
        return pd.Series(np.where(series.notna(), f"{prefix}_all", "missing"), index=series.index)
    try:
        bucketed = pd.qcut(series, q=q, duplicates="drop")
        return bucketed.astype(str).fillna("missing")
    except ValueError:
        return pd.Series(np.where(series.notna(), f"{prefix}_all", "missing"), index=series.index)


def _compute_trade_features(trades: pd.DataFrame, bars: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    window = int(cfg.get("window", 150))
    feat_rows: list[dict[str, Any]] = []
    bars = bars.copy()

    for _, trade in trades.iterrows():
        ts = cast(pd.Timestamp, pd.Timestamp(trade["ts"]))
        if pd.isna(ts):
            continue
        if ts not in bars.index:
            continue
        pos = bars.index.get_loc(ts)
        if isinstance(pos, slice):
            pos = pos.stop - 1
        if pos < window - 1:
            continue
        w = bars.iloc[pos - window + 1 : pos + 1]
        last15 = w.iloc[-15:] if len(w) >= 15 else w
        last30 = w.iloc[-30:] if len(w) >= 30 else w
        last60 = w.iloc[-60:] if len(w) >= 60 else w
        last90 = w.iloc[-90:] if len(w) >= 90 else w
        close = float(w["close"].iloc[-1])
        prev_close = float(w["close"].iloc[-2]) if len(w) >= 2 else close
        range_150 = float(w["high"].max() - w["low"].min())
        swing_15 = float(last15["high"].max() - last15["low"].min())
        move_15 = float(last15["close"].iloc[-1] - last15["close"].iloc[0]) if len(last15) else 0.0
        move_30 = float(last30["close"].iloc[-1] - last30["close"].iloc[0]) if len(last30) else 0.0
        move_60 = float(last60["close"].iloc[-1] - last60["close"].iloc[0]) if len(last60) else 0.0
        returns_30 = last30["close"].pct_change().dropna()
        vol_30 = float(returns_30.std()) if len(returns_30) else 0.0
        vol_mean_30 = float(last30["volume"].mean()) if len(last30) else 0.0
        vol_rel = float(last15["volume"].iloc[-1] / vol_mean_30) if vol_mean_30 > 0 and len(last15) else np.nan
        wr_high = float(last90["high"].max()) if len(last90) else np.nan
        wr_low = float(last90["low"].min()) if len(last90) else np.nan
        wr_span = wr_high - wr_low if pd.notna(wr_high) and pd.notna(wr_low) else np.nan
        wr90_last = float(-100.0 * ((wr_high - close) / wr_span)) if pd.notna(wr_span) and wr_span > 0 else np.nan
        bar_return = float(close - prev_close)

        feat_rows.append(
            {
                "ts": ts,
                "entry_time": trade["entry_time"],
                "side": trade.get("side"),
                "pnl": float(trade.get("pnl", 0.0)),
                "entry_signal_prob": float(trade.get("entry_signal_prob", np.nan)),
                "exit_reason": trade.get("exit_reason"),
                "range_150": range_150,
                "swing_15": swing_15,
                "move_15": move_15,
                "move_30": move_30,
                "move_60": move_60,
                "volatility_30": vol_30,
                "volume_rel_last": vol_rel,
                "wr90_last": wr90_last,
                "bar_return": bar_return,
                "hour_hkt": ts.tz_convert(HK_TZ).hour,
                "weekday_hkt": ts.tz_convert(HK_TZ).day_name(),
                "session_hkt": _session_bucket_hkt(ts),
            }
        )

    feat = pd.DataFrame(feat_rows)
    if feat.empty:
        raise ValueError("No feature rows were built from promoted trades.")

    feat["side_bucket"] = feat["side"].astype(str)
    feat["exit_reason_bucket"] = feat["exit_reason"].astype(str)
    feat["prob_bucket"] = _safe_qcut(feat["entry_signal_prob"], q=5, prefix="prob")
    feat["range_150_bucket"] = _safe_qcut(feat["range_150"], q=5, prefix="range150")
    feat["swing_15_bucket"] = _safe_qcut(feat["swing_15"], q=5, prefix="swing15")
    feat["volatility_30_bucket"] = _safe_qcut(feat["volatility_30"], q=5, prefix="vol30")
    feat["volume_rel_bucket"] = _safe_qcut(feat["volume_rel_last"], q=5, prefix="volrel")
    feat["wr90_bucket"] = feat["wr90_last"].apply(_bucket_wr90)
    feat["move_15_bucket"] = feat["move_15"].apply(lambda v: _bucket_signed_move(v, (5.0, 15.0)))
    feat["move_30_bucket"] = feat["move_30"].apply(lambda v: _bucket_signed_move(v, (8.0, 20.0)))
    feat["move_60_bucket"] = feat["move_60"].apply(lambda v: _bucket_signed_move(v, (12.0, 30.0)))
    feat["bar_return_bucket"] = feat["bar_return"].apply(lambda v: _bucket_signed_move(v, (2.0, 6.0)))
    feat["hour_bucket_hkt"] = feat["hour_hkt"].apply(lambda h: f"{int(h):02d}:00")
    return feat


def _bucket_report(df: pd.DataFrame, feature_col: str, baseline_avg_trade: float, baseline_pf: float | None) -> list[dict[str, Any]]:
    grouped = (
        df.groupby(feature_col, dropna=False)
        .agg(
            trades=("pnl", "size"),
            total_pnl=("pnl", "sum"),
            avg_trade=("pnl", "mean"),
            win_rate_pct=("pnl", lambda s: float((s > 0).mean() * 100.0)),
            gross_profit=("pnl", lambda s: float(s[s > 0].sum())),
            gross_loss=("pnl", lambda s: float(s[s < 0].sum())),
            avg_prob=("entry_signal_prob", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["trades"] >= MIN_BUCKET_TRADES].copy()
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        gross_loss = float(row["gross_loss"])
        pf = (float(row["gross_profit"]) / abs(gross_loss)) if gross_loss < 0 else None
        rows.append(
            {
                "feature": feature_col,
                "bucket": str(row[feature_col]),
                "trades": int(row["trades"]),
                "total_pnl": float(row["total_pnl"]),
                "avg_trade": float(row["avg_trade"]),
                "avg_trade_uplift": float(row["avg_trade"] - baseline_avg_trade),
                "profit_factor": float(pf) if pf is not None else None,
                "profit_factor_uplift": None if pf is None or baseline_pf is None else float(pf - baseline_pf),
                "win_rate_pct": float(row["win_rate_pct"]),
                "avg_prob": float(row["avg_prob"]) if pd.notna(row["avg_prob"]) else None,
            }
        )
    return rows


def main() -> int:
    report = _read_json(REPORT_PATH)
    cfg = dict(report.get("config") or {})
    trades = _read_trades(TRADES_PATH)
    bars = _load_bars(cfg)
    feat = _compute_trade_features(trades, bars, cfg)

    baseline_avg_trade = float(feat["pnl"].mean())
    gross_profit = float(feat.loc[feat["pnl"] > 0, "pnl"].sum())
    gross_loss = float(feat.loc[feat["pnl"] < 0, "pnl"].sum())
    baseline_pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    pre_entry_feature_cols = [
        "side_bucket",
        "prob_bucket",
        "session_hkt",
        "hour_bucket_hkt",
        "weekday_hkt",
        "range_150_bucket",
        "swing_15_bucket",
        "move_15_bucket",
        "move_30_bucket",
        "move_60_bucket",
        "wr90_bucket",
        "volatility_30_bucket",
        "volume_rel_bucket",
        "bar_return_bucket",
    ]
    outcome_feature_cols = [
        "exit_reason_bucket",
    ]

    ranked_rows: list[dict[str, Any]] = []
    for feature_col in pre_entry_feature_cols:
        ranked_rows.extend(_bucket_report(feat, feature_col, baseline_avg_trade, baseline_pf))

    outcome_rows: list[dict[str, Any]] = []
    for feature_col in outcome_feature_cols:
        outcome_rows.extend(_bucket_report(feat, feature_col, baseline_avg_trade, baseline_pf))

    ranked_df = pd.DataFrame(ranked_rows)
    ranked_df = ranked_df.sort_values(["avg_trade_uplift", "total_pnl", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    outcome_df = pd.DataFrame(outcome_rows)
    if not outcome_df.empty:
        outcome_df = outcome_df.sort_values(["avg_trade_uplift", "total_pnl", "trades"], ascending=[False, False, False]).reset_index(drop=True)

    top_positive = ranked_df.head(TOP_N).to_dict(orient="records")
    top_negative = ranked_df.sort_values(["avg_trade_uplift", "total_pnl"], ascending=[True, True]).head(TOP_N).to_dict(orient="records")
    outcome_summary = outcome_df.to_dict(orient="records") if not outcome_df.empty else []

    result = {
        "model": {
            "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "trades_path": str(TRADES_PATH.relative_to(PROJECT_ROOT)),
            "signal_model_path": report.get("config", {}).get("model_out") or "training/backtest_model_best_base_corrected.joblib",
        },
        "coverage": {
            "trade_count": int(len(feat)),
            "trade_start_hkt": _to_hkt_str(feat["entry_time"].min()),
            "trade_end_hkt": _to_hkt_str(feat["entry_time"].max()),
        },
        "baseline": {
            "avg_trade": baseline_avg_trade,
            "profit_factor": baseline_pf,
            "total_pnl": float(feat["pnl"].sum()),
            "win_rate_pct": float((feat["pnl"] > 0).mean() * 100.0),
        },
        "pre_entry_feature_columns": pre_entry_feature_cols,
        "top_positive_buckets": top_positive,
        "top_negative_buckets": top_negative,
        "outcome_metadata_buckets": outcome_summary,
    }

    json_out = RUNTIME_DIR / "best_base_regime_feature_report.json"
    md_out = RUNTIME_DIR / "best_base_regime_feature_report.md"
    csv_out = RUNTIME_DIR / "best_base_regime_feature_buckets.csv"

    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_payload = ranked_df.copy()
    csv_payload["feature_scope"] = "pre_entry"
    if not outcome_df.empty:
        extra = outcome_df.copy()
        extra["feature_scope"] = "outcome_metadata"
        csv_payload = pd.concat([csv_payload, extra], ignore_index=True)
    csv_payload.to_csv(csv_out, index=False)

    def _fmt_rows(rows: list[dict[str, Any]]) -> str:
        return "\n".join(
            [
                f"- `{row['feature']} = {row['bucket']}` trades={row['trades']} avg_trade={row['avg_trade']:.2f} "
                f"uplift={row['avg_trade_uplift']:.2f} total_pnl={row['total_pnl']:.2f} pf={row['profit_factor']}"
                for row in rows
            ]
        )

    md = f"""# Best-base regime feature report

## Coverage
- Trades analyzed: `{result['coverage']['trade_count']}`
- Start (HKT): `{result['coverage']['trade_start_hkt']}`
- End (HKT): `{result['coverage']['trade_end_hkt']}`

## Baseline
- Avg trade: `{result['baseline']['avg_trade']:.2f}`
- Profit factor: `{result['baseline']['profit_factor']}`
- Total pnl: `{result['baseline']['total_pnl']:.2f}`
- Win rate %: `{result['baseline']['win_rate_pct']:.2f}`

## Top positive regime-feature buckets
{_fmt_rows(top_positive)}

## Top negative regime-feature buckets
{_fmt_rows(top_negative)}

## Outcome metadata buckets (not usable as forward filters)
{_fmt_rows(outcome_summary[:6]) if outcome_summary else '- none'}

## Interpretation
- These are reconstructable regime/context proxies, not raw model feature importances.
- The ranked positive/negative lists above are restricted to **entry-available** features.
- Positive buckets indicate conditions where the promoted best-base trades materially outperform the overall average trade.
- Negative buckets indicate conditions that may be worth filtering or deprioritizing in future regime gating experiments.
"""
    md_out.write_text(md, encoding="utf-8")

    print(f"Saved: {json_out}")
    print(f"Saved: {md_out}")
    print(f"Saved: {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

