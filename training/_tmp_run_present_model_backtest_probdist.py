from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path("/Users/alpha/Desktop/python/AlphaGold")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training"))

from data import DataLoader

try:
    from training.image_trend_ml import (
        LABEL_DOWN,
        LABEL_FLAT,
        LABEL_RISKY,
        LABEL_UP,
        TREND_LABELS,
        _prepare_ohlcv,
        _resolve_directional_stops,
        _resolve_directional_targets,
        build_dataset,
        directional_pnl_report,
        flatten_tensors,
        predict_two_stage_details,
    )
    from training.sweep_wf_probs_no_retrain import _build_cycle_augmented_features, _cycle_slice_bounds, _find_slice
except ModuleNotFoundError:
    from image_trend_ml import (
        LABEL_DOWN,
        LABEL_FLAT,
        LABEL_RISKY,
        LABEL_UP,
        TREND_LABELS,
        _prepare_ohlcv,
        _resolve_directional_stops,
        _resolve_directional_targets,
        build_dataset,
        directional_pnl_report,
        flatten_tensors,
        predict_two_stage_details,
    )
    from sweep_wf_probs_no_retrain import _build_cycle_augmented_features, _cycle_slice_bounds, _find_slice


def main() -> int:
    report_path = PROJECT_ROOT / "training/backtest_report_wf_final_alias.json"
    report = json.loads(report_path.read_text())
    cfg = dict(report["config"])

    print(f"report: {report_path}")
    print(f"model_out: {cfg.get('model_out')}")
    print(f"evaluation_mode: {report.get('evaluation_mode')}")

    raw = DataLoader().load_data(cfg["table"], start_date=cfg["start_date"], end_date=cfg["end_date"])
    bars = _prepare_ohlcv(raw, cfg["timeframe"])
    long_thr, short_thr = _resolve_directional_targets(
        float(cfg["trend_threshold"]),
        cfg.get("long_target_threshold"),
        cfg.get("short_target_threshold"),
    )
    long_stop, short_stop = _resolve_directional_stops(
        float(cfg["adverse_limit"]),
        cfg.get("long_adverse_limit"),
        cfg.get("short_adverse_limit"),
    )

    X_tensor, y, ts, curr_a, entry_a, fut_a, entry_ts_a, fut_ts_a, _, _ = build_dataset(
        bars,
        int(cfg["window"]),
        float(cfg["min_window_range"]),
        int(cfg["horizon"]),
        float(cfg["trend_threshold"]),
        float(cfg["adverse_limit"]),
        long_target_threshold=cfg.get("long_target_threshold"),
        short_target_threshold=cfg.get("short_target_threshold"),
        long_adverse_limit=cfg.get("long_adverse_limit"),
        short_adverse_limit=cfg.get("short_adverse_limit"),
        min_15m_drop=float(cfg.get("min_15m_drop", 0.0) or 0.0),
        min_15m_rise=float(cfg.get("min_15m_rise", 0.0) or 0.0),
        window_15m=int(cfg.get("window_15m", 0) or 0),
        apply_time_filter=not bool(cfg.get("disable_time_filter", False)),
    )
    X_flat = flatten_tensors(X_tensor)

    pred_all = np.full(len(y), LABEL_FLAT, dtype=np.int64)
    signal_prob_all = np.zeros(len(y), dtype=np.float64)
    trend_prob_all = np.zeros(len(y), dtype=np.float64)
    up_prob_all = np.full(len(y), np.nan, dtype=np.float64)
    mask = np.zeros(len(y), dtype=bool)

    for c in report.get("walkforward_cycles", []):
        cyc_cfg = cfg.copy()
        cyc_cfg.update(c.get("selected_config", {}))
        cyc_model = Path(c["cycle_model_out"])
        if not cyc_model.is_absolute():
            cyc_model = PROJECT_ROOT / cyc_model
        bundle = joblib.load(cyc_model)
        m1, m2 = bundle["stage1"], bundle["stage2"]
        test_start_bound, test_end_bound = _cycle_slice_bounds(c)
        sl = _find_slice(ts, test_start_bound, test_end_bound)
        x_raw = X_flat[sl]
        x_aug = _build_cycle_augmented_features(
            x_raw=x_raw,
            curr=curr_a[sl],
            entry_px=entry_a[sl],
            m1=m1,
            m2=m2,
            stage1_gate=float(c["selected_config"]["stage1_min_prob"]),
            stage2_gate=float(c["selected_config"]["stage2_min_prob"]),
            cfg=cyc_cfg,
            long_thr=long_thr,
            short_thr=short_thr,
            long_stop=long_stop,
            short_stop=short_stop,
            state_feature_names=list(bundle.get("state_feature_names", [])),
            max_state_iters=3,
        )
        gate_up = float(
            c["selected_config"].get(
                "stage2_min_prob_up",
                c["selected_config"]["stage2_min_prob"],
            )
        )
        gate_down = float(
            c["selected_config"].get(
                "stage2_min_prob_down",
                c["selected_config"]["stage2_min_prob"],
            )
        )
        pm = predict_two_stage_details(
            x_aug,
            m1,
            m2,
            stage1_min_prob=float(c["selected_config"]["stage1_min_prob"]),
            stage2_min_prob=float(c["selected_config"]["stage2_min_prob"]),
            stage2_min_prob_up=gate_up,
            stage2_min_prob_down=gate_down,
            stage1_min_prob_1m=cfg.get("stage1_min_prob_1m"),
            stage1_min_prob_15m=cfg.get("stage1_min_prob_15m"),
            stage2_min_prob_1m=cfg.get("stage2_min_prob_1m"),
            stage2_min_prob_15m=cfg.get("stage2_min_prob_15m"),
        )
        pred_all[sl] = pm["pred"]
        signal_prob_all[sl] = pm["signal_prob"]
        trend_prob_all[sl] = pm["trend_prob"]
        up_prob_all[sl] = pm["up_prob"]
        mask[sl] = True

    eval_idx = np.where(mask)[0]
    y_true = y[eval_idx]
    y_pred = pred_all[eval_idx]
    signal_prob = signal_prob_all[eval_idx]
    trend_prob = trend_prob_all[eval_idx]
    up_prob = up_prob_all[eval_idx]

    pnl, _ = directional_pnl_report(
        ts[eval_idx],
        entry_ts_a[eval_idx],
        fut_ts_a[eval_idx],
        y_pred,
        curr_a[eval_idx],
        entry_a[eval_idx],
        fut_a[eval_idx],
        signal_prob=signal_prob,
        adverse_limit=float(cfg["adverse_limit"]),
        long_adverse_limit=long_stop,
        short_adverse_limit=short_stop,
        allow_overlap=bool(cfg.get("allow_overlap_backtest", False)),
        reverse_exit_prob=float(cfg.get("reverse_exit_prob", 0.7)),
    )

    overall_correct = y_pred == y_true
    non_risky_mask = y_true != LABEL_RISKY
    sig_mask = np.isin(y_pred, TREND_LABELS)
    sig_probs = signal_prob[sig_mask]
    up_mask = y_pred == LABEL_UP
    down_mask = y_pred == LABEL_DOWN
    sig_correct = y_pred[sig_mask] == y_true[sig_mask]

    def _print_direction_block(name: str, pred_mask: np.ndarray, true_label: int) -> None:
        probs = signal_prob[pred_mask]
        correct = y_true[pred_mask] == true_label
        print(f"{name}_only:")
        print(f"  correct_pct={correct.mean() * 100:.2f}% ({int(correct.sum())}/{len(correct)})")
        print(
            f"  prob_summary: count={len(probs)} mean={probs.mean():.4f} median={np.median(probs):.4f} "
            f"min={probs.min():.4f} max={probs.max():.4f}"
        )
        quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        qvals = np.quantile(probs, quantiles)
        print("  quantiles: " + ", ".join(f"q{int(q * 100):02d}={v:.4f}" for q, v in zip(quantiles, qvals)))
        bins = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001])
        hist, edges = np.histogram(probs, bins=bins)
        print("  prob_histogram:")
        for n, lo, hi in zip(hist, edges[:-1], edges[1:]):
            hi_disp = min(hi, 1.0)
            bracket = "]" if hi >= 1.0 else ")"
            in_bin = (probs >= lo) & ((probs <= 1.0) if hi >= 1.0 else (probs < hi))
            hit_txt = "n/a"
            if int(in_bin.sum()) > 0:
                hit_txt = f"{float(correct[in_bin].mean() * 100.0):.2f}%"
            print(f"    [{lo:.2f}, {hi_disp:.2f}{bracket}: count={int(n)} hit_rate={hit_txt}")
        print()

    print(f"eval_samples: {len(eval_idx)}")
    print(f"pred_counts: down={int(down_mask.sum())} flat={int((y_pred == LABEL_FLAT).sum())} up={int(up_mask.sum())}")
    print(f"backtest_trades: {int(pnl.get('trades', 0))}")
    print(f"backtest_total_pnl: {float(pnl.get('total_pnl', 0.0)):.2f}")
    print()

    print("correct_prediction_percentages:")
    print(f"  overall_exact={overall_correct.mean() * 100:.2f}% ({int(overall_correct.sum())}/{len(overall_correct)})")
    if non_risky_mask.any():
        nr_correct = (y_pred[non_risky_mask] == y_true[non_risky_mask])
        print(f"  non_risky_exact={nr_correct.mean() * 100:.2f}% ({int(nr_correct.sum())}/{len(nr_correct)})")
    if sig_mask.any():
        print(f"  signal_only_hit_rate={sig_correct.mean() * 100:.2f}% ({int(sig_correct.sum())}/{len(sig_correct)})")
    if down_mask.any():
        down_correct = y_true[down_mask] == LABEL_DOWN
        print(f"  predicted_down_hit_rate={down_correct.mean() * 100:.2f}% ({int(down_correct.sum())}/{len(down_correct)})")
    if up_mask.any():
        up_correct = y_true[up_mask] == LABEL_UP
        print(f"  predicted_up_hit_rate={up_correct.mean() * 100:.2f}% ({int(up_correct.sum())}/{len(up_correct)})")
    print()

    if down_mask.any():
        _print_direction_block("short", down_mask, LABEL_DOWN)
    if up_mask.any():
        _print_direction_block("long", up_mask, LABEL_UP)

    if len(sig_probs) == 0:
        print("No non-flat predictions; no final prediction probability distribution to show.")
    else:
        quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        qvals = np.quantile(sig_probs, quantiles)
        print("final_prediction_signal_prob_summary:")
        print(f"  count={len(sig_probs)} mean={sig_probs.mean():.4f} std={sig_probs.std():.4f}")
        for q, v in zip(quantiles, qvals):
            print(f"  q{int(q * 100):02d}={v:.4f}")
        print()

        bins = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001])
        hist, edges = np.histogram(sig_probs, bins=bins)
        print("final_prediction_signal_prob_histogram:")
        for n, lo, hi in zip(hist, edges[:-1], edges[1:]):
            hi_disp = min(hi, 1.0)
            bracket = "]" if hi >= 1.0 else ")"
            print(f"  [{lo:.2f}, {hi_disp:.2f}{bracket}: {int(n)}")
        print()

        print("directional_breakdown:")
        for name, arr in (("down", signal_prob[down_mask]), ("up", signal_prob[up_mask])):
            if len(arr) == 0:
                print(f"  {name}: count=0")
            else:
                print(
                    f"  {name}: count={len(arr)} mean={arr.mean():.4f} "
                    f"median={np.median(arr):.4f} min={arr.min():.4f} max={arr.max():.4f}"
                )
        print()

        bins = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001])
        print("correct_prediction_pct_by_signal_prob_bucket:")
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (sig_probs >= lo) & (sig_probs < hi)
            if hi >= 1.0:
                in_bin = (sig_probs >= lo) & (sig_probs <= 1.0)
            count = int(in_bin.sum())
            hi_disp = min(hi, 1.0)
            bracket = "]" if hi >= 1.0 else ")"
            if count == 0:
                print(f"  [{lo:.2f}, {hi_disp:.2f}{bracket}: count=0 hit_rate=n/a")
                continue
            hit_rate = float(sig_correct[in_bin].mean() * 100.0)
            print(f"  [{lo:.2f}, {hi_disp:.2f}{bracket}: count={count} hit_rate={hit_rate:.2f}%")
        print()

    trend_nonzero = trend_prob[trend_prob > 0]
    if len(trend_nonzero):
        print("stage1_trend_prob_summary_on_eval:")
        print(
            f"  mean={trend_nonzero.mean():.4f} median={np.median(trend_nonzero):.4f} "
            f"p90={np.quantile(trend_nonzero, 0.9):.4f} max={trend_nonzero.max():.4f}"
        )

    valid_up = up_prob[~np.isnan(up_prob)]
    if len(valid_up):
        print("stage2_up_prob_summary_on_trend_candidates:")
        print(
            f"  mean={valid_up.mean():.4f} median={np.median(valid_up):.4f} "
            f"p10={np.quantile(valid_up, 0.1):.4f} p90={np.quantile(valid_up, 0.9):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

