#!/usr/bin/env python3
"""Train the 15-minute asymmetric-target alpha model.

Target definition:
- Build 15-minute candles from raw data.
- Label a bar positive when, within the next N bars, max high is >= +threshold
  and min low does not breach -max-adverse-low.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import DataLoader
try:
    from training.ml_alpha_model_15m_nextbar import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        UptrendRecognitionSystem15mNextBar,
        chronological_train_test_split,
        plot_results,
        prepare_gold_data_15m,
    )
except ModuleNotFoundError:
    from ml_alpha_model_15m_nextbar import (
        DEFAULT_FORWARD_BARS,
        DEFAULT_MAX_ADVERSE_LOW_PCT,
        DEFAULT_THRESHOLD_PCT,
        UptrendRecognitionSystem15mNextBar,
        chronological_train_test_split,
        plot_results,
        prepare_gold_data_15m,
    )


ROOT_DIR = Path(__file__).resolve().parent.parent
plt.style.use("default")
plt.rcParams["figure.figsize"] = (15, 10)


def analyze_prediction_accuracy(system, data: pd.DataFrame, artifacts_dir: Path, test_size: float = 0.2) -> None:
    print("\n" + "=" * 80)
    print("PREDICTION ACCURACY ANALYSIS")
    print("=" * 80)

    X = data[system.extractor.feature_names].fillna(0)
    y = data["is_uptrend"]
    _, X_test, _, y_test = chronological_train_test_split(X, y, test_size=test_size)

    from sklearn.metrics import auc, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    rf_probs = system.models["random_forest"].predict_proba(X_test)

    for model_name in ("random_forest", "gradient_boosting"):
        model = system.models[model_name]
        probs = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves - 15m Next-Bar Target")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(rf_probs[y_test == 0], bins=30, alpha=0.5, label="No Uptrend", color="red")
    ax2.hist(rf_probs[y_test == 1], bins=30, alpha=0.5, label="Uptrend", color="green")
    ax2.axvline(x=0.5, color="k", linestyle="--", label="Threshold")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("RF Probability Distribution by Actual Label")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "roc_analysis_15m_nextbar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved ROC analysis to {out_path}")


def run_feature_correlation_analysis(
    data: pd.DataFrame,
    feature_names: list[str],
    artifacts_dir: Path,
    corr_threshold: float = 0.90,
    top_n: int = 25,
) -> None:
    """Save feature-vs-target and feature-vs-feature correlation diagnostics."""
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    feature_df = data[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target = pd.to_numeric(data["is_uptrend"], errors="coerce").fillna(0.0)

    corr_to_target = feature_df.corrwith(target).fillna(0.0)
    corr_summary = pd.DataFrame(
        {
            "feature": corr_to_target.index,
            "corr_with_target": corr_to_target.values,
            "abs_corr_with_target": corr_to_target.abs().values,
        }
    ).sort_values("abs_corr_with_target", ascending=False)

    corr_csv = artifacts_dir / "feature_target_correlation_15m_nextbar.csv"
    corr_summary.to_csv(corr_csv, index=False)
    print(f"✅ Saved feature-target correlation table to {corr_csv}")

    feat_corr = feature_df.corr().fillna(0.0)
    pairs: list[dict[str, float | str]] = []
    for a, b in combinations(feat_corr.columns.tolist(), 2):
        c = float(feat_corr.loc[a, b])
        if abs(c) >= corr_threshold:
            pairs.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "corr": c,
                    "abs_corr": abs(c),
                }
            )

    pairs_df = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "corr", "abs_corr"]
    )
    pairs_csv = artifacts_dir / "highly_correlated_feature_pairs_15m_nextbar.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"✅ Saved high-correlation feature pairs to {pairs_csv} (threshold={corr_threshold:.2f})")

    top_features = corr_summary.head(max(5, int(top_n)))["feature"].tolist()
    plot_df = feature_df[top_features].copy()
    plot_df["is_uptrend"] = target.to_numpy()
    matrix = plot_df.corr().to_numpy()
    labels = plot_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Correlation Matrix (Top Features + Target)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heatmap_path = artifacts_dir / "correlation_matrix_top_features_15m_nextbar.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved correlation heatmap to {heatmap_path}")

    print("Top 10 features by |corr with target|:")
    print(corr_summary.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def visualize_feature_patterns(system, artifacts_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE VISUALIZATION")
    print("=" * 80)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rf_fig = system.models["random_forest"].plot_feature_importance(top_n=25)
    rf_path = artifacts_dir / "feature_importance_rf_15m_nextbar.png"
    plt.savefig(rf_path, dpi=150, bbox_inches="tight")
    plt.close(rf_fig)
    print(f"✅ Saved Random Forest feature importance to {rf_path}")

    gb_fig = system.models["gradient_boosting"].plot_feature_importance(top_n=25)
    gb_path = artifacts_dir / "feature_importance_gb_15m_nextbar.png"
    plt.savefig(gb_path, dpi=150, bbox_inches="tight")
    plt.close(gb_fig)
    print(f"✅ Saved Gradient Boosting feature importance to {gb_path}")


def test_on_recent_data(df: pd.DataFrame, system, artifacts_dir: Path, lookback: int = 120) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("TESTING ON RECENT 15-MINUTE DATA")
    print("=" * 80)

    predictions = system.scan_recent_patterns(df, model_type="random_forest", lookback=lookback)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pred_path = artifacts_dir / "recent_predictions_15m_nextbar.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"✅ Saved recent predictions to {pred_path}")

    if not predictions.empty:
        start_idx = max(0, len(df) - lookback - 30)
        df_recent = df.iloc[start_idx:].reset_index(drop=True)
        predictions_adjusted = predictions.copy()
        predictions_adjusted["idx"] = predictions_adjusted["idx"] - start_idx
        fig = plot_results(df_recent, predictions_adjusted, title=f"15m Next-Bar Predictions (Last {lookback} Bars)")
        viz_path = artifacts_dir / "recent_predictions_15m_nextbar.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved recent prediction chart to {viz_path}")
    else:
        print("⚠️ No recent RF signals found on the selected period")

    return predictions


def save_signal_cache(df: pd.DataFrame, data: pd.DataFrame, system, out_path: Path, test_size: float | None = None) -> None:
    X = data[system.extractor.feature_names].fillna(0)
    rows = data[["idx", "is_uptrend", "future_gain_pct"]].copy()

    if test_size is not None:
        _, X_sel, _, y_sel = chronological_train_test_split(X, data["is_uptrend"], test_size=test_size)
        rows = rows.loc[X_sel.index].copy()
        rows["actual"] = y_sel.values
        X = X_sel
    else:
        rows["actual"] = rows["is_uptrend"].astype(int)

    rows["timestamp"] = df.index[rows["idx"].astype(int)].astype(str).values
    rows["rf_probability"] = system.models["random_forest"].predict_proba(X)
    rows["gb_probability"] = system.models["gradient_boosting"].predict_proba(X)
    rows["rf_signal"] = system.models["random_forest"].predict(X).astype(int)
    rows["gb_signal"] = system.models["gradient_boosting"].predict(X).astype(int)
    rows = rows.sort_values("idx").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_path, index=False)
    print(f"✅ Saved signal cache to {out_path} ({len(rows):,} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the 15-minute next-bar-high alpha model.")
    parser.add_argument("--table", default="gold_prices")
    parser.add_argument("--start-date", default="2025-05-20")
    parser.add_argument("--end-date", default="2026-02-04")
    parser.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    parser.add_argument("--max-adverse-low-pct", type=float, default=DEFAULT_MAX_ADVERSE_LOW_PCT)
    parser.add_argument("--forward-bars", type=int, default=DEFAULT_FORWARD_BARS)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--correlation-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--corr-top-n", type=int, default=25)
    parser.add_argument("--model-dir", default="ml_models_15m_nextbar")
    parser.add_argument("--artifacts-dir", default="training/artifacts_15m_nextbar")
    parser.add_argument("--test-cache-out", default="training/test_period_signals_15m_nextbar.csv")
    parser.add_argument("--full-cache-out", default="training/full_period_signals_15m_nextbar.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts_dir = ROOT_DIR / args.artifacts_dir

    print("=" * 80)
    print("TRAIN 15-MINUTE NEXT-BAR-HIGH MODEL")
    print("=" * 80)
    print("Configuration:")
    print("  Candle timeframe: 15 minutes")
    print(
        "  Target: future max high >= "
        f"+{args.threshold_pct:.2f}% and future min low >= -{args.max_adverse_low_pct:.2f}% "
        f"within {args.forward_bars} bars"
    )
    print(f"  Validation split: chronological {(1 - args.test_size) * 100:.0f}/{args.test_size * 100:.0f}")
    print()

    print("Loading raw gold data...")
    raw = DataLoader().load_data(args.table, start_date=args.start_date, end_date=args.end_date)
    df = prepare_gold_data_15m(raw)
    print(f"✅ Prepared {len(df):,} aggregated 15-minute candles")

    system = UptrendRecognitionSystem15mNextBar(
        threshold_pct=args.threshold_pct,
        max_adverse_low_pct=args.max_adverse_low_pct,
        forward_bars=args.forward_bars,
    )
    data = system.prepare_data(df)

    if args.correlation_analysis:
        run_feature_correlation_analysis(
            data=data,
            feature_names=system.extractor.feature_names,
            artifacts_dir=artifacts_dir,
            corr_threshold=float(args.corr_threshold),
            top_n=int(args.corr_top_n),
        )

    system.train_models(data, test_size=args.test_size)

    analyze_prediction_accuracy(system, data, artifacts_dir=artifacts_dir, test_size=args.test_size)
    visualize_feature_patterns(system, artifacts_dir=artifacts_dir)
    test_on_recent_data(df, system, artifacts_dir=artifacts_dir)

    save_signal_cache(
        df,
        data,
        system,
        out_path=ROOT_DIR / args.test_cache_out,
        test_size=args.test_size,
    )
    save_signal_cache(
        df,
        data,
        system,
        out_path=ROOT_DIR / args.full_cache_out,
        test_size=None,
    )

    model_dir = ROOT_DIR / args.model_dir
    system.save_models(str(model_dir))
    print(f"✅ Saved models to {model_dir}")

    print("\nTraining complete.")
    print("Generated:")
    print(f"  - Models: {model_dir}")
    print(f"  - Artifacts: {artifacts_dir}")
    print(f"  - Test cache: {ROOT_DIR / args.test_cache_out}")
    print(f"  - Full cache: {ROOT_DIR / args.full_cache_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

