#!/usr/bin/env python3
"""Summarize price-action feature importance by group (fvg / wick / fake)."""
import glob
import os
import joblib

MODEL_DIR = os.environ.get(
    "V14_MODEL_OUTPUT_DIR", "runtime/bot_assets/wf_models_v14_pa_all"
)

GROUPS = {
    "fvg": ["dist_fvg", "time_from_fvg", "fvg_bull", "fvg_bear"],
    "wick": ["time_from_long_upper_wick", "time_from_long_lower_wick", "long_upper_wick", "long_lower_wick"],
    "fake": ["time_from_fake_up", "time_from_fake_down", "fake_up", "fake_down"],
}


def classify(name: str) -> str:
    for group, keys in GROUPS.items():
        if any(k in name for k in keys):
            return group
    return "other"


def score_model(path: str, title: str):
    print(f"\n=== {title} ===")
    model = joblib.load(path)
    xgb = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    raw = xgb.get_booster().get_score(importance_type="gain")
    fn = getattr(xgb, "feature_names_in_", None)
    names = list(fn) if fn is not None and len(fn) else []
    by_group = {"fvg": 0.0, "wick": 0.0, "fake": 0.0, "other": 0.0}
    rows = []
    for key, gain in raw.items():
        if key.startswith("f") and key[1:].isdigit() and names:
            idx = int(key[1:])
            fname = names[idx] if idx < len(names) else key
        else:
            fname = key
        g = classify(fname)
        by_group[g] += gain
        if g != "other":
            rows.append((fname, gain, g))
    rows.sort(key=lambda x: x[1], reverse=True)
    total_pa = by_group["fvg"] + by_group["wick"] + by_group["fake"]
    print("Group totals (gain):")
    for g in ("fvg", "wick", "fake"):
        pct = 100 * by_group[g] / total_pa if total_pa else 0
        print(f"  {g:5s}: {by_group[g]:10.2f}  ({pct:5.1f}% of PA features)")
    print("\nTop 10 PA features:")
    for i, (fname, gain, g) in enumerate(rows[:10], 1):
        print(f"  {i:2d}. [{g}] {fname:<28} {gain:.4f}")


def main():
    s1 = sorted(glob.glob(f"{MODEL_DIR}/filter_v14_cycle_*.joblib"), key=os.path.getmtime)
    s2 = sorted(glob.glob(f"{MODEL_DIR}/directional_v14_cycle_*.joblib"), key=os.path.getmtime)
    if s1:
        score_model(s1[-1], "Stage 1 — latest cycle")
    if s2:
        score_model(s2[-1], "Stage 2 — latest cycle")


if __name__ == "__main__":
    main()
