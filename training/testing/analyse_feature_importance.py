"""
analyse_feature_importance.py
------------------------------
Extract and visualise feature importances from the saved best-base model bundle.

Features are flattened as:  feature_idx = channel_idx * n_bars + bar_idx
  channel order : image_channel_names  (9 channels)
  bar order     : bar 0 = oldest, bar 149 = most recent (current bar)

Outputs
-------
  runtime/feature_importance/
    feature_importance_by_channel.png   – mean importance per channel (bar chart)
    feature_importance_heatmap.png      – importance heatmap: channel × bar position
    feature_importance_top50.png        – top-50 individual features
    feature_importance.csv              – full table (channel, bar, importance)
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ──────────────────────────────────────────────────────────────────
BUNDLE_PATH = PROJECT_ROOT / "runtime/bot_assets/backtest_model_best_base_weak_nostate.joblib"
OUT_DIR = PROJECT_ROOT / "runtime/feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load bundle ────────────────────────────────────────────────────────────
print("Loading bundle …")
bundle = joblib.load(BUNDLE_PATH)
channel_names: list[str] = bundle["image_channel_names"]
feature_shape: list[int] = bundle["feature_shape"]   # [n_channels, n_bars]
n_channels, n_bars = feature_shape[0], feature_shape[1]
n_1m_feats: int = bundle.get("n_1m_feats", n_channels * n_bars)

m1 = bundle["stage1"]
m2 = bundle["stage2"]

print(f"Channels  : {channel_names}")
print(f"Bars      : {n_bars}")
print(f"1-min feats: {n_1m_feats}")
print(f"Stage1 type: {type(m1).__name__}")

# ── feature importances ────────────────────────────────────────────────────
def get_importances(model, label: str) -> np.ndarray:
    """Extract split-gain importances from HistGradientBoostingClassifier internal trees."""
    inner = getattr(model, "estimator_", model)
    inner = getattr(inner, "named_steps", {}).get("clf", inner)

    # Try built-in first
    fi = getattr(inner, "feature_importances_", None)
    if fi is not None:
        return np.asarray(fi[:n_1m_feats], dtype=np.float64)

    # For HistGradientBoostingClassifier: walk internal tree nodes
    predictors = getattr(inner, "_predictors", None)
    if predictors is None:
        print(f"[WARN] {label}: cannot extract importances")
        return np.zeros(n_1m_feats)

    importance = np.zeros(n_1m_feats, dtype=np.float64)
    for class_predictors in predictors:        # one list per class / per iteration
        for tree in class_predictors:
            nodes = tree.nodes
            for node in nodes:
                # internal nodes have feature_idx >= 0 and gain > 0
                feat = int(node["feature_idx"]) if "feature_idx" in node.dtype.names else -1
                gain = float(node["gain"]) if "gain" in node.dtype.names else 0.0
                if feat >= 0 and feat < n_1m_feats and gain > 0:
                    importance[feat] += gain

    total = importance.sum()
    if total > 0:
        importance /= total
    print(f"[INFO] {label}: extracted split-gain importances from {len(predictors)} iterations")
    return importance

fi_s1 = get_importances(m1, "stage1")
fi_s2 = get_importances(m2, "stage2")
fi_avg = (fi_s1 + fi_s2) / 2.0

# ── build DataFrame ────────────────────────────────────────────────────────
records = []
for idx in range(n_1m_feats):
    ch_idx = idx // n_bars
    bar_idx = idx % n_bars
    ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"
    records.append({
        "feature_idx": idx,
        "channel": ch_name,
        "bar":  bar_idx,
        "bars_ago": n_bars - 1 - bar_idx,   # 0 = current bar
        "importance_s1":  fi_s1[idx],
        "importance_s2":  fi_s2[idx],
        "importance_avg": fi_avg[idx],
    })

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
print(f"Saved CSV → {OUT_DIR / 'feature_importance.csv'}")

# ── 1) Bar chart: mean importance per channel ──────────────────────────────
ch_mean = df.groupby("channel")["importance_avg"].sum().reindex(channel_names)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(channel_names)))
bars = ax.bar(channel_names, ch_mean.values, color=colors)
ax.set_title("Total Feature Importance by Channel\n(avg of stage-1 & stage-2, best-base model)", fontsize=13)
ax.set_xlabel("Channel")
ax.set_ylabel("Sum of importance")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
plt.xticks(rotation=30, ha="right")
for bar, val in zip(bars, ch_mean.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
out = OUT_DIR / "feature_importance_by_channel.png"
fig.savefig(out, dpi=150)
plt.close()
print(f"Saved → {out}")

# ── 2) Heatmap: channel × bar ──────────────────────────────────────────────
heat = df.pivot_table(index="channel", columns="bars_ago", values="importance_avg", aggfunc="mean")
heat = heat.reindex(channel_names)          # keep channel order
heat = heat[sorted(heat.columns)]           # bars_ago ascending

fig, ax = plt.subplots(figsize=(18, 4))
im = ax.imshow(heat.values, aspect="auto", cmap="hot", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Importance")
ax.set_yticks(range(len(channel_names)))
ax.set_yticklabels(channel_names)
# x-axis: show every 10 bars
step = 10
xtick_pos = list(range(0, n_bars, step))
ax.set_xticks(xtick_pos)
ax.set_xticklabels([str(v) for v in xtick_pos], fontsize=7)
ax.set_xlabel("bars ago (0 = most recent)")
ax.set_title("Feature Importance Heatmap: Channel × Time Position\n(brighter = more important)", fontsize=13)
plt.tight_layout()
out = OUT_DIR / "feature_importance_heatmap.png"
fig.savefig(out, dpi=150)
plt.close()
print(f"Saved → {out}")

# ── 3) Top-50 features ────────────────────────────────────────────────────
top50 = df.nlargest(50, "importance_avg").reset_index(drop=True)
labels = [f"{r['channel']}\nbars_ago={int(r['bars_ago'])}" for _, r in top50.iterrows()]

fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(range(50), top50["importance_avg"].values[::-1],
        color=plt.cm.viridis(np.linspace(0.2, 0.9, 50)))
ax.set_yticks(range(50))
ax.set_yticklabels(labels[::-1], fontsize=7)
ax.set_xlabel("Importance (avg s1+s2)")
ax.set_title("Top-50 Most Important Features (best-base model)", fontsize=13)
plt.tight_layout()
out = OUT_DIR / "feature_importance_top50.png"
fig.savefig(out, dpi=150)
plt.close()
print(f"Saved → {out}")

# ── 4) Print summary ──────────────────────────────────────────────────────
print("\n── Channel ranking (by total importance) ──")
ranking = ch_mean.sort_values(ascending=False)
for rank, (ch, val) in enumerate(ranking.items(), 1):
    pct = val / ch_mean.sum() * 100
    print(f"  {rank}. {ch:<20s}  {val:.5f}  ({pct:.1f}%)")

print("\n── Top-20 individual features ──")
for _, row in top50.head(20).iterrows():
    print(f"  ch={row['channel']:<20s}  bars_ago={int(row['bars_ago']):3d}  imp={row['importance_avg']:.6f}")

print(f"\nAll outputs saved to: {OUT_DIR}")

