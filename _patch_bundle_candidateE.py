#!/usr/bin/env python3
"""Patch the model bundle joblib with Candidate E parameters."""
import joblib, shutil
from pathlib import Path

path = Path("runtime/bot_assets/backtest_model_best_base_weak_nostate.joblib")

# Backup first
backup = Path(str(path) + ".bak_pre_candidateE")
shutil.copy(path, backup)
print(f"Backup saved: {backup}")

b = joblib.load(path)

updates = {
    "stage1_min_prob":        0.55,
    "stage2_min_prob":        0.58,
    "stage2_min_prob_up":     0.65,
    "stage2_min_prob_down":   0.62,
    "max_hold_minutes":       60.0,
    "long_target_threshold":  0.008,
    "long_adverse_limit":     15.0,
}

for k, v in updates.items():
    b[k] = v
    if isinstance(b.get("config"), dict):
        b["config"][k] = v

joblib.dump(b, path)
print("Bundle patched. Verifying...")

b2 = joblib.load(path)
for k in updates:
    print(f"  {k}: {b2[k]}")

print("Done.")

