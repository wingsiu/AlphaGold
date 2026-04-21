#!/usr/bin/env python3
"""Patch the model bundle joblib with current best-base parameters.

History:
  Candidate E (2026-04-20): max_hold_minutes=60, long_adverse_limit=15
  Stop update  (2026-04-21): long_adverse_limit=12, short_adverse_limit=18, max_hold_minutes=50
    → h25+cap50 sweep p2 PnL=3114, WR=50.2%, PF=1.466, DD=-143
    NOTE: horizon stays 25 — bundle was trained on h25 labels.
"""
import joblib, shutil
from pathlib import Path

path = Path("runtime/bot_assets/backtest_model_best_base_weak_nostate.joblib")

# Backup first
backup = Path(str(path) + ".bak_pre_candidateE")
if not backup.exists():
    shutil.copy(path, backup)
    print(f"Backup saved: {backup}")
else:
    print(f"Backup already exists, skipping: {backup}")

b = joblib.load(path)

updates = {
    "stage1_min_prob":        0.55,
    "stage2_min_prob":        0.58,
    "stage2_min_prob_up":     0.65,
    "stage2_min_prob_down":   0.62,
    "max_hold_minutes":       50.0,   # h25+cap50 sweep (was 60 for Candidate E)
    "long_target_threshold":  0.008,
    "long_adverse_limit":     12.0,   # updated from 15 → 12
    "short_adverse_limit":    18.0,   # explicitly set
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

