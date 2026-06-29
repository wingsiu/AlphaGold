#!/usr/bin/env python3
"""v14 Energetic Model — Targeted Parameter Relaxation Test
================================================================
Each config: patch config/v14_config.py, run backtest_patterns_v14.py,
read CSV, report energetic-only stats.
"""
import os, sys, subprocess, shutil
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = PROJECT_ROOT / "runtime" / "v14_pattern_backtest_trades.csv"
CONFIG_PY = PROJECT_ROOT / "config" / "v14_config.py"
CONFIG_BAK = PROJECT_ROOT / "config" / "v14_config.py.bak"

# Preserve original
shutil.copy(CONFIG_PY, CONFIG_BAK)

def patch_config(bm, mv, s1, s2):
    """Inline patch FILTER_CONFIG and ENERGETIC_EXECUTION_CONFIG."""
    text = CONFIG_PY.read_text()
    text = text.replace('"min_bar_move": 3.0', f'"min_bar_move": {bm}')
    text = text.replace('"min_bar_move": 4.0', f'"min_bar_move": {bm}')
    text = text.replace('"min_bar_move": 2.0', f'"min_bar_move": {bm}')
    # More robust: find and replace by pattern
    import re
    text = re.sub(r'"min_bar_move":\s*[\d.]+', f'"min_bar_move": {bm}', text)
    text = re.sub(r'"min_volume":\s*\d+', f'"min_volume": {mv}', text)
    # ENERGETIC_EXECUTION_CONFIG s1/s2
    text = re.sub(
        r'("s1_threshold":\s*)[\d.]+(.*#\s*ENERGETIC_EXECUTION_CONFIG)',
        rf'\g<1>{s1}\g<2>', text
    )
    text = re.sub(
        r'(ENERGETIC_EXECUTION_CONFIG.*?"s1_threshold":\s*)[\d.]+',
        rf'\g<1>{s1}', text, flags=re.DOTALL
    )
    text = re.sub(
        r'(ENERGETIC_EXECUTION_CONFIG.*?"s2_threshold":\s*)[\d.]+',
        rf'\g<1>{s2}', text, flags=re.DOTALL
    )
    CONFIG_PY.write_text(text)

def run_and_get_stats():
    """Run hybrid backtest, return (n_trades, pnl, wr) for energetic only."""
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "V14_HYBRID": "1", "V14_FVG_MIN_GAP": "0"}
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "oil" / "backtest" / "pattern_backtest.py")],
        cwd=str(PROJECT_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    if TRADES_CSV.exists():
        tdf = pd.read_csv(TRADES_CSV)
        if "source" in tdf.columns:
            en = tdf[tdf["source"] == "energetic"]
        elif "pattern" in tdf.columns:
            en = tdf[tdf["pattern"] == "energetic"]
        else:
            en = pd.DataFrame()
        if len(en) > 0:
            n = len(en)
            pnl = float(en["pnl"].sum())
            wr = float((en["pnl"] > 0).mean() * 100)
            return n, pnl, wr
    return 0, 0.0, 0.0

print("="*72)
print("  v14 Energetic Parameter Relaxation Test")
print("="*72)

# 1) Production baseline
print("\n[1/5] Production baseline (bm=3.0, mv=200, s1=0.50, s2=0.55)...")
shutil.copy(CONFIG_BAK, CONFIG_PY)
t, pnl, wr = run_and_get_stats()
prod_t, prod_pnl, prod_wr = t, pnl, wr
print(f"  Energetic: {t}t, PnL={pnl:+.1f}, WR={wr:.1f}%")

# 2) Relax bar_move
print("\n[2/5] Relax min_bar_move to 1.5...")
patch_config(1.5, 200, 0.50, 0.55)
t, pnl, wr = run_and_get_stats()
print(f"  Energetic: {t}t, PnL={pnl:+.1f}, WR={wr:.1f}%  (vs prod: {pnl-prod_pnl:+.0f}pts)")

# 3) Relax volume
print("\n[3/5] Relax min_volume to 100...")
patch_config(3.0, 100, 0.50, 0.55)
t, pnl, wr = run_and_get_stats()
print(f"  Energetic: {t}t, PnL={pnl:+.1f}, WR={wr:.1f}%  (vs prod: {pnl-prod_pnl:+.0f}pts)")

# 4) Relax S1
print("\n[4/5] Relax s1_threshold to 0.45...")
patch_config(3.0, 200, 0.45, 0.55)
t, pnl, wr = run_and_get_stats()
print(f"  Energetic: {t}t, PnL={pnl:+.1f}, WR={wr:.1f}%  (vs prod: {pnl-prod_pnl:+.0f}pts)")

# 5) Relax all
print("\n[5/5] Relax all: bm=1.5, mv=100, s1=0.45, s2=0.52...")
patch_config(1.5, 100, 0.45, 0.52)
t, pnl, wr = run_and_get_stats()
print(f"  Energetic: {t}t, PnL={pnl:+.1f}, WR={wr:.1f}%  (vs prod: {pnl-prod_pnl:+.0f}pts)")

# Restore original
shutil.copy(CONFIG_BAK, CONFIG_PY)
CONFIG_BAK.unlink()

print(f"\n{'='*72}")
print(f"  Summary")
print(f"{'='*72}")
print(f"  Production: {prod_t}t, +{prod_pnl:.0f}pts, WR={prod_wr:.1f}%")
print(f"\nDONE.")
