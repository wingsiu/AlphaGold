import os
import subprocess
import re
import sys
from pathlib import Path

CONFIG_PATH = "config/v14_config.py"

def update_thresholds(s1, s2):
    with open(CONFIG_PATH, "r") as f:
        config = f.read()
        
    lines = config.split("\n")
    in_exec = False
    for i, line in enumerate(lines):
        if "EXECUTION_CONFIG = {" in line:
            in_exec = True
        elif "}" in line:
            in_exec = False
            
        if in_exec:
            if '"s1_threshold":' in line:
                lines[i] = re.sub(r'(:\s*)[0-9.]+', rf'\g<1>{s1}', line, count=1)
            elif '"s2_threshold":' in line:
                lines[i] = re.sub(r'(:\s*)[0-9.]+', rf'\g<1>{s2}', line, count=1)

    with open(CONFIG_PATH, "w") as f:
        f.write("\n".join(lines))

def run_sweep():
    s1_vals = [0.45, 0.50, 0.55]
    s2_vals = [0.50, 0.55, 0.60]
    
    combos = []
    for s1 in s1_vals:
        for s2 in s2_vals:
            combos.append((s1, s2))
                
    print(f"Starting threshold sweep of {len(combos)} combinations...")
    print(f"{'S1':>5} | {'S2':>5} | {'Trades':>6} | {'Win%':>6} | {'Net PnL':>8}")
    print("-" * 45)
    sys.stdout.flush()
    
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache_thresh"
    
    with open("sweep_thresholds_results_v14.csv", "w") as f:
        f.write("S1,S2,Trades,WinRate,NetPnL\n")
    
    for s1, s2 in combos:
        update_thresholds(s1, s2)
        
        print(f"Running S1={s1}, S2={s2}... ", end="")
        sys.stdout.flush()
        
        # Only run backtest, no need to retrain!
        result = subprocess.run(
            [sys.executable, "backtest_v14.py", "2026-01-01", "2026-05-21"],
            env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Error for S1={s1}, S2={s2}: {result.stderr}")
        
        trades = 0
        win_rate = 0.0
        pnl = 0.0
        
        for line in result.stdout.split("\n"):
            if "Trades       :" in line:
                m = re.search(r'Trades\s*:\s*(\d+)', line)
                if m: trades = int(m.group(1))
            elif "Win Rate     :" in line:
                m = re.search(r'Win Rate\s*:\s*([0-9.]+)', line)
                if m: win_rate = float(m.group(1))
            elif "Net PnL      :" in line:
                m = re.search(r'Net PnL\s*:\s*([0-9.-]+)', line)
                if m: pnl = float(m.group(1))
                
        print(f"\r{s1:5.2f} | {s2:5.2f} | {trades:6d} | {win_rate:5.1f}% | {pnl:8.1f}")
        sys.stdout.flush()
        
        with open("sweep_thresholds_results_v14.csv", "a") as f:
            f.write(f"{s1},{s2},{trades},{win_rate},{pnl}\n")

if __name__ == "__main__":
    run_sweep()
