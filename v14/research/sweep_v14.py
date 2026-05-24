import os
import subprocess
import re
import sys
from pathlib import Path

CONFIG_PATH = "config/v14_config.py"

def update_config(horizon, tp, sl):
    with open(CONFIG_PATH, "r") as f:
        config = f.read()
        
    # Update TARGET_CONFIG
    config = re.sub(r'("tp":\s*)[0-9.]+(,)', rf'\g<1>{float(tp)}\g<2>', config, count=1)
    config = re.sub(r'("sl":\s*)[0-9.]+(,)', rf'\g<1>{float(sl)}\g<2>', config, count=1)
    config = re.sub(r'("horizon":\s*)[0-9]+(,)', rf'\g<1>{int(horizon)}\g<2>', config, count=1)
    
    # Update EXECUTION_CONFIG
    # We use a slightly different regex to target the second occurrence (which has comments)
    # Actually, let's just do a global replace for the specific lines if they match the pattern
    # A safer way: split by lines and replace
    lines = config.split("\n")
    in_target = False
    in_exec = False
    for i, line in enumerate(lines):
        if "TARGET_CONFIG = {" in line:
            in_target = True
        elif "EXECUTION_CONFIG = {" in line:
            in_exec = True
        elif "}" in line:
            in_target = False
            in_exec = False
            
        if in_target or in_exec:
            if '"tp":' in line:
                lines[i] = re.sub(r'(:\s*)[0-9.]+', rf'\g<1>{float(tp)}', line, count=1)
            elif '"sl":' in line:
                lines[i] = re.sub(r'(:\s*)[0-9.]+', rf'\g<1>{float(sl)}', line, count=1)
            elif '"horizon":' in line:
                lines[i] = re.sub(r'(:\s*)[0-9]+', rf'\g<1>{int(horizon)}', line, count=1)

    with open(CONFIG_PATH, "w") as f:
        f.write("\n".join(lines))

def run_sweep():
    # Define the parameter grid
    horizons = [15, 30, 45]
    targets = [15, 30, 45]
    stops = [10, 15, 25]
    
    combos = []
    for h in horizons:
        for t in targets:
            for s in stops:
                # Skip illogical combinations (e.g. stop > target, or 1:1 ratios if we want 2:1)
                # Let's just run them all, or filter out SL >= TP
                if s >= t:
                    continue
                combos.append((h, t, s))
                
    print(f"Starting sweep of {len(combos)} combinations...")
    print(f"{'Horizon':>8} | {'TP':>5} | {'SL':>5} | {'Trades':>6} | {'Win%':>6} | {'Net PnL':>8}")
    print("-" * 60)
    sys.stdout.flush()
    
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
    
    with open("sweep_results_v14.csv", "w") as f:
        f.write("Horizon,TP,SL,Trades,WinRate,NetPnL\n")
    
    for h, t, s in combos:
        update_config(h, t, s)
        
        print(f"Running H={h}, TP={t}, SL={s}... ", end="")
        sys.stdout.flush()
        
        # 1. Train S1
        subprocess.run(
            [sys.executable, "xgboost_filter_model/train_filter_v14.py"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # 2. Train S2
        subprocess.run(
            [sys.executable, "xgboost_filter_model/train_stage2_v14_directional.py"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # 3. Backtest
        result = subprocess.run(
            [sys.executable, "backtest_v14.py", "2026-01-01", "2026-05-21"],
            env=env, capture_output=True, text=True
        )
        
        # Parse output
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
                
        # Clear the "Running..." text and print the result
        print(f"\r{h:8d} | {t:5.1f} | {s:5.1f} | {trades:6d} | {win_rate:5.1f}% | {pnl:8.1f}")
        sys.stdout.flush()
        
        with open("sweep_results_v14.csv", "a") as f:
            f.write(f"{h},{t},{s},{trades},{win_rate},{pnl}\n")

if __name__ == "__main__":
    run_sweep()
