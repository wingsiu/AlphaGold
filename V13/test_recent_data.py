import sys
import pandas as pd
from datetime import datetime, timezone
import logging
from pathlib import Path
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("/Users/alpha/Desktop/python/AlphaGold")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading_bot_v13 import AlphaGoldV13Bot, prepare_v13_features, extract_image_payload, _session_info, HK_TZ, LONDON_TZ, NY_TZ
from config.v13_config import FILTER_CONFIG
import numpy as np

def run_test():
    try:
        bot = AlphaGoldV13Bot()
        df = bot.cached_df.copy()

        if len(df) < bot.img_win + 10:
            print("Not enough data to run simulation.")
            return

        print(f"Loaded {len(df)} bars. Preparing features...")
        df = prepare_v13_features(df)

        # Process last 2000 bars
        start_idx = max(bot.img_win, len(df) - 2000)

        print(f"Evaluating signals for the last {len(df) - start_idx} minutes...")
        signals = 0
        s1_passed = 0
        filtered_bars = 0
        for i in range(start_idx, len(df)):
            idx = i

            # Apply same bar-quality filter as training/backtest (skip quiet bars)
            bar_move = abs(df['close'].iloc[idx] - df['open'].iloc[idx])
            bar_vol  = df['volume'].iloc[idx]
            if bar_move <= FILTER_CONFIG["min_bar_move"] or bar_vol <= FILTER_CONFIG["min_volume"]:
                filtered_bars += 1
                continue

            img_vec = extract_image_payload(df, idx, bot.img_win)
            ts = df.index[idx]

            asia_f, asia_p = _session_info(ts, HK_TZ, 8, 0, 16, 0)
            lon_f, lon_p = _session_info(ts, LONDON_TZ, 8, 0, 16, 30)
            ny_f, ny_p = _session_info(ts, NY_TZ, 9, 30, 16, 0)
            extra = [
                df["Dchange_utc2_rel"].iloc[idx],
                df["Dupper_wick_utc2_rel"].iloc[idx],
                df["Dlower_wick_utc2_rel"].iloc[idx],
                asia_f, asia_p, lon_f, lon_p, ny_f, ny_p
            ]

            img_s1_prob = bot.img_s1_model.predict_proba(np.concatenate([img_vec, extra]).reshape(1, -1))[0][1]
            df.loc[ts, "image_s1_prob"] = img_s1_prob

            s1_p = bot.s1_model.predict_proba(df.loc[[ts], bot.v13_s1_cols])[0][1]

            side = 0
            s2_p = None
            if s1_p >= 0.50:
                s1_passed += 1
                s2_p = bot.s2_model.predict_proba(df.loc[[ts], bot.v13_s2_cols])[0][1]
                if s2_p >= 0.55:
                    side = 1
                elif s2_p <= 0.45:
                    side = -1

            if side != 0:
                signals += 1
                print(f"🚀 SIGNAL HIT on {ts}: side={side}, S1={s1_p:.4f}, S2={s2_p:.4f}")

        print("\n--- TEST SUMMARY ---")
        print(f"Total minutes evaluated: {len(df) - start_idx}")
        print(f"Bars filtered (quiet/low-vol): {filtered_bars}")
        print(f"Energetic bars evaluated: {len(df) - start_idx - filtered_bars}")
        print(f"Amount of times S1 threshold (0.50) was passed: {s1_passed}")
        print(f"Total valid Entry Signals triggered: {signals}")

    except Exception as e:
        print("Error during test:", e)

if __name__ == "__main__":
    run_test()

