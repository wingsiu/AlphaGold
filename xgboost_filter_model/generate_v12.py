import sys
from pathlib import Path
# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
def generate():
    v10_path = PROJECT_ROOT / "xgboost_filter_model" / "train_filter_v10.py"
    v12_path = PROJECT_ROOT / "xgboost_filter_model" / "train_filter_v12_image.py"
    with open(v10_path, "r") as f:
        v10_lines = f.readlines()
    v10_lines.insert(10, "from zoneinfo import ZoneInfo\n")
    img_func = r"""
def add_image_model_predictions(df: pd.DataFrame) -> pd.DataFrame:
    print("Adding image model prediction features...")
    model_path = PROJECT_ROOT / "training" / "image_trend_model.joblib"
    if not model_path.exists():
        print(f"Warning: Image model not found at {model_path}. Skipping.")
        return df
    bundle = joblib.load(model_path)
    s1_model = bundle["stage1"]
    cfg = bundle["config"]
    window_1m = cfg.get("window", 150)
    df = df.copy()
    def _session_info(ts, timezone, start_h, start_m, end_h, end_m):
        local_ts = ts.tz_convert(timezone)
        minute_of_day = local_ts.hour * 60 + local_ts.minute
        s = start_h * 60 + start_m
        e = end_h * 60 + end_m
        if s <= minute_of_day < e:
            return 1.0, (minute_of_day - s) / (e - s)
        return 0.0, 0.0
    HK_TZ = ZoneInfo("Asia/Hong_Kong")
    LONDON_TZ = ZoneInfo("Europe/London")
    NY_TZ = ZoneInfo("America/New_York")
    day_start_offset = pd.Timedelta(hours=2)
    df["day_utc2"] = (df.index + day_start_offset).floor("D")
    df["day_open"] = df.groupby("day_utc2")["open"].transform("first")
    df["day_high_rolling"] = df.groupby("day_utc2")["high"].cummax()
    df["day_low_rolling"] = df.groupby("day_utc2")["low"].cummin()
    df["Dchange_utc2_rel"] = (df["close"] - df["day_open"]) / df["day_open"]
    df["Dupper_wick_utc2_rel"] = (df["day_high_rolling"] - df[["day_open", "close"]].max(axis=1)) / df["day_open"]
    df["Dlower_wick_utc2_rel"] = (df[["day_open", "close"]].min(axis=1) - df["day_low_rolling"]) / df["day_open"]
    df["bar_move"] = (df["close"] - df["open"]).abs()
    mask = (df["bar_move"] > 3) & (df["volume"] > 250)
    indices = np.where(mask)[0]
    s1_probs = np.full(len(df), np.nan)
    print(f"Calculating image model predictions for {len(indices)} samples...")
    for idx, i in enumerate(indices):
        if i < window_1m - 1:
            continue
        if idx % 2000 == 0:
            print(f"  Progress: {idx}/{len(indices)}")
        w = df.iloc[i - window_1m + 1 : i + 1]
        c0 = float(w["close"].iloc[0]) or 1.0
        open_rel  = w["open"].to_numpy()  / c0 - 1.0
        high_rel  = w["high"].to_numpy()  / c0 - 1.0
        low_rel   = w["low"].to_numpy()   / c0 - 1.0
        close_rel = w["close"].to_numpy() / c0 - 1.0
        body_rel  = (w["close"].to_numpy() - w["open"].to_numpy()) / c0
        range_rel = (w["high"].to_numpy()  - w["low"].to_numpy())  / c0
        vol = w["volume"].to_numpy(dtype=float)
        vol_mean, vol_std = np.mean(vol), np.std(vol)
        vol_z = np.zeros_like(vol) if vol_std < 1e-9 else (vol - vol_mean) / vol_std
        v0 = float(vol[0])
        vol_rel = np.zeros_like(vol) if abs(v0) < 1e-9 else vol / v0 - 1.0
        vd = np.diff(vol, prepend=vol[0])
        vd_std = np.std(vd)
        vol_diff_norm = np.zeros_like(vd) if vd_std < 1e-9 else vd / vd_std
        img = np.stack([open_rel, high_rel, low_rel, close_rel, body_rel, range_rel, vol_z, vol_rel, vol_diff_norm], axis=0).flatten()
        ts = df.index[i]
        is_asia, asia_prog = _session_info(ts, HK_TZ, 8, 0, 16, 0)
        is_london, lon_prog = _session_info(ts, LONDON_TZ, 8, 0, 16, 30)
        is_ny, ny_prog = _session_info(ts, NY_TZ, 9, 30, 16, 0)
        extra = [
            df["Dchange_utc2_rel"].iloc[i],
            df["Dupper_wick_utc2_rel"].iloc[i],
            df["Dlower_wick_utc2_rel"].iloc[i],
            is_asia, asia_prog,
            is_london, lon_prog,
            is_ny, ny_prog
        ]
        full_input = np.concatenate([img, extra]).reshape(1, -1)
        if hasattr(s1_model, "predict_proba"):
             s1_probs[i] = s1_model.predict_proba(full_input)[0][1]
    df["image_s1_prob"] = s1_probs
    return df
"""
    for i, line in enumerate(v10_lines):
        if line.startswith("def add_liquidity_indicators"):
            v10_lines.insert(i, img_func + "\n")
            break
    full_text = "".join(v10_lines)
    full_text = full_text.replace("def train_v10_filter():", "def train_v12_filter():")
    full_text = full_text.replace("filter_model_v10.joblib", "filter_model_v12_image.joblib")
    full_text = full_text.replace("df = load_price_data(start_date=FULL_START, end_date=\"2026-05-07\")", 
                                "df = load_price_data(start_date=FULL_START, end_date=\"2026-05-10\")")
    full_text = full_text.replace("df = add_liquidity_indicators(df)", 
                                "df = add_liquidity_indicators(df)\n    df = add_image_model_predictions(df)")
    full_text = full_text.replace("train_v10_filter()", "train_v12_filter()")
    old_ex = "exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',"
    new_ex = "exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',\n" \
             "               'day_high_rolling', 'day_low_rolling', 'day_open',\n" \
             "               'Dchange_utc2_rel', 'Dupper_wick_utc2_rel', 'Dlower_wick_utc2_rel',"
    full_text = full_text.replace(old_ex, new_ex)
    with open(v12_path, "w") as f:
        f.write(full_text)
    print(f"Generated {v12_path}")
if __name__ == "__main__":
    generate()
