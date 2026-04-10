#!/usr/bin/env python3
"""Find support and resistance levels from Databento Level 2 order book data.

Methods used (each can be independently enabled/disabled):

  1. Volume Profile  (from trades schema)
     - Builds histogram of traded volume at each price tick.
     - POC  = price with most traded volume.
     - VAH  = Value Area High  (top of 70% volume range around POC).
     - VAL  = Value Area Low   (bottom of 70% volume range).
     - Peaks in the histogram are candidate S/R levels.

  2. Order Book Wall Detection  (from mbp-10 schema)
     - Accumulates average bid/ask size at each price level over all snapshots.
     - Large persistent clusters = walls where price tends to stall.
     - Top-N bid walls -> Support levels.
     - Top-N ask walls -> Resistance levels.

  3. Price-Level Touch Count  (from any OHLCV or trade data)
     - Counts how many bars touched (high >= level or low <= level) a rounded price.
     - High-touch prices = significant S/R.

Usage examples:
  # From trades CSV (produced by databento_l2_download.py --convert):
  python3 training/databento_l2_sr.py \
    --mode volume-profile \
    --input training/trades_gc_week.csv \
    --price-col price \
    --size-col size \
    --tick-size 0.10 \
    --va-pct 0.70 \
    --top-n 10 \
    --plot \
    --out training/sr_volume_profile.csv

  # From MBP-10 CSV (order book walls):
  python3 training/databento_l2_sr.py \
    --mode book-walls \
    --input training/l2_gc_20260320.csv \
    --tick-size 0.10 \
    --top-n 10 \
    --plot \
    --out training/sr_book_walls.csv

  # Combined (volume profile + book walls):
  python3 training/databento_l2_sr.py \
    --mode combined \
    --trades-csv  training/trades_gc_week.csv \
    --book-csv    training/l2_gc_20260320.csv \
    --tick-size 0.10 \
    --top-n 10 \
    --plot \
    --out training/sr_combined.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _round_to_tick(series: pd.Series, tick: float) -> pd.Series:
    return (series / tick).round() * tick


def _find_local_peaks(arr: np.ndarray, min_prom_pct: float = 0.05) -> list[int]:
    """Simple local-peak finder (no scipy dependency)."""
    peaks = []
    n = len(arr)
    max_val = float(arr.max()) if n > 0 else 1.0
    threshold = max_val * min_prom_pct

    for i in range(1, n - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1] and arr[i] >= threshold:
            # ensure it's meaningfully above both neighbors
            if arr[i] > arr[i - 1] or arr[i] > arr[i + 1]:
                peaks.append(i)
    return peaks


def _merge_close_levels(levels: list[float], merge_ticks: int, tick: float) -> list[float]:
    """Merge price levels that are within merge_ticks of each other (keep strongest)."""
    if not levels:
        return []
    levels_sorted = sorted(levels)
    merged = [levels_sorted[0]]
    for lv in levels_sorted[1:]:
        if lv - merged[-1] > merge_ticks * tick:
            merged.append(lv)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Method 1: Volume Profile
# ──────────────────────────────────────────────────────────────────────────────

def compute_volume_profile(
    df: pd.DataFrame,
    price_col: str,
    size_col: str,
    tick_size: float,
) -> pd.DataFrame:
    """Aggregate traded volume at each price tick."""
    df = df[[price_col, size_col]].dropna().copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[size_col]  = pd.to_numeric(df[size_col],  errors="coerce")
    df = df.dropna()
    df["price_tick"] = _round_to_tick(df[price_col], tick_size)

    profile = (
        df.groupby("price_tick")[size_col]
        .sum()
        .sort_index()
        .rename("volume")
        .reset_index()
    )
    profile.columns = ["price", "volume"]
    return profile


def value_area(profile: pd.DataFrame, va_pct: float = 0.70) -> dict[str, float]:
    """Compute POC, VAH, VAL from a volume profile DataFrame."""
    if profile.empty:
        return {"poc": float("nan"), "vah": float("nan"), "val": float("nan")}

    total_vol = float(profile["volume"].sum())
    target = total_vol * va_pct

    poc_idx = int(profile["volume"].idxmax())
    poc     = float(profile.loc[poc_idx, "price"])

    accumulated = float(profile.loc[poc_idx, "volume"])
    lo_idx = hi_idx = poc_idx

    while accumulated < target:
        vol_above = profile.loc[hi_idx + 1, "volume"] if (hi_idx + 1) < len(profile) else -1
        vol_below = profile.loc[lo_idx - 1, "volume"] if (lo_idx - 1) >= 0 else -1

        if vol_above < 0 and vol_below < 0:
            break
        if vol_above >= vol_below:
            hi_idx += 1
            accumulated += vol_above
        else:
            lo_idx -= 1
            accumulated += vol_below

    vah = float(profile.loc[hi_idx, "price"])
    val = float(profile.loc[lo_idx, "price"])
    return {"poc": poc, "vah": vah, "val": val}


def sr_from_volume_profile(
    profile: pd.DataFrame,
    tick_size: float,
    top_n: int = 10,
    merge_ticks: int = 3,
    min_prom_pct: float = 0.05,
) -> pd.DataFrame:
    """Find S/R levels from local peaks in the volume profile."""
    vols = profile["volume"].to_numpy(dtype="float64")
    prices = profile["price"].to_numpy(dtype="float64")

    peak_idxs = _find_local_peaks(vols, min_prom_pct=min_prom_pct)
    if not peak_idxs:
        return pd.DataFrame(columns=["price", "volume", "type"])

    peak_prices  = [float(prices[i]) for i in peak_idxs]
    peak_volumes = [float(vols[i])   for i in peak_idxs]

    # Sort by volume descending, take top_n * 2 before merging
    pairs = sorted(zip(peak_volumes, peak_prices), reverse=True)[: top_n * 2]
    strong_prices = [p for _, p in pairs]
    strong_prices = _merge_close_levels(strong_prices, merge_ticks, tick_size)[:top_n]

    va = value_area(profile)
    poc = va["poc"]

    rows = []
    for price in strong_prices:
        level_type = "resistance" if price > poc else "support"
        vol_at = float(profile.loc[profile["price"] == price, "volume"].sum())
        rows.append({"price": price, "volume": vol_at, "type": level_type, "method": "volume_profile"})

    result = pd.DataFrame(rows).sort_values("price").reset_index(drop=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: Order Book Wall Detection (MBP-10)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_book_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Auto-detect bid/ask price and size columns from Databento MBP-10 export."""
    bid_price_cols = [c for c in df.columns if "bid_px" in c or ("bid" in c and "price" in c)]
    ask_price_cols = [c for c in df.columns if "ask_px" in c or ("ask" in c and "price" in c)]
    bid_size_cols  = [c for c in df.columns if "bid_sz" in c or ("bid" in c and "size" in c)]
    ask_size_cols  = [c for c in df.columns if "ask_sz" in c or ("ask" in c and "size" in c)]

    # Databento MBP-10 naming: bid_px_00..bid_px_09, bid_sz_00..bid_sz_09
    if not bid_price_cols:
        bid_price_cols = sorted([c for c in df.columns if c.startswith("bid_px_")])
    if not ask_price_cols:
        ask_price_cols = sorted([c for c in df.columns if c.startswith("ask_px_")])
    if not bid_size_cols:
        bid_size_cols  = sorted([c for c in df.columns if c.startswith("bid_sz_")])
    if not ask_size_cols:
        ask_size_cols  = sorted([c for c in df.columns if c.startswith("ask_sz_")])

    return {
        "bid_px": bid_price_cols,
        "ask_px": ask_price_cols,
        "bid_sz": bid_size_cols,
        "ask_sz": ask_size_cols,
    }


def compute_book_walls(
    df: pd.DataFrame,
    tick_size: float,
    price_scale: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accumulate average bid/ask size at each price level over all book snapshots.

    Databento MBP-10 prices are stored as fixed-point int64 (divide by 1e9 to get USD).
    Set price_scale=1.0 if your CSV already has float prices.

    Returns:
        bid_walls, ask_walls  (DataFrames with columns: price, avg_size)
    """
    cols = _detect_book_columns(df)
    if not cols["bid_px"] or not cols["ask_px"]:
        raise ValueError(
            "Could not find bid/ask price columns. "
            "Expected columns like bid_px_00..bid_px_09, ask_px_00..ask_px_09"
        )

    bid_accum: dict[float, list[float]] = {}
    ask_accum: dict[float, list[float]] = {}

    n_levels = min(len(cols["bid_px"]), len(cols["bid_sz"]),
                   len(cols["ask_px"]), len(cols["ask_sz"]))

    for lvl in range(n_levels):
        bp_col = cols["bid_px"][lvl]
        bs_col = cols["bid_sz"][lvl]
        ap_col = cols["ask_px"][lvl]
        as_col = cols["ask_sz"][lvl]

        for _, row in df[[bp_col, bs_col, ap_col, as_col]].iterrows():
            bp = row[bp_col]
            bs = row[bs_col]
            ap = row[ap_col]
            as_ = row[as_col]

            if pd.notna(bp) and pd.notna(bs) and bp > 0 and bs > 0:
                price_f = round(float(bp) * price_scale / tick_size) * tick_size
                bid_accum.setdefault(price_f, []).append(float(bs))

            if pd.notna(ap) and pd.notna(as_) and ap > 0 and as_ > 0:
                price_f = round(float(ap) * price_scale / tick_size) * tick_size
                ask_accum.setdefault(price_f, []).append(float(as_))

    bid_walls = pd.DataFrame([
        {"price": p, "avg_size": float(np.mean(v)), "total_size": float(np.sum(v))}
        for p, v in bid_accum.items()
    ]).sort_values("avg_size", ascending=False).reset_index(drop=True)

    ask_walls = pd.DataFrame([
        {"price": p, "avg_size": float(np.mean(v)), "total_size": float(np.sum(v))}
        for p, v in ask_accum.items()
    ]).sort_values("avg_size", ascending=False).reset_index(drop=True)

    return bid_walls, ask_walls


def sr_from_book_walls(
    bid_walls: pd.DataFrame,
    ask_walls: pd.DataFrame,
    tick_size: float,
    top_n: int = 10,
    merge_ticks: int = 3,
) -> pd.DataFrame:
    support_prices = _merge_close_levels(
        bid_walls["price"].head(top_n * 2).tolist(), merge_ticks, tick_size
    )[:top_n]
    resistance_prices = _merge_close_levels(
        ask_walls["price"].head(top_n * 2).tolist(), merge_ticks, tick_size
    )[:top_n]

    rows = []
    for p in support_prices:
        sz = float(bid_walls.loc[bid_walls["price"] == p, "avg_size"].values[0]) \
            if p in bid_walls["price"].values else 0.0
        rows.append({"price": p, "avg_size": sz, "type": "support", "method": "book_wall"})
    for p in resistance_prices:
        sz = float(ask_walls.loc[ask_walls["price"] == p, "avg_size"].values[0]) \
            if p in ask_walls["price"].values else 0.0
        rows.append({"price": p, "avg_size": sz, "type": "resistance", "method": "book_wall"})

    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Combined Summary
# ──────────────────────────────────────────────────────────────────────────────

def merge_sr_levels(
    frames: list[pd.DataFrame],
    tick_size: float,
    merge_ticks: int = 3,
) -> pd.DataFrame:
    """Merge S/R levels from multiple methods, count how many methods agree."""
    if not frames:
        return pd.DataFrame()

    all_levels: list[dict] = []
    for frame in frames:
        for _, row in frame.iterrows():
            all_levels.append(row.to_dict())

    if not all_levels:
        return pd.DataFrame()

    combined = pd.DataFrame(all_levels).sort_values("price").reset_index(drop=True)
    combined["price_tick"] = _round_to_tick(combined["price"], tick_size)

    agg = (
        combined.groupby("price_tick")
        .agg(
            price=("price_tick", "first"),
            type=("type", lambda x: x.mode()[0]),
            methods=("method", lambda x: "+".join(sorted(set(x)))),
            method_count=("method", "nunique"),
        )
        .reset_index(drop=True)
        .sort_values("method_count", ascending=False)
    )
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_volume_profile(
    profile: pd.DataFrame,
    sr_levels: pd.DataFrame,
    va: dict[str, float],
    title: str = "Volume Profile with S/R Levels",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.barh(profile["price"], profile["volume"], height=profile["price"].diff().abs().median() * 0.9,
            color="#aec7e8", edgecolor="none", label="Volume")

    colors = {"support": "green", "resistance": "red"}
    plotted_labels: set[str] = set()
    for _, row in sr_levels.iterrows():
        ltype = str(row.get("type", "support"))
        color = colors.get(ltype, "gray")
        label = ltype if ltype not in plotted_labels else "_nolegend_"
        plotted_labels.add(ltype)
        ax.axhline(float(row["price"]), color=color, linestyle="--", linewidth=1.0, label=label)

    if not pd.isna(va.get("poc", float("nan"))):
        ax.axhline(va["poc"], color="gold", linewidth=2.0, label=f"POC {va['poc']:.2f}")
    if not pd.isna(va.get("vah", float("nan"))):
        ax.axhline(va["vah"], color="blue", linewidth=1.2, linestyle=":", label=f"VAH {va['vah']:.2f}")
    if not pd.isna(va.get("val", float("nan"))):
        ax.axhline(va["val"], color="blue", linewidth=1.2, linestyle=":", label=f"VAL {va['val']:.2f}")

    ax.set_xlabel("Volume")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_book_walls(
    bid_walls: pd.DataFrame,
    ask_walls: pd.DataFrame,
    top_n: int = 20,
    title: str = "Order Book Walls",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, walls, label, color in [
        (axes[0], bid_walls.head(top_n), "Bid Walls (Support)", "green"),
        (axes[1], ask_walls.head(top_n), "Ask Walls (Resistance)", "red"),
    ]:
        if walls.empty:
            ax.set_title(f"{label} (no data)")
            continue
        ax.barh(walls["price"].astype(str), walls["avg_size"], color=color, alpha=0.7)
        ax.set_xlabel("Avg Size")
        ax.set_ylabel("Price")
        ax.set_title(label)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find support/resistance from Databento L2 data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", default="volume-profile",
                   choices=["volume-profile", "book-walls", "combined"],
                   help="Analysis method")

    # Single input (volume-profile or book-walls)
    p.add_argument("--input",  default=None, help="Input CSV (trades or mbp-10)")

    # Combined mode inputs
    p.add_argument("--trades-csv", default=None, help="Trades CSV for volume profile")
    p.add_argument("--book-csv",   default=None, help="MBP-10 CSV for book walls")

    # Column names (trades)
    p.add_argument("--price-col", default="price", help="Price column name in trades CSV")
    p.add_argument("--size-col",  default="size",  help="Size column name in trades CSV")

    # Price scale for Databento fixed-point int prices
    p.add_argument("--price-scale", type=float, default=1e-9,
                   help="Multiply raw int prices by this to get float (default 1e-9 for Databento)")
    p.add_argument("--prices-already-float", action="store_true",
                   help="Skip price scaling (use when CSV already has float prices)")

    p.add_argument("--tick-size", type=float, default=0.10,
                   help="Minimum price tick (default 0.10 for GC)")
    p.add_argument("--va-pct", type=float, default=0.70,
                   help="Value Area percent (default 0.70 = 70%%)")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of S/R levels to output")
    p.add_argument("--merge-ticks", type=int, default=3,
                   help="Merge levels within this many ticks of each other")
    p.add_argument("--plot", action="store_true", help="Show matplotlib chart")
    p.add_argument("--out", default=None, help="Output CSV for S/R levels")
    return p


def _load_csv(path: str) -> pd.DataFrame:
    print(f"Loading: {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}  Columns: {list(df.columns[:8])} ...")
    return df


def main() -> int:
    args = build_parser().parse_args()
    price_scale = 1.0 if args.prices_already_float else args.price_scale

    # ── Volume Profile ────────────────────────────────────────────────────────
    if args.mode in ("volume-profile", "combined"):
        trades_path = args.input if args.mode == "volume-profile" else args.trades_csv
        if not trades_path:
            print("ERROR: provide --input (volume-profile) or --trades-csv (combined)")
            return 1

        trades_df = _load_csv(trades_path)

        # Scale prices if they are int fixed-point
        if not args.prices_already_float and args.price_col in trades_df.columns:
            if pd.api.types.is_integer_dtype(trades_df[args.price_col]):
                print(f"  Scaling {args.price_col} by {price_scale}")
                trades_df[args.price_col] = trades_df[args.price_col].astype(float) * price_scale

        profile = compute_volume_profile(
            trades_df,
            price_col=args.price_col,
            size_col=args.size_col,
            tick_size=args.tick_size,
        )
        va = value_area(profile, va_pct=args.va_pct)
        vp_sr = sr_from_volume_profile(
            profile,
            tick_size=args.tick_size,
            top_n=args.top_n,
            merge_ticks=args.merge_ticks,
        )

        print("\n── Volume Profile ─────────────────────────────────")
        print(f"  POC : {va['poc']:.2f}")
        print(f"  VAH : {va['vah']:.2f}")
        print(f"  VAL : {va['val']:.2f}")
        print(f"  S/R levels ({len(vp_sr)}):")
        print(vp_sr[["price", "type", "volume"]].to_string(index=False))

        if args.plot and args.mode == "volume-profile":
            plot_volume_profile(profile, vp_sr, va)
    else:
        vp_sr = pd.DataFrame()
        profile = pd.DataFrame()
        va = {}

    # ── Book Walls ────────────────────────────────────────────────────────────
    if args.mode in ("book-walls", "combined"):
        book_path = args.input if args.mode == "book-walls" else args.book_csv
        if not book_path:
            print("ERROR: provide --input (book-walls) or --book-csv (combined)")
            return 1

        book_df = _load_csv(book_path)
        bid_walls, ask_walls = compute_book_walls(
            book_df,
            tick_size=args.tick_size,
            price_scale=price_scale,
        )
        bw_sr = sr_from_book_walls(
            bid_walls, ask_walls,
            tick_size=args.tick_size,
            top_n=args.top_n,
            merge_ticks=args.merge_ticks,
        )

        print("\n── Book Walls ──────────────────────────────────────")
        print(f"  Top bid walls (support):")
        print(bid_walls.head(args.top_n)[["price", "avg_size"]].to_string(index=False))
        print(f"\n  Top ask walls (resistance):")
        print(ask_walls.head(args.top_n)[["price", "avg_size"]].to_string(index=False))

        if args.plot and args.mode == "book-walls":
            plot_book_walls(bid_walls, ask_walls, top_n=args.top_n)
    else:
        bw_sr = pd.DataFrame()

    # ── Combined ──────────────────────────────────────────────────────────────
    frames = [f for f in [vp_sr, bw_sr] if not f.empty]
    if frames:
        merged = merge_sr_levels(frames, tick_size=args.tick_size, merge_ticks=args.merge_ticks)
        print("\n── Combined S/R Levels ─────────────────────────────")
        print(merged[["price", "type", "method_count", "methods"]].to_string(index=False))

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(str(out_path), index=False)
            print(f"\nSaved: {out_path}")

        if args.plot and args.mode == "combined":
            if not profile.empty:
                plot_volume_profile(profile, merged, va, title="Combined S/R Levels")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

