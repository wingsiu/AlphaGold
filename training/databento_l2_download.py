#!/usr/bin/env python3
"""Download Level 2 (MBP-10) order book data from Databento.

Supported schemas:
  mbp-10   : Top-10 levels of market-by-price (L2 order book snapshots + trades)
  trades   : Trade-only feed (L1 trades, good for volume profile)
  ohlcv-1m : 1-minute OHLCV bars with volume (fast / cheapest option)

Default target: CME Gold futures (GC.c.0) via GLBX.MDP3

Usage examples:
  # Download MBP-10 (full L2) for one day:
  python3 training/databento_l2_download.py \
    --dataset GLBX.MDP3 \
    --symbol GC.c.0 \
    --schema mbp-10 \
    --start 2026-03-20 \
    --end   2026-03-21 \
    --out   training/l2_gc_20260320.dbn.zst

  # Download trades only (cheaper, still has volume-at-price):
  python3 training/databento_l2_download.py \
    --dataset GLBX.MDP3 \
    --symbol GC.c.0 \
    --schema trades \
    --start 2026-03-17 \
    --end   2026-03-21 \
    --out   training/trades_gc_week.dbn.zst

  # Convert saved .dbn.zst to CSV for inspection:
  python3 training/databento_l2_download.py \
    --convert training/l2_gc_20260320.dbn.zst \
    --out-csv  training/l2_gc_20260320.csv

Notes:
  - Set DATABENTO_API_KEY in .env or export it as an environment variable.
  - Databento charges by number of records; MBP-10 is large.
    Use --cost-check to get an estimated cost before downloading.
  - For gold futures the continuous front contract is GC.c.0
    or specific expiry e.g. GCM6 (June 2026).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env")


def _get_client():
    try:
        import databento as db  # type: ignore
    except ModuleNotFoundError:
        print("ERROR: databento package not installed.")
        print("  Run:  pip install 'databento>=0.44.0,<1.0.0'")
        sys.exit(1)

    api_key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        print("ERROR: DATABENTO_API_KEY not set.")
        print("  Edit .env and add your key:  DATABENTO_API_KEY=db-xxxxxxxxxxxx")
        sys.exit(1)

    return db.Historical(api_key)


def cmd_cost_check(args: argparse.Namespace) -> None:
    client = _get_client()
    cost = client.metadata.get_cost(
        dataset=args.dataset,
        symbols=[args.symbol],
        schema=args.schema,
        start=args.start,
        end=args.end,
        stype_in="continuous" if ".c." in args.symbol else "raw_symbol",
    )
    print(f"Estimated cost: ${cost:.4f} USD")
    print(f"  dataset={args.dataset}  symbol={args.symbol}")
    print(f"  schema={args.schema}  start={args.start}  end={args.end}")


def cmd_download(args: argparse.Namespace) -> None:
    client = _get_client()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stype = "continuous" if ".c." in args.symbol else "raw_symbol"

    print(f"Downloading {args.schema} data...")
    print(f"  dataset : {args.dataset}")
    print(f"  symbol  : {args.symbol}  (stype={stype})")
    print(f"  range   : {args.start} -> {args.end}")
    print(f"  output  : {out_path}")

    client.timeseries.get_range(
        dataset=args.dataset,
        symbols=[args.symbol],
        schema=args.schema,
        start=args.start,
        end=args.end,
        stype_in=stype,
        path=str(out_path),
    )

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Saved: {out_path}  ({size_mb:.2f} MB)")


def cmd_convert(args: argparse.Namespace) -> None:
    try:
        import databento as db  # type: ignore
    except ModuleNotFoundError:
        print("ERROR: databento package not installed.")
        sys.exit(1)

    in_path = Path(args.convert)
    if not in_path.exists():
        print(f"ERROR: File not found: {in_path}")
        sys.exit(1)

    out_csv = Path(args.out_csv) if args.out_csv else in_path.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting {in_path} -> {out_csv} ...")
    dbn = db.DBNStore.from_file(str(in_path))
    df = dbn.to_df()

    # Normalise column names for downstream scripts.
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.to_csv(str(out_csv), index=True)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved CSV: {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download Databento Level 2 order book data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common download args
    p.add_argument("--dataset", default="GLBX.MDP3",
                   help="Databento dataset (default: GLBX.MDP3 = CME Globex)")
    p.add_argument("--symbol", default="GC.c.0",
                   help="Symbol (default: GC.c.0 = gold front-month continuous)")
    p.add_argument("--schema", default="mbp-10",
                   choices=["mbp-10", "mbp-1", "trades", "ohlcv-1m", "ohlcv-1s", "mbo"],
                   help="Data schema (default: mbp-10)")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default=None, help="End date YYYY-MM-DD (exclusive)")
    p.add_argument("--out",   default=None, help="Output .dbn.zst file path")

    # Cost check (no download)
    p.add_argument("--cost-check", action="store_true",
                   help="Estimate cost without downloading")

    # Conversion only
    p.add_argument("--convert", default=None,
                   help="Convert existing .dbn.zst file to CSV (skip download)")
    p.add_argument("--out-csv", default=None,
                   help="Output CSV path when using --convert")

    return p


def main() -> int:
    args = build_parser().parse_args()

    # Convert-only mode
    if args.convert:
        cmd_convert(args)
        return 0

    # All other modes need start/end
    if not args.start or not args.end:
        print("ERROR: --start and --end are required for download/cost-check.")
        build_parser().print_help()
        return 1

    if args.cost_check:
        cmd_cost_check(args)
        return 0

    if not args.out:
        safe_sym = args.symbol.replace(".", "_")
        args.out = f"training/l2_{safe_sym}_{args.start}_{args.end}.dbn.zst"

    cmd_download(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

