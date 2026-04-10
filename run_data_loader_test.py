import argparse
from datetime import datetime, timedelta, timezone

from data import DataLoader

TRADING_TZ = timezone(timedelta(hours=6))


def _format_epoch_ms(epoch_ms: int) -> tuple[str, str]:
    dt_utc = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    dt_trading = dt_utc.astimezone(TRADING_TZ)
    return dt_utc.isoformat(), dt_trading.isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DataLoader query and print sample results.")
    parser.add_argument("--table", default="gold_prices", help="MySQL table name")
    parser.add_argument("--start-date", required=True, help="Trading start date (UTC+6), format YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Trading end date (UTC+6), format YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=10, help="Max rows to fetch")
    args = parser.parse_args()

    loader = DataLoader()
    df = loader.load_data(
        table_name=args.table,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )

    print("--- Query Parameters ---")
    print(f"table={args.table}")
    print(f"trading_range_utc_plus_6={args.start_date} to {args.end_date}")
    print(f"limit={args.limit}")
    print("\n--- Results ---")
    print(f"rows={len(df)}")

    if df.empty:
        print("No rows returned.")
        return

    sample = df.head(min(5, len(df))).copy()
    sample["timestamp_utc"] = sample["timestamp"].apply(lambda x: _format_epoch_ms(int(x))[0])
    sample["timestamp_utc_plus_6"] = sample["timestamp"].apply(lambda x: _format_epoch_ms(int(x))[1])

    columns = [
        "timestamp",
        "timestamp_utc",
        "timestamp_utc_plus_6",
        "openPrice",
        "closePrice",
        "highPrice",
        "lowPrice",
        "lastTradedVolume",
    ]
    present_columns = [c for c in columns if c in sample.columns]

    print("\n--- Sample (first rows) ---")
    print(sample[present_columns].to_string(index=False))


if __name__ == "__main__":
    main()

