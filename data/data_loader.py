import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load .env from project root so running from any cwd still works.
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Database Configuration from .env
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

_VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HKT = timezone(timedelta(hours=8))
DEFAULT_TRADING_TIMEZONE = HKT
DEFAULT_TRADING_START_HOUR = 6
TABLE_TRADING_BOUNDARIES = {
    "aud_prices": (ZoneInfo("Australia/Sydney"), 6, "Australia/Sydney 06:00"),
}
TABLE_WEEKDAY_ONLY = {"aud_prices"}


def _resolve_trading_boundary(table_name: str):
    """Resolve the timezone and local start hour for a table's trading day."""
    return TABLE_TRADING_BOUNDARIES.get(
        table_name,
        (DEFAULT_TRADING_TIMEZONE, DEFAULT_TRADING_START_HOUR, "HKT 06:00"),
    )


def _date_boundary_to_utc_ms(table_name: str, date_str: str, *, end_exclusive: bool = False) -> int:
    """Convert a table-specific trading date to UTC epoch milliseconds."""
    trading_date = datetime.strptime(date_str, "%Y-%m-%d")
    if end_exclusive:
        trading_date = trading_date + timedelta(days=1)

    trading_tz, trading_start_hour, _ = _resolve_trading_boundary(table_name)

    trading_boundary = datetime(
        year=trading_date.year,
        month=trading_date.month,
        day=trading_date.day,
        hour=trading_start_hour,
        minute=0,
        second=0,
        tzinfo=trading_tz,
    )
    return int(trading_boundary.astimezone(timezone.utc).timestamp() * 1000)


def _filter_to_trading_weekdays(table_name: str, data: pd.DataFrame) -> pd.DataFrame:
    """Keep only Monday-Friday trading dates for tables that trade on weekdays only."""
    if table_name not in TABLE_WEEKDAY_ONLY or data.empty:
        return data

    trading_tz, trading_start_hour, _ = _resolve_trading_boundary(table_name)
    local_time = pd.to_datetime(data["timestamp"], unit="ms", utc=True).dt.tz_convert(trading_tz)
    trading_date = (local_time.dt.tz_localize(None) - pd.Timedelta(hours=trading_start_hour)).dt.normalize()
    return data.loc[trading_date.dt.weekday < 5].reset_index(drop=True)


class DataLoader:
    def __init__(self):
        """
        Initialize the DataLoader. It uses the database configuration from the environment variables
        and establishes the SQLAlchemy engine.
        """
        missing = [key for key, value in DB_CONFIG.items() if not value]
        if missing:
            raise ValueError(f"Missing DB environment variables: {', '.join(missing)}")

        password = quote_plus(DB_CONFIG["password"])
        self.engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{password}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        )

    def load_data(
        self,
        table_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load data from the specified table in the MySQL database using SQLAlchemy.

        Args:
        - table_name (str): The name of the table to query data from.
        - start_date (Optional[str]): Optional start trading date. `aud_prices` uses
          Australia/Sydney 06:00; other tables default to HKT 06:00. Format: 'YYYY-MM-DD'.
        - end_date (Optional[str]): Optional end trading date using the same table-specific
          boundary. Format: 'YYYY-MM-DD'.

        Returns:
        - pd.DataFrame: The loaded data as a Pandas DataFrame.
        """
        if not _VALID_TABLE_NAME.match(table_name):
            raise ValueError("Invalid table name. Use only letters, numbers, and underscores.")

        # Keep the projected columns consistent with your IG schema.
        query = f"""
            SELECT
                timestamp,
                openPrice, openPrice_ask, openPrice_bid,
                closePrice, closePrice_ask, closePrice_bid,
                highPrice, highPrice_ask, highPrice_bid,
                lowPrice, lowPrice_ask, lowPrice_bid,
                lastTradedVolume
            FROM {table_name}
        """
        conditions = []
        params = {}

        # Apply trading-day filters using the table's local session boundary.
        if start_date:
            start_timestamp = _date_boundary_to_utc_ms(table_name, start_date)
            conditions.append("timestamp >= :start_timestamp")
            params["start_timestamp"] = start_timestamp
        if end_date:
            end_exclusive_timestamp = _date_boundary_to_utc_ms(table_name, end_date, end_exclusive=True)
            conditions.append("timestamp < :end_exclusive_timestamp")
            params["end_exclusive_timestamp"] = end_exclusive_timestamp
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp"
        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        # Execute the SQL query and load into a DataFrame
        data = pd.read_sql(text(query), self.engine, params=params)
        data = _filter_to_trading_weekdays(table_name, data)
        print(f"Data successfully loaded: {len(data)} rows.")
        return data
