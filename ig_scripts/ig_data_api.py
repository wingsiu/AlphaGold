import json
from enum import Enum
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import pandas as pd
import requests
import mysql.connector
from dotenv import load_dotenv
import os
import keyring

# Ensure all columns are displayed in pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Load environment variables from .env
load_dotenv()

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
}

API_CONFIG = {
    'api_key': os.getenv('IG_API_KEY'),
    'username': os.getenv('IG_USERNAME'),
    'password': os.getenv('IG_PASSWORD'),
    'base_url': os.getenv('IG_BASE_URL', "https://deal.ig.com/"),  # Default is the IG API base URL
}

PRICE_COLUMNS = [
    'timestamp',
    'openPrice', 'openPrice_ask', 'openPrice_bid',
    'closePrice', 'closePrice_ask', 'closePrice_bid',
    'highPrice', 'highPrice_ask', 'highPrice_bid',
    'lowPrice', 'lowPrice_ask', 'lowPrice_bid',
    'lastTradedVolume',
]

# Price Instruments Enum

class Price(Enum):
    Oil = ("CC.D.CL.BMU.IP", "prices")
    AUD = ("CS.D.AUDUSD.MINI.IP", "aud_prices")
    Gold = ("CS.D.CFDGOLD.BMU.IP", "gold_prices")

    def __init__(self, epic: str, db_name: str):
        self._value_ = epic
        self.db_name = db_name

    @property
    def epic(self) -> str:
        return self._value_

# Modernized IG Service Class
class IGService:
    def __init__(self, api_key, username, password, base_url):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json; charset=UTF-8',
            'Accept': 'application/json; charset=UTF-8',
            'VERSION': '2',
            'X-IG-API-KEY': self.api_key,
        }
        self.authenticate()

    def authenticate(self):
        """Authenticate and store CST and X-SECURITY-TOKEN."""
        payload = {"identifier": self.username, "password": self.password}
        url = f"{self.base_url}/session"
        #url = f"https://api.ig.com/gateway/deal/session"


        response = requests.post(url, json=payload, headers=self.headers)
        #print("login response:", response.headers)
        if response.status_code == 200:
            self.headers['CST'] = response.headers.get('CST')
            self.headers['X-SECURITY-TOKEN'] = response.headers.get('X-SECURITY-TOKEN')
            # Store tokens securely
            keyring.set_password('ig_api', 'cst_token', self.headers['CST'])
            keyring.set_password('ig_api', 'security_token', self.headers['X-SECURITY-TOKEN'])
        else:
            raise Exception(f"Authentication failed: {response.status_code} - {response.text}")

    def refresh_tokens_if_needed(self):
        """Refresh tokens if not available."""
        cst_token = keyring.get_password('ig_api', 'cst_token')
        security_token = keyring.get_password('ig_api', 'security_token')

        if not cst_token or not security_token:
            self.authenticate()
        else:
            self.headers['CST'] = cst_token
            self.headers['X-SECURITY-TOKEN'] = security_token


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _utc_epoch_ms(dt: datetime) -> int:
    utc_dt = dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(utc_dt.timestamp() * 1000)


def _ensure_utc_datetime(value: Optional[datetime]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _iso_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return _ensure_utc_datetime(value).isoformat()


def _price_rows_period(prices_data: list[dict[str, Any]]) -> tuple[Optional[datetime], Optional[datetime]]:
    if not prices_data:
        return None, None
    timestamps = [row.get('timestamp') for row in prices_data if row.get('timestamp') is not None]
    if not timestamps:
        return None, None
    ts_min = min(int(float(str(ts))) for ts in timestamps)
    ts_max = max(int(float(str(ts))) for ts in timestamps)
    return (
        datetime.fromtimestamp(ts_min / 1000, tz=timezone.utc),
        datetime.fromtimestamp(ts_max / 1000, tz=timezone.utc),
    )


def snapshot_to_price_row(snapshot_record: dict[str, Any]) -> dict[str, Any]:
    """Map a live market snapshot into the existing OHLC MySQL schema.

    The existing `prices`/`gold_prices`/`aud_prices` tables store minute-level OHLC data.
    A single snapshot does not provide a true full minute bar, so we persist the best available
    quote values into the same schema using the current minute bucket as the row timestamp.
    """
    fetch_ts = pd.Timestamp(snapshot_record.get('fetch_time_utc') or datetime.now(timezone.utc))
    if fetch_ts.tzinfo is None:
        fetch_ts = fetch_ts.tz_localize(timezone.utc)
    else:
        fetch_ts = fetch_ts.tz_convert(timezone.utc)
    bucket_ts = fetch_ts.floor('min')
    if bucket_ts is pd.NaT:
        raise ValueError("Could not derive minute bucket from snapshot fetch_time_utc")
    bucket_dt = datetime.fromtimestamp(bucket_ts.timestamp(), tz=timezone.utc)

    bid = _safe_float(snapshot_record.get('bid'))
    offer = _safe_float(snapshot_record.get('offer'))
    mid = _safe_float(snapshot_record.get('mid'))
    if mid is None and bid is not None and offer is not None:
        mid = (bid + offer) / 2.0

    high = _safe_float(snapshot_record.get('high'))
    low = _safe_float(snapshot_record.get('low'))
    base_mid = _coalesce(mid, bid, offer, high, low)
    if base_mid is None:
        raise ValueError("Snapshot does not contain any usable bid/offer/mid/high/low price")
    ask_side = _coalesce(offer, base_mid)
    bid_side = _coalesce(bid, base_mid)
    # Snapshot high/low are session-level fields, not minute OHLC highs/lows.
    # Using them here corrupts minute bars with unrealistic extremes.
    high_side = max(float(ask_side), float(bid_side))
    low_side = min(float(ask_side), float(bid_side))

    return {
        'timestamp': _utc_epoch_ms(bucket_dt),
        'openPrice': base_mid,
        'openPrice_ask': ask_side,
        'openPrice_bid': bid_side,
        'closePrice': base_mid,
        'closePrice_ask': ask_side,
        'closePrice_bid': bid_side,
        'highPrice': high_side,
        'highPrice_ask': high_side,
        'highPrice_bid': high_side,
        'lowPrice': low_side,
        'lowPrice_ask': low_side,
        'lowPrice_bid': low_side,
        'lastTradedVolume': snapshot_record.get('lastTradedVolume', 0) or 0,
    }


def fetch_market_snapshot(service: IGService, instrument: Price, fetch_time: Optional[datetime] = None) -> dict[str, Any]:
    """Fetch the latest IG market snapshot for one instrument.

    Returns a normalised record suitable for local persistence.
    """
    service.refresh_tokens_if_needed()
    fetch_dt = fetch_time.astimezone(timezone.utc) if fetch_time is not None else datetime.now(timezone.utc)
    url = f"{service.base_url.rstrip('/')}/markets/{instrument.epic}"
    headers = dict(service.headers)
    headers['VERSION'] = '3'

    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 401:
        service.authenticate()
        headers = dict(service.headers)
        headers['VERSION'] = '3'
        response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch market snapshot: {response.status_code} - {response.text}")

    payload = response.json()
    snapshot = payload.get('snapshot') or {}
    instrument_info = payload.get('instrument') or {}
    bid = _safe_float(snapshot.get('bid'))
    offer = _safe_float(snapshot.get('offer'))
    mid = None if bid is None or offer is None else (bid + offer) / 2.0
    return {
        'instrument': instrument.name.lower(),
        'epic': instrument.epic,
        'fetch_time_utc': fetch_dt.isoformat(),
        'bucket_minute_utc': fetch_dt.replace(second=0, microsecond=0).isoformat(),
        'market_status': snapshot.get('marketStatus'),
        'update_time_utc': snapshot.get('updateTimeUTC') or snapshot.get('updateTime'),
        'bid': bid,
        'offer': offer,
        'mid': mid,
        'high': _safe_float(snapshot.get('high')),
        'low': _safe_float(snapshot.get('low')),
        'net_change': _safe_float(snapshot.get('netChange')),
        'percentage_change': _safe_float(snapshot.get('percentageChange')),
        'lastTradedVolume': snapshot.get('lastTradedVolume') or 0,
        'instrument_name': instrument_info.get('name'),
        'raw_json': json.dumps(payload, separators=(',', ':')),
    }


def fetch_and_store_market_snapshot(
    service: IGService,
    instrument: Price,
    fetch_time: Optional[datetime] = None,
) -> dict[str, Any]:
    """Fetch one live snapshot and upsert it into the instrument's existing price table."""
    snapshot_record = fetch_market_snapshot(service, instrument, fetch_time=fetch_time)
    price_row = snapshot_to_price_row(snapshot_record)
    insert_prices([price_row], instrument)
    return price_row


def fetch_and_store_prices_from_latest(
    service: IGService,
    instrument: Price,
    end_time: Optional[datetime] = None,
) -> dict[str, Any]:
    """Backfill minute data from the latest MySQL timestamp up to the requested end time."""
    end_dt = _ensure_utc_datetime(end_time)
    latest_before = fetch_last_date(instrument)
    request_start = latest_before if latest_before is not None else end_dt - timedelta(days=5)
    prices = fetch_prices(service, instrument, start_time=request_start, end_time=end_dt)
    written = insert_prices(prices, instrument)
    latest_before_ms = None if latest_before is None else _utc_epoch_ms(latest_before)
    inserted_rows = len(
        {
            int(price.get('timestamp'))
            for price in prices
            if price.get('timestamp') is not None and (latest_before_ms is None or int(price.get('timestamp')) > latest_before_ms)
        }
    )
    period_start, period_end = _price_rows_period(prices)
    return {
        'instrument': instrument.name.lower(),
        'table_name': instrument.db_name,
        'latest_db_before_utc': _iso_utc(latest_before),
        'requested_start_utc': _iso_utc(request_start),
        'requested_end_utc': _iso_utc(end_dt),
        'fetched_rows': len(prices),
        'written_rows': int(written),
        'inserted_rows': int(inserted_rows),
        'fetched_period_start_utc': _iso_utc(period_start),
        'fetched_period_end_utc': _iso_utc(period_end),
    }

# Database Utilities
def fetch_last_date(instrument: Price):
    """Fetch the most recent date from the database for the given instrument."""
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor(dictionary=True)
    query = f"SELECT * FROM {instrument.db_name} ORDER BY timestamp DESC LIMIT 1"
    cursor.execute(query)
    result = cursor.fetchone()
    connection.close()

    if result:
        ts_val = result.get('timestamp')
        if ts_val is None:
            return None
        return datetime.fromtimestamp(float(str(ts_val)) / 1000, tz=timezone.utc)
    return None

def insert_prices(prices_data, instrument: Price):
    """Upsert prices into the existing instrument table using the current IG schema."""
    if not prices_data:
        return 0

    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    columns_sql = ', '.join(PRICE_COLUMNS)
    placeholders = ', '.join(['%s'] * len(PRICE_COLUMNS))
    update_columns = [column for column in PRICE_COLUMNS if column != 'timestamp']
    update_sql = ', '.join(f"{column}=VALUES({column})" for column in update_columns)
    rows = [tuple(data.get(column) for column in PRICE_COLUMNS) for data in prices_data]

    cursor.executemany(f"""
        INSERT INTO {instrument.db_name} ({columns_sql})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_sql}
    """, rows)
    connection.commit()
    written = cursor.rowcount
    connection.close()
    return int(written)

# Fetch Prices
def fetch_prices(
    service: IGService,
    instrument: Price,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    """Fetch historical prices from the latest MySQL timestamp (or provided start) up to the requested end time."""
    service.refresh_tokens_if_needed()
    start_dt = _ensure_utc_datetime(start_time) if start_time is not None else fetch_last_date(instrument)
    end_dt = _ensure_utc_datetime(end_time)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=5)
    if end_dt < start_dt:
        return []

    # Prepare fetch URL
    url = (
        f"https://deal.ig.com/chart/snapshot/{instrument.epic}/1/MINUTE/batch/"
        f"start/{start_dt.year}/{start_dt.month}/{start_dt.day}/{start_dt.hour}/{start_dt.minute}/{start_dt.second}/{start_dt.microsecond // 1000}/"
        f"end/{end_dt.year}/{end_dt.month}/{end_dt.day}/{end_dt.hour}/{end_dt.minute}/{end_dt.second}/{end_dt.microsecond // 1000}?format=json"
        #&siteId=inm&locale=en_GB&version=61"
    )

    response = requests.get(url, headers=service.headers, timeout=30)
    if response.status_code == 401:
        service.authenticate()
        response = requests.get(url, headers=service.headers, timeout=30)
    if response.status_code == 200:
        data = response.json()
        #print("data:" ,data)
        results = []
        for data_point in data["intervalsDataPoints"]:
            for price in data_point['dataPoints']:
                try:
                    results.append({
                        'timestamp': price['timestamp'],
                        'openPrice': (price['openPrice']['ask'] + price['openPrice']['bid']) / 2,
                        'openPrice_ask': price['openPrice']['ask'],
                        'openPrice_bid': price['openPrice']['bid'],
                        'closePrice': (price['closePrice']['ask'] + price['closePrice']['bid']) / 2,
                        'closePrice_ask': price['closePrice']['ask'],
                        'closePrice_bid': price['closePrice']['bid'],
                        'highPrice': (price['highPrice']['ask'] + price['highPrice']['bid']) / 2,
                        'highPrice_ask': price['highPrice']['ask'],
                        'highPrice_bid': price['highPrice']['bid'],
                        'lowPrice': (price['lowPrice']['ask'] + price['lowPrice']['bid']) / 2,
                        'lowPrice_ask': price['lowPrice']['ask'],
                        'lowPrice_bid': price['lowPrice']['bid'],
                        'lastTradedVolume': price['lastTradedVolume']
                    })
                except KeyError:
                    continue
        #insert_prices(results, instrument)
        return results
    else:
        raise Exception(f"Failed to fetch prices: {response.status_code} - {response.text}")


def _request_with_reauth(
    service: IGService,
    method: str,
    path: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
    version: str = "2",
    timeout: int = 30,
) -> dict[str, Any]:
    service.refresh_tokens_if_needed()
    url = f"{service.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _do_request() -> requests.Response:
        headers = dict(service.headers)
        headers['VERSION'] = str(version)
        return requests.request(method.upper(), url, headers=headers, json=payload, params=params, timeout=timeout)

    response = _do_request()
    if response.status_code == 401:
        service.authenticate()
        response = _do_request()

    if response.status_code < 200 or response.status_code >= 300:
        raise Exception(f"IG request failed {method.upper()} {path}: {response.status_code} - {response.text}")

    body: dict[str, Any] = {}
    if response.text.strip():
        try:
            body = response.json()
        except ValueError:
            body = {"raw_text": response.text}
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": body,
    }


def fetch_accounts(service: IGService) -> list[dict[str, Any]]:
    response = _request_with_reauth(service, "GET", "/accounts", version="1")
    body = dict(response.get("body") or {})
    accounts = body.get("accounts")
    return list(accounts) if isinstance(accounts, list) else []


def fetch_primary_account_summary(service: IGService) -> dict[str, Any]:
    accounts = fetch_accounts(service)
    if not accounts:
        return {"status": "unavailable"}

    preferred: Optional[dict[str, Any]] = None
    for account in accounts:
        if not isinstance(account, dict):
            continue
        if bool(account.get("preferred")):
            preferred = account
            break
    if preferred is None:
        preferred = next((a for a in accounts if isinstance(a, dict)), None)
    if preferred is None:
        return {"status": "unavailable"}

    balance = dict(preferred.get("balance") or {})
    balance_amt = _safe_float(_coalesce(balance.get("balance"), balance.get("availableCash")))
    pnl_amt = _safe_float(_coalesce(balance.get("profitLoss"), balance.get("profitLossLR"), 0.0))
    equity_amt = None if balance_amt is None else balance_amt + float(pnl_amt or 0.0)
    available_amt = _safe_float(_coalesce(balance.get("available"), balance.get("availableToDeal"), balance.get("availableCash")))
    deposit_amt = _safe_float(balance.get("deposit"))

    return {
        "account_id": preferred.get("accountId"),
        "account_name": preferred.get("accountName"),
        "account_type": preferred.get("accountType"),
        "status": preferred.get("status") or preferred.get("accountStatus") or "unknown",
        "currency": preferred.get("currency") or preferred.get("currencyIsoCode"),
        "balance": balance_amt,
        "equity": equity_amt,
        "available": available_amt,
        "deposit": deposit_amt,
        "profit_loss": pnl_amt,
    }


def confirm_deal_reference(service: IGService, deal_reference: str) -> dict[str, Any]:
    response = _request_with_reauth(service, "GET", f"/confirms/{deal_reference}", version="1")
    return dict(response.get("body") or {})


def place_otc_market_order(
    service: IGService,
    instrument: Price,
    direction: str,
    *,
    size: float,
    stop_distance: Optional[float] = None,
    limit_distance: Optional[float] = None,
    currency_code: str = "USD",
    force_open: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epic": instrument.epic,
        "expiry": "-",
        "direction": str(direction).upper(),
        "size": float(size),
        "orderType": "MARKET",
        "currencyCode": currency_code,
        "forceOpen": bool(force_open),
        "guaranteedStop": False,
    }
    if stop_distance is not None and float(stop_distance) > 0:
        payload["stopDistance"] = float(stop_distance)
    if limit_distance is not None and float(limit_distance) > 0:
        payload["limitDistance"] = float(limit_distance)

    response = _request_with_reauth(service, "POST", "/positions/otc", payload=payload, version="2")
    body = dict(response.get("body") or {})
    headers = dict(response.get("headers") or {})
    deal_reference = str(body.get("dealReference") or headers.get("deal-reference") or "")
    confirm = confirm_deal_reference(service, deal_reference) if deal_reference else {}
    return {
        "request": payload,
        "deal_reference": deal_reference or None,
        "deal_id": confirm.get("dealId"),
        "deal_status": confirm.get("dealStatus"),
        "reason": confirm.get("reason"),
        "response": body,
        "confirm": confirm,
    }


def fetch_open_positions(service: IGService) -> list[dict[str, Any]]:
    response = _request_with_reauth(service, "GET", "/positions", version="2")
    body = dict(response.get("body") or {})
    positions = body.get("positions")
    return list(positions) if isinstance(positions, list) else []


def get_open_position_by_deal_id(service: IGService, deal_id: str) -> Optional[dict[str, Any]]:
    wanted = str(deal_id or "").strip()
    if not wanted:
        return None
    for position in fetch_open_positions(service):
        pos = position.get("position") if isinstance(position, dict) else None
        if isinstance(pos, dict) and str(pos.get("dealId") or "").strip() == wanted:
            return position
    return None


def amend_position_levels(
    service: IGService,
    deal_id: str,
    *,
    stop_level: Optional[float] = None,
    limit_level: Optional[float] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if stop_level is not None:
        payload["stopLevel"] = float(stop_level)
    if limit_level is not None:
        payload["limitLevel"] = float(limit_level)
    if not payload:
        raise ValueError("At least one of stop_level/limit_level must be provided.")

    response = _request_with_reauth(
        service,
        "PUT",
        f"/positions/otc/{deal_id}",
        payload=payload,
        version="2",
    )
    body = dict(response.get("body") or {})
    headers = dict(response.get("headers") or {})
    deal_reference = str(body.get("dealReference") or headers.get("deal-reference") or "")
    confirm = confirm_deal_reference(service, deal_reference) if deal_reference else {}
    return {
        "deal_id": deal_id,
        "deal_reference": deal_reference or None,
        "request": payload,
        "response": body,
        "confirm": confirm,
    }


def close_otc_position(
    service: IGService,
    *,
    deal_id: str,
    direction: str,
    size: float,
    order_type: str = "MARKET",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dealId": str(deal_id),
        "direction": str(direction).upper(),
        "size": float(size),
        "orderType": str(order_type).upper(),
        "timeInForce": "FILL_OR_KILL",
    }
    response = _request_with_reauth(
        service,
        "DELETE",
        "/positions/otc",
        payload=payload,
        version="1",
    )
    body = dict(response.get("body") or {})
    headers = dict(response.get("headers") or {})
    deal_reference = str(body.get("dealReference") or headers.get("deal-reference") or "")
    confirm = confirm_deal_reference(service, deal_reference) if deal_reference else {}
    return {
        "request": payload,
        "deal_reference": deal_reference or None,
        "response": body,
        "confirm": confirm,
        "deal_status": confirm.get("dealStatus"),
        "reason": confirm.get("reason"),
        "close_level": _safe_float(confirm.get("level")),
        "close_time": confirm.get("date"),
    }


def _parse_ig_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts is pd.NaT:
            return None
        return ts.to_pydatetime().astimezone(timezone.utc) if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _fetch_recent_transactions(service: IGService, *, lookback_hours: int = 72) -> list[dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    from_utc = now_utc - timedelta(hours=max(int(lookback_hours), 1))
    tx_rows: list[dict[str, Any]] = []

    # Common IG path shape.
    from_str = from_utc.strftime("%Y-%m-%dT%H:%M:%S")
    to_str = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        response = _request_with_reauth(
            service,
            "GET",
            f"/history/transactions/ALL/{from_str}/{to_str}",
            version="2",
        )
        body = dict(response.get("body") or {})
        rows = body.get("transactions")
        if isinstance(rows, list):
            tx_rows.extend([row for row in rows if isinstance(row, dict)])
    except Exception:
        pass

    # Fallback path shape used by some environments.
    if not tx_rows:
        try:
            response = _request_with_reauth(
                service,
                "GET",
                "/history/transactions",
                params={"from": from_str, "to": to_str, "detailed": "true", "pageSize": 500},
                version="2",
            )
            body = dict(response.get("body") or {})
            rows = body.get("transactions")
            if isinstance(rows, list):
                tx_rows.extend([row for row in rows if isinstance(row, dict)])
        except Exception:
            pass

    return tx_rows


def get_closed_trade_by_deal_id(service: IGService, deal_id: str, *, lookback_hours: int = 72) -> Optional[dict[str, Any]]:
    wanted = str(deal_id or "").strip()
    if not wanted:
        return None
    rows = _fetch_recent_transactions(service, lookback_hours=lookback_hours)
    if not rows:
        return None

    matched: list[dict[str, Any]] = []
    for row in rows:
        row_deal = str(row.get("dealId") or row.get("reference") or "").strip()
        if not row_deal:
            continue
        if row_deal == wanted or wanted in row_deal or row_deal in wanted:
            matched.append(row)

    if not matched:
        return None

    def _row_dt(item: dict[str, Any]) -> datetime:
        dt = _parse_ig_datetime(item.get("date") or item.get("dateUtc") or item.get("timestamp"))
        return dt if dt is not None else datetime.fromtimestamp(0, tz=timezone.utc)

    matched.sort(key=_row_dt, reverse=True)
    row = matched[0]
    exit_dt = _parse_ig_datetime(row.get("date") or row.get("dateUtc") or row.get("timestamp"))
    exit_price = _safe_float(row.get("closeLevel"))
    if exit_price is None:
        exit_price = _safe_float(row.get("level"))
    if exit_price is None:
        exit_price = _safe_float(row.get("price"))

    reason = str(row.get("transactionType") or row.get("type") or row.get("channel") or "")
    pnl = _safe_float(row.get("profitAndLoss"))

    return {
        "deal_id": wanted,
        "exit_time": exit_dt.isoformat() if exit_dt is not None else None,
        "exit_price": exit_price,
        "reason": reason or None,
        "pnl": pnl,
        "raw": row,
    }

# Main Execution
if __name__ == "__main__":
    ig_service = IGService(
        api_key=API_CONFIG['api_key'],
        username=API_CONFIG['username'],
        password=API_CONFIG['password'],
        base_url=API_CONFIG['base_url']
    )
    try:
        oil_summary = fetch_and_store_prices_from_latest(ig_service, Price.Oil)
        print("oil sync summary:\n", pd.DataFrame([oil_summary]))

        gold_summary = fetch_and_store_prices_from_latest(ig_service, Price.Gold)
        print("gold sync summary:\n", pd.DataFrame([gold_summary]))

        latest_gold_snapshot = fetch_and_store_market_snapshot(ig_service, Price.Gold)
        print("latest gold snapshot stored as existing-schema row:\n", pd.DataFrame([latest_gold_snapshot]))

        print("Data fetch and insertion completed using the existing MySQL tables/schema.")
    except Exception as e:
        print(f"Error: {e}")

