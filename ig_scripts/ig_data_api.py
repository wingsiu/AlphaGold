import json
from enum import Enum
from datetime import datetime, timedelta, timezone
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
        return self.value

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
        return datetime.fromtimestamp(result['timestamp'] / 1000, tz=timezone.utc)
    return None

def insert_prices(prices_data, instrument: Price):
    """Insert prices into the database."""
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    for data in prices_data:
        cursor.execute(f"""
            INSERT INTO {instrument.db_name} (
                timestamp, openPrice, openPrice_ask, openPrice_bid,
                closePrice, closePrice_ask, closePrice_bid,
                highPrice, highPrice_ask, highPrice_bid,
                lowPrice, lowPrice_ask, lowPrice_bid,
                lastTradedVolume
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data['timestamp'], data['openPrice'], data['openPrice_ask'], data['openPrice_bid'],
            data['closePrice'], data['closePrice_ask'], data['closePrice_bid'],
            data['highPrice'], data['highPrice_ask'], data['highPrice_bid'],
            data['lowPrice'], data['lowPrice_ask'], data['lowPrice_bid'],
            data['lastTradedVolume']
        ))
    connection.commit()
    connection.close()

# Fetch Prices
def fetch_prices(service: IGService, instrument: Price):
    """Fetch historical prices for a given instrument."""
    service.refresh_tokens_if_needed()
    #last_date = fetch_last_date(instrument) or datetime.utcnow() - timedelta(days=5)
    last_date = fetch_last_date(instrument)  - timedelta(days=5)

    now = datetime.now(tz=timezone.utc) - timedelta(minutes=1)

    # Prepare fetch URL
    url = (
        f"https://deal.ig.com/chart/snapshot/{instrument.epic}/1/MINUTE/batch/"
        f"start/{last_date.year}/{last_date.month}/{last_date.day}/{last_date.hour}/{last_date.minute}/0/0/"
        f"end/{now.year}/{now.month}/{now.day}/{now.hour}/{now.minute}/59/999?format=json"
        #&siteId=inm&locale=en_GB&version=61"
    )

    response = requests.get(url, headers=service.headers)
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

# Main Execution
if __name__ == "__main__":
    ig_service = IGService(
        api_key=API_CONFIG['api_key'],
        username=API_CONFIG['username'],
        password=API_CONFIG['password'],
        base_url=API_CONFIG['base_url']
    )
    try:
        oil = fetch_prices(ig_service, Price.Oil)
        oil_df = pd.DataFrame(oil)
        print("oil data:\n", oil_df)
        #insert_prices(oil, Price.Oil)
        gold = fetch_prices(ig_service, Price.Gold)
        gold_df = pd.DataFrame(gold)
        print("gold data:\n", gold_df)

        print("Data fetch and insertion completed.")
    except Exception as e:
        print(f"Error: {e}")

