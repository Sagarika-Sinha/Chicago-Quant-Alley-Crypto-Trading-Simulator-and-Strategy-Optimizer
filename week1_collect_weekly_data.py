import requests
import sqlite3
from datetime import datetime, timedelta, timezone

DB_NAME = "tables.db"

HEADERS = {
    'Accept': 'application/json'
}

def get_btc_price():
    url = "https://api.delta.exchange/v2/tickers"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch tickers")
    data = response.json()['result']
    for ticker in data:
        if ticker['symbol'] == "BTCUSDT":
            return float(ticker['spot_price'])
    raise Exception("BTCUSDT ticker not found")

def fetch_option_data():
    url = "https://api.delta.exchange/v2/products"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch product list")
    return response.json()['result']

def fetch_candles(symbol, start_dt, end_dt, resolution="5m"):
    url = "https://api.india.delta.exchange/v2/history/candles"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": int(start_dt.timestamp()),
        "end": int(end_dt.timestamp())
    }
    response = requests.get(url, params=params, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch candles: {response.status_code}")
    data = response.json()
    return data.get('result', [])

def store_in_db(date_str, call_rows, put_rows):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS options_{date_str} (
            symbol TEXT,
            strike_price INTEGER,
            expiry TEXT,
            option_type TEXT,
            volume REAL
        )
    ''')
    if call_rows:
        cursor.executemany(f'''
            INSERT INTO options_{date_str} (symbol, strike_price, expiry, option_type, volume)
            VALUES (?, ?, ?, ?, ?)
        ''', call_rows)
    if put_rows:
        cursor.executemany(f'''
            INSERT INTO options_{date_str} (symbol, strike_price, expiry, option_type, volume)
            VALUES (?, ?, ?, ?, ?)
        ''', put_rows)
    conn.commit()
    conn.close()

def store_candles_in_db(date_str, candles):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS candles_{date_str} (
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    rows = [(c['time'], c['open'], c['high'], c['low'], c['close'], c['volume']) for c in candles]
    cursor.executemany(f'''
        INSERT INTO candles_{date_str} (timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', rows)
    conn.commit()
    conn.close()

def main():
    start_date_str = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date_str = input("Enter end date (YYYY-MM-DD): ").strip()

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    if end_date < start_date:
        print("End date cannot be earlier than start date.")
        return

    btc_price = get_btc_price()
    print(f"Current BTC Price: {btc_price}")

    all_products = fetch_option_data()
    delta_days = (end_date - start_date).days + 1

    for i in range(delta_days):
        current_date = start_date + timedelta(days=i)
        expiry_date = current_date + timedelta(days=3)
        expiry_str = expiry_date.strftime("%d%m%y")
        date_str = current_date.strftime("%Y%m%d")

        print(f"\nCollecting options expiring on {expiry_date.date()} from options data dated {current_date.date()}")

        lower = int((btc_price - 15000) // 100) * 100
        upper = int((btc_price + 15000) // 100) * 100
        strike_range = set(range(lower, upper + 1, 100))

        call_options = []
        put_options = []

        for product in all_products:
            if product['underlying_asset']['symbol'] != "BTC":
                continue
            if product['contract_type'] not in ['put_options', 'call_options']:
                continue
            if not product['symbol'].endswith(expiry_str):
                continue

            strike_price = product.get('strike_price')
            if strike_price is None or int(strike_price) not in strike_range:
                continue

            volume = float(product.get('volume', 0.0))
            option_type = "call" if product['contract_type'] == "call_options" else "put"
            record = (product['symbol'], int(strike_price), expiry_str, option_type, volume)

            (call_options if option_type == "call" else put_options).append(record)

        store_in_db(date_str, call_options, put_options)
        print(f"Stored {len(call_options)} call options and {len(put_options)} put options for {date_str}")

        # Fetch and store candle data
        start_candle_dt = datetime(current_date.year, current_date.month, current_date.day, 0, 0, 0, tzinfo=timezone.utc)
        end_candle_dt = start_candle_dt + timedelta(days=1) - timedelta(seconds=1)

        candles = fetch_candles(symbol="BTCUSD", start_dt=start_candle_dt, end_dt=end_candle_dt)
        store_candles_in_db(date_str, candles)
        print(f"Stored {len(candles)} candles for {date_str}")

if __name__ == "__main__":
    main()
