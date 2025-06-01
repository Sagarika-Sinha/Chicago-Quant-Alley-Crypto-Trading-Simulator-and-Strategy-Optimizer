import requests
import sqlite3
from datetime import datetime, timedelta, timezone

DB_NAME = "btc_options_week.db"

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

def store_in_db(date_str, data_rows):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS options_{date_str} (symbol TEXT,strike_price INTEGER,expiry TEXT,option_type TEXT,volume REAL)''')
    cursor.executemany(f'''INSERT INTO options_{date_str} (symbol, strike_price, expiry, option_type, volume)VALUES (?, ?, ?, ?, ?)''',data_rows)
    conn.commit()
    conn.close()

def main():
    btc_price = get_btc_price()
    print(f"Current BTC Price: {btc_price}")

    all_products = fetch_option_data()
    base_date = datetime(2025, 5, 25, tzinfo=timezone.utc)

    for i in range(7):
        current_date = base_date + timedelta(days=i)
        expiry_date = current_date + timedelta(days=3)
        expiry_str = expiry_date.strftime("%d%m%y")
        date_str = current_date.strftime("%Y%m%d")
        print(f"Collecting for {date_str}, expiry: {expiry_str}")

        lower = int((btc_price - 15000) // 100) * 100
        upper = int((btc_price + 15000) // 100) * 100
        strike_range = set(range(lower, upper + 1, 100))

        collected = []
        for product in all_products:
            if product['underlying_asset']['symbol'] != "BTC":
                continue
            if product['contract_type'] not in ['put_options', 'call_options']:
                continue

            symbol = product['symbol']
            if not symbol.endswith(expiry_str):
                continue

            strike_price = product.get('strike_price')
            if strike_price is None:
                continue
            if int(strike_price) not in strike_range:
                continue

            volume = float(product.get('volume', 0.0))
            option_type = "call" if product['contract_type'] == "call_options" else "put"
            collected.append((symbol, int(strike_price), expiry_str, option_type, volume))

        store_in_db(date_str, collected)
        print(f"Stored {len(collected)} options for {date_str}")

if __name__ == "__main__":
    main()
