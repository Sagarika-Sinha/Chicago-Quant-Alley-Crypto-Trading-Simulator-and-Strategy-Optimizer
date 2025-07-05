import requests
import os
import csv
from datetime import datetime, timedelta, timezone
HEADERS = {'Accept':'application/json'}
DATA_DIR = "data"
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
    params = {"symbol": symbol,"resolution": resolution,"start": int(start_dt.timestamp()),"end": int(end_dt.timestamp())}
    response = requests.get(url, params=params, headers=HEADERS)
    if response.status_code != 200:
        print(f"Warning: Failed to fetch candles for {symbol}")
        return []
    data = response.json()
    return data.get('result', [])
def save_combined_option_and_candles(path, metadata, candles):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "strike_price", "expiry", "option_type", "volume", "timestamp", "open", "high", "low", "close", "volume_candle"])
        for c in candles:
            writer.writerow(list(metadata) + [c['time'], c['open'], c['high'], c['low'], c['close'], c['volume']])
def save_btc_candles(path, candles):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c['time'], c['open'], c['high'], c['low'], c['close'], c['volume']])
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
        expiry_date = current_date
        target_expiry = expiry_date.date()
        date_str = current_date.strftime("%Y%m%d")
        folder = os.path.join(DATA_DIR, date_str)
        print(f"\nCollecting options expiring on {target_expiry} for folder {date_str}")
        saved_options = 0
        start_candle_dt = datetime(current_date.year, current_date.month, current_date.day, 0, 0, 0, tzinfo=timezone.utc)
        end_candle_dt = start_candle_dt + timedelta(days=1) - timedelta(seconds=1)
        btc_candles = fetch_candles(symbol="BTCUSD", start_dt=start_candle_dt, end_dt=end_candle_dt)
        btc_path = os.path.join(folder, "MARK:BTCUSDT.csv")
        if btc_candles:
            save_btc_candles(btc_path, btc_candles)
        print(f"Saved {len(btc_candles)} BTC candles for {date_str}")
        for product in all_products:
            if product['underlying_asset']['symbol'] != "BTC":
                continue
            if product['contract_type'] not in ['put_options', 'call_options']:
                continue
            symbol = product['symbol']
            symbol_parts = symbol.split("-")
            if len(symbol_parts) < 4:
                continue
            strike_price = product.get('strike_price')
            if strike_price is None:
                continue
            volume = float(product.get('volume', 0.0))
            option_type = "call" if product['contract_type'] == "call_options" else "put"
            expiry_str = symbol_parts[-1]
            try:
                symbol_expiry_date = datetime.strptime(expiry_str, "%d%m%y").date()
            except ValueError:
                continue

            if symbol_expiry_date != target_expiry:
                continue
            option_candles = fetch_candles(symbol=symbol, start_dt=start_candle_dt, end_dt=end_candle_dt)
            if not option_candles:
                continue
            metadata = (symbol, int(strike_price), expiry_str, option_type, volume)
            file_name = f"MARK:{symbol[:-6]}-{date_str}.csv"
            file_path = os.path.join(folder, file_name)
            save_combined_option_and_candles(file_path, metadata, option_candles)
            saved_options += 1
        print(f"Saved {saved_options} option files with candles for {date_str}")
if __name__ == "__main__":
    main()
