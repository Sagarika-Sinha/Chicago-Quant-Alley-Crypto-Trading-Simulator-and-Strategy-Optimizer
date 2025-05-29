
import requests
from datetime import datetime,timedelta,timezone
import random
url_for_symbols="https://api.delta.exchange/v2/products"
symbols_response=requests.get(url_for_symbols)
if symbols_response.status_code==200:
    products=symbols_response.json()['result']
    active_symbols=[p['symbol'] for p in products if 'symbol' in p]
    if not active_symbols:
        raise Exception("No active symbols found")
    symbol=random.choice(active_symbols)
else:
    raise Exception("Failed to fetch product list from Delta Exchange.")
print(f"Collecting data for symbol: {symbol}")

resolution="1h"
end_time=datetime.now(timezone.utc)
start_time=end_time-timedelta(days=7)
start_timestamp=int(start_time.timestamp())
end_timestamp=int(end_time.timestamp())

url="https://api.delta.exchange/v2/history/candles"
params={"symbol":symbol,"resolution":resolution,"start":start_timestamp,"end":end_timestamp}
response=requests.get(url,params=params)
if response.status_code==200:
    data=response.json()
    print(f"Retrieved {len(data['result'])} candles:")
    for candle in data['result']:
        print(candle)
else:
    print("Error:",response.status_code,response.text)
