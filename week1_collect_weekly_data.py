import requests
from datetime import datetime,timedelta,timezone
symbol='BTCUSD'
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
