import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from Strategy import Strategy
import config
class Simulator:
    def __init__(self, configFilePath=None):
        self.startDate = datetime.strptime(config.simStartDate, "%Y%m%d")
        self.endDate = datetime.strptime(config.simEndDate, "%Y%m%d")
        self.symbols = config.symbols
        self.df = pd.DataFrame()
        self.currentPrice = {symbol: None for symbol in self.symbols}
        self.currQuantity = {symbol: 0 for symbol in self.symbols}
        self.buyValue = {symbol: 0.0 for symbol in self.symbols}
        self.sellValue = {symbol: 0.0 for symbol in self.symbols}
        self.slippage = 0.0001
        self.strategy = Strategy(self)
        self.readData()
        self.startSimulation()
    def readData(self):
        all_data = []
        curr_date = self.startDate
        while curr_date <= self.endDate:
            date_str = curr_date.strftime("%Y%m%d")
            conn = sqlite3.connect("tables.db")
            for symbol in self.symbols:
                if symbol == "BTCUSDT":
                    table_name = f"candles_{date_str}"
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                        df["price"] = df["close"]
                        df["Symbol"] = symbol
                        all_data.append(df)
                    except Exception:
                        print(f"[WARNING] BTC candle data missing for {date_str}")
                else:
                    table_name = f"options_{date_str}"
                    try:
                        df = pd.read_sql_query(f"""SELECT * FROM {table_name} WHERE symbol = '{symbol}'""", conn)
                        df["timestamp"] = pd.Timestamp(curr_date)
                        df["price"] = df["volume"]  # placeholder: real price should come from tickers or quotes
                        df["Symbol"] = symbol
                        all_data.append(df)
                    except Exception:
                        print(f"[WARNING] Option data missing for {symbol} on {date_str}")
            conn.close()
            curr_date += timedelta(days=1)
        if all_data:
            self.df = pd.concat(all_data, ignore_index=True)
            self.df.sort_values("timestamp", inplace=True)
        else:
            print("No data loaded from the database.")
    def startSimulation(self):
        for _, row in self.df.iterrows():
            symbol = row["Symbol"]
            price = row["price"]
            self.currentPrice[symbol] = price
            self.strategy.onMarketData(row)
    def onOrder(self, symbol, side, quantity, price):
        if side == "BUY":
            adjusted_price = price * (1 + self.slippage)
        elif side == "SELL":
            adjusted_price = price * (1 - self.slippage)
        else:
            raise ValueError(f"Invalid order side: {side}")
        trade_value = adjusted_price * quantity
        if side == "BUY":
            self.currQuantity[symbol] += quantity
            self.buyValue[symbol] += trade_value
        else:
            self.currQuantity[symbol] -= quantity
            self.sellValue[symbol] += trade_value
        self.strategy.onTradeConfirmation(symbol, side, quantity, adjusted_price)
    def printPnl(self):
        total_pnl = 0.0
        for symbol in self.symbols:
            realized = self.sellValue[symbol] - self.buyValue[symbol]
            latest_price = self.currentPrice.get(symbol, 0)
            unrealized = self.currQuantity[symbol] * latest_price
            total_pnl += realized + unrealized
        print(f"[PnL] Current Total PnL: {total_pnl:.2f}")
if __name__ == "__main__":
    Simulator()
