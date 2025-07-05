import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
def printStats(pnl_csv):
    df = pd.read_csv(pnl_csv)
    df.rename(columns={"PnL": "pnl"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["drawdown"] = df["cumulative_pnl"] - df["cumulative_pnl"].cummax()
    print("=== Performance Metrics ===")
    print(f"Mean PnL          : {df['pnl'].mean():.4f}")
    print(f"Median PnL        : {df['pnl'].median():.4f}")
    print(f"Std Dev PnL       : {df['pnl'].std():.4f}")
    df['date'] = df["timestamp"].dt.date
    daily_pnl = df.groupby('date')["pnl"].sum()
    daily_returns = daily_pnl.pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else np.nan
    print(f"Sharpe Ratio      : {sharpe_ratio:.4f}")
    max_dd = df["drawdown"].min()
    print(f"Max Drawdown      : {max_dd:.4f}")
    var_95 = np.percentile(daily_returns, 5)
    es_95 = daily_returns[daily_returns <= var_95].mean()
    print(f"VaR (95%)         : {var_95:.4f}")
    print(f"Expected Shortfall: {es_95:.4f}")
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["cumulative_pnl"], label="Cumulative PnL", color="blue")
    plt.title("Cumulative PnL")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cumulative_pnl.png")
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["drawdown"], label="Drawdown", color="red")
    plt.fill_between(df["timestamp"], df["drawdown"], 0, color="red", alpha=0.3)
    plt.title("Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("drawdown.png")
    plt.close()

    print("[INFO] Plots saved to cumulative_pnl.png and drawdown.png")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "output", "timestamped_pnl.csv")
    printStats(csv_path)
