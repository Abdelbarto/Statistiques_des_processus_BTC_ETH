import requests
import pandas as pd 
import numpy as np 

def get_binance_klines(symbol, interval, start, end):
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end).timestamp() * 1000)
    all_rows = []

    while start_ms < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     1000
        }
        r = requests.get(url, params=params).json()
        if not r:
            break
        all_rows += r
        start_ms = r[-1][0] + 1   # repart depuis la dernière bougie +1ms
        if len(r) < 1000:
            break

    df = pd.DataFrame(all_rows, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    df["time"]  = pd.to_datetime(df["time"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df[["time","close"]].set_index("time")

# Télécharge toute l'année 2025 heure par heure (~8760 points)
btc = get_binance_klines("BTCUSDT", "1h", "2025-01-01", "2025-12-31")
eth = get_binance_klines("ETHUSDT", "1h", "2025-01-01", "2025-12-31")
print(f"BTC: {len(btc)} bougies | ETH: {len(eth)} bougies")

#Log rendement

btc["return"] = np.log(btc["close"] / btc["close"].shift(1))
eth["return"] = np.log(eth["close"] / eth["close"].shift(1))

btc["loss"] = -btc["return"]
eth["loss"] = -eth["return"]

df = pd.DataFrame({
    "btc_return": btc["return"],
    "eth_return": eth["return"],
    "btc_loss": btc["loss"],
    "eth_loss": eth["loss"]
}).dropna()

df.to_csv("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/returns_btc_eth.csv")
print(df.describe())
print(f"\n Nombre d'observations:: {len(df)}")

        