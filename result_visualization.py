from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mprl.decision_io import iter_decision_records

path = Path("logs/eval_2021_2024/decisions.jsonl")
records = list(iter_decision_records(path))
df = pd.DataFrame(records).sort_values("date")

weights = np.stack(df["action"].to_numpy())
tickers = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT",
]
w_df = pd.DataFrame(weights, columns=tickers, index=pd.to_datetime(df["date"]))

summary = w_df.agg(["mean", "std", "min", "max"]).T
print(summary.sort_values("std", ascending=False).head(10))  # 변동성 상위 종목

fig, ax = plt.subplots(figsize=(12, 4))
(w_df.diff().abs().sum(axis=1)).plot(ax=ax, title="Daily turnover (L1)")
ax.set_ylabel("Sum |Δweights|")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(np.cov(weights.T), cmap="coolwarm")
plt.xticks(range(len(tickers)), tickers, rotation=90)
plt.yticks(range(len(tickers)), tickers)
plt.title("Weight covariance heatmap")
plt.colorbar(label="Covariance")
plt.tight_layout()
plt.show()
