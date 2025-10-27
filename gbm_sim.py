#importing all the libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats

ticker = "^GSPC" #ticker for S&P 500 in yfinance lib        
past_years = 10          
sims = 100000            
trading_days = 252       
end = datetime.today()
start = end - timedelta(days=int(365 * past_years))

df = yf.download(
    ticker,
    start=start.date(),
    end=end.date(),
    auto_adjust=True,
    progress=False
)

if df.empty or "Close" not in df.columns:
    raise ValueError("No price data downloaded. Check ticker/date range/network.")

prices_hist = df["Close"].dropna().copy()
if len(prices_hist) < trading_days:
    raise ValueError("Not enough data to estimate daily parameters.")

S0 = float(prices_hist.iloc[-1]) #Starting Price 
dlog = np.log(prices_hist / prices_hist.shift(1)).dropna() #log of the daily retunrns

mu_daily = float(dlog.mean())             
sigma_daily = float(dlog.std(ddof=1))     

print(".................................................")
print(f"Estimated from {len(dlog)} daily obs")
print(f"Dailuy mu  ≈ {mu_daily:.4f}")
print(f"Daily vol ≈ {sigma_daily:.4f}")
print(".................................................")

all_paths = []    # list of lists, each inner list is one price path (length 253 incl. day 0)
final_prices = [] # list of terminal prices (day 252)

for i in range(sims):
    price = S0
    sim_prices = [price]
    for t in range(trading_days):
        eps = np.random.normal()
        log_return = mu_daily + (sigma_daily * eps)
        price *= np.exp(log_return)
        sim_prices.append(price)
    all_paths.append(sim_prices)
    final_prices.append(price)

paths = np.array(all_paths, dtype=float)

final_prices_arr = np.array(final_prices, dtype=float)
above_start = (final_prices_arr > S0).mean()
expected_final = final_prices_arr.mean()
median_final = np.median(final_prices_arr)
expected_return = ((expected_final-S0)/S0)
p10, p25, p75, p90 = np.percentile(final_prices_arr, [10, 25, 75, 90])

print(".................................................")
print("One-year Monte Carlo summary")
print(f"Start price (S0)      : {S0:,.2f}")
print(f"Expected final price  : {expected_final:,.2f}")
print(f"Expected returns  : {expected_return:.2%}")
print(".................................................")

print(".................................................")
print("5 Number Summary")
print(f"Median final price    : {median_final:,.2f}")
print(f"10th / 25th pct       : {p10:,.2f} / {p25:,.2f}")
print(f"75th / 90th pct       : {p75:,.2f} / {p90:,.2f}")
print(".................................................")


plt.figure(figsize=(10, 6))
max_to_show = min(500, sims)
for j in range(max_to_show):
    plt.plot(paths[j], linewidth=0.7, alpha=1)
plt.xlabel("Trading day")
plt.ylabel("Simulated price")
plt.title(f"{ticker} Monte Carlo paths ({paths.shape[0]} sims)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(final_prices_arr, bins=60, density=True, alpha=0.65)
plt.title("Distribution of Simulated Year-End Prices (Day 252)")
plt.xlabel("Price")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
