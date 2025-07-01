import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import zscore
import numpy as np
import sqlite3

# Load updated data with predictions
updated_data_file = '/home/a/Fin Project/Financial Web Scraping/updated_data_with_predictions.csv'
data = pd.read_csv(updated_data_file)

# Ensure the required columns are present
if 'ticker' not in data.columns or 'predicted_next_day_return' not in data.columns:
    raise ValueError("Required columns 'ticker' or 'predicted_next_day_return' missing in data.")

# Connect to the database
conn = sqlite3.connect('/home/a/Fin Project/Financial Web Scraping/equity_bse.db')  # Replace with your actual database path
cursor = conn.cursor()

# Fetch historical prices for all tickers
def fetch_historical_prices(tickers, days=31):
    results = []
    for ticker in tickers:
        query = '''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ?
            ORDER BY date DESC
            LIMIT ?
        '''
        cursor.execute(query, (ticker, days))
        rows = cursor.fetchall()
        if len(rows) < days:
            print(f"Not enough data for ticker {ticker}. Skipping...")
            continue
        results.append([row[1] for row in rows][::-1])  # Reverse to ascending dates
    return results

# Get tickers from the data
tickers = data['ticker'].unique()

# Fetch prices
price_data = fetch_historical_prices(tickers)

# Calculate daily average returns and std devs
daily_avg_returns = []
daily_std_devs = []

for day in range(1, 31):  # Last 30 days
    returns = []
    for prices in price_data:
        daily_return = (prices[day] - prices[day - 1]) / prices[day - 1]
        returns.append(daily_return)
    daily_avg_returns.append(np.mean(returns))
    daily_std_devs.append(np.std(returns))

# Train ARIMA models on daily averages
try:
    # ARIMA for average returns
    avg_return_model = ARIMA(daily_avg_returns, order=(1, 1, 1))
    avg_return_model_fit = avg_return_model.fit()
    predicted_avg_return = avg_return_model_fit.forecast(steps=1)[0]

    # ARIMA for standard deviations
    std_dev_model = ARIMA(daily_std_devs, order=(1, 1, 1))
    std_dev_model_fit = std_dev_model.fit()
    predicted_std_dev = std_dev_model_fit.forecast(steps=1)[0]
except Exception as e:
    print(f"ARIMA model training failed: {e}")
    predicted_avg_return = np.mean(daily_avg_returns)  # Fallback to mean
    predicted_std_dev = np.mean(daily_std_devs)       # Fallback to mean
      # Fallback to mean

print(avg_return_model_fit.summary())
print(std_dev_model_fit.summary())
print(predicted_avg_return)
print(predicted_std_dev)

# Rescale predicted returns in the CSV
data['predicted_next_day_return'] = (
    zscore(data['predicted_next_day_return']) * predicted_std_dev + predicted_avg_return
)

# Save the updated file
output_file = '/home/a/Fin Project/Financial Web Scraping/updated_data_with_predictions.csv'
data.to_csv(output_file, index=False)

print(f"Updated data saved to {output_file}")
