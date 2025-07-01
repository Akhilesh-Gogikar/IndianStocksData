import os
import json
import sqlite3
from datetime import datetime, timedelta
import csv
import pandas as pd
import time
from tqdm import tqdm
from extract_data_from_entry import fetch_additional_info
import concurrent.futures
import threading
from functools import partial
import tempfile
import shutil
import numpy as np
from technical_indicators import fetch_ohlcv_features

# Path to the folder containing processed JSON files
PROCESSED_RESULTS_FOLDER = '/home/a/Fin Project/Financial Web Scraping/data/processed_results'
LATEST_FOLDER = '/home/a/Fin Project/Financial Web Scraping/data/latest'  # New folder to store latest processed results
# Paths to store train, test, validation folders
TRAIN_FOLDER = '/home/a/Fin Project/Financial Web Scraping/train'
TEST_FOLDER = '/home/a/Fin Project/Financial Web Scraping/test'
VAL_FOLDER = '/home/a/Fin Project/Financial Web Scraping/val'
# Path to store the summary CSV file
SUMMARY_CSV_PATH = '/home/a/Fin Project/Financial Web Scraping/data/latest/stock_news_event_summary_latest.csv'
# Path to the SQLite database containing OHLCV data
DATABASE_FILE = '/home/a/Fin Project/Financial Web Scraping/equity_bse.db'

# Create a thread-local storage for database connections
thread_local = threading.local()

def get_db_connection():
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    return thread_local.connection


def get_rsi(cursor, ticker, event_date, period=14):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'rsi': 0}

        closes = [row[1] for row in rows]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        return {'rsi': rsi}
    except Exception as e:
        print(f"Error fetching RSI: {e}")
        return {'rsi': 0}

def get_macd(cursor, ticker, event_date, short_window=12, long_window=26, signal_window=9):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=long_window * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < long_window:
            return {'macd': 0, 'macd_signal': 0}

        closes = pd.Series([row[1] for row in rows])
        short_ema = closes.ewm(span=short_window, min_periods=short_window).mean()
        long_ema = closes.ewm(span=long_window, min_periods=long_window).mean()
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=signal_window, min_periods=signal_window).mean()

        return {'macd': macd.iloc[-1], 'macd_signal': macd_signal.iloc[-1]}
    except Exception as e:
        print(f"Error fetching MACD: {e}")
        return {'macd': 0, 'macd_signal': 0}

def get_bollinger_bands(cursor, ticker, event_date, window=20):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=window * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < window:
            return {'bollinger_upper': 0, 'bollinger_middle': 0, 'bollinger_lower': 0}

        closes = pd.Series([row[1] for row in rows])
        rolling_mean = closes.rolling(window=window).mean().iloc[-1]
        rolling_std = closes.rolling(window=window).std().iloc[-1]

        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        return {'bollinger_upper': upper_band, 'bollinger_middle': rolling_mean, 'bollinger_lower': lower_band}
    except Exception as e:
        print(f"Error fetching Bollinger Bands: {e}")
        return {'bollinger_upper': 0, 'bollinger_middle': 0, 'bollinger_lower': 0}

def get_atr(cursor, ticker, event_date, period=14):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, high, low, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'atr': 0}

        highs = pd.Series([row[1] for row in rows])
        lows = pd.Series([row[2] for row in rows])
        closes = pd.Series([row[3] for row in rows])

        high_low = highs - lows
        high_close = abs(highs - closes.shift(1))
        low_close = abs(lows - closes.shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return {'atr': atr}
    except Exception as e:
        print(f"Error fetching ATR: {e}")
        return {'atr': 0}

def get_recent_trend_features(cursor, ticker, event_date):
    try:
        # Exclude the event date to prevent data leakage
        end_date = event_date - timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(days=6)  # Total of 7 days

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date_str))

        rows = cursor.fetchall()
        prev_closes = [row[0] for row in rows if row[0] is not None]

        if len(prev_closes) < 2:
            return {
                'recent_price_change_pct': 0,
                'recent_avg_volume': 0,
                'recent_avg_volatility': 0,
            }

        price_change_pct = ((prev_closes[-1] - prev_closes[0]) / prev_closes[0]) * 100

        return {
            'recent_price_change_pct': price_change_pct,
            'recent_avg_volume': 0,  # Placeholder as volume data is not used
            'recent_avg_volatility': 0,  # Placeholder as volatility is not calculated
        }
    except Exception as e:
        print(f"Error fetching recent trend features: {e}")
        return {
            'recent_price_change_pct': 0,
            'recent_avg_volume': 0,
            'recent_avg_volatility': 0,
        }

def get_mid_term_trend_features(cursor, ticker, event_date):
    try:
        # Exclude the event date
        end_date = event_date - timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(days=29)  # Total of 30 days

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date_str))

        rows = cursor.fetchall()
        prev_closes = [row[0] for row in rows if row[0] is not None]

        if not prev_closes:
            return {
                'ma_15': 0,
                'ma_30': 0,
                'vwap': 0,  # Placeholder as VWAP is not calculated
            }

        # Calculate moving averages
        ma_15 = sum(prev_closes[-15:]) / min(len(prev_closes), 15)
        ma_30 = sum(prev_closes[-30:]) / min(len(prev_closes), 30)

        return {
            'ma_15': ma_15,
            'ma_30': ma_30,
            'vwap': 0  # Placeholder as VWAP is not calculated
        }
    except Exception as e:
        print(f"Error fetching mid-term trend features: {e}")
        return {
            'ma_15': 0,
            'ma_30': 0,
            'vwap': 0,
        }

def get_long_term_trend_features(cursor, ticker, event_date):
    try:
        # Exclude the event date
        end_date = event_date - timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(days=364)  # Total of 365 days

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date_str))

        rows = cursor.fetchall()
        prev_closes = [row[0] for row in rows if row[0] is not None]

        if not prev_closes:
            return {
                'ma_200': 0,
                'annual_volatility': 0,
                'pct_above_52_week_low': 0,
                'pct_below_52_week_high': 0,
            }

        # Calculate 200-day Moving Average
        ma_200 = sum(prev_closes[-200:]) / min(len(prev_closes), 200)

        # Calculate Annual Volatility
        if len(prev_closes) > 1:
            daily_returns = [(prev_closes[i+1] - prev_closes[i]) / prev_closes[i] for i in range(len(prev_closes)-1)]
            annual_volatility = pd.Series(daily_returns).std() * (252 ** 0.5)
        else:
            annual_volatility = 0

        # 52-week High/Low
        high_52_week = max(prev_closes)
        low_52_week = min(prev_closes)
        current_price = prev_closes[-1]

        pct_above_52_week_low = ((current_price - low_52_week) / low_52_week) * 100 if low_52_week != 0 else 0
        pct_below_52_week_high = ((high_52_week - current_price) / high_52_week) * 100 if high_52_week != 0 else 0

        return {
            'ma_200': ma_200,
            'annual_volatility': annual_volatility,
            'pct_above_52_week_low': pct_above_52_week_low,
            'pct_below_52_week_high': pct_below_52_week_high
        }
    except Exception as e:
        print(f"Error fetching long-term trend features: {e}")
        return {
            'ma_200': 0,
            'annual_volatility': 0,
            'pct_above_52_week_low': 0,
            'pct_below_52_week_high': 0
        }

def get_next_day_return(cursor, ticker, event_date):
    try:
        # Calculate return for the day after the event
        next_date = event_date + timedelta(days=1)
        next_date_str = next_date.strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date = ?
        ''', (ticker, next_date_str))

        row_next = cursor.fetchone()
        if not row_next or row_next[0] is None:
            return {
                'next_day_return': 0
            }

        next_day_close = row_next[0]

        # Get the following day's close (i.e., prevClose of two days after the event date)
        following_date = next_date + timedelta(days=1)
        following_date_str = following_date.strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date = ?
        ''', (ticker, following_date_str))

        row_following = cursor.fetchone()
        if not row_following or row_following[0] is None:
            return {
                'next_day_return': 0
            }

        following_close = row_following[0]

        # Calculate next day return
        next_day_return = ((following_close - next_day_close) / next_day_close) * 100 if next_day_close else 0

        return {
            'next_day_return': next_day_return
        }
    except Exception as e:
        print(f"Error fetching next day return: {e}")
        return {
            'next_day_return': 0
        }

def load_and_sort_entries(folder_path):
    entries = []
    total_files = len([f for f in os.listdir(folder_path) if f.endswith('.json')])

    # Iterate over all files in the specified folder using tqdm for progress visualization
    for filename in tqdm(os.listdir(folder_path), total=total_files, desc="Loading and sorting entries"):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Ensure data is a list of dictionaries
                    if isinstance(data, list):
                        latest_entry = None
                        for entry in data:
                            # Extract only necessary fields to save memory
                            if 'news_date' in entry:
                                entry_type = 'news'
                                date = entry['news_date']
                            elif 'event_date' in entry:
                                entry_type = 'event'
                                date = entry['event_date']
                            else:
                                continue

                            sid = entry.get('sid', None)
                            ticker = entry.get('data_before_date', {}).get('securityInfo', {}).get('info', {}).get('ticker', None)

                            # Fetch additional information using external function
                            additional_info = fetch_additional_info(entry, entry_type, look_back=1)

                            if sid and date:
                                current_entry = {
                                    'sid': sid,
                                    'date': datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ"),
                                    'type': entry_type,
                                    'ticker': ticker,
                                    **additional_info
                                }
                                # Keep only the latest record per file
                                if latest_entry is None or current_entry['date'] > latest_entry['date']:
                                    latest_entry = current_entry
                        if latest_entry:
                            entries.append(latest_entry)
            except Exception as e:
                print(e)

    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x['date'])
    return sorted_entries

def store_summary_csv(entries, csv_path):
    # Define CSV header
    header = ['sid', 'date', 'type', 'ticker'] + list(entries[0].keys())[4:]
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        # Write each entry to the CSV file
        for entry in entries:
            row = [entry.get(col, '') for col in header]
            writer.writerow(row)

if not os.path.exists(LATEST_FOLDER):
    os.makedirs(LATEST_FOLDER)

# Function to get the next day return and check if it's available
def get_next_day_return(cursor, ticker, event_date):
    try:
        # Calculate return for the day after the event
        next_date = event_date + timedelta(days=1)
        next_date_str = next_date.strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date = ?
        ''', (ticker, next_date_str))

        row_next = cursor.fetchone()
        if not row_next or row_next[0] is None:
            return None  # No next day return available

        next_day_close = row_next[0]

        # Get the following day's close (i.e., prevClose of two days after the event date)
        following_date = next_date + timedelta(days=1)
        following_date_str = following_date.strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date = ?
        ''', (ticker, following_date_str))

        row_following = cursor.fetchone()
        if not row_following or row_following[0] is None:
            return None  # No following day data available

        following_close = row_following[0]

        # Calculate next day return
        next_day_return = ((following_close - next_day_close) / next_day_close) * 100 if next_day_close else 0

        return {
            'next_day_return': next_day_return
        }
    except Exception as e:
        print(f"Error fetching next day return: {e}")
        return None

def process_and_store_data(summary_csv_path, output_file):
    # Load summary CSV with robustness to mixed types
    try:
        summary_df = pd.read_csv(
            summary_csv_path,
            parse_dates=['date'],
            dtype={'ticker': str},
            low_memory=False
        )
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Add index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data (tickerSymbol, date);
    ''')
    conn.commit()

    # Get the list of unique tickers
    unique_tickers = summary_df['ticker'].unique()
    top_tickers = pd.read_csv("/home/a/Fin Project/Financial Web Scraping/top_stocks_by_trading_volume.csv", usecols=["tickerSymbol"])["tickerSymbol"].dropna().tolist()


    # Open the output file and write the header once
    with open(output_file, mode='w', newline='') as file:
        writer = None  # Initialize writer
        print("Processing tickers...")
        for ticker in tqdm(unique_tickers, desc="Tickers", unit="ticker"):
            if ticker not in top_tickers:
                print(f"{ticker} not in top traded stocks")
                continue
            ticker_df = summary_df[summary_df['ticker'] == ticker].sort_values(by='date').reset_index(drop=True)
            num_entries = len(ticker_df)

            if num_entries < 1:
                # tqdm.write allows messages within the progress bar
                tqdm.write(f"No entries for ticker {ticker} to proceed.")
                continue

            # Process only the latest entry for each ticker
            current_entry = ticker_df.iloc[-1]
            current_date = current_entry['date'].date()

            # Define the date range starting from the current date up to today
            today = datetime.today().date()
            date_range = pd.date_range(start=current_date, end=today, freq='D')

            # Process each date in the date range in reverse order
            for date in date_range[::-1]:
                date = date.date()
                ohlcv_features = fetch_ohlcv_features(cursor, ticker, date)
                if ohlcv_features:
                    enriched_entry = current_entry.to_dict()
                    enriched_entry.update(ohlcv_features)
                    enriched_entry['days_from_last'] = (date - current_entry['date'].date()).days
                    enriched_entry['processing_date'] = date  # Add processing date

                    # Initialize the CSV DictWriter if not already done
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=enriched_entry.keys())
                        writer.writeheader()

                    # Write the enriched entry to the file
                    writer.writerow(enriched_entry)
                    # Break the loop since we only need the entry with the highest days_from_last
                    break  # Added break here to ensure only one entry per ticker

        print(f"Data processing completed. Output saved to {output_file}")
        conn.close()


# Example Usage
if __name__ == "__main__":
    sorted_entries = load_and_sort_entries(PROCESSED_RESULTS_FOLDER)
    if not os.path.exists(LATEST_FOLDER):
        os.makedirs(LATEST_FOLDER)
    store_summary_csv(sorted_entries, SUMMARY_CSV_PATH)
    process_and_store_data(SUMMARY_CSV_PATH, "/home/a/Fin Project/Financial Web Scraping/latest_data.csv")
