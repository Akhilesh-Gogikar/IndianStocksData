import os
import json
import sqlite3
from datetime import datetime, timedelta
import csv
import time
from tqdm import tqdm
from extract_data_from_entry import fetch_additional_info
import concurrent.futures
import threading
from functools import partial
import tempfile
import shutil
import pandas as pd
import numpy as np
import logging
from technical_indicators import fetch_ohlcv_features

# Path to the folder containing processed JSON files
PROCESSED_RESULTS_FOLDER = '/home/a/Fin Project/Financial Web Scraping/data/processed_results'
# Paths to store train, test, validation folders
TRAIN_FOLDER = '/home/a/Fin Project/Financial Web Scraping/train'
TEST_FOLDER = '/home/a/Fin Project/Financial Web Scraping/test'
VAL_FOLDER = '/home/a/Fin Project/Financial Web Scraping/val'
# Path to store the summary CSV file
SUMMARY_CSV_PATH = '/home/a/Fin Project/Financial Web Scraping/stock_news_event_summary.csv'
# Path to the SQLite database containing OHLCV data
DATABASE_FILE = '/home/a/Fin Project/Financial Web Scraping/equity_bse.db'

# Create a thread-local storage for database connections
thread_local = threading.local()

def get_db_connection():
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    return thread_local.connection


# Function to load relevant data from JSON files and sort by date
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
                            # print(additional_info)

                            if sid and date:
                                entries.append({
                                    'sid': sid,
                                    'date': datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ"),
                                    'type': entry_type,
                                    'ticker': ticker,
                                    **additional_info
                                })
            except Exception as e:
                print(e)

    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x['date'])
    return sorted_entries

# Function to store summary information in a CSV file
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

# Check if the summary CSV file exists, if not, create it
if not os.path.exists(SUMMARY_CSV_PATH):
    sorted_entries = load_and_sort_entries(PROCESSED_RESULTS_FOLDER)
    store_summary_csv(sorted_entries, SUMMARY_CSV_PATH)
else:
    # Load the sorted entries from the summary CSV
    summary_df = pd.read_csv(SUMMARY_CSV_PATH, usecols=['sid', 'date', 'type', 'ticker'], parse_dates=['date'])
    sorted_entries = summary_df.to_dict('records')

# Connect to SQLite database
conn = sqlite3.connect(DATABASE_FILE)
cursor = conn.cursor()




# Function to get the earliest date from the database
def get_earliest_date_from_db():
    cursor.execute('''
        SELECT MIN(date) FROM stock_data
    ''')
    result = cursor.fetchone()
    if result and result[0]:
        return datetime.strptime(result[0], '%Y-%m-%d').date()
    else:
        return None



# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


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

            if num_entries < 2:
                # tqdm.write allows messages within the progress bar
                tqdm.write(f"Not enough entries for ticker {ticker} to proceed.")
                continue

            for i in range(num_entries - 1):
                current_entry = ticker_df.iloc[i]
                next_entry = ticker_df.iloc[i + 1]

                current_date = current_entry['date'].date()
                next_date = next_entry['date'].date()

                # Generate all dates between current_date and next_date (excluding next_date)
                date_range = pd.date_range(start=current_date, end=next_date - timedelta(days=1), freq='D')

                # Process each date in the date range
                for date in date_range:
                    date = date.date()
                    ohlcv_features = fetch_ohlcv_features(cursor, ticker, date)
                    if ohlcv_features:
                        enriched_entry = current_entry.to_dict()
                        enriched_entry.update(ohlcv_features)
                        enriched_entry['days_from_last'] = (date - current_entry['date'].date()).days
                        enriched_entry['processing_date'] = date
                          # Add processing date

                        # Initialize the CSV DictWriter if not already done
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=enriched_entry.keys())
                            writer.writeheader()

                        # Write the enriched entry to the file
                        writer.writerow(enriched_entry)

    print(f"Data processing completed. Output saved to {output_file}")
    conn.close()


# Example Usage
if __name__ == "__main__":
    process_and_store_data(SUMMARY_CSV_PATH, "/home/a/Fin Project/Financial Web Scraping/output/ohlcv_data.csv")