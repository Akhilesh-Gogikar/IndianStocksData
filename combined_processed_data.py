import os
import json
import sqlite3
import pandas as pd
import numpy as np
import pickle
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# File paths and constants
SUMMARY_CSV_PATH = 'stock_news_event_summary.csv'
scaler_path = 'scalers.pkl'
ordinal_encodings_path = 'ordinal_encodings.pkl'
text_features_path = 'text_features.json'
ohlcv_features_path = 'ohlcv_features.json'
output_dataset_dir = 'processed_data_chunks/'
entries_per_chunk = 10000

# Ensure output directory exists
os.makedirs(output_dataset_dir, exist_ok=True)

# Load precomputed scalers, encodings, and features
try:
    logger.info("Loading scalers, ordinal encodings, text features, and OHLCV features.")
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    with open(ordinal_encodings_path, 'rb') as f:
        ordinal_encodings = pickle.load(f)
    with open(text_features_path, 'r') as f:
        text_features = json.load(f)
    with open(ohlcv_features_path, 'r') as f:
        ohlcv_features = json.load(f)
except FileNotFoundError as e:
    logger.error(f"Error: {e}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Error loading JSON data: {e}")
    raise

# Load summary CSV
try:
    logger.info("Loading summary CSV.")
    summary_df = pd.read_csv(SUMMARY_CSV_PATH, parse_dates=['date'], dtype={'ticker': str}, low_memory=False)
except Exception as e:
    logger.error(f"Error loading CSV: {e}")
    raise

# Extract column classifications
numeric_columns = scalers.keys()
categorical_columns = ordinal_encodings.keys()
target_column = 'next_day_return'  # Adjust this to match the actual target column in your CSV

# Process the summary CSV and apply transformations
logger.info("Processing summary CSV and applying transformations.")
processed_data = []
chunk_index = 0

tickers = summary_df['ticker'].unique()
ticker_indices = {ticker: 0 for ticker in tickers}

# Iterate over rows to ensure at least one entry per ticker in each chunk
while any(ticker_indices[ticker] < len(summary_df[summary_df['ticker'] == ticker]) for ticker in tickers):
    for ticker in tickers:
        group = summary_df[summary_df['ticker'] == ticker]
        if ticker_indices[ticker] < len(group):
            row = group.iloc[ticker_indices[ticker]]
            ticker_indices[ticker] += 1

            logger.debug(f"Processing row for ticker: {ticker}, date: {row['date']}")

            date_str = row['date'].strftime('%Y-%m-%d')

            # Process numeric columns
            numeric_data = []
            for col in numeric_columns:
                if col in ohlcv_features.get(f"{ticker}_{date_str}", {}):
                    value = ohlcv_features[f"{ticker}_{date_str}"][col]
                    scaled_value = scalers[col].transform([[value]])[0][0]
                    numeric_data.append(scaled_value)
                    logger.debug(f"Numeric column '{col}': original value = {value}, scaled value = {scaled_value}")
                else:
                    numeric_data.append(0)  # Default value if data is not available
                    logger.debug(f"Numeric column '{col}': default value used (0)")

            # Process categorical columns
            categorical_data = []
            for col in categorical_columns:
                if col in row:
                    category = row[col]
                else:
                    category = 'Unknown'
                encoding = ordinal_encodings[col].get(category, -1)
                categorical_data.append(encoding)
                logger.debug(f"Categorical column '{col}': category = {category}, encoding = {encoding}")

            # Process text features
            text_data = []
            for col in text_features:
                text_key = f"{ticker}_{col}"
                if text_key in text_features and date_str in text_features[text_key]:
                    text_data.append(np.array(text_features[text_key][date_str]).tolist())  # Convert ndarray to list for JSON serialization
                    logger.debug(f"Text feature '{col}': data found for date {date_str}")
                else:
                    text_data.append(np.zeros(len(text_features[list(text_features.keys())[0]][list(text_features[list(text_features.keys())[0]].keys())[0]])).tolist())  # Default embedding if not available
                    logger.debug(f"Text feature '{col}': default embedding used")

            # Extract target value
            target_value = row[target_column] if target_column in row else 0  # Default target value if not available
            logger.debug(f"Target value for row: {target_value}")

            # Combine all processed data
            combined_row = numeric_data + categorical_data
            if text_data:
                combined_row.extend([item for sublist in text_data for item in sublist])  # Flatten text_data and add to combined_row

            # Append target value as the last element
            combined_row.append(target_value)

            processed_data.append(combined_row)
            logger.debug(f"Combined row appended. Current number of processed rows: {len(processed_data)}")

            # Save chunk if entries reach the specified limit
            if len(processed_data) >= entries_per_chunk:
                chunk_file_path = os.path.join(output_dataset_dir, f'processed_data_chunk_{chunk_index}.npz')
                logger.info(f"Saving chunk {chunk_index} to '{chunk_file_path}'.")
                processed_data_array = np.array(processed_data, dtype=np.float32)
                np.savez_compressed(chunk_file_path, data=processed_data_array)
                processed_data = []
                chunk_index += 1

# Save any remaining data
if processed_data:
    chunk_file_path = os.path.join(output_dataset_dir, f'processed_data_chunk_{chunk_index}.npz')
    logger.info(f"Saving final chunk {chunk_index} to '{chunk_file_path}'.")
    processed_data_array = np.array(processed_data, dtype=np.float32)
    np.savez_compressed(chunk_file_path, data=processed_data_array)

logger.info("Data processing and saving completed.")
