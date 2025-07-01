import os
import json
import sqlite3
import threading
import csv
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import logging
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load column classifications from JSON
try:
    logger.info("Loading column classifications from JSON.")
    with open('column_types.json', 'r') as f:
        columns_classification = json.load(f)
except FileNotFoundError:
    logger.error("Error: column_types.json file not found.")
    raise
except json.JSONDecodeError:
    logger.error("Error: column_types.json file is not a valid JSON.")
    raise

# Extract column types
numeric_columns = columns_classification.get('numeric_columns', [])
categorical_columns = columns_classification.get('categorical_columns', [])
text_columns = columns_classification.get('text_columns', [])
target_column = columns_classification.get('target_column', 'next_day_return')

# File paths and constants
SUMMARY_CSV_PATH = 'stock_news_event_summary.csv'
DATABASE_FILE = 'equity_bse.db'
scaler_path = 'scalers.pkl'
ordinal_encodings_path = 'ordinal_encodings.pkl'
text_features_path = 'text_features.json'
ohlcv_features_path = 'ohlcv_features.json'

# Create a thread-local storage for database connections
thread_local = threading.local()

def get_db_connection():
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    return thread_local.connection

def fetch_ohlcv_features(ticker, current_date, cursor):
    try:
        cursor.execute('''
            SELECT * FROM stock_data
            WHERE tickerSymbol = ? AND date = ?
        ''', (ticker, current_date.strftime('%Y-%m-%d')))
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
    except Exception as e:
        logger.error(f"Error fetching OHLCV features for {ticker} on {current_date}: {e}")
    return {}

def calculate_returns(cursor, ticker):
    try:
        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ?
            ORDER BY date ASC
        ''', (ticker,))
        rows = cursor.fetchall()
        if not rows:
            return {}

        returns = {}
        prev_price = None
        for row in rows:
            current_date = row[0]
            current_price = row[1]
            if prev_price is not None and current_price is not None and prev_price!=0:
                daily_return = ((current_price - prev_price) / prev_price) * 100
                returns[current_date] = daily_return
            prev_price = current_price
        return returns
    except Exception as e:
        logger.error(f"Error calculating returns for ticker '{ticker}': {e}")
        return {}

def calculate_ordinal_mappings(summary_df, returns_dict, categorical_columns):
    ordinal_encodings = {}
    for col in categorical_columns:
        logger.info(f"Calculating ordinal mapping for column '{col}' based on cumulative returns.")
        try:
            grouped = summary_df.groupby(col)
            cumulative_returns = []
            for group_name, group_data in grouped:
                cumulative_return = 0
                for _, row in group_data.iterrows():
                    date = row['date'].strftime('%Y-%m-%d')
                    ticker = row['ticker']
                    if ticker in returns_dict and date in returns_dict[ticker]:
                        cumulative_return += returns_dict[ticker][date]
                cumulative_returns.append((group_name, cumulative_return))
            # Sort by cumulative return in descending order
            cumulative_returns.sort(key=lambda x: x[1], reverse=True)
            max_rank = len(cumulative_returns) - 1
            ordinal_encodings[col] = {name: (max_rank - idx) / max_rank for idx, (name, _) in enumerate(cumulative_returns)}
        except Exception as e:
            logger.error(f"Error calculating ordinal mapping for column '{col}': {e}")
            ordinal_encodings[col] = {}
    return ordinal_encodings

def process_and_store_data(summary_csv_path, scaler_path, ordinal_encodings_path, text_features_path, ohlcv_features_path):
    try:
        summary_df = pd.read_csv(
            summary_csv_path,
            parse_dates=['date'],
            dtype={'ticker': str},
            low_memory=False
        )
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    # Add index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data (tickerSymbol, date);
    ''')
    conn.commit()

    # Calculate returns for each ticker
    returns_dict = {}
    unique_tickers = summary_df['ticker'].unique()
    for ticker in tqdm(unique_tickers, desc="Calculating returns", unit="ticker"):
        returns_dict[ticker] = calculate_returns(cursor, ticker)

    # Calculate ordinal mappings for categorical columns based on returns
    ordinal_encodings = calculate_ordinal_mappings(summary_df, returns_dict, categorical_columns)

    # Initialize MinMaxScalers for numeric columns
    scalers = {col: MinMaxScaler() for col in numeric_columns}
    text_features = {}
    ohlcv_features = {}

    # Set up NLTK lemmatizer
    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    # Load pre-trained word embedding model (e.g., Word2Vec)
    embedding_model = Word2Vec.load('word2vec_word_level.model')
    embedding_dim = embedding_model.vector_size
    default_embedding = np.zeros(embedding_dim)  # Default embedding for unknown tokens

    for ticker in tqdm(unique_tickers, desc="Processing tickers", unit="ticker"):
        ticker_df = summary_df[summary_df['ticker'] == ticker].sort_values(by='date').reset_index(drop=True)
        num_entries = len(ticker_df)

        if num_entries < 2:
            tqdm.write(f"Not enough entries for ticker {ticker} to proceed.")
            continue

        # Process text columns once per ticker
        for col in text_columns:
            if col in ticker_df.columns:
                logger.info(f"Processing text column '{col}' with NLTK lemmatization and averaging embeddings for ticker '{ticker}'.")
                ticker_df[col] = ticker_df[col].fillna('').astype(str)
                lemmatized_texts = []
                embeddings_per_date = {}
                for idx, text in enumerate(ticker_df[col]):
                    current_date = ticker_df.iloc[idx]['date'].date()
                    tokens = word_tokenize(text.lower())
                    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
                    # Compute embeddings for the current text entry
                    embeddings = []
                    for token in lemmatized_tokens:
                        if token in embedding_model.wv:
                            embeddings.append(embedding_model.wv[token])
                        else:
                            embeddings.append(default_embedding)
                    if embeddings:
                        mean_embedding = np.mean(embeddings, axis=0)
                    else:
                        mean_embedding = default_embedding
                    embeddings_per_date[str(current_date)] = mean_embedding
                # Store embeddings with ticker and date key
                text_features[f"{ticker}_{col}"] = embeddings_per_date

        # Update scalers with numeric features for each date and store OHLCV features
        for i in range(num_entries - 1):
            current_entry = ticker_df.iloc[i]
            current_date = current_entry['date'].date()

            # Fetch OHLCV features
            ohlcv = fetch_ohlcv_features(ticker, current_date, cursor)
            if ohlcv:
                # Store OHLCV features with ticker and date key
                ohlcv_features[f"{ticker}_{current_date}"] = ohlcv

                # Update scalers with numeric features
                for col in numeric_columns:
                    if col in ohlcv:
                        scalers[col].partial_fit(np.array(ohlcv[col]).reshape(-1, 1))

    # Save scalers
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)

    # Save ordinal encodings
    with open(ordinal_encodings_path, 'wb') as f:
        pickle.dump(ordinal_encodings, f)

    # Save text features as word embeddings
    with open(text_features_path, 'w') as f:
        text_features_serializable = {
            k: {date: embedding.tolist() for date, embedding in v.items()} 
            if isinstance(v, dict) else v
            for k, v in text_features.items()
        }
        json.dump(text_features_serializable, f)


    # Save OHLCV features
    with open(ohlcv_features_path, 'w') as f:
        json.dump({k: v for k, v in ohlcv_features.items()}, f)

    logger.info(f"Data processing completed. Scalers, encodings, text features, and OHLCV features saved.")
    conn.close()

if __name__ == "__main__":
    process_and_store_data(SUMMARY_CSV_PATH, scaler_path, ordinal_encodings_path, text_features_path, ohlcv_features_path)
