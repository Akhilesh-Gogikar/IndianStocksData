import pandas as pd
import numpy as np
import json
import pickle
import os
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Set up logging
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

# Identify date columns based on column names
logger.info("Identifying date columns.")
date_columns = [col for col in numeric_columns + categorical_columns + text_columns if 'date' in col.lower()]

# Remove date columns from other categories
numeric_columns = [col for col in numeric_columns if col not in date_columns]
categorical_columns = [col for col in categorical_columns if col not in date_columns]
text_columns = [col for col in text_columns if col not in date_columns]

# File paths and constants
csv_file = 'output/combined_data.csv'
chunksize = 100000
scaler_path = 'scalers.pkl'
ordinal_encodings_path = 'ordinal_encodings.pkl'
processed_data_dir = 'processed_chunks/'
vocab_path = 'vocab.pkl'

# Create directory for processed chunks if it doesn't exist
os.makedirs(processed_data_dir, exist_ok=True)

# Dictionary to store MinMaxScalers for each numeric column
scalers = {}

# Dictionary to store ordinal encodings for categorical columns
ordinal_encodings = {}

# Dictionary to store word counts for text columns
word_counts = {}

# Step 1: Iterate through the entire dataset column-wise to calculate global encodings, scalers, vocabularies, etc.

logger.info("Calculating global encodings and scalers column-wise.")

# Process numeric columns for scaling
if numeric_columns:
    for col in numeric_columns:
        logger.info(f"Processing numeric column '{col}' for scaling.")
        scaler = MinMaxScaler()
        for chunk in tqdm(pd.read_csv(csv_file, usecols=[col], chunksize=chunksize)):
            scaler.partial_fit(chunk[[col]])
        scalers[col] = scaler

# Process categorical columns for ordinal encoding
for col in categorical_columns:
    logger.info(f"Processing categorical column '{col}' for ordinal encoding.")
    unique_values = set()
    for chunk in tqdm(pd.read_csv(csv_file, usecols=[col], chunksize=chunksize)):
        unique_values.update(chunk[col].dropna().unique())
    ordinal_encodings[col] = {val: idx for idx, val in enumerate(sorted(unique_values))}

# Process text columns for vocabulary
for col in text_columns:
    logger.info(f"Processing text column '{col}' for vocabulary building.")
    for chunk in tqdm(pd.read_csv(csv_file, usecols=[col], chunksize=chunksize)):
        chunk[col] = chunk[col].fillna('').astype(str)
        for text in chunk[col]:
            tokens = text.split()
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

# Save ordinal encodings
logger.info("Saving ordinal encodings.")
with open(ordinal_encodings_path, 'wb') as f:
    pickle.dump(ordinal_encodings, f)

# Save the scalers
logger.info("Saving scalers for numeric columns.")
with open(scaler_path, 'wb') as f:
    pickle.dump(scalers, f)

