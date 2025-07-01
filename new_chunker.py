import pandas as pd
import numpy as np
import json
import pickle
import os
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For WordNet lemmatizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

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
chunksize = 10000
processed_data_dir = 'processed_chunks/'
scaler_path = 'scalers.pkl'
ordinal_encodings_path = 'ordinal_encodings.pkl'

# Update paths for embeddings
embedding_model_path = '/home/a/Fin Project/Financial Web Scraping/word2vec_word_level.model'  # Path to your pre-trained embedding model

# Load the precomputed scalers and encodings
try:
    logger.info("Loading scalers.")
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)

    logger.info("Loading ordinal encodings.")
    with open(ordinal_encodings_path, 'rb') as f:
        ordinal_encodings = pickle.load(f)
except FileNotFoundError as e:
    logger.error(f"Error: {e}")
    raise

# Load pre-trained word embeddings using Gensim
try:
    logger.info("Loading pre-trained word embeddings using Gensim.")
    embedding_model = gensim.models.Word2Vec.load(embedding_model_path)
    embedding_dim = embedding_model.vector_size
except FileNotFoundError:
    logger.error(f"Error: Pre-trained embedding model not found at '{embedding_model_path}'.")
    raise

# Create directory for updated processed chunks if it doesn't exist
updated_processed_data_dir = 'updated_processed_chunks/'
os.makedirs(updated_processed_data_dir, exist_ok=True)

# Parameters for text processing
default_embedding = np.zeros(embedding_dim)  # Default embedding for unknown tokens

# Prepare list of processed columns for later filtering
# This list will include:
# - New date feature columns
# - Scaled numeric columns
# - Encoded categorical columns
# Note: We exclude text columns and the target column because they are dropped before filtering

# Initialize list to store new date feature columns
new_date_feature_columns = []
for date_col in date_columns:
    new_date_feature_columns.extend([
        f'{date_col}_year',
        f'{date_col}_month',
        f'{date_col}_day',
        f'{date_col}_weekday'
    ])

# All processed columns excluding text columns and target column
processed_columns = new_date_feature_columns + numeric_columns + categorical_columns

# Process chunks from scratch
logger.info("Processing CSV file in chunks with NLTK lemmatization and pre-trained embeddings.")
chunk_index = 0

for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunksize)):
    # Handle date columns: Convert to datetime and extract features
    for date_col in date_columns:
        logger.info(f"Processing date column '{date_col}'.")
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
        # Extract date features
        chunk[f'{date_col}_year'] = chunk[date_col].dt.year.fillna(-1).astype(int)
        chunk[f'{date_col}_month'] = chunk[date_col].dt.month.fillna(-1).astype(int)
        chunk[f'{date_col}_day'] = chunk[date_col].dt.day.fillna(-1).astype(int)
        chunk[f'{date_col}_weekday'] = chunk[date_col].dt.weekday.fillna(-1).astype(int)
        # Drop the original date column
        chunk.drop(columns=[date_col], inplace=True)

    # Extract target column before modifying the rest of the chunk
    if target_column in chunk.columns:
        target_data = chunk[target_column].to_numpy().reshape(-1, 1)
        # Handle missing target values if any
        if np.isnan(target_data).any():
            logger.warning(f"Missing values found in target column '{target_column}'. Filling with 0.")
            target_data = np.nan_to_num(target_data, nan=0.0)
        chunk = chunk.drop(columns=[target_column])
    else:
        logger.error(f"Error: Target column '{target_column}' not found in chunk.")
        raise KeyError(f"Target column '{target_column}' not found in chunk.")

    # Handle numeric columns: Fill missing values and then transform using the corresponding MinMaxScaler
    for col in numeric_columns:
        if col in chunk.columns:
            logger.info(f"Processing numeric column '{col}'.")
            # Fill missing values with the median from the scaler (assuming scaler was fitted with median)
            # If scaler does not store median, use a default strategy
            if hasattr(scalers[col], 'feature_names_in_'):
                # Example: if scaler has been fitted with feature names
                median = np.nanmedian(chunk[col])
                chunk[col] = chunk[col].fillna(median)
            else:
                # Default strategy: fill NaNs with 0
                chunk[col] = chunk[col].fillna(0)
            # Scale the column
            chunk[[col]] = scalers[col].transform(chunk[[col]])
        else:
            logger.warning(f"Numeric column '{col}' not found in chunk.")
            # Optionally, add a column filled with default scaled value (e.g., 0)
            chunk[col] = 0
            # Ensure the column is scaled
            chunk[[col]] = scalers[col].transform(chunk[[col]])

    # Handle categorical columns: Fill missing values and apply global ordinal encodings
    for col in categorical_columns:
        if col in chunk.columns:
            logger.info(f"Processing categorical column '{col}'.")
            # Fill missing values with a placeholder, e.g., 'Unknown'
            chunk[col] = chunk[col].fillna('Unknown')
            # Apply encoding; unknown categories will result in NaN
            chunk[col] = chunk[col].map(ordinal_encodings[col])
            # Replace NaN encodings with a special value, e.g., -1
            chunk[col] = chunk[col].fillna(-1).astype(int)
        else:
            logger.warning(f"Categorical column '{col}' not found in chunk.")
            # Optionally, add a column filled with the special encoding
            chunk[col] = -1

    # Handle text columns with NLTK lemmatization and averaging embeddings
    text_embeddings = []
    for col in text_columns:
        if col in chunk.columns:
            logger.info(f"Processing text column '{col}' with NLTK lemmatization and averaging embeddings.")
            chunk[col] = chunk[col].fillna('').astype(str)
            lemmatized_texts = []
            for text in chunk[col]:
                # Tokenize the text
                tokens = word_tokenize(text.lower())
                # Lemmatize the tokens and filter out non-alphabetic tokens
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
                lemmatized_texts.append(lemmatized_tokens)
            # Compute embeddings
            averaged_embeddings = []
            for tokens in lemmatized_texts:
                embeddings = []
                for token in tokens:
                    if token in embedding_model.wv:
                        embeddings.append(embedding_model.wv[token])
                    else:
                        embeddings.append(default_embedding)
                if embeddings:
                    mean_embedding = np.mean(embeddings, axis=0)
                else:
                    mean_embedding = default_embedding
                averaged_embeddings.append(mean_embedding)
            # Convert to NumPy array
            averaged_embeddings = np.array(averaged_embeddings)
            text_embeddings.append(averaged_embeddings)
            # Drop the original text column after processing
            chunk.drop(columns=[col], inplace=True)
        else:
            logger.warning(f"Text column '{col}' not found in chunk.")
            # If the text column is missing, append default embeddings
            text_embeddings.append(np.tile(default_embedding, (len(chunk), 1)))

    # Drop any columns that have not been processed
    # This ensures that only the processed columns are retained
    # Any other columns present in the chunk will be removed
    logger.info("Dropping unprocessed columns.")
    # Adjust processed_columns to include only columns currently in chunk
    columns_to_keep = [col for col in processed_columns if col in chunk.columns]
    chunk = chunk[columns_to_keep]

    # Fill any remaining NaNs in processed columns with 0
    if chunk.isnull().values.any():
        logger.warning("NaN values found in processed numeric/categorical columns. Filling with 0.")
        chunk = chunk.fillna(0)

    # Convert the rest of the chunk to a NumPy array
    chunk_values = chunk.values.astype(np.float32)

    # Concatenate all text embeddings into a single array
    if text_embeddings:
        # If multiple text columns, concatenate along the last axis
        text_embeddings_combined = np.concatenate(text_embeddings, axis=1)
        # Concatenate the processed chunk with the text embeddings
        chunk_values = np.hstack((chunk_values, text_embeddings_combined))

    # Concatenate the target data
    chunk_values = np.hstack((chunk_values, target_data))

    # Final check for any NaN values in the entire chunk
    if np.isnan(chunk_values).any():
        logger.error(f"NaN values detected in processed chunk {chunk_index}. Applying final fill.")
        chunk_values = np.nan_to_num(chunk_values, nan=0.0)

    # Save processed chunk as a NumPy array
    updated_chunk_path = os.path.join(updated_processed_data_dir, f'processed_chunk_updated_{chunk_index}.npz')
    logger.info(f"Saving processed chunk {chunk_index} to '{updated_chunk_path}'.")
    np.savez_compressed(updated_chunk_path, data=chunk_values)

    chunk_index += 1

logger.info("Data processing complete.")
