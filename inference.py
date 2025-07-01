import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data (only needed once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Paths
scaler_path = '/home/a/Fin Project/Financial Web Scraping/scalers.pkl'
ordinal_encodings_path = '/home/a/Fin Project/Financial Web Scraping/ordinal_encodings.pkl'
embedding_model_path = '/home/a/Fin Project/Financial Web Scraping/word2vec_word_level.model'
model_path = '/home/a/Fin Project/Financial Web Scraping/checkpoints/latest_checkpoint.pth'  # Adjusted to match the training script
column_types_path = '/home/a/Fin Project/Financial Web Scraping/column_types.json'

# Load scalers
try:
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    logger.info("Loaded scalers.")
except FileNotFoundError:
    logger.error(f"Scalers not found at {scaler_path}")
    exit(1)

# Load encodings
try:
    with open(ordinal_encodings_path, 'rb') as f:
        ordinal_encodings = pickle.load(f)
    logger.info("Loaded ordinal encodings.")
except FileNotFoundError:
    logger.error(f"Ordinal encodings not found at {ordinal_encodings_path}")
    exit(1)

# Load word embeddings
try:
    embedding_model = gensim.models.Word2Vec.load(embedding_model_path)
    embedding_dim = embedding_model.vector_size
    logger.info("Loaded word embeddings.")
except FileNotFoundError:
    logger.error(f"Word embeddings not found at {embedding_model_path}")
    exit(1)

# Load column types
try:
    with open(column_types_path, 'r') as f:
        columns_classification = json.load(f)
    logger.info("Loaded column types.")
except FileNotFoundError:
    logger.error(f"Column types JSON not found at {column_types_path}")
    exit(1)

numeric_columns = columns_classification.get('numeric_columns', [])
categorical_columns = columns_classification.get('categorical_columns', [])
text_columns = columns_classification.get('text_columns', [])
target_column = columns_classification.get('target_column', 'next_day_return')

# Identify date columns
date_columns = [col for col in numeric_columns + categorical_columns + text_columns if 'date' in col.lower()]

# Remove date columns from other categories
numeric_columns = [col for col in numeric_columns if col not in date_columns]
categorical_columns = [col for col in categorical_columns if col not in date_columns]
text_columns = [col for col in text_columns if col not in date_columns]

# Compute input_dim
num_numeric = len(numeric_columns)
num_categorical = len(categorical_columns)
num_date_features = len(date_columns) * 4  # Each date column produces 4 features
num_text_embeddings = len(text_columns) * embedding_dim
input_dim = num_numeric + num_categorical + num_date_features + num_text_embeddings

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Residual Block for Improved Gradient Flow
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return x + self.linear2(self.relu(self.linear1(x)))

# Improved Feedforward Regressor Model
class ImprovedFeedforwardRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super(ImprovedFeedforwardRegressor, self).__init__()
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.batch_norm(self.initial_layer(x)))
        for block in self.residual_blocks:
            x = block(x)
        x = self.dropout(x)
        return self.tanh(self.output_layer(x))

# Weight Initialization Function
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim = 64      # Adjusted to match the training script
num_layers = 4
nhead = 2
dropout = 0.1        # Adjusted to match the training script

model = ImprovedFeedforwardRegressor(input_dim, hidden_dim, num_layers, dropout=dropout).to(device)

try:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded trained model from latest checkpoint.")
except FileNotFoundError:
    logger.error(f"Model not found at {model_path}")
    exit(1)
except KeyError:
    logger.error("Key 'model_state_dict' not found in checkpoint.")
    exit(1)

model.eval()

# Function to process data for inference
def process_data_for_inference(df):
    # Initialize NLTK lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Handle date columns: Convert to datetime and extract features
    for date_col in date_columns:
        if date_col in df.columns:
            logger.info(f"Processing date column '{date_col}'.")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Extract date features
            df[f'{date_col}_year'] = df[date_col].dt.year.fillna(-1).astype(int)
            df[f'{date_col}_month'] = df[date_col].dt.month.fillna(-1).astype(int)
            df[f'{date_col}_day'] = df[date_col].dt.day.fillna(-1).astype(int)
            df[f'{date_col}_weekday'] = df[date_col].dt.weekday.fillna(-1).astype(int)
            # Drop the original date column
            df.drop(columns=[date_col], inplace=True)
        else:
            logger.warning(f"Date column '{date_col}' not found in data. Using default date features.")
            # Create default date features with -1
            df[f'{date_col}_year'] = -1
            df[f'{date_col}_month'] = -1
            df[f'{date_col}_day'] = -1
            df[f'{date_col}_weekday'] = -1

    # Handle numeric columns: Fill missing values and then transform using the corresponding MinMaxScaler
    for col in numeric_columns:
        if col in df.columns:
            logger.info(f"Processing numeric column '{col}'.")
            df[col] = df[col].fillna(0)
            df[[col]] = scalers[col].transform(df[[col]])
        else:
            logger.warning(f"Numeric column '{col}' not found in data. Filling with 0.")
            df[col] = 0
            df[[col]] = scalers[col].transform(df[[col]])
    
    # Handle categorical columns: Fill missing values and apply global ordinal encodings
    for col in categorical_columns:
        if col in df.columns:
            logger.info(f"Processing categorical column '{col}'.")
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].map(ordinal_encodings[col])
            df[col] = df[col].fillna(-1).astype(int)
        else:
            logger.warning(f"Categorical column '{col}' not found in data. Filling with -1.")
            df[col] = -1
    
    # Handle text columns with NLTK lemmatization and averaging embeddings
    text_embeddings = []
    default_embedding = np.zeros(embedding_dim)  # Default embedding for unknown tokens
    for col in text_columns:
        if col in df.columns:
            logger.info(f"Processing text column '{col}' with NLTK lemmatization and averaging embeddings.")
            df[col] = df[col].fillna('').astype(str)
            lemmatized_texts = []
            for text in df[col]:
                tokens = word_tokenize(text.lower())
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
            df.drop(columns=[col], inplace=True)
        else:
            logger.warning(f"Text column '{col}' not found in data. Using default embeddings.")
            text_embeddings.append(np.tile(default_embedding, (len(df), 1)))
    
    # Prepare the list of columns to keep
    new_date_feature_columns = []
    for date_col in date_columns:
        new_date_feature_columns.extend([
            f'{date_col}_year',
            f'{date_col}_month',
            f'{date_col}_day',
            f'{date_col}_weekday'
        ])
    processed_columns = new_date_feature_columns + numeric_columns + categorical_columns

    # Ensure only the processed columns are kept
    df = df[processed_columns]
    
    # Fill any remaining NaNs in processed columns with 0
    if df.isnull().values.any():
        logger.warning("NaN values found in processed numeric/categorical columns. Filling with 0.")
        df = df.fillna(0)
    
    # Convert the rest of the df to a NumPy array
    df_values = df.values.astype(np.float32)
    
    # Concatenate all text embeddings into a single array
    if text_embeddings:
        text_embeddings_combined = np.concatenate(text_embeddings, axis=1)
        df_values = np.hstack((df_values, text_embeddings_combined))
    
    # Final check for any NaN values in the entire data
    if np.isnan(df_values).any():
        logger.error("NaN values detected in processed data. Applying final fill.")
        df_values = np.nan_to_num(df_values, nan=0.0)
    
    return df_values


if __name__ == '__main__':
    # Load new data
    new_data_file = '/home/a/Fin Project/Financial Web Scraping/latest_data.csv'  # Replace with your new data file path
    try:
        new_data = pd.read_csv(new_data_file)
        logger.info(f"Loaded new data from {new_data_file}")
    except FileNotFoundError:
        logger.error(f"New data file not found at {new_data_file}")
        exit(1)
    
    # Extract and store the tickers
    if 'ticker' in new_data.columns:
        tickers = new_data['ticker'].tolist()
    else:
        logger.error("'ticker' column not found in new data.")
        exit(1)
    
    # Process data (excluding non-numeric columns)
    processed_data = process_data_for_inference(new_data)
    logger.info("Data processing for inference completed.")
    
    # Check if input dimensions match
    if processed_data.shape[1] != input_dim:
        logger.error(f"Input dimension mismatch: expected {input_dim}, got {processed_data.shape[1]}")
        exit(1)
    
    # Convert processed_data to torch tensor
    features = torch.tensor(processed_data, dtype=torch.float32).to(device)
    
    # Run inference in batches if data is large
    batch_size = 4
    num_samples = features.shape[0]
    predictions = []
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_features = features[start_idx:end_idx]
            outputs = model(batch_features)
            batch_predictions = torch.atanh(outputs).cpu().numpy().squeeze()  # Apply inverse of tanh
            predictions.extend(batch_predictions)

    try:
        data = pd.read_csv(new_data_file)
        logger.info(f"Loaded new data from {new_data_file}")
    except FileNotFoundError:
        logger.error(f"New data file not found at {new_data_file}")
        exit(1)
    
    # Attach predictions to the original DataFrame
    data['predicted_next_day_return'] = predictions
    
    # Save the updated DataFrame
    output_file = '/home/a/Fin Project/Financial Web Scraping/updated_data_with_predictions.csv'
    data.to_csv(output_file, index=False)
    
    logger.info(f"Inference completed. Updated data saved to {output_file}")


# # Define file paths
# latest_data_path = "latest_data.csv"  # Path to the latest_data.csv
# combined_data_path = "combined_data.csv"  # Path to the combined_data.csv

# # Load the 'ticker' column from both files
# latest_tickers = pd.read_csv(latest_data_path, usecols=["ticker"])["ticker"].dropna().unique()
# combined_tickers = pd.read_csv(combined_data_path, usecols=["ticker"])["ticker"].dropna().unique()

# # Find tickers in latest_data.csv but not in combined_data.csv
# missing_tickers = set(latest_tickers) - set(combined_tickers)

# # Count of missing tickers
# missing_count = len(missing_tickers)

# missing_count
