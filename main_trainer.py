import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from data_processor import DataProcessor
from ml_model import MLModel
from backtester import Backtester
import pandas as pd
from io import StringIO
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np



class CSVRowDataset(Dataset):
    """Custom Dataset to read CSV files row by row without loading the entire file into memory."""
    def __init__(self, file_list, data_processor):
        self.file_list = file_list
        self.data_processor = data_processor
        self.index_map = self._create_index_map()

    def _create_index_map(self):
        """Creates a mapping from a global index to (file_path, line_number), skipping empty or invalid files."""
        index_map = []
        for file_path in self.file_list:
            try:
                # Open and read the file in a safe way
                with open(file_path, 'r') as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract 1 for header
                    if line_count > 0:
                        for line_number in range(1, line_count + 1):  # Start from 1 to skip header
                            index_map.append((file_path, line_number))
                    else:
                        print(f"Skipping empty file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        print(f"Total entries in index_map: {len(index_map)}")
        return index_map



    def _initialize_data_processor(self):
        # Initialize vectorizer using text data from training files
        combined_df = pd.DataFrame()  # Create an empty DataFrame to accumulate all chunks

        for file_path in self.file_list:
            try:
                # Read file in chunks and concatenate to the main DataFrame
                for chunk in pd.read_csv(file_path, chunksize=1000):
                    combined_df = pd.concat([combined_df, chunk], ignore_index=True)
                    
            except pd.errors.EmptyDataError:
                print(f"Warning: {file_path} is empty or has no valid data.")
                continue
            except FileNotFoundError:
                print(f"Error: File {file_path} not found.")
                continue

        # Once all files are read and combined, initialize vectorizer and encoders
        if not combined_df.empty:
            self.data_processor._initialize_vectorizer_and_encoders(combined_df)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.index_map)}")
        
        file_path, line_number = self.index_map[idx]
        with open(file_path, 'r') as f:
            # Read header to get column names
            header_line = f.readline()
            header = header_line.strip().split(',')
            # Skip to the desired line
            for i, line in enumerate(f):
                if i == line_number - 1:
                    try:
                        # Clean the line
                        line = line.strip()
                        # Combine header and line
                        data_str = header_line + line + '\n'
                        # Read the data
                        data = pd.read_csv(
                            StringIO(data_str),
                            sep=',',
                            header=0,
                            error_bad_lines=False,  # Skip bad lines
                            warn_bad_lines=True     # Optional: Warn about bad lines
                        )
                    except pd.errors.ParserError as e:
                        print(f"ParserError at line {line_number} in file {file_path}: {e}")
                        return None  # Skip malformed lines
                    if data is not None and not data.empty:
                        # Ensure that data_row is a Series
                        data_row = data.iloc[0]
                        result = self.data_processor.process_row(data_row)  # Pass Series
                        if result is None:
                            return None
                        text_tensor, num_tensor, target = result
                        return text_tensor, num_tensor, target
        # If we reach here, we didn't return data
        # Instead of raising IndexError, return None
        print(f"Warning: Line {line_number} not found in file {file_path}")
        return None




def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None, None
    text_tensors, num_tensors, targets = zip(*batch)
    text_tensors = torch.stack(text_tensors)
    num_tensors = torch.stack(num_tensors)
    targets = torch.stack(targets)
    return text_tensors, num_tensors, targets



def main():
    # Define paths to data folders
    train_folder = 'train'
    val_folder = 'val'
    test_folder = 'test'

    # Get list of CSV files
    train_files = glob.glob(os.path.join(train_folder, "*.csv"))
    val_files = glob.glob(os.path.join(val_folder, "*.csv"))
    test_files = glob.glob(os.path.join(test_folder, "*.csv"))

    # Initialize data processor
    data_processor = DataProcessor()

    # Read all training data and initialize data_processor
    combined_df = pd.DataFrame()
    for file_path in train_files:
        try:
            for chunk in pd.read_csv(file_path, chunksize=1000):
                combined_df = pd.concat([combined_df, chunk], ignore_index=True)
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty or has no valid data.")
            continue
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            continue

    # Initialize vectorizer and encoders using the combined training data
    if not combined_df.empty:
        data_processor._initialize_vectorizer_and_encoders(combined_df)
    else:
        print("Error: Training data is empty. Cannot initialize DataProcessor.")
        return

    # Create Datasets
    train_dataset = CSVRowDataset(train_files, data_processor)
    val_dataset = CSVRowDataset(val_files, data_processor)
    test_dataset = CSVRowDataset(test_files, data_processor)

    # Create DataLoaders
    batch_size = 32  # Adjust batch size based on memory constraints
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
      # Not needed anymore
    ml_model = MLModel(
        text_input_size=len(data_processor.vectorizer.get_feature_names()),
        num_input_size=len(data_processor.expanded_numerical_columns),
        embedding_dim=32  # Adjust based on your constraints
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ml_model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 5  # Adjust as needed
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        ml_model.train()
        total_loss = 0

        # Create a tqdm progress bar for the training loop
        train_loader_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", leave=False, ncols=100)

        for batch_idx, (text_batch, num_batch, y_batch) in enumerate(train_loader_bar):
            if text_batch is None:
                continue
            text_batch = text_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = ml_model(text_batch, num_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate and update the rolling average training loss
            avg_loss = total_loss / (batch_idx + 1)
            train_loader_bar.set_postfix(avg_loss=avg_loss)

        print(f'Training Loss after Epoch {epoch + 1}: {avg_loss:.4f}')

        # Validation loop
        ml_model.eval()
        total_val_loss = 0

        val_loader_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", leave=False, ncols=100)

        with torch.no_grad():
            for batch_idx, (text_val_batch, num_val_batch, y_val_batch) in enumerate(val_loader_bar):
                if text_val_batch is None:
                    continue
                text_val_batch = text_val_batch.to(device)
                num_val_batch = num_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                val_outputs = ml_model(text_val_batch, num_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / (batch_idx + 1)
                val_loader_bar.set_postfix(avg_val_loss=avg_val_loss)


        print(f'Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}\n')

    print("Training complete.")
    # Testing loop
    # Testing loop
    print("Evaluating the model on the test data...")
        # After training is complete
    # Save the model
    model_save_path = 'ml_model.pth'
    torch.save(ml_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the DataProcessor's vectorizer and encoders
    vectorizer_save_path = 'vectorizer.joblib'
    joblib.dump(data_processor.vectorizer, vectorizer_save_path)
    print(f"Vectorizer saved to {vectorizer_save_path}")

    # Save the DataProcessor's scaler
    scaler_save_path = 'scaler.joblib'
    joblib.dump({'scaler':data_processor.num_scaler, 'num_scaler':len(data_processor.expanded_numerical_columns)}, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

        
    ml_model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_test_batch, num_test_batch, y_test_batch in test_loader:
            if X_test_batch is None:
                continue
            X_test_batch = X_test_batch.to(device)
            num_test_batch = num_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            test_outputs = ml_model(X_test_batch, num_test_batch)
            predictions.extend(test_outputs.cpu().numpy().flatten())
            actuals.extend(y_test_batch.cpu().numpy().flatten())


    # Convert lists to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Compute evaluation metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("Test Set Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R^2): {r2:.4f}")

    # Perform backtesting using the predictions
    # Perform backtesting using the predictions
    backtester = Backtester()
    print("Running backtesting...")
    backtesting_results = backtester.backtest(predictions, actuals, test_data)

    # Output the backtesting results
    print("Backtesting Results:")
    for key, value in backtesting_results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
