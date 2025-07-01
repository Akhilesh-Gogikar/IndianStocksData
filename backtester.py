import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_processor import DataProcessor
from ml_model import MLModel
from tqdm import tqdm
import joblib
import numpy as np
from io import StringIO

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
                        print("Skipping empty file: {}".format(file_path))
            except Exception as e:
                print("Error processing file {}: {}".format(file_path, e))
                continue
        print("Total entries in index_map: {}".format(len(index_map)))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if idx >= len(self.index_map):
            raise IndexError("Index {} out of range for dataset with length {}".format(idx, len(self.index_map)))
        
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
                        print("ParserError at line {} in file {}: {}".format(line_number, file_path, e))
                        return None  # Skip malformed lines
                    if data is not None and not data.empty:
                        # Ensure that data_row is a Series
                        data_row = data.iloc[0]
                        result = self.data_processor.process_row(data_row)  # Pass Series
                        if result is None:
                            return None
                        text_tensor, num_tensor, target = result
                        return text_tensor, num_tensor, target, data_row['date'], data_row['ticker']
        # If we reach here, we didn't return data
        # Instead of raising IndexError, return None
        print("Warning: Line {} not found in file {}".format(line_number, file_path))
        return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None, None, None
    text_tensors, num_tensors, targets, dates, scrip_codes = zip(*batch)
    text_tensors = torch.stack(text_tensors)
    num_tensors = torch.stack(num_tensors)
    targets = torch.stack(targets)
    return text_tensors, num_tensors, targets, dates, scrip_codes

class Backtester:
    def __init__(self):
        self.results_df = pd.DataFrame(columns=['date', 'ticker', 'prediction', 'actual'])

    def backtest(self, predictions, actuals, dates, scrip_codes):
        # Convert lists to NumPy arrays for easier calculations
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        dates = np.array(dates)
        scrip_codes = np.array(scrip_codes)

        # Save results to DataFrame
        self.results_df = pd.DataFrame({
            'date': dates,
            'ticker': scrip_codes,
            'prediction': predictions,
            'actual': actuals
        })

        # Implement your trading strategy logic here
        # For example, generate signals based on predictions
        signals = np.where(predictions > 0, 1, -1)  # Buy if prediction > 0, else sell

        # Assume returns are the actual percentage changes
        returns = actuals

        # Calculate strategy returns
        strategy_returns = signals * returns

        # Calculate cumulative returns
        cumulative_returns = np.cumsum(strategy_returns)

        # Calculate performance metrics
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        sharpe_ratio = (
            np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            if np.std(strategy_returns) != 0 else 0
        )
        max_drawdown = (
            np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
            if len(cumulative_returns) > 0 else 0
        )

        # Compile results
        results = {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
        }

        return results

def main():
    # Define path to test data folder
    test_folder = 'test'

    # Get list of CSV files in the test folder
    test_files = glob.glob(os.path.join(test_folder, "*.csv"))

    # Initialize data processor
    data_processor = DataProcessor()

    # Load the DataProcessor's vectorizer and encoders
    vectorizer_save_path = 'vectorizer.joblib'
    data_processor.vectorizer = joblib.load(vectorizer_save_path)
    print("Vectorizer loaded from {}".format(vectorizer_save_path))

    # Save the DataProcessor's scaler
    scaler_save_path = 'scaler.joblib'
    joblib.dump({'scaler':data_processor.num_scaler, 'num_scaler':len(data_processor.expanded_numerical_columns)}, scaler_save_path)
    print("Scaler saved to {}".format(scaler_save_path))

    data_processor._initialized = True

    # Create Dataset and DataLoader for test data
    test_dataset = CSVRowDataset(test_files, data_processor)
    batch_size = 16  # Adjust based on memory constraints
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Load the trained model
    model_save_path = 'ml_model.pth'
    ml_model = MLModel(
        text_input_size=len(data_processor.vectorizer.get_feature_names()),
        num_input_size=len(data_processor.expanded_numerical_columns),
        embedding_dim=32  # Adjust based on your constraints
    )
    ml_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ml_model.to(device)
    ml_model.eval()
    print("Model loaded from {}".format(model_save_path))

    # Evaluate the model on the test data
    print("Evaluating the model on the test data...")
    predictions = []
    actuals = []
    dates = []
    scrip_codes = []

    # Create a tqdm progress bar for the test loop
    test_loader_bar = tqdm(test_loader, desc="Testing", ncols=100)

    with torch.no_grad():
        for batch_idx, (text_test_batch, num_test_batch, y_test_batch, batch_dates, batch_scrip_codes) in enumerate(test_loader_bar):
            if text_test_batch is None:
                continue
            text_test_batch = text_test_batch.to(device)
            num_test_batch = num_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            test_outputs = ml_model(text_test_batch, num_test_batch)
            predictions.extend(test_outputs.cpu().numpy().flatten())
            actuals.extend(y_test_batch.cpu().numpy().flatten())
            dates.extend(batch_dates)
            scrip_codes.extend(batch_scrip_codes)

    # Convert lists to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    dates = np.array(dates)
    scrip_codes = np.array(scrip_codes)

    # Compute evaluation metrics
    mse = np.mean((actuals - predictions) ** 2) if len(actuals) > 0 else float('inf')
    mae = np.mean(np.abs(actuals - predictions)) if len(actuals) > 0 else float('inf')
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)) if len(actuals) > 0 else float('-inf')

    print("\nTest Set Evaluation Metrics:")
    print("Mean Squared Error (MSE): {:.4f}".format(mse))
    print("Mean Absolute Error (MAE): {:.4f}".format(mae))
    print("R-squared (R^2): {:.4f}".format(r2))

    # Perform backtesting using the predictions
    backtester = Backtester()
    print("\nRunning backtesting...")
    backtesting_results = backtester.backtest(predictions, actuals, dates, scrip_codes)

    # Output the backtesting results
    print("\nBacktesting Results:")
    for key, value in backtesting_results.items():
        print("{}: {}".format(key, value))

    # Save the backtesting results to a CSV file
    backtester.results_df.to_csv('backtesting_results.csv', index=False)
    print("Backtesting results saved to backtesting_results.csv")

if __name__ == "__main__":
    main()
