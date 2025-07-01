import pandas as pd
import json

def segregate_columns_large_csv(csv_file, json_file, chunksize=10000, nan_threshold=90):
    """
    Segregate columns of a large CSV file into numeric, categorical, and text columns,
    remove columns with 90% or more NaN values, and store the column names in a JSON file.

    Parameters:
    csv_file (str): Path to the input CSV file.
    json_file (str): Path to the output JSON file where column names will be stored.
    chunksize (int): Number of rows per chunk to read at a time.
    nan_threshold (float): Percentage threshold to remove columns with NaN values.
    """
    # Initialize dictionaries to hold data types and counts for each column
    column_types = {}
    non_nan_counts = {}
    total_rows = 0

    # Initialize sets to collect column names
    numeric_columns = set()
    categorical_columns = set()
    text_columns = set()

    # Read the CSV file in chunks
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        # Update total number of rows
        total_rows += len(chunk)

        # For the first chunk, initialize column_types and non_nan_counts
        if not column_types:
            for column in chunk.columns:
                column_types[column] = {'is_numeric': True, 'word_counts': []}
                non_nan_counts[column] = 0  # Initialize non-NaN counts

        # Process each column in the chunk
        for column in chunk.columns:
            col_data = chunk[column]
            # Update non-NaN count
            non_nan_counts[column] += col_data.notna().sum()

            # Check if the column is numeric
            if column_types[column]['is_numeric']:
                try:
                    pd.to_numeric(col_data.dropna())
                except ValueError:
                    column_types[column]['is_numeric'] = False

            # If not numeric, collect word counts
            if not column_types[column]['is_numeric']:
                word_counts = col_data.dropna().astype(str).apply(lambda x: len(x.split()))
                column_types[column]['word_counts'].extend(word_counts.tolist())

    # Determine columns to exclude based on NaN percentage
    columns_to_exclude = set()
    for column, count in non_nan_counts.items():
        nan_percentage = 100 * (total_rows - count) / total_rows
        if nan_percentage >= nan_threshold:
            columns_to_exclude.add(column)

    # After processing all chunks, classify columns
    for column, info in column_types.items():
        if column in columns_to_exclude:
            continue  # Skip columns with high NaN percentage

        if info['is_numeric']:
            numeric_columns.add(column)
        else:
            # Calculate the average word count
            if info['word_counts']:
                avg_words = sum(info['word_counts']) / len(info['word_counts'])
            else:
                avg_words = 0  # If the column only has NaNs

            if avg_words < 5:
                categorical_columns.add(column)
            else:
                text_columns.add(column)

    # Store the column names in a dictionary
    column_types_output = {
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
        "text_columns": list(text_columns)
    }

    # Write the column names to a JSON file
    with open(json_file, 'w') as f:
        json.dump(column_types_output, f, indent=4)

    print(f"Column names have been stored in {json_file}.")
    print(f"Excluded columns due to high NaN percentage ({nan_threshold}% or more):")
    print(columns_to_exclude)
    
# Example usage:
if __name__ == "__main__":
    # Replace 'your_large_file.csv' with the path to your CSV file
    segregate_columns_large_csv('latest_data.csv', 'column_types.json', chunksize=10000)
