import os
import pandas as pd
from datetime import datetime

# Load CSV file with ticker and predicted_next_day return columns
def load_csv(csv_filename):
    return pd.read_csv(csv_filename)

# Load TXT file as CSV with ScripCode and ScripID columns
def load_txt_as_csv(txt_filename):
    return pd.read_csv(txt_filename)

# Match ticker to ScripID and get corresponding ScripCode, then save the desired columns to a new CSV file
def match_and_save(ticker_csv, scrip_txt, output_csv):
    # Load dataframes
    df_ticker = load_csv(ticker_csv)
    df_scrip = load_txt_as_csv(scrip_txt)

    # Merge dataframes based on ticker column from df_ticker and ScripID column from df_scrip
    df_merged = pd.merge(df_ticker, df_scrip, how='left', left_on='ticker', right_on='ScripID')

    # Filter out rows where no match is found
    df_filtered = df_merged.dropna(subset=['ShortName'])

    # Add the current date as a new column
    current_date = datetime.now().strftime("%Y-%m-%d")
    df_filtered['Date'] = current_date

    # Select desired columns for the output file
    df_output = df_filtered[['ShortName', 'ScripID', 'predicted_next_day_return', 'Date']]

    # Save the result to a new CSV file
    df_output.to_csv(output_csv, index=False)
    
    # Save the result with a date appendage in the archive folder
    archive_folder = "/home/a/Fin Project/Financial Web Scraping/Archive"
    os.makedirs(archive_folder, exist_ok=True)
    archive_file = os.path.join(archive_folder, f"trade_data_{current_date}.csv")
    df_output.to_csv(archive_file, index=False)
    print(f"Archived results to {archive_file}")

if __name__ == "__main__":
    ticker_csv_filename = "/home/a/Fin Project/Financial Web Scraping/updated_data_with_predictions.csv"  # Replace with the name of your CSV file
    scrip_txt_filename = "/home/a/Fin Project/Financial Web Scraping/BSEScripMaster.txt"    # Replace with the name of your TXT file
    output_csv_filename = "/home/a/Fin Project/Financial Web Scraping/trade_data.csv"  # Replace with the desired name of the output CSV file

    # Run the match and save function
    match_and_save(ticker_csv_filename, scrip_txt_filename, output_csv_filename)
    print(f"Saved results to {output_csv_filename}")
