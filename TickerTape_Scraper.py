import requests
import re
import json
import os
import time
import random
import shutil
from datetime import datetime

def fetch_html(ticker_suffix):
    url = "https://www.tickertape.in/stocks/{}".format(ticker_suffix)
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to fetch page for {}: {}".format(ticker_suffix, response.status_code))
            return None
    except requests.exceptions.RequestException as e:
        print("Error fetching page for {}: {}".format(ticker_suffix, e))
        return None

def extract_json_data(html_content):
    # Define the start and end markers
    start_marker = '<script id="__NEXT_DATA__" type="application/json">'
    end_marker = '</script>'
    # Use regex to extract the JSON content
    pattern = re.escape(start_marker) + '(.*?)' + re.escape(end_marker)
    match = re.search(pattern, html_content, re.DOTALL)
    if match:
        json_text = match.group(1)
        # Parse the JSON string
        data = json.loads(json_text)
        return data
    else:
        print("JSON data not found in the HTML content.")
        return None

def save_data_to_file(data, ticker_suffix):
    # Ensure the 'data' and 'archive' directories exist
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('archive'):
        os.makedirs('archive')
    
    try:
        # Extract 'pageProps' from 'data'
        page_props = data['props']['pageProps']
        
        # Remove 'dehydratedState' if it exists
        if 'dehydratedState' in page_props:
            del page_props['dehydratedState']
        
        # File path for the raw JSON data
        raw_json_file = "/home/a/Fin Project/Financial Web Scraping/data/{}_raw.json".format(ticker_suffix)
        
        # Archive existing raw JSON file if it exists
        if os.path.exists(raw_json_file):
            # Create an archive file name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_file = "archive/{}_raw_{}.json".format(ticker_suffix, timestamp)
            shutil.copy2(raw_json_file, archive_file)
            print("Archived existing raw data to {}".format(archive_file))
        
        # Save the cleaned 'pageProps' data to the raw JSON file
        with open(raw_json_file, 'w', encoding='utf-8') as f:
            json.dump(page_props, f, indent=4)
        print("Updated raw data saved to {}".format(raw_json_file))
    except KeyError as e:
        print("Key not found during data processing: {}".format(e))
    except Exception as e:
        print("Error saving data for {}: {}".format(ticker_suffix, e))

def scrape_ticker_data(ticker_suffix):
    html_content = fetch_html(ticker_suffix)
    if html_content:
        data = extract_json_data(html_content)
        if data:
            save_data_to_file(data, ticker_suffix)
        else:
            print("No data extracted for {}.".format(ticker_suffix))
    else:
        print("No HTML content fetched for {}.".format(ticker_suffix))

def read_tickers_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        tickers = json.load(f)
    return tickers

# Main function to read tickers and process them
if __name__ == "__main__":
    json_file = '/home/a/Fin Project/Financial Web Scraping/full-company-list.json'
    tickers = read_tickers_from_json(json_file)
    for ticker in tickers:
        name = ticker.get('name')
        ticker_type = ticker.get('type')
        subdirectory = ticker.get('subdirectory')

        if not subdirectory:
            print("Subdirectory missing for ticker: {}".format(name))
            continue

        print("\nProcessing {} ({})...".format(name, subdirectory))
        scrape_ticker_data(subdirectory)

        # Random sleep between 0.5 and 1.5 seconds
        sleep_duration = random.uniform(0.5, 1.5)
        print("Sleeping for {:.2f} seconds...".format(sleep_duration))
        time.sleep(sleep_duration)
