from breeze_connect import BreezeConnect
import os
import json
import datetime
import time
import random
from dateutil.parser import parse

# Load or save consecutive misses from/to a JSON file
CONSECUTIVE_MISSES_FILE = 'consecutive_misses.json'

def load_consecutive_misses(filename=CONSECUTIVE_MISSES_FILE):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

def save_consecutive_misses(consecutive_misses, filename=CONSECUTIVE_MISSES_FILE):
    with open(filename, 'w') as file:
        json.dump(consecutive_misses, file, indent=2)

# Function to load API credentials
def load_api_credentials(filename="breeze_api_creds.txt"):
    with open(filename, 'r') as file:
        lines = file.readlines()
        api_key = lines[0].strip().split(':')[-1]  # First line contains api_key
        secret_key = lines[1].strip().split(':')[-1]  # Second line contains secret_key
    return api_key, secret_key

# Load credentials
api_key, secret_key = load_api_credentials()

# Initialize Breeze API
breeze = BreezeConnect(api_key=api_key)
breeze.generate_session(api_secret=secret_key, session_token="48847809")

# Constants
PROCESSED_FOLDER = 'data/processed_results'
FINAL_RESULTS_FOLDER = 'data/breeze_results'

# Sets and lists to track various information
stocks_with_news = set()
missing_ohlcv_entries = []
invalid_date_entries = []
processed_files = 0
saved_files = 0

# Load consecutive misses from the JSON file
consecutive_misses = load_consecutive_misses()

def get_ohlcv_data(scrip_code, event_date):
    """
    Retrieves OHLCV data for the day following the event date from 12:00 AM to 3:55 PM.

    Args:
        scrip_code (str): The stock's scrip code.
        event_date (datetime): The date of the event.

    Returns:
        list: List containing OHLCV data from 12:00 AM to 3:55 PM the day after the event,
              or an empty list if no data is found.
    """
    start_time = (event_date + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0)
    end_time = start_time.replace(hour=15, minute=55)

    from_date = start_time.isoformat() + 'Z'
    to_date = end_time.isoformat() + 'Z'

    # Fetch OHLCV data from Breeze API
    ohlcv_data = breeze.get_historical_data(
        interval="1minute",
        from_date=from_date,
        to_date=to_date,
        stock_code=scrip_code,
        exchange_code="NSE",
        product_type="cash"
    )

    # Check if data is found; return empty list if "Success" key is None or empty
    if not ohlcv_data.get('Success'):
        print(f"No data found for {scrip_code} from {from_date} to {to_date}")
        return []

    # Structure data according to the response format
    return [
        {
            'date': entry['datetime'],
            'stock_code': entry['stock_code'],
            'open': float(entry['open']),
            'high': float(entry['high']),
            'low': float(entry['low']),
            'close': float(entry['close']),
            'volume': int(entry['volume'])
        } for entry in ohlcv_data['Success']
    ]

def process_files():
    global processed_files, saved_files
    os.makedirs(FINAL_RESULTS_FOLDER, exist_ok=True)

    for filename in os.listdir(PROCESSED_FOLDER):
        if filename.endswith('.json'):
            processed_files += 1
            file_path = os.path.join(PROCESSED_FOLDER, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file '{filename}': {e}")
                    continue

            extracted_data_with_ohlcv = []

            for item in data:
                scrip_code = item.get("data_before_date").get("securityInfo").get("info").get("ticker")
                exchange = item.get("data_before_date").get("securityInfo").get("info").get("exchange")
                
                # Skip if this code has been unavailable for 3 consecutive times
                if consecutive_misses.get(scrip_code, 0) >= 3:
                    print(f"Skipping {scrip_code} as it has no data for 3 consecutive attempts.")
                    continue

                if scrip_code:
                    stocks_with_news.add(scrip_code)

                date_str = item.get('news_date') or item.get('event_date')
                if date_str:
                    try:
                        event_date = parse(date_str)
                        ohlcv_data = get_ohlcv_data(scrip_code, event_date)
                        if ohlcv_data:
                            item['ohlcv_data'] = ohlcv_data
                            extracted_data_with_ohlcv.append(item)
                            # Reset consecutive miss count since data is found
                            consecutive_misses[scrip_code] = 0
                        else:
                            # Increment consecutive miss count if no data found
                            consecutive_misses[scrip_code] = consecutive_misses.get(scrip_code, 0) + 1
                            save_consecutive_misses(consecutive_misses)  # Save immediately after updating
                            missing_ohlcv_entries.append({
                                'filename': filename,
                                'sid': scrip_code,
                                'date': date_str,
                                'reason': "No OHLCV data found for specified time range."
                            })
                    except ValueError:
                        invalid_date_entries.append({
                            'filename': filename,
                            'sid': scrip_code,
                            'date': date_str,
                            'reason': "Invalid date format."
                        })
                        continue

                # Introduce a random sleep between 1 to 3 seconds to avoid rate limits
                time.sleep(random.uniform(0.1, 1.5))

            if not extracted_data_with_ohlcv:
                print(f"Issue: No OHLCV data for any entries in '{filename}'.")
            else:
                output_file_path = os.path.join(FINAL_RESULTS_FOLDER, f"final_{filename}")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    json.dump(extracted_data_with_ohlcv, output_file, indent=2)
                saved_files += 1
                print(f"Processed data saved to '{output_file_path}'")

    print("\nSummary:")
    print(f"  - Total files processed: {processed_files}")
    print(f"  - Total files saved: {saved_files}")
    print(f"  - Total stocks with news: {len(stocks_with_news)}")

if __name__ == "__main__":
    process_files()
