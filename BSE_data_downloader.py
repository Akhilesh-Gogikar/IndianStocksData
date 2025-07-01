import requests
import zipfile
import io
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Constants
DATABASE_FILE = '/home/a/Fin Project/Financial Web Scraping/equity_bse.db'
OLD_URL_TEMPLATE = "https://www.bseindia.com/download/BhavCopy/Equity/EQ{date}_CSV.ZIP"
NEW_URL_TEMPLATE = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{date}_F_0000.CSV"

# Connect to SQLite database
conn = sqlite3.connect(DATABASE_FILE)
cursor = conn.cursor()

def reset_database():
    """Drops and recreates the stock_data table to start fresh and accommodate both data formats."""
    cursor.execute('DROP TABLE IF EXISTS stock_data')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            date TEXT,
            scripCode TEXT,
            tickerSymbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            last REAL,
            prevClose REAL,
            totalTrades INTEGER,
            totalSharesTraded INTEGER,
            netTurnover REAL,
            scripType TEXT,
            securityID TEXT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_scripCode ON stock_data (scripCode);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON stock_data (date);')
    conn.commit()

def insert_data(df, date, is_new_format):
    """Inserts rows from the DataFrame into the stock_data table."""
    for _, row in df.iterrows():
        if is_new_format:
            cursor.execute('''
                INSERT INTO stock_data (date, scripCode, tickerSymbol, open, high, low, close, prevClose, totalTrades, totalSharesTraded, netTurnover, scripType, securityID)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                row.get('FinInstrmId'),       # Scrip Code (Financial Instrument ID)
                row.get('TckrSymb'),          # Ticker Symbol
                row.get('OpnPric'),           # Open price
                row.get('HghPric'),           # High price
                row.get('LwPric'),            # Low price
                row.get('ClsgPric'),          # Close price
                row.get('PrvsClsgPric'),      # Previous close price
                row.get('TtlNbOfTxsExctd'),   # Total trades
                row.get('TtlTradgVol'),       # Total shares traded
                row.get('TtlTrfVal'),         # Net turnover
                'Equity',                     # Scrip Type (assumed as Equity)
                row.get('ISIN')               # Security ID (ISIN)
            ))
        else:
            cursor.execute('''
                INSERT INTO stock_data (date, scripCode, tickerSymbol, open, high, low, close, last, prevClose, totalTrades, totalSharesTraded, netTurnover, scripType, securityID)
                VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                row.get('SC_CODE'),
                row.get('OPEN'),
                row.get('HIGH'),
                row.get('LOW'),
                row.get('CLOSE'),
                row.get('LAST'),
                row.get('PREVCLOSE'),
                row.get('NO_OF_TRADES'),
                row.get('NO_OF_SHRS'),
                row.get('NET_TURNOV'),
                'Equity',
                row.get('ISIN')
            ))
    conn.commit()

def download_and_process_data(date):
    formatted_date_new = date.strftime('%Y%m%d')
    formatted_date_old = date.strftime('%d%m%y')
    url_new = NEW_URL_TEMPLATE.format(date=formatted_date_new)
    url_old = OLD_URL_TEMPLATE.format(date=formatted_date_old)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url_new, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        insert_data(df, date.strftime('%Y-%m-%d'), is_new_format=True)
        print("Data stored for {} in new format".format(date.strftime('%Y-%m-%d')))
    except requests.HTTPError:
        print("New format not available for {}. Trying old format.".format(date.strftime('%Y-%m-%d')))
        try:
            response = requests.get(url_old, headers=headers)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for filename in z.namelist():
                    with z.open(filename) as file:
                        df = pd.read_csv(file)
                        break
            insert_data(df, date.strftime('%Y-%m-%d'), is_new_format=False)
            print("Data stored for {} in old format".format(date.strftime('%Y-%m-%d')))
        except requests.HTTPError as e:
            print("Failed to download data for {} in both formats due to HTTP Error: {}".format(date.strftime('%Y-%m-%d'), e))
        except zipfile.BadZipFile:
            print("Corrupted ZIP file for date: {}".format(date.strftime('%Y-%m-%d')))
        except pd.errors.ParserError:
            print("Parsing error with the CSV file for date: {}".format(date.strftime('%Y-%m-%d')))

def get_last_date():
    cursor.execute("SELECT MAX(date) FROM stock_data")
    result = cursor.fetchone()
    if result[0]:
        return datetime.strptime(result[0], '%Y-%m-%d')
    return datetime(2012, 12, 28)

def main():
    # reset_database()
    last_date = get_last_date()
    current_date = last_date + timedelta(days=1)
    end_date = datetime.now().date()

    while current_date.date() <= end_date:
        if current_date.weekday() < 5:  # Only process data on weekdays
            try:
                download_and_process_data(current_date)
            except Exception as e:
                print(e)
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
    conn.close()
