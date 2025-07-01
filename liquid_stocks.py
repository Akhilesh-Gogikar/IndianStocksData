import pandas as pd
import sqlite3
import numpy as np

# Connect to the SQLite database
DATABASE_FILE = '/home/a/Fin Project/Financial Web Scraping/equity_bse.db'

conn = sqlite3.connect(DATABASE_FILE)

# Constants for Sharpe ratio calculation
average_market_return = 0.12 / 252  # Assuming 252 trading days in a year
risk_free_rate = 0.0686 / 252

# Step 1: Get the top 50% of tickers by trading volume
try:
    query_top_volume = """
    SELECT scripCode, tickerSymbol, AVG(totalSharesTraded) AS avg_trading_volume
    FROM stock_data
    WHERE scripType = 'Equity'
    GROUP BY scripCode, tickerSymbol
    HAVING avg_trading_volume IS NOT NULL
    ORDER BY avg_trading_volume DESC
    """
    
    df_volume = pd.read_sql_query(query_top_volume, conn)
    median_volume = df_volume['avg_trading_volume'].median()
    df_top_volume = df_volume[(df_volume['avg_trading_volume'] >= median_volume) & (df_volume['tickerSymbol'].notna())]
    top_tickers = df_top_volume['scripCode'].tolist()
    top_tickers_syms = df_top_volume['tickerSymbol'].tolist()

    # Zip ticker symbols and stock codes
    tickers_and_symbols = list(zip(top_tickers, top_tickers_syms))

    # Step 2: Fetch previous close values for each ticker and calculate returns and Sharpe ratios
    sharpe_data = []
  
    for ticker, symbol in tickers_and_symbols:
        query_returns = f"""
        SELECT date, prevClose
        FROM stock_data
        WHERE scripCode = '{ticker}' AND prevClose > 0 AND scripType = 'Equity'
        ORDER BY date
        """
        df_ticker = pd.read_sql_query(query_returns, conn)
        
        # Calculate daily returns
        df_ticker['daily_return'] = df_ticker['prevClose'].pct_change()
        avg_daily_return = df_ticker['daily_return'].mean()
        std_daily_return = df_ticker['daily_return'].std()
        
        # Calculate Sharpe ratio
        if std_daily_return is not None and std_daily_return != 0:
            sharpe_ratio = (avg_daily_return - risk_free_rate) / std_daily_return
        else:
            sharpe_ratio = None
        
        # Append data
        if sharpe_ratio is not None:
            avg_trading_volume = df_top_volume[df_top_volume['scripCode'] == ticker]['avg_trading_volume'].values[0]
            sharpe_data.append((ticker, symbol, avg_daily_return, std_daily_return, sharpe_ratio, avg_trading_volume))

    # Create DataFrame from sharpe_data
    df_sharpe = pd.DataFrame(sharpe_data, columns=['scripCode', 'tickerSymbol', 'avg_daily_return', 'std_daily_return', 'sharpe_ratio', 'avg_trading_volume'])

    # Sort by Sharpe ratio in descending order
    df_sharpe = df_sharpe.sort_values(by='sharpe_ratio', ascending=False)

    # Select the top 1000 stocks with the highest Sharpe ratio
    df_top_1000 = df_sharpe.head(1000)

    # Zip ticker symbols and stock codes
    top_1000_zipped = list(zip(df_top_1000['scripCode'], df_top_volume.set_index('scripCode').loc[df_top_1000['scripCode']]['tickerSymbol']))

    # Save the result to a CSV file
    output_file = '/home/a/Fin Project/Financial Web Scraping/top_stocks_by_trading_volume.csv'  # Update with your desired output path
    df_top_1000.to_csv(output_file, index=False)
    print(f"CSV file saved at: {output_file}")
    print("Top 1000 Ticker Symbols and Stock Codes:")
    print(top_1000_zipped)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the connection
    conn.close()
