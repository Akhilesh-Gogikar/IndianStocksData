import sqlite3
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

DATABASE_FILE = '/home/a/Fin Project/Financial Web Scraping/equity_bse.db'

# Create a thread-local storage for database connections
thread_local = threading.local()

def get_db_connection():
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    return thread_local.connection

def get_price_data(ticker, start_date, end_date):
    conn = get_db_connection()
    query = '''
        SELECT date, prevClose FROM stock_data
        WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
        ORDER BY date ASC
    '''
    params = (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=['prevClose'], index=pd.to_datetime([]))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def compute_returns(df):
    df['return'] = df['prevClose'].pct_change()
    return df

def fit_arima_model(returns, order=(1, 0, 1)):
    """Fit an ARIMA model on the returns series."""
    if len(returns.dropna()) < 30:  # Use a shorter window to reduce memory usage
        return None
    try:
        model = ARIMA(returns.dropna().tail(180), order=order)  # Limit lookback window to 180 days
        model_fit = model.fit(method_kwargs={"warn_convergence": False}, disp=0)
        return model_fit
    except:
        return None

def predict_next_return(model_fit):
    """Forecast the next return (1-step ahead)."""
    if model_fit is None:
        return np.nan
    try:
        forecast = model_fit.forecast(steps=1)
        return forecast.iloc[0]
    except:
        return np.nan

if __name__ == "__main__":
    # Assuming we have a CSV with tickers
    ticker_df = pd.read_csv('/home/a/Fin Project/Financial Web Scraping/top_stocks_by_trading_volume.csv')  # Must have column 'tickerSymbol'
    tickers = ticker_df['tickerSymbol'].unique()

    # Define the overall backtest period (6 years for example)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6 * 365)  # approx 6 years

    # Load data for all tickers
    all_data = {}
    for ticker in tickers:
        df = get_price_data(ticker, start_date, end_date)
        if df.empty:
            continue
        # Resample to business days and forward fill missing
        df = df.asfreq('B')
        df['prevClose'] = df['prevClose'].ffill()
        df = compute_returns(df)
        all_data[ticker] = df

    if not all_data:
        print("No ticker data available.")
        exit()

    # Combine all into panels
    all_dates = sorted(set().union(*[df.index for df in all_data.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    prevClose_panel = pd.DataFrame(index=all_dates, columns=tickers)
    returns_panel = pd.DataFrame(index=all_dates, columns=tickers)

    for ticker, df in all_data.items():
        prevClose_panel[ticker] = df['prevClose']
        returns_panel[ticker] = df['return']

    prevClose_panel = prevClose_panel.ffill()
    returns_panel = returns_panel.ffill()

    # Backtest parameters
    initial_capital = 100000.0
    portfolio_values = []
    portfolio_value = initial_capital

    # Start backtest after 1 year of data to allow model warmup
    start_backtest_date = start_date + timedelta(days=365)
    start_backtest_date = max(start_backtest_date, all_dates[0])

    # We will hold positions overnight and close them out the next day.
    # That means for each day we form a new portfolio and realize returns the following day.
    for current_day in all_dates:
        if current_day < start_backtest_date:
            continue
        idx = np.searchsorted(all_dates, current_day)
        if idx + 1 >= len(all_dates):
            # No next day to realize returns
            break
        next_day = all_dates[idx + 1]

        # Predict returns for next_day using data up to current_day
        predictions = {}
        for ticker in tickers:
            history = returns_panel.loc[:current_day, ticker].dropna().tail(180)  # Limit lookback to 180 days
            if len(history) < 30:
                predictions[ticker] = np.nan
                continue
            model_fit = fit_arima_model(history)
            pred_return = predict_next_return(model_fit)
            predictions[ticker] = pred_return

        # Select top 100 opportunities by absolute predicted return
        valid_predictions = {t: p for t, p in predictions.items() if not np.isnan(p)}
        if len(valid_predictions) == 0:
            # No trades
            portfolio_values.append((next_day, portfolio_value))
            continue

        # Sort by absolute predicted return, pick top 100
        sorted_by_opportunity = sorted(valid_predictions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_100 = sorted_by_opportunity[:100]

        # From these 100, sort by prevClose price
        day_prices = prevClose_panel.loc[current_day, [t for t, _ in top_100]]
        opportunity_df = pd.DataFrame(top_100, columns=['ticker', 'pred_return'])
        opportunity_df['prevClose'] = [day_prices[t] for t in opportunity_df['ticker']]
        opportunity_df.sort_values('prevClose', inplace=True)

        # Allocate capital greedily
        current_capital = portfolio_value
        position_sizes = {t: 0 for t in opportunity_df['ticker']}

        while True:
            purchased_any = False
            for i, row in opportunity_df.iterrows():
                ticker = row['ticker']
                p_return = row['pred_return']
                price = row['prevClose']

                if current_capital >= price:
                    # Buy/sell one share
                    position_sizes[ticker] += 1 * (1 if p_return > 0 else -1)
                    current_capital -= price
                    purchased_any = True
            if not purchased_any:
                break

        # Realize returns the next day
        next_day_prices = prevClose_panel.loc[next_day, position_sizes.keys()]
        next_day_rets = returns_panel.loc[next_day, position_sizes.keys()]

        total_pl = 0.0
        for t, shares in position_sizes.items():
            if shares != 0:
                price_today = prevClose_panel.loc[current_day, t]
                realized_ret = returns_panel.loc[next_day, t]
                pl = shares * price_today * realized_ret
                total_pl += pl

        # Update portfolio value
        portfolio_value += total_pl
        portfolio_values.append((next_day, portfolio_value))

    # Convert results to a DataFrame
    results_df = pd.DataFrame(portfolio_values, columns=['date', 'portfolio_value'])
    results_df.set_index('date', inplace=True)
    results_df['return'] = results_df['portfolio_value'].pct_change()
    cumulative_return = results_df['portfolio_value'].iloc[-1] / initial_capital - 1.0

    print("Backtest completed.")
    print(f"Final Portfolio Value: {results_df['portfolio_value'].iloc[-1]:.2f}")
    print(f"Cumulative Return: {cumulative_return * 100:.2f}%")
