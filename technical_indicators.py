import pandas as pd
import numpy as np
import sqlite3
from datetime import timedelta, datetime
import logging

def get_rsi(cursor, ticker, event_date, period=14):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'rsi': 0}

        closes = [row[1] for row in rows]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        return {'rsi': rsi}
    except Exception as e:
        print(f"Error fetching RSI: {e}")
        return {'rsi': 0}

def get_macd(cursor, ticker, event_date, short_window=12, long_window=26, signal_window=9):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=long_window * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < long_window:
            return {'macd': 0, 'macd_signal': 0}

        closes = pd.Series([row[1] for row in rows])
        short_ema = closes.ewm(span=short_window, min_periods=short_window).mean()
        long_ema = closes.ewm(span=long_window, min_periods=long_window).mean()
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=signal_window, min_periods=signal_window).mean()

        return {'macd': macd.iloc[-1], 'macd_signal': macd_signal.iloc[-1]}
    except Exception as e:
        print(f"Error fetching MACD: {e}")
        return {'macd': 0, 'macd_signal': 0}

def get_bollinger_bands(cursor, ticker, event_date, window=20):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=window * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < window:
            return {'bollinger_upper': 0, 'bollinger_middle': 0, 'bollinger_lower': 0}

        closes = pd.Series([row[1] for row in rows])
        rolling_mean = closes.rolling(window=window).mean().iloc[-1]
        rolling_std = closes.rolling(window=window).std().iloc[-1]

        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        return {'bollinger_upper': upper_band, 'bollinger_middle': rolling_mean, 'bollinger_lower': lower_band}
    except Exception as e:
        print(f"Error fetching Bollinger Bands: {e}")
        return {'bollinger_upper': 0, 'bollinger_middle': 0, 'bollinger_lower': 0}

def get_atr(cursor, ticker, event_date, period=14):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)  # Need enough data for calculation

        cursor.execute('''
            SELECT date, high, low, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'atr': 0}

        highs = pd.Series([row[1] for row in rows])
        lows = pd.Series([row[2] for row in rows])
        closes = pd.Series([row[3] for row in rows])

        high_low = highs - lows
        high_close = abs(highs - closes.shift(1))
        low_close = abs(lows - closes.shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return {'atr': atr}
    except Exception as e:
        print(f"Error fetching ATR: {e}")
        return {'atr': 0}
    

def get_recent_trend_features(cursor, ticker, event_date):
    try:
        # Exclude the event date to prevent data leakage
        end_date = event_date - timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(days=6)  # Total of 7 days

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date_str))

        rows = cursor.fetchall()
        prev_closes = [row[0] for row in rows if row[0] is not None]

        if len(prev_closes) < 2:
            return {
                'recent_price_change_pct': 0,
                'recent_avg_volume': 0,
                'recent_avg_volatility': 0,
            }

        price_change_pct = ((prev_closes[-1] - prev_closes[0]) / prev_closes[0]) * 100

        return {
            'recent_price_change_pct': price_change_pct,
            'recent_avg_volume': 0,  # Placeholder as volume data is not used
            'recent_avg_volatility': 0,  # Placeholder as volatility is not calculated
        }
    except Exception as e:
        print(f"Error fetching recent trend features: {e}")
        return {
            'recent_price_change_pct': 0,
            'recent_avg_volume': 0,
            'recent_avg_volatility': 0,
        }

def get_mid_term_trend_features(cursor, ticker, event_date):
    try:
        # Exclude the event date
        end_date = event_date - timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(days=29)  # Total of 30 days

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date_str))

        rows = cursor.fetchall()
        prev_closes = [row[0] for row in rows if row[0] is not None]

        if not prev_closes:
            return {
                'ma_15': 0,
                'ma_30': 0,
                'vwap': 0,  # Placeholder as VWAP is not calculated
            }

        # Calculate moving averages
        ma_15 = sum(prev_closes[-15:]) / min(len(prev_closes), 15)
        ma_30 = sum(prev_closes[-30:]) / min(len(prev_closes), 30)

        return {
            'ma_15': ma_15,
            'ma_30': ma_30,
            'vwap': 0  # Placeholder as VWAP is not calculated
        }
    except Exception as e:
        print(f"Error fetching mid-term trend features: {e}")
        return {
            'ma_15': 0,
            'ma_30': 0,
            'vwap': 0,
        }

def get_long_term_trend_features(cursor, ticker, event_date):
    try:
        # Exclude the event date
        end_date = event_date - timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(days=364)  # Total of 365 days

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date_str))

        rows = cursor.fetchall()
        prev_closes = [row[0] for row in rows if row[0] is not None]

        if not prev_closes:
            return {
                'ma_200': 0,
                'annual_volatility': 0,
                'pct_above_52_week_low': 0,
                'pct_below_52_week_high': 0,
            }

        # Calculate 200-day Moving Average
        ma_200 = sum(prev_closes[-200:]) / min(len(prev_closes), 200)

        # Calculate Annual Volatility
        if len(prev_closes) > 1:
            daily_returns = [(prev_closes[i+1] - prev_closes[i]) / prev_closes[i] for i in range(len(prev_closes)-1)]
            annual_volatility = pd.Series(daily_returns).std() * (252 ** 0.5)
        else:
            annual_volatility = 0

        # 52-week High/Low
        high_52_week = max(prev_closes)
        low_52_week = min(prev_closes)
        current_price = prev_closes[-1]

        pct_above_52_week_low = ((current_price - low_52_week) / low_52_week) * 100 if low_52_week != 0 else 0
        pct_below_52_week_high = ((high_52_week - current_price) / high_52_week) * 100 if high_52_week != 0 else 0

        return {
            'ma_200': ma_200,
            'annual_volatility': annual_volatility,
            'pct_above_52_week_low': pct_above_52_week_low,
            'pct_below_52_week_high': pct_below_52_week_high
        }
    except Exception as e:
        print(f"Error fetching long-term trend features: {e}")
        return {
            'ma_200': 0,
            'annual_volatility': 0,
            'pct_above_52_week_low': 0,
            'pct_below_52_week_high': 0
        }

def get_next_day_return(cursor, ticker, event_date):
    try:
        # Calculate return for the day after the event
        next_date = event_date + timedelta(days=1)
        next_date_str = next_date.strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date = ?
        ''', (ticker, next_date_str))

        row_next = cursor.fetchone()
        if not row_next or row_next[0] is None:
            return {
                'next_day_return': 0
            }

        next_day_close = row_next[0]

        # Get the following day's close (i.e., prevClose of two days after the event date)
        following_date = next_date + timedelta(days=1)
        following_date_str = following_date.strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date = ?
        ''', (ticker, following_date_str))

        row_following = cursor.fetchone()
        if not row_following or row_following[0] is None:
            return {
                'next_day_return': 0
            }

        following_close = row_following[0]

        # Calculate next day return
        next_day_return = ((following_close - next_day_close) / next_day_close) * 100 if next_day_close else 0

        return {
            'next_day_return': next_day_return
        }
    except Exception as e:
        print(f"Error fetching next day return: {e}")
        return {
            'next_day_return': 0
        }

# Function to check OHLCV data availability in the database
def has_ohlcv_data(ticker, date, cursor):
    # Ensure the `date` is a datetime object
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    
    next_day = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    date_str = date.strftime('%Y-%m-%d')
    
    cursor.execute('''
        SELECT 1 FROM stock_data
        WHERE tickerSymbol = ? AND date IN (?, ?)
        LIMIT 1
    ''', (ticker, date_str, next_day))
    
    return cursor.fetchone() is not None

def get_volume_features(cursor, ticker, event_date, period=14):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)

        cursor.execute('''
            SELECT date, totalSharesTraded FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'avg_volume': 0, 'volume_change_pct': 0}

        volumes = pd.Series([row[1] for row in rows])
        avg_volume = volumes[-period:].mean()

        # Calculate percentage change in volume over the last period
        volume_change_pct = ((volumes.iloc[-1] - volumes.iloc[-period]) / volumes.iloc[-period]) * 100

        return {'avg_volume': avg_volume, 'volume_change_pct': volume_change_pct}
    except Exception as e:
        print(f"Error fetching volume features: {e}")
        return {'avg_volume': 0, 'volume_change_pct': 0}

def get_ema(cursor, ticker, event_date, short_window=12, long_window=26):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=long_window * 2)

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < long_window:
            return {'ema_short': 0, 'ema_long': 0}

        closes = pd.Series([row[1] for row in rows])
        ema_short = closes.ewm(span=short_window, min_periods=short_window).mean().iloc[-1]
        ema_long = closes.ewm(span=long_window, min_periods=long_window).mean().iloc[-1]

        return {'ema_short': ema_short, 'ema_long': ema_long}
    except Exception as e:
        print(f"Error fetching EMA: {e}")
        return {'ema_short': 0, 'ema_long': 0}

def get_volatility(cursor, ticker, event_date, period=30):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'volatility': 0}

        closes = pd.Series([row[1] for row in rows])
        returns = closes.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility

        return {'volatility': volatility}
    except Exception as e:
        print(f"Error fetching volatility: {e}")
        return {'volatility': 0}

def get_momentum(cursor, ticker, event_date, period=14):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'momentum': 0}

        closes = pd.Series([row[1] for row in rows])
        momentum = closes.iloc[-1] - closes.iloc[-period]

        return {'momentum': momentum}
    except Exception as e:
        print(f"Error fetching momentum: {e}")
        return {'momentum': 0}

def get_stochastic_oscillator(cursor, ticker, event_date, period=14):
    try:
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=period * 2)

        cursor.execute('''
            SELECT date, high, low, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < period:
            return {'stochastic_k': 0}

        highs = pd.Series([row[1] for row in rows])
        lows = pd.Series([row[2] for row in rows])
        closes = pd.Series([row[3] for row in rows])

        lowest_low = lows.rolling(window=period).min().iloc[-1]
        highest_high = highs.rolling(window=period).max().iloc[-1]
        current_close = closes.iloc[-1]

        stochastic_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100 if (highest_high - lowest_low) != 0 else 0

        return {'stochastic_k': stochastic_k}
    except Exception as e:
        print(f"Error fetching Stochastic Oscillator: {e}")
        return {'stochastic_k': 0}

def get_weekly_return(cursor, ticker, event_date):
    try:
        # Calculate the start and end date for the last 7 days
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=6)  # The last week (7 days total)

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < 2:
            return {'weekly_return': 0}

        # Extract the closing prices for the start and end of the week
        start_price = rows[0][1]
        end_price = rows[-1][1]

        # Calculate the weekly return
        weekly_return = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0

        return {'weekly_return': weekly_return}
    except Exception as e:
        print(f"Error fetching weekly return: {e}")
        return {'weekly_return': 0}

def get_lagged_daily_returns(cursor, ticker, event_date):
    try:
        # Calculate the start and end date for the last 7 days
        end_date = event_date - timedelta(days=1)
        start_date = end_date - timedelta(days=6)  # The last 7 days

        cursor.execute('''
            SELECT date, prevClose FROM stock_data
            WHERE tickerSymbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        rows = cursor.fetchall()
        if len(rows) < 2:
            return {f'lagged_return_day_{i+1}': 0 for i in range(len(rows))}

        # Calculate daily returns
        prev_closes = pd.Series([row[1] for row in rows])
        daily_returns = prev_closes.pct_change().dropna() * 100

        # Create a dictionary with lagged returns for each day (last 7 days)
        lagged_returns = {}
        for i in range(1, 8):
            if i <= len(daily_returns):
                lagged_returns[f'lagged_return_day_{i}'] = daily_returns.iloc[-i]
            else:
                lagged_returns[f'lagged_return_day_{i}'] = 0

        return lagged_returns
    except Exception as e:
        print(f"Error fetching lagged daily returns: {e}")
        return {f'lagged_return_day_{i+1}': 0 for i in range(7)}

def fetch_ohlcv_features(cursor, ticker, current_date):
    if has_ohlcv_data(ticker, current_date, cursor):
        return {
            **get_recent_trend_features(cursor, ticker, current_date),
            **get_mid_term_trend_features(cursor, ticker, current_date),
            **get_long_term_trend_features(cursor, ticker, current_date),
            **get_rsi(cursor, ticker, current_date),
            **get_macd(cursor, ticker, current_date),
            **get_bollinger_bands(cursor, ticker, current_date),
            **get_atr(cursor, ticker, current_date),
            **get_ema(cursor, ticker, current_date),
            **get_volatility(cursor, ticker, current_date),
            **get_momentum(cursor, ticker, current_date),
            **get_volume_features(cursor, ticker, current_date),
            **get_stochastic_oscillator(cursor, ticker, current_date),
            # **get_lagged_daily_returns(cursor, ticker, current_date),  # Include lagged returns here
            **get_next_day_return(cursor, ticker, current_date)
        }
    return None
