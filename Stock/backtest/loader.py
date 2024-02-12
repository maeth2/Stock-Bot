import datetime as dt
import time
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import strategies
import sys

time_frame = {
    '1m'    : 24 * 60,
    '5m'    : 24 * 60 / 5,
    '15m'   : 24 * 60 / 15,
    '30m'   : 24 * 60 / 30,
    '1h'    : 24,
    '1d'    : 1,
}

def resample(df, prev_timeframe, new_timeframe):
    new_data = []
    offset = int(time_frame[prev_timeframe] / time_frame[new_timeframe])
    high = df['High'].rolling(window=offset).max()
    low = df['Low'].rolling(window=offset).max()
    volume = df['Volume'].rolling(window=offset).sum()
    for i in range(0, len(df), offset):
        o = strategies.get_data(df, 'Open', i)
        h = high.iloc[i - offset - 1]
        l = low.iloc[i - offset - 1]
        c = strategies.get_data(df, 'Close', i + offset - 1)
        v = volume.iloc[i - offset - 1]
        time = df.index[i]
        new_data.append([time, o, h, c, l, v])

    data = pd.DataFrame(new_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    data['Type'] = np.where(
        data['Open'] > data['Close'],
        'Bear',
        'Bull'
    )
    data.index = pd.DatetimeIndex(data['Date'])
    del data['Date']
    return data

def get_real_time_stock_data(code, period, interval='1m'):
    start_date = dt.datetime.now() - dt.timedelta(days = period)
    unix_time = int(time.mktime((start_date).timetuple()) * 1000)
    df = yf.download(tickers=code, start=pd.to_datetime(unix_time, unit='ms'), interval=interval).round(2)
    df['Type'] = np.where(
        df['Open'] > df['Close'],
        'Bear',
        'Bull'
    )
    print(df)
    return df

def get_historical_crypto_data_candles(exchange, code, candles=500, interval='1m'):
    ohlcv = exchange.fetch_ohlcv(code, interval, limit=candles)
    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Type'] = np.where(
        df['Open'] > df['Close'],
        'Bear',
        'Bull'
    )
    df['Date'] = pd.to_datetime(df['Date'], unit=('ms')).dt.tz_localize('UTC')
    df['Date'] = df['Date'].dt.tz_convert(pytz.timezone('Asia/Singapore'))
    df.index = pd.DatetimeIndex(df['Date'])
    del df['Date']
    print(df)
    print(f"Data Start Date: {list(df.index.values)[0]}")
    print(f"Data End Date: {list(df.index.values)[-1]}")
    min_diff = (list(df.index.values)[-1] - list(df.index.values)[0]).astype('timedelta64[m]')
    print(f"Minutes Difference: {min_diff}")
    intervals = min_diff / np.timedelta64(5, 'm')
    print(f"Number of Candles: {intervals}")
    return df

def get_historical_crypto_data(exchange, code, period, interval='1m', debug=True):
    start_date = dt.datetime.now() - dt.timedelta(days = period)
    print(start_date)
    unix_time = int(time.mktime((start_date).timetuple()) * 1000)
    total_ticks = period * time_frame[interval]
    ticks = total_ticks
    ohlcv_list = []
    print(f"Loading: {int(100 - ticks / total_ticks * 100)}%")
    sys.stdout.write("\033[F") 
    while(ticks > 0):
        ohlcv = exchange.fetch_ohlcv(code, interval, since=unix_time)
        ohlcv_list.extend(ohlcv[1:])
        unix_time = ohlcv_list[-1][0]
        ticks -= len(ohlcv)
        print(f"Loading: {int(100 - ticks / total_ticks * 100)}%")
        sys.stdout.write("\033[F") 
    
    df = pd.DataFrame(ohlcv_list, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    increase = (int(df.iloc[-1]['Close']) / int(df.iloc[0]['Open']) - 1) * 100
    print(f"Price increased {increase} for {code}, Starting Price: {int(df.iloc[0]['Open'])}, Final Price: {int(df.iloc[-1]['Close'])}")
    df['Type'] = np.where(
        df['Open'] > df['Close'],
        'Bear',
        'Bull'
    )
    df['Date'] = pd.to_datetime(df['Date'], unit=('ms')).dt.tz_localize('UTC')
    df['Date'] = df['Date'].dt.tz_convert(pytz.timezone('Asia/Singapore'))
    df.index = pd.DatetimeIndex(df['Date'])
    del df['Date']
    if debug:
        print(df)
        print(f"Data Start Date: {list(df.index.values)[0]}")
        print(f"Data End Date: {list(df.index.values)[-1]}")
        min_diff = (list(df.index.values)[-1] - list(df.index.values)[0]).astype('timedelta64[m]')
        print(f"Minutes Difference: {min_diff}")
        intervals = min_diff / np.timedelta64(5, 'm')
        print(f"Number of Candles: {intervals}")

    return df