import datetime as dt
import time
import pandas as pd
import numpy as np
import pytz
import sys
import ccxt

time_frame = {
    '1m'    : 24 * 60,
    '5m'    : 24 * 60 / 5,
    '15m'   : 24 * 60 / 15,
    '30m'   : 24 * 60 / 30,
    '1h'    : 24,
    '1d'    : 1,
}

def get_data(df, col, index):
    return df.iloc[index, df.columns.get_loc(col)]

def resample(df, prev_timeframe, new_timeframe):
    new_data = []
    offset = int(time_frame[prev_timeframe] / time_frame[new_timeframe])
    high = df['High'].rolling(window=offset).max()
    low = df['Low'].rolling(window=offset).max()
    volume = df['Volume'].rolling(window=offset).sum()
    for i in range(0, len(df), offset):
        o = get_data(df, 'Open', i)
        h = high.iloc[i - offset - 1]
        l = low.iloc[i - offset - 1]
        c = get_data(df, 'Close', i + offset - 1)
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

def fetch_ohlcv_data(code, interval, period=0):
    exchange = ccxt.phemex()
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
    df['Type'] = np.where(
        df['Open'] > df['Close'],
        'Bear',
        'Bull'
    )
    df['Swing_H'] = df['High'].rolling(window=7).max()
    df['Swing_L'] = df['Low'].rolling(window=7).min()
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

def get_historical_crypto_data_candles(code, candles=500, interval='1m'):
    exchange = ccxt.phemex()
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
