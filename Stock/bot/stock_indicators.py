import numpy as np
import loader

def get_data(df, col, index):
    return df.iloc[index, df.columns.get_loc(col)]

def set_data(data, index, col, value):
    data.iloc[index, data.columns.get_loc(col)] = value

def EMA(df, name='EMA', period=10, column='Close'):
    df[name] = df[column].ewm(span=period, adjust=False).mean()
    return {name : df[name]}

def SMA(df, name='SMA', period=10, column='Close'):
    df[name] = df[column].rolling(window=period).mean()
    return {name : df[name]}

def ATR(df, name='ATR', period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    df['tr0'] = abs(high - low)
    df['tr1'] = abs(high - close.shift())
    df['tr2'] = abs(low - close.shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df[name] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    del df['tr0'], df['tr1'], df['tr2'], df['tr']
    return {name : df[name]}

def RSI(df, name='RSI', period=14, low=20, high=60):
    change = df['Close'] - df['Open']
    df['gain'] = change.copy().apply(lambda x: x if x > 0 else 0)
    df['loss'] = change.copy().apply(lambda x: -x if x < 0 else 0)
    df['ema_gain'] = df['gain'].ewm(span=period, min_periods=period).mean()
    df['ema_loss'] = df['loss'].ewm(span=period, min_periods=period).mean()
    df['rs'] = df['ema_gain'] / df['ema_loss']
    df[name] = 100 - (100 / (df['rs'] + 1))
    df['Oversold'] = df[name].copy().apply(lambda x: True if x < low else False)
    df['Overbought'] = df[name].copy().apply(lambda x: True if x > high else False)
    del df['gain'], df['loss'], df['ema_gain'], df['ema_loss'], df['rs']
    return {name : df[name]}

def ADF(df, MA, name='ADF', period=14):
    df['DFS'] = abs(df['Close'] - MA)
    df[name] = df['DFS'].rolling(window=period).mean()
    df[name] = df[name] / ATR(df, period=period)['ATR']
    del df['DFS']
    return {name: df[name]}

def ADX(df, name='ADX', period=14, smoothing=14):
    alpha = 1 / period
    df['atr'] = ATR(df, name='atr', period=period)['atr']
    df['H'] = df['High'] - df['High'].shift()
    df['L'] = df['Low'].shift() - df['Low']
    df['+DX'] = np.where(
        (df['H'] > df['L']) & (df['H'] > 0),
        df['H'],
        0
    )
    df['-DX'] = np.where(
        (df['H'] < df['L']) & (df['L'] > 0),
        df['L'],
        0
    )
    df['S+DX'] = df['+DX'].ewm(alpha=1/smoothing, adjust=False).mean()
    df['S-DX'] = df['-DX'].ewm(alpha=1/smoothing, adjust=False).mean()
    df['+DMI'] = (df['S+DX'] / df['atr']) * 100
    df['-DMI'] = (df['S-DX'] / df['atr']) * 100
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI']) / (df['+DMI'] + df['-DMI'])) * 100
    df[name] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['H'], df['L'], df['S+DX'], df['S-DX'], df['+DX'], df['-DX'], df['atr']
    return {name : df[name]}

def bollinger_band(df, period=20, std=2, bw_threshold=0):
    df['STD'] = df['Close'].rolling(window=period).std(ddof=0)
    df['Bol_M'] = SMA(df, period=period)['SMA']
    df['Bol_H'] = df['Bol_M'] + (df['STD'] * std)
    df['Bol_L'] = df['Bol_M'] - (df['STD'] * std)
    df['Bol_BW'] = (df['Bol_H'] - df['Bol_L']) / df['Bol_M'] * 100
    df['BW_TH'] = df['Bol_BW'].copy().apply(lambda x: True if x > bw_threshold else False)
    df['BW_G'] = np.where(
        df['BW_TH'],
        df['Bol_BW'],
        np.nan
    )
    del df['BW_G'], df['STD'], df['BW_TH']
    return {'Bol_H' : df['Bol_H'], 'Bol_M' : df['Bol_M'], 'Bol_L' : df['Bol_L'], 'Bol_BW' : df['Bol_BW']}

def check_above_below_trend(df, MA, window=10, above='upt', below='dwt', column1='Open', column2='Close'):
    df['max'] = np.maximum(df[column1], df[column2])
    df['min'] = np.minimum(df[column1], df[column2])
    df['a'] = 0
    df['b'] = 0
    df.loc[(df['min'] >= MA), 'a'] = 1
    df.loc[(df['max'] <= MA), 'b'] = 1
    df[above] = df['a'].rolling(window=window).min() == 1
    df[below] = df['b'].rolling(window=window).min() == 1
    del df['a'], df['b'], df['max'], df['min']
    return {above : df[above], below : df[below]}

def engulfing_candlestick(df):
    df['Engulfing'] = np.nan
    for i in range(1, len(df)):
        if get_data(df, 'Type', i - 1) == 'Bear' and get_data(df, 'Type', i) == 'Bull':
            if get_data(df, 'Open', i - 1) < get_data(df, 'Close', i) and get_data(df, 'Close', i - 1) > get_data(df, 'Open', i):
                set_data(df, i, 'Engulfing', 2)
        elif get_data(df, 'Type', i - 1) == 'Bull' and get_data(df, 'Type', i) == 'Bear':
            if get_data(df, 'Close', i - 1) < get_data(df, 'Open', i) and get_data(df, 'Open', i - 1) > get_data(df, 'Close', i):
                set_data(df, i, 'Engulfing', 1)
    return {'Engulfing' : df['Engulfing']}

def signal(df, name, data=None):
    if data == None:
        data = [np.nan] * len(df)
    df[name] = [data] * len(df)
    return {name : data}

def VWAP(df, name='VWAP'):
    df['price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VXP'] = df['Volume'] * df['price']
    df['VXPA'] = df['VXP'].groupby(df['VXP'].index.date).cumsum()
    df['VA'] = df['Volume'].groupby(df['Volume'].index.date).cumsum()
    df[name] = df['VXPA'] / df['VA']
    del df['price'], df['VXP'], df['VXPA'], df['VA']
    return {name : df[name]}

def swing_L_H(df, period=7):
    df['Swing_H'] = df['High'].rolling(window=period).max()
    df['Swing_L'] = df['Low'].rolling(window=period).min()
    return {'Swing_L' : df['Swing_L'], 'Swing_H' : df['Swing_H']}

def MACD(df, slow=26, fast=12, signal=9):
    df['slow'] = EMA(df, 'slow', period=slow)['slow']
    df['fast'] = EMA(df, 'fast', period=fast)['fast']

    df['MACD'] = df['fast'] - df['slow']
    df['MACD_signal'] = EMA(df, 'MACD_signal', period=signal, column='MACD')['MACD_signal']

    del df['slow'], df['fast']

    return {'MACD' : df['MACD'], 'MACD_signal' : df['MACD_signal']}

def strend(df, name='strend', period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    df['atr'] = ATR(df, name='atr', period=period)['atr']
    matr = multiplier * df['atr']
    df['upperband'] = hl2 + matr
    df['lowerband'] = hl2 - matr
    df['in_uptrend'] = True

    for curr in range(1, len(df.index)):
        prev = curr - 1
        if df['Close'].iloc[curr] > df['upperband'].iloc[prev]:
           set_data(df, curr, 'in_uptrend', True)
        elif df['Close'].iloc[curr] < df['lowerband'].iloc[prev]:
           set_data(df, curr, 'in_uptrend', False)
        else:
            set_data(df, curr, 'in_uptrend', df['in_uptrend'].iloc[prev])
            if df['in_uptrend'].iloc[curr] and df['lowerband'].iloc[curr] < df['lowerband'].iloc[prev]:
                set_data(df, curr, 'lowerband', df['lowerband'].iloc[prev])
            if not df['in_uptrend'].iloc[curr] and df['upperband'].iloc[curr] > df['upperband'].iloc[prev]:
                set_data(df, curr, 'upperband', df['upperband'].iloc[prev])
    
    df[name] = df['in_uptrend']
    del df['in_uptrend'], df['upperband'], df['lowerband']

    return {name : df[name]}

def multi_timeframe(df, prev_timeframe, new_timeframe, indicator, kwargs):
    offset = int(loader.time_frame[prev_timeframe] / loader.time_frame[new_timeframe])
    df_new = loader.resample(df, prev_timeframe=prev_timeframe, new_timeframe=new_timeframe)
    kwargs['df'] = df_new
    series = indicator(**kwargs)
    for s in series.keys():
        print(s)
        df[s] = 0
        for i in range(len(series[s])):
            for j in range(offset):
                if i * offset + j < len(df): set_data(df, i * offset + j, s, series[s].iloc[i])
    return series

def stochastic(df, name='stochastic', period=14, k_smoothing=1, d_smoothing=3):
    df['H'] = df['High'].rolling(window=period).max()
    df['L'] = df['Low'].rolling(window=period).min()
    df['C-L'] = df['Close'] - df['L']
    df['H-L'] = df['H'] - df['L']
    df[name] = df['C-L'] / df['H-L'] * 100
    df[name] = df[name].rolling(window=k_smoothing).mean()
    df['slow_'+name] = df[name].rolling(window=d_smoothing).mean()

    del df['H'], df['L'], df['C-L'], df['H-L']

    return {name : df[name], 'slow_'+name : df['slow_'+name]}