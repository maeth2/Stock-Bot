import numpy as np
import plot
import strategies as strat
import loader
import sys

def set_data(data, index, col, value):
    data.iloc[index, data.columns.get_loc(col)] = value

def get_data(df, col, index):
    if index >= len(df):
        return -1
    return df.iloc[index, df.columns.get_loc(col)]

def ADF(df, MA, name='ADF', period=14, panel=1, color='white', display=True):
    df['DFS'] = abs(df['Close'] - MA)
    df[name] = df['DFS'].rolling(window=period).mean()
    df[name] = df[name] / ATR(df, period=period, display=False)['ATR']
    del df['DFS']
    if display: plot.add_plot(df[name], color=color, panel=panel, ylabel=name)
    return {name: df[name]}

def ADX(df, name='ADX', period=14, smoothing=14, panel=1, color1='white', display=True):
    alpha = 1 / period
    df['atr'] = ATR(df, name='atr', period=period, display=False)['atr']
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
    if display:
        plot.add_plot(df['+DMI'], color='green', panel=panel)
        plot.add_plot(df['-DMI'], color='red', panel=panel)
        plot.add_plot(df[name], color=color1, panel=panel, ylabel=name)
    return {name : df[name]}
        
def ATR(df, name='ATR', period=14, panel=1, display=True, color='white'):
    high = df['High']
    low = df['Low']
    close = df['Close']
    df['tr0'] = abs(high - low)
    df['tr1'] = abs(high - close.shift())
    df['tr2'] = abs(low - close.shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df[name] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    del df['tr0'], df['tr1'], df['tr2'], df['tr']
    if display: plot.add_plot(df[name], color=color, panel=panel, ylabel=name)
    return {name : df[name]}

def bollinger_band(df, name='Bol', period=20, std=2, bw_threshold=0, panel1=0, panel2=1, color1='#FF0089', color2='white', color3='#00FF76', color4='white', show_high=True, show_mid=True, show_low=True, show_bw=True, display=True):
    high = name + '_H'
    mid = name + '_M'
    low = name + '_L'
    bw = name + '_BW'
    df['STD'] = df['Close'].rolling(window=period).std(ddof=0)
    df[mid] = SMA(df, period=period, display=False)['SMA']
    df[high] = df[mid] + (df['STD'] * std)
    df[low] = df[mid] - (df['STD'] * std)
    df[bw] = (df[high] - df[low]) / df[mid]
    df['BW_TH'] = df[bw].copy().apply(lambda x: True if x > bw_threshold else False)
    df['BW_G'] = np.where(
        df['BW_TH'],
        df[bw],
        np.nan
    )
    if display:
        if show_low: plot.add_plot(df[low], color=color1, panel=panel1)
        if show_mid: plot.add_plot(df[mid], color=color2, panel=panel1)
        if show_high: plot.add_plot(df[high], color=color3, panel=panel1)
        if show_bw: 
            plot.add_plot(df['Bol_BW'], color=color4, panel=panel2, ylabel='BBW')
            if df['BW_TH'].sum(): plot.add_plot(df['BW_G'], color='green', panel=panel2)
    del df['BW_G'], df['STD'], df['BW_TH']
    return {high : df[high], mid : df[mid], low : df[low], bw : df[bw]}

#TODO: FIX THIS
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

def DEMA(df, name='DEMA', period=10, panel=0, column='Close', color='pink', display=True):
    df['ema'] = df[column].ewm(span=period, adjust=False).mean()
    df['sema'] = df['ema'].ewm(span=period, adjust=False).mean()
    df[name] = 2 * df['ema'] - df['sema']
    if display: plot.add_plot(df[name], color=color, panel=panel)
    del df['ema'], df['sema']
    return {name : df[name]}

def EMA(df, name='EMA', period=10, column='Close', panel=0, color='blue', display=True):
    df[name] = df[column].ewm(span=period, adjust=False).mean()
    if display: plot.add_plot(df[name], color=color, panel=panel, ylabel=name)
    return {name : df[name]}

def engulfing_candlestick(df, color='purple', panel=0):
    df['Engulfing'] = np.nan
    for i in range(1, len(df)):
        if strat.get_data(df, 'Type', i - 1) == 'Bear' and strat.get_data(df, 'Type', i) == 'Bull':
            if strat.get_data(df, 'Open', i - 1) <= strat.get_data(df, 'Close', i) and strat.get_data(df, 'Close', i - 1) >= strat.get_data(df, 'Open', i):
                set_data(df, i, 'Engulfing', 2)
        elif strat.get_data(df, 'Type', i - 1) == 'Bull' and strat.get_data(df, 'Type', i) == 'Bear':
            if strat.get_data(df, 'Close', i - 1) <= strat.get_data(df, 'Open', i) and strat.get_data(df, 'Open', i - 1) >= strat.get_data(df, 'Close', i):
                set_data(df, i, 'Engulfing', 1)             
    df['Engulfing_plot'] = np.where(
        df['Engulfing'] > 0,
        df['Close'],
        np.nan
    )
    if df['Engulfing_plot'].sum(): plot.add_marker(df['Engulfing_plot'], marker_size=20, symbol='o', color=color, panel=panel)
    del df['Engulfing_plot']
    return {'Engulfing' : df['Engulfing']}

def hammer_candle(df, color='lime', panel=0):
    df['Hammer'] = np.nan
    for i in range(1, len(df)):
        open = strat.get_data(df, 'Open', i)
        close = strat.get_data(df, 'Close', i)
        low = strat.get_data(df, 'Low', i)
        body = abs(open - close)
        if strat.get_data(df, 'Type', i) == 'Bull':
            if (body * 2) < (open - low):
                set_data(df, i, 'Hammer', 1)
        elif strat.get_data(df, 'Type', i) == 'Bear':
            if (body * 2) < (close - low):
                set_data(df, i, 'Hammer', 1)
    df['Hammer_plot'] = np.where(
        df['Hammer'] == 1,
        df['Close'],
        np.nan
    )
    if df['Hammer_plot'].sum(): plot.add_marker(df['Hammer_plot'], marker_size=20, symbol='o', color=color, panel=panel)
    del df['Hammer_plot']
    return {'Hammer' : df['Hammer']}

def MACD(df, slow=26, fast=12, signal=9,  color='white', signal_color='red', panel=1, display=True):
    df['slow'] = EMA(df, 'slow', period=slow, display=False)['slow']
    df['fast'] = EMA(df, 'fast', period=fast, display=False)['fast']

    df['MACD'] = df['fast'] - df['slow']
    df['MACD_signal'] = EMA(df, 'MACD_signal', period=signal, column='MACD', color=signal_color, panel=panel, display=display)['MACD_signal']

    del df['slow'], df['fast']
    if display: 
        plot.add_plot(df['MACD'], color=color, panel=panel, ylabel='MACD')
    return {'MACD' : df['MACD'], 'MACD_signal' : df['MACD_signal']}

def resistance(df, i, n1, n2):
    for j in range(i - n1 + 1, i + 1):
        if strat.get_data(df, 'High', j) <= strat.get_data(df, 'High', j - 1):
            return 0
    for j in range(i + 1, i + n2 + 1):
        if strat.get_data(df, 'High', j) >= strat.get_data(df, 'High', j - 1):
            return 0
    return 1

def RSI(df, name='RSI', period=14, low=20, high=60, panel=1, color='white', display=True):
    change = df['Close'] - df['Open']
    df['gain'] = change.copy().apply(lambda x: x if x > 0 else 0)
    df['loss'] = change.copy().apply(lambda x: -x if x < 0 else 0)
    df['ema_gain'] = df['gain'].ewm(span=period, min_periods=period).mean()
    df['ema_loss'] = df['loss'].ewm(span=period, min_periods=period).mean()
    df['rs'] = df['ema_gain'] / df['ema_loss']
    df[name] = 100 - (100 / (df['rs'] + 1))
    df['RSI_oversold'] = df[name].copy().apply(lambda x: True if x < low else False)
    df['RSI_overbought'] = df[name].copy().apply(lambda x: True if x > high else False)
    df['oversold_plot'] = np.where(
        df['RSI_oversold'],
        df[name],
        np.nan
    )
    df['overbought_plot'] = np.where(
        df['RSI_overbought'],
        df[name],
        np.nan
    )
    del df['gain'], df['loss'], df['ema_gain'], df['ema_loss'], df['rs']
    if display:
        plot.add_plot(df[name], color=color, panel=panel, ylabel=name)
        plot.add_plot(df['oversold_plot'], color='green', panel=panel, ylabel=name)
        plot.add_plot(df['overbought_plot'], color='red', panel=panel, ylabel=name)
    return {name : df[name]}

def multi_timeframe(df, prev_timeframe, new_timeframe, indicator, kwargs, display={}, panel=0, color='white'):
    offset = int(loader.time_frame[prev_timeframe] / loader.time_frame[new_timeframe])
    df_new = loader.resample(df, prev_timeframe=prev_timeframe, new_timeframe=new_timeframe)
    kwargs['df'] = df_new
    kwargs['display'] = False
    series = indicator(**kwargs)
    for s in series.keys():
        print(s)
        df[s] = 0
        for i in range(len(series[s])):
            for j in range(offset):
                if i * offset + j < len(df): set_data(df, i * offset + j, s, series[s].iloc[i])
        if s in display and display[s]: plot.add_plot(df[s], color=color, panel=panel, ylabel=s)
    return series

def signal(df, name, data="None", panel=1, color='white', display=False):
    if data == "None":
        df[name] = [np.nan] * len(df)
    else:
        df[name] = df[data]
        if display: plot.add_plot(df[name], color=color, panel=panel, ylabel=name)
    return {name : data}

def SMA(df, name='SMA', period=10, panel=0, column='Close', color='yellow', display=True):
    df[name] = df[column].rolling(window=period).mean()
    if display: plot.add_plot(df[name], color=color, panel=panel, ylabel=name)
    return {name : df[name]}

def stochastic(df, name='stochastic', period=14, k_smoothing=1, d_smoothing=3, panel=1, color1='white', color2='red', display=True):
    df['H'] = df['High'].rolling(window=period).max()
    df['L'] = df['Low'].rolling(window=period).min()
    df['C-L'] = df['Close'] - df['L']
    df['H-L'] = df['H'] - df['L']
    df[name] = df['C-L'] / df['H-L'] * 100
    df[name] = df[name].rolling(window=k_smoothing).mean()
    df['slow_'+name] = df[name].rolling(window=d_smoothing).mean()
    del df['H'], df['L'], df['C-L'], df['H-L']
    if display: 
        plot.add_plot(df[name], color=color1, panel=panel)
        plot.add_plot(df['slow_'+name], color=color2, panel=panel)
    return {name : df[name], 'slow_'+name : df['slow_'+name]}

def strend(df, name='strend', period=10, multiplier=3, display=True):
    hl2 = (df['High'] + df['Low']) / 2
    df['atr'] = ATR(df, name='atr', period=period, display=False)['atr']
    matr = multiplier * df['atr']
    df['upperband'] = hl2 + matr
    df['lowerband'] = hl2 - matr
    df['in_uptrend'] = True

    for curr in range(1, len(df.index)):
        prev = curr - 1
        print(f"{name}, Computing Progress: %{int(curr / len(df.index) * 100)}")
        sys.stdout.write("\033[F") 
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

    if display:
        df['show_upper'] = df['upperband'].copy()
        df['show_lower'] = df['lowerband'].copy()

        for i in range(1, len(df.index)):
            if df['in_uptrend'].iloc[i]:
                set_data(df, i, 'show_upper', None)
            else:
                set_data(df, i, 'show_lower', None)
        
        plot.add_plot(df['show_upper'], color='red')
        plot.add_plot(df['show_lower'], color='green')
        del df['show_upper'], df['show_lower']
    
    df[name] = df['in_uptrend']
    del df['in_uptrend']

    return {name : df[name]}

def swing_L_H(df, period=7, display=False):
    df['Swing_H'] = df['High'].rolling(window=period).max()
    df['Swing_L'] = df['Low'].rolling(window=period).min()
    if display:
        plot.add_plot(df['Swing_H'], 'green')
        plot.add_plot(df['Swing_L'], 'red')
    return {'Swing_L' : df['Swing_L'], 'Swing_H' : df['Swing_H']}

def support(df, i, n1, n2):
    for j in range(i - n1 + 1, i + 1):
        if strat.get_data(df, 'Low', j) >= strat.get_data(df, 'Low', j - 1):
            return 0
    for j in range(i + 1, i + n2 + 1):
        if strat.get_data(df, 'Low', j) <= strat.get_data(df, 'Low', j - 1):
            return 0
    return 1

def support_resistance(df, n1=3, n2=2, threshold=0.01, min_strength=1):
    s = []
    r = []
    color = []
    for i in range(len(df)):
        if support(df, i, n1, n2): s.append(strat.get_data(df, 'Low', i))
        if resistance(df, i, n1, n2): r.append(strat.get_data(df, 'High', i))
    
    s.sort()
    r.sort()

    merged_s = []
    merged_r = []
    support_strength = []
    resistance_strength = []

    for i in range(0, len(s)):
        if i == 0: 
            merged_s.append(s[i])
            support_strength.append(1)
        else:
            if (s[i] - merged_s[-1]) / s[i] < threshold:
                merged_s[-1] = (s[i] + merged_s[-1]) / 2
                support_strength[-1] += 1
            else:
                merged_s.append(s[i])
                support_strength.append(1)

    for i in range(0, len(r)):
        if i == 0: 
            merged_r.append(r[i])
            resistance_strength.append(1)
        else:
            if (r[i] - merged_r[-1]) / r[i] < threshold:
                merged_r[-1] = (r[i] + merged_r[-1]) / 2
                resistance_strength[-1] += 1
            else:
                merged_r.append(r[i])
                resistance_strength.append(1)
    
    merged_s = [merged_s[i] for i in range(len(support_strength)) if support_strength[i] > min_strength]
    merged_r = [merged_r[i] for i in range(len(resistance_strength)) if resistance_strength[i] > min_strength]

    plot.add_hlines(merged_s, 'g')
    plot.add_hlines(merged_r, 'r')
    return merged_s, merged_r

def three_line_strike(df, color='yellow', panel=0):
    df['Three_line_strike'] = np.nan
    for i in range(3, len(df)):
        be = 1
        bu = 1
        for j in range(1, 4):
            if strat.get_data(df, 'Type', i - j) == 'Bull':
                if j < 3:
                    be = be and strat.get_data(df, 'Open', i - j) > strat.get_data(df, 'Open', i - j - 1)
                    be = be and strat.get_data(df, 'Close', i - j) > strat.get_data(df, 'Close', i - j - 1)
                bu = 0
            elif strat.get_data(df, 'Type', i - j) == 'Bear':
                if j < 3:
                    bu = bu and strat.get_data(df, 'Open', i - j) < strat.get_data(df, 'Open', i - j - 1)
                    bu = bu and strat.get_data(df, 'Close', i - j) < strat.get_data(df, 'Close', i - j - 1)
                be = 0
        
        bu = bu and strat.get_data(df, 'Type', i) == 'Bull'
        bu = bu and strat.get_data(df, 'Close', i) > strat.get_data(df, 'Open', i - 1)

        be = be and strat.get_data(df, 'Type', i) == 'Bear'
        be = be and strat.get_data(df, 'Close', i) < strat.get_data(df, 'Open', i - 1)

        if bu and be: set_data(df, i, 'Three_line_strike', np.nan)
        elif bu: set_data(df, i, 'Three_line_strike', 2)
        elif be: set_data(df, i, 'Three_line_strike', 1)
    df['TLS_plot'] = np.where(
        df['Three_line_strike'] > 0,
        df['Close'],
        np.nan
    )
    if df['TLS_plot'].sum(): plot.add_marker(df['TLS_plot'], marker_size=20, symbol='o', color=color, panel=panel)
    del df['TLS_plot']
    return {'Three_line_strike' : df['Three_line_strike']}

def trend_detector(df, candles=5, display=False, draw_hl_lines=False):
    df['max'] = df['High'].rolling(window=candles).max()
    df['min'] = df['Low'].rolling(window=candles).min()
    df['is_peak'] = np.nan
    for i in range(len(df) - candles):
        if get_data(df, 'max', i) == get_data(df, 'max', i + candles - 1) and get_data(df, 'max', i) == get_data(df, 'High', i):
            set_data(df, i, 'is_peak', 2)
        if get_data(df, 'min', i) == get_data(df, 'min', i + candles - 1) and get_data(df, 'min', i) == get_data(df, 'Low', i):
            set_data(df, i, 'is_peak', 1)

    df['h'] = 1
    df['l'] = 1

    highs = []
    lows = []
    hm = 0
    lm = 0
    for i in range(len(df)):
        is_empty = len(lows) == 0 or len(highs) == 0

        if get_data(df, 'is_peak', i) == 2:
            if not is_empty and lows[-1][0] < highs[-1][0]:
                if(highs[-1][1] < get_data(df, 'High', i)):
                    highs[-1] = (i, get_data(df, 'High', i))
            else:
                highs.append((i, get_data(df, 'High', i)))
            if len(highs) > 1: hm = (highs[-1][1] - highs[-2][1]) / (highs[-1][0] - highs[-2][0])
        
        if get_data(df, 'is_peak', i) == 1:
            if not is_empty and highs[-1][0] < lows[-1][0]:
                if(lows[-1][1] > get_data(df, 'Low', i)):
                    lows[-1] = (i, get_data(df, 'Low', i))
            else:
                lows.append((i, get_data(df, 'Low', i)))
            if len(lows) > 1: lm = (lows[-1][1] - lows[-2][1]) / (lows[-1][0] - lows[-2][0])
            
        if hm <= -1: set_data(df, i, 'h', 1)
        elif hm >= 1: set_data(df, i, 'h', 2)
        if lm <= -1: set_data(df, i, 'l', 1)
        elif lm >= 1: set_data(df, i, 'l', 2)

    df['trend'] = np.where(
        df['h'] == df['l'],
        df['h'],
        0
    )
    
    if display:
        if draw_hl_lines:
            high_line = []
            low_line = []
            df['is_high'] = np.nan
            df['is_low'] = np.nan

            for i in range(1, len(highs)):
                set_data(df, highs[i][0], 'is_high', get_data(df, 'High', highs[i][0]))
                high_line.append((highs[i - 1], highs[i]))

            for i in range(1, len(lows)):
                set_data(df, lows[i][0], 'is_low', get_data(df, 'Low', lows[i][0]))
                low_line.append((lows[i - 1], lows[i]))

            plot.add_marker(df['is_high'], 200, '.', 'green')
            plot.add_marker(df['is_low'], 200, '.', 'red')
            plot.add_p_to_p_lines(high_line, max_length=len(df), color='green')
            plot.add_p_to_p_lines(low_line, max_length=len(df), color='red')
            del df['is_high'], df['is_low']

        df['uptrend'] = np.where(
            df['trend'] == 2,
            df['High'],
            np.nan
        )
        df['downtrend'] = np.where(
            df['trend'] == 1,
            df['Low'],
            np.nan
        )
        df['range'] = np.where(
            df['trend'] == 0,
            (df['High'] + df['Low']) / 2,
            np.nan
        )
        plot.add_plot(df['uptrend'], 'green')
        plot.add_plot(df['downtrend'], 'red')
        plot.add_plot(df['range'], 'yellow')

        del df['uptrend'], df['downtrend'], df['range']

    del df['max'], df['min'], df['h'], df['l'], df['is_peak']
    
    return {'trend' : df['trend']}

def VWAP(df, name='VWAP', panel=0, color='blue', display=True):
    df['price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VXP'] = df['Volume'] * df['price']
    df['VXPA'] = df['VXP'].groupby(df['VXP'].index.date).cumsum()
    df['VA'] = df['Volume'].groupby(df['Volume'].index.date).cumsum()
    df[name] = df['VXPA'] / df['VA']
    del df['price'], df['VXP'], df['VXPA'], df['VA']
    if display: plot.add_plot(df[name], color=color, panel=panel)
    return {name : df[name]}

def anchored_VWAP(df, name='anchored_VWAP', candles=10, panel=0, color='blue', display=True):
    df['price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VXP'] = df['Volume'] * df['price']
    df['max'] = df['High'].rolling(window=candles).max()
    df['min'] = df['Low'].rolling(window=candles).min()
    df['is_peak'] = 0
    for i in range(len(df) - candles):
        if get_data(df, 'max', i) == get_data(df, 'max', i + candles - 1) and get_data(df, 'max', i) == get_data(df, 'High', i):
            set_data(df, i, 'is_peak', 2)
        if get_data(df, 'min', i) == get_data(df, 'min', i + candles - 1) and get_data(df, 'min', i) == get_data(df, 'Low', i):
            set_data(df, i, 'is_peak', 1)

    df['VXPA'] = np.nan
    df['VA'] = np.nan
    vxpa = 0
    va = 0
    for i in range(len(df)):
        if get_data(df, 'is_peak', i) != 0:
            vxpa = get_data(df, 'VXP', i)
            va = get_data(df, 'Volume', i)
        else:
            vxpa += get_data(df, 'VXP', i)
            va += get_data(df, 'Volume', i)
        set_data(df, i, 'VXPA', vxpa)
        set_data(df, i, 'VA', va)

    df[name] = df['VXPA'] / df['VA']
    del df['price'], df['VXP'], df['VXPA'], df['VA'], df['max'], df['min'], df['is_peak']
    if display: plot.add_plot(df[name], color=color, panel=panel)
    return {name : df[name]}