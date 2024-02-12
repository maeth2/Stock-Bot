import indicators as ind

def get_indicator(data, id, index):
    return data[id].iloc[index]

def load_indicators(df, function, kwargs={}, display_ind=True):
    name = function.__name__
    ind.signal(df, 'long')
    ind.signal(df, 'short')
    ind.ATR(df, period=7, display=False)
    ind.swing_L_H(df, period=15)

    if name == 'vwap_bol_strat_5m':
        ind.EMA(df, 'EMA_200', period=200, color='yellow', display=display_ind)
        ind.bollinger_band(df, period=kwargs['Bol_period'], std=2, panel1=0, show_high=display_ind, show_low=display_ind, show_mid=display_ind, show_bw=False)
        ind.RSI(df, name='RSI_vwap', period=14, low=45, high=55, panel=1, display=display_ind)
        ind.SMA(df, 'SMA_RSI_vwap', period=14, column='RSI_vwap', panel=1, color='blue', display=display_ind)
        ind.stochastic(df, panel=2, display=display_ind)
        ind.VWAP(df, name='VWAP', display=display_ind)
        ind.ADF(df, MA=df['VWAP'], display=False)
        ind.check_above_below_trend(df, MA=df['VWAP'], window=5, above='dwt_vwap', below='upt_vwap', column1='Open', column2='Close')
    if name == 'vwap_bol_adf_strat_5m':
        ind.EMA(df, 'EMA_200', period=200, color='yellow', display=display_ind)
        ind.bollinger_band(df, period=kwargs['Bol_period'], std=2, panel1=0, show_high=display_ind, show_low=display_ind, show_mid=display_ind, show_bw=False)
        ind.RSI(df, name='RSI_vwap', period=14, low=45, high=55, panel=1, display=display_ind)
        ind.SMA(df, 'SMA_RSI_vwap', period=14, column='RSI_vwap', panel=1, color='blue', display=display_ind)
        ind.VWAP(df, name='VWAP', display=display_ind)
        ind.ADF(df, MA=df['VWAP'], display=False)
        ind.check_above_below_trend(df, MA=df['VWAP'], window=5, above='dwt_vwap', below='upt_vwap', column1='Open', column2='Close')
    if name == 'vwap_bol_strat_30m':
        ind.bollinger_band(df, period=14, std=2, panel1=0, panel2=1, show_high=display_ind, show_low=display_ind, show_mid=display_ind, show_bw=display_ind, bw_threshold=0.015)
        ind.RSI(df, name='RSI_vwap', period=14, low=45, high=55, panel=2, display=display_ind)
        ind.SMA(df, 'SMA_RSI_vwap', period=14, column='RSI_vwap', panel=2, color='blue', display=display_ind)
        ind.EMA(df, 'EMA_200', period=200, color='yellow', display=display_ind)
        ind.VWAP(df, name='VWAP', display=display_ind)
        ind.ADF(df, MA=df['VWAP'], display=False)
        ind.check_above_below_trend(df, MA=df['VWAP'], window=1, above='dwt_vwap', below='upt_vwap', column1='Open', column2='Close')
    if name == 'test_strat':
        ind.bollinger_band(df, period=30, std=2, panel1=0, panel2=1, show_high=False, show_low=display_ind, show_mid=False, show_bw=display_ind, bw_threshold=0.015)
        ind.RSI(df, panel=2, low=25, high=75)
        ind.EMA(df, 'SMA_RSI', period=14, column='RSI', panel=2, color='blue', display=display_ind)
    if name == 'rsi_macd_stoch_strat':
        backcandles = 5
        ind.EMA(df, 'EMA_200', period=200, color='yellow', display=False)
        ind.stochastic(df, period=14, k_smoothing=3, d_smoothing=3, panel=1, display=display_ind)
        ind.MACD(df, panel=2, color='white', signal_color='orange', display=display_ind)
        ind.RSI(df, name='RSI_14', period=14, panel=3, display=display_ind)
        ind.SMA(df, name='SMA_RSI_14', period=14, column='RSI_14', panel=3, color='white', display=display_ind)
        ind.SMA(df, name='SMA_volume', period=30, column='Volume', display=False)
        ind.strend(df, period=10, multiplier=2.5, display=False)
        ind.check_above_below_trend(df, MA =df['EMA_200'], window=8, above='upt_EMA', below='dwt_EMA', column1='Open', column2='Close')
        df['long_stoch_signal'] = df['stochastic'].rolling(window=backcandles).min() < 20
        df['long_slow_stoch_signal'] = df['slow_stochastic'].rolling(window=backcandles).min() < 20
        # ind.multi_timeframe(df, prev_timeframe='5m', new_timeframe='1h', indicator=ind.strend, kwargs={'name' : 'strend_1h', 'period' : 30, 'multiplier' : 15})

def get_data(df, col, index):
    if index >= len(df):
        return -1
    return df.iloc[index, df.columns.get_loc(col)]

#Profitable
def vwap_bol_strat_30m(df, i, atr_multiplier=0.5, profit_margin=1):
    orders = []
    exits = {'force_close_long' : False, 'force_close_short' : False}

    long_trade_signal = False
    short_trade_signal = False

    dwt = 1
    upt = 1
      
    high = max(get_data(df, 'Open', i), get_data(df, 'Close', i)) >= get_data(df, 'Bol_H', i)
    low = min(get_data(df, 'Open', i), get_data(df, 'Close', i)) <= get_data(df, 'Bol_L', i)
    
    entry_price = get_data(df, 'Close', i)
    short_stop_loss = entry_price + get_data(df, 'ATR', i) * atr_multiplier
    short_take_profit = entry_price - (short_stop_loss - entry_price) * profit_margin
    long_stop_loss = entry_price - get_data(df,'ATR', i) * atr_multiplier
    long_take_profit = entry_price + (entry_price - long_stop_loss) * profit_margin

    max_backcandles = 2
    backcandles = 0
    found = False
    for j in reversed(range(i - max_backcandles, i + 1)):
        low = min(get_data(df, 'Open', j), get_data(df, 'Close', j)) <= get_data(df, 'Bol_L', j)
        high = max(get_data(df, 'Open', j), get_data(df, 'Close', j)) >= get_data(df, 'Bol_H', j)
        if low and not high:
            found = True
            dwt = 0
            break
        if high and not low:
            found = True
            upt = 0
            break
        backcandles+=1

    if not found:
        dwt = 0
        upt = 0
    else:
        for j in range(i - backcandles, i + 1):
            if get_data(df, 'Type', j) == 'Bear' or get_data(df, 'Close', j) < get_data(df, 'Close', j - 1):
                upt = 0
            if get_data(df, 'Type', j) == 'Bull' or get_data(df, 'Close', j) > get_data(df, 'Close', j - 1):
                dwt = 0

    long_trade_signal = upt and not dwt
    long_trade_signal = long_trade_signal and get_data(df, 'RSI_vwap', i) < get_data(df, 'SMA_RSI_vwap', i)
    long_trade_signal = long_trade_signal and get_data(df, 'Close', i) > get_data(df, 'High', i - backcandles - 1)
    long_trade_signal = long_trade_signal and get_data(df, 'Bol_H', i) < get_data(df, 'EMA_200', i)
    long_trade_signal = long_trade_signal and max(get_data(df, 'Open', i), get_data(df, 'Close', i)) < get_data(df, 'EMA_200', i)

    short_trade_signal = dwt and not upt
    short_trade_signal = short_trade_signal and get_data(df, 'RSI_vwap', i) > get_data(df, 'SMA_RSI_vwap', i)
    short_trade_signal = short_trade_signal and get_data(df, 'Close', i) < get_data(df, 'Low', i - backcandles - 1)
    short_trade_signal = short_trade_signal and get_data(df, 'Bol_L', i) > get_data(df, 'EMA_200', i)
    short_trade_signal = short_trade_signal and min(get_data(df, 'Open', i), get_data(df, 'Close', i)) > get_data(df, 'EMA_200', i)

    if long_trade_signal: orders.append({'Type' : 'Long', 'Entry' : entry_price, 'stop_loss' : long_stop_loss, 'take_profit' : long_take_profit})
    if short_trade_signal: orders.append({'Type' : 'Short', 'Entry' : entry_price, 'stop_loss' : short_stop_loss, 'take_profit' : short_take_profit})

    return orders, exits

#Profitable
def vwap_bol_strat_5m(df, i, atr_multiplier=0.5, profit_margin=1):
    orders = []
    exits = {'force_close_long' : False, 'force_close_short' : False}

    long_trade_signal = False
    short_trade_signal = False

    dwt = get_data(df, 'dwt_vwap', i)
    upt = get_data(df, 'upt_vwap', i)
      
    high = max(get_data(df, 'Open', i), get_data(df, 'Close', i)) >= get_data(df, 'Bol_H', i)
    low = min(get_data(df, 'Open', i), get_data(df, 'Close', i)) <= get_data(df, 'Bol_L', i)

    max_backcandles = 1
    backcandles = 0
    found = False
    for j in reversed(range(i - max_backcandles, i + 1)):
        low = min(get_data(df, 'Open', j), get_data(df, 'Close', j)) <= get_data(df, 'Bol_L', j)
        high = max(get_data(df, 'Open', j), get_data(df, 'Close', j)) >= get_data(df, 'Bol_H', j)
        if low and not high:
            found = True
            dwt = 0
            break
        if high and not low:
            found = True
            upt = 0
            break
        backcandles+=1

    if not found:
        dwt = 0
        upt = 0
    else:
        for j in range(i - backcandles, i + 1):
            if get_data(df, 'Type', j) == 'Bear' or get_data(df, 'Close', j) < get_data(df, 'Close', j - 1):
                upt = 0
            if get_data(df, 'Type', j) == 'Bull' or get_data(df, 'Close', j) > get_data(df, 'Close', j - 1):
                dwt = 0

    trigger_candle = i - backcandles - 1

    entry_price = get_data(df, 'Close', i)
    short_stop_loss = get_data(df, 'High', trigger_candle) + get_data(df, 'ATR', trigger_candle) * atr_multiplier
    short_take_profit = entry_price - (short_stop_loss - entry_price) * profit_margin
    long_stop_loss = get_data(df, 'Low', trigger_candle) - get_data(df,'ATR', trigger_candle) * atr_multiplier
    long_take_profit = entry_price + (entry_price - long_stop_loss) * profit_margin

    hit = min(get_data(df, 'Open', i), get_data(df, 'Close', i)) <= get_data(df, 'Bol_L', i) or max(get_data(df, 'Open', i), get_data(df, 'Close', i)) >= get_data(df, 'Bol_H', i)
    exits['force_close_long'] = hit
    exits['force_close_short'] = hit

    long_trade_signal = upt and not dwt
    long_trade_signal = long_trade_signal and min(get_data(df, 'Open', trigger_candle), get_data(df, 'Close', trigger_candle)) <= get_data(df, 'Bol_L', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI_vwap', trigger_candle) < get_data(df, 'SMA_RSI_vwap', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI_vwap', i) < get_data(df, 'SMA_RSI_vwap', i)
    long_trade_signal = long_trade_signal and get_data(df, 'Close', i) > get_data(df, 'Bol_L', i)
    long_trade_signal = long_trade_signal and get_data(df, 'Close', i) > get_data(df, 'High', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'Bol_H', i) < get_data(df, 'EMA_200', i)
    long_trade_signal = long_trade_signal and max(get_data(df, 'Open', i), get_data(df, 'Close', i)) < get_data(df, 'EMA_200', i)

    short_trade_signal = dwt and not upt
    short_trade_signal = short_trade_signal and max(get_data(df, 'Open', trigger_candle), get_data(df, 'Close', trigger_candle)) >= get_data(df, 'Bol_H', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'RSI_vwap', trigger_candle) > get_data(df, 'SMA_RSI_vwap', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'RSI_vwap', i) > get_data(df, 'SMA_RSI_vwap', i)
    short_trade_signal = short_trade_signal and get_data(df, 'Close', i) < get_data(df, 'Bol_H', i)
    short_trade_signal = short_trade_signal and get_data(df, 'Close', i) < get_data(df, 'Low', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'Bol_L', i) > get_data(df, 'EMA_200', i)
    short_trade_signal = short_trade_signal and min(get_data(df, 'Open', i), get_data(df, 'Close', i)) > get_data(df, 'EMA_200', i)
    
    if long_trade_signal: orders.append({'Type' : 'Long', 'Entry' : get_data(df, 'Close', i), 'stop_loss' : long_stop_loss, 'take_profit' : long_take_profit})
    if short_trade_signal: orders.append({'Type' : 'Short', 'Entry' : get_data(df, 'Close', i), 'stop_loss' : short_stop_loss, 'take_profit' : short_take_profit})

    return orders, exits

#Profitable
def vwap_bol_adf_strat_5m(df, i, atr_multiplier=0.5, profit_margin=1):
    orders = []
    exits = {'force_close_long' : False, 'force_close_short' : False}

    long_trade_signal = False
    short_trade_signal = False

    dwt = get_data(df, 'dwt_vwap', i)
    upt = get_data(df, 'upt_vwap', i)
      
    high = max(get_data(df, 'Open', i), get_data(df, 'Close', i)) >= get_data(df, 'Bol_H', i)
    low = min(get_data(df, 'Open', i), get_data(df, 'Close', i)) <= get_data(df, 'Bol_L', i)

    max_backcandles = 1
    backcandles = 0
    found = False
    for j in reversed(range(i - max_backcandles, i + 1)):
        low = min(get_data(df, 'Open', j), get_data(df, 'Close', j)) <= get_data(df, 'Bol_L', j)
        high = max(get_data(df, 'Open', j), get_data(df, 'Close', j)) >= get_data(df, 'Bol_H', j)
        if low and not high:
            found = True
            dwt = 0
            break
        if high and not low:
            found = True
            upt = 0
            break
        backcandles+=1

    if not found:
        dwt = 0
        upt = 0
    else:
        for j in range(i - backcandles, i + 1):
            if get_data(df, 'Type', j) == 'Bear' or get_data(df, 'Close', j) < get_data(df, 'Close', j - 1):
                upt = 0
            if get_data(df, 'Type', j) == 'Bull' or get_data(df, 'Close', j) > get_data(df, 'Close', j - 1):
                dwt = 0

    trigger_candle = i - backcandles - 1

    entry_price = get_data(df, 'Close', i)
    short_stop_loss = entry_price + get_data(df, 'ATR', i) * get_data(df, 'ADF', i) * atr_multiplier
    short_take_profit = entry_price - (short_stop_loss - entry_price) * profit_margin
    long_stop_loss = entry_price - get_data(df,'ATR', i) * get_data(df, 'ADF', i) * atr_multiplier
    long_take_profit = entry_price + (entry_price - long_stop_loss) * profit_margin

    exits['force_close_long'] = min(get_data(df, 'Open', i), get_data(df, 'Close', i)) <= get_data(df, 'Bol_L', i)
    exits['force_close_short'] = max(get_data(df, 'Open', i), get_data(df, 'Close', i)) >= get_data(df, 'Bol_H', i)

    long_trade_signal = upt and not dwt
    long_trade_signal = long_trade_signal and min(get_data(df, 'Open', trigger_candle), get_data(df, 'Close', trigger_candle)) <= get_data(df, 'Bol_L', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI_vwap', trigger_candle) < get_data(df, 'SMA_RSI_vwap', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI_vwap', i) < get_data(df, 'SMA_RSI_vwap', i)
    long_trade_signal = long_trade_signal and get_data(df, 'Close', i) > get_data(df, 'High', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'Bol_H', i) < get_data(df, 'EMA_200', i)
    long_trade_signal = long_trade_signal and max(get_data(df, 'Open', i), get_data(df, 'Close', i)) < get_data(df, 'EMA_200', i)

    short_trade_signal = dwt and not upt
    short_trade_signal = short_trade_signal and max(get_data(df, 'Open', trigger_candle), get_data(df, 'Close', trigger_candle)) >= get_data(df, 'Bol_H', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'RSI_vwap', trigger_candle) > get_data(df, 'SMA_RSI_vwap', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'RSI_vwap', i) > get_data(df, 'SMA_RSI_vwap', i)
    short_trade_signal = short_trade_signal and get_data(df, 'Close', i) < get_data(df, 'Low', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'Bol_L', i) > get_data(df, 'EMA_200', i)
    short_trade_signal = short_trade_signal and min(get_data(df, 'Open', i), get_data(df, 'Close', i)) > get_data(df, 'EMA_200', i)
    
    if long_trade_signal: orders.append({'Type' : 'Long', 'Entry' : get_data(df, 'Close', i), 'stop_loss' : long_stop_loss, 'take_profit' : long_take_profit})
    if short_trade_signal: orders.append({'Type' : 'Short', 'Entry' : get_data(df, 'Close', i), 'stop_loss' : short_stop_loss, 'take_profit' : short_take_profit})

    return orders, exits

def rsi_macd_stoch_strat(df, i, atr_multiplier=1, profit_margin=1):
    orders = []
    exits = {'force_close_long' : False, 'force_close_short' : False}

    long_trade_signal = False

    entry_price = get_data(df, 'Close', i)

    long_stop_loss = get_data(df, 'Swing_L', i) - get_data(df, 'ATR', i) * atr_multiplier
    long_take_profit = entry_price + (entry_price - long_stop_loss) * profit_margin

    long_trade_signal = get_data(df, 'long_stoch_signal', i) and get_data(df, 'long_slow_stoch_signal', i)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI_14', i) > get_data(df, 'SMA_RSI_14', i)
    long_trade_signal = long_trade_signal and get_data(df, 'MACD', i) > get_data(df, 'MACD_signal', i)
    long_trade_signal = long_trade_signal and get_data(df, 'Volume', i) > get_data(df, 'SMA_volume', i)
    long_trade_signal = long_trade_signal and get_data(df, 'upt_EMA', i)
    long_trade_signal = long_trade_signal and get_data(df, 'strend', i)
    long_trade_signal = long_trade_signal and get_data(df, 'stochastic', i) > 20 and get_data(df, 'stochastic', i) < 80

    if long_trade_signal: orders.append({'Type' : 'Long', 'Entry' : entry_price, 'stop_loss' : long_stop_loss, 'take_profit' : long_take_profit})

    return orders, exits

def template_strat(df, i, atr_multiplier=1, profit_margin=1):
    orders = []
    exits = {'force_close_long' : False, 'force_close_short' : False}

    long_trade_signal = False
    short_trading = False

    entry_price = get_data(df, 'Close', i)
    short_stop_loss = entry_price + get_data(df, 'ATR', i) * atr_multiplier
    short_take_profit = entry_price - (short_stop_loss - entry_price) * profit_margin
    long_stop_loss = entry_price - get_data(df,'ATR', i) * atr_multiplier 
    long_take_profit = entry_price + (entry_price - long_stop_loss) * profit_margin

    if long_trade_signal: orders.append({'Type' : 'Long', 'Entry' : entry_price, 'stop_loss' : long_stop_loss, 'take_profit' : long_take_profit})
    if short_trading: orders.append({'Type' : 'Short', 'Entry' : entry_price, 'stop_loss' : short_stop_loss, 'take_profit' : short_take_profit})
    
    return orders, exits

def test_strat(df, i, atr_multiplier=1, profit_margin=1):
    orders = []
    exits = {'force_close_long' : False, 'force_close_short' : False}

    upt = get_data(df, 'upt_vwap', i)
    dwt = get_data(df, 'dwt_vwap', i)

    entry_price = get_data(df, 'Open', i)
    short_stop_loss = entry_price + get_data(df, 'ATR', i) * atr_multiplier
    short_take_profit = entry_price - (short_stop_loss - entry_price) * profit_margin
    long_stop_loss = entry_price - get_data(df,'ATR', i) * atr_multiplier
    long_take_profit = entry_price + (entry_price - long_stop_loss) * profit_margin

    long_trade_signal = upt and not dwt
    long_trade_signal = long_trade_signal and get_data(df, 'Type', i) == 'Bull' and  get_data(df, 'Type', i - 1) == 'Bear'
    long_trade_signal = long_trade_signal and get_data(df, 'Close', i - 1) < get_data(df, 'Bol_L', i - 1)
    long_trade_signal = long_trade_signal and get_data(df, 'Close', i) > get_data(df, 'Bol_L', i)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI', i - 1) < 45

    short_trade_signal = dwt and not upt
    short_trade_signal = short_trade_signal and get_data(df, 'Type', i) == 'Bear' and get_data(df, 'Type', i - 1) == 'Bull'
    short_trade_signal = short_trade_signal and get_data(df, 'Close', i - 1) > get_data(df, 'Bol_H', i - 1)
    short_trade_signal = short_trade_signal and get_data(df, 'Close', i) < get_data(df, 'Bol_H', i)
    short_trade_signal = short_trade_signal and get_data(df, 'RSI', i - 1) > 55

    # if long_trade_signal: ind.set_data(df, i, 'long', get_data(df, 'Close', i))
    # if short_trade_signal: ind.set_data(df, i, 'short', get_data(df, 'Close', i))

    if long_trade_signal: orders.append({'Type' : 'Long', 'Entry' : get_data(df, 'Close', i), 'stop_loss' : long_stop_loss, 'take_profit' : long_take_profit})
    if short_trade_signal: orders.append({'Type' : 'Short', 'Entry' : get_data(df, 'Close', i), 'stop_loss' : short_stop_loss, 'take_profit' : short_take_profit})

    return orders, exits