import numpy as np
import pandas as pd
import pytz
import sys
import stock_indicators as ind
import datetime as dt
import loader
import exchange

stop_loss = np.nan
take_profit = np.nan
position = 'nan'
strat = 'nan'
shares = np.nan
value = np.nan
held_time = np.nan
entry_time = np.nan
max_hold = 12 * 10
prev = None
max_spend = 1000

commisions = {
    'entry' : 0.01 / 100,
    'stop_loss' : 0.01 / 100,
    'take_profit' : 0.01 / 100,
}

VERSION = '1.18.0'

def get_data(df, col, index):
    return df.iloc[index, df.columns.get_loc(col)]

def set_data(data, index, col, value):
    data.iloc[index, data.columns.get_loc(col)] = value

def set_max_spend(money):
    global max_spend
    max_spend = money
    
def calculate_minute_diff(t1, t2):
    diff = (t1 - t2).astype('timedelta64[m]')
    diff = diff / np.timedelta64(1, 'm')
    return diff

def calculate_position_size(balance, entry_price, stop_loss, risk):
    global commisions
    maximum_loss = abs(entry_price - stop_loss)
    exit_fee = stop_loss * commisions['stop_loss']
    entry_fee = entry_price * commisions['entry']
    maximum_loss = maximum_loss + exit_fee + entry_fee
    shares = (balance * risk) / maximum_loss 
    return shares

def vwap_boll_strat(df, i, profit_margin=1):
    long_trade_signal = False
    short_trade_signal = False
    exits = {}

    atr_multiplier = 1
    
    ind.ATR(df, period=7)
    ind.bollinger_band(df, period=14, std=2)
    ind.RSI(df, period=14)
    ind.SMA(df, 'SMA_RSI', period=14, column='RSI')
    ind.EMA(df, period=200)
    ind.VWAP(df, name='VWAP')
    ind.ADF(df, MA=df['VWAP'])
    ind.check_above_below_trend(df, MA=df['VWAP'], window=5, above='dwt', below='upt', column1='Open', column2='Close')

    dwt = get_data(df, 'dwt', i)
    upt = get_data(df, 'upt', i)
      
    high = max(get_data(df, 'Open', i), get_data(df, 'Close', i)) >= get_data(df, 'Bol_H', i)
    low = min(get_data(df, 'Open', i), get_data(df, 'Close', i)) <= get_data(df, 'Bol_L', i)
    
    entry_price = get_data(df, 'Close', i)
    exits['short_stop_loss'] = entry_price + get_data(df, 'ATR', i) * get_data(df, 'ADF', i) * atr_multiplier
    exits['short_take_profit'] = entry_price - (exits['short_stop_loss'] - entry_price) * profit_margin
    exits['long_stop_loss'] = entry_price - get_data(df,'ATR', i) * get_data(df, 'ADF', i) * atr_multiplier
    exits['long_take_profit'] = entry_price + (entry_price - exits['long_stop_loss']) * profit_margin

    exits['force_close_long'] = min(get_data(df, 'Open', i), get_data(df, 'Close', i)) <= get_data(df, 'Bol_L', i)
    exits['force_close_short'] = max(get_data(df, 'Open', i), get_data(df, 'Close', i)) >= get_data(df, 'Bol_H', i)

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

    trigger_candle = i - backcandles - 1

    long_trade_signal = upt and not dwt
    long_trade_signal = long_trade_signal and min(get_data(df, 'Open', trigger_candle), get_data(df, 'Close', trigger_candle)) <= get_data(df, 'Bol_L', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI', trigger_candle) < get_data(df, 'SMA_RSI', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI', i) < get_data(df, 'SMA_RSI', i)
    long_trade_signal = long_trade_signal and get_data(df, 'Close', i) > get_data(df, 'High', trigger_candle)
    long_trade_signal = long_trade_signal and get_data(df, 'Bol_H', i) < get_data(df, 'EMA', i)
    long_trade_signal = long_trade_signal and max(get_data(df, 'Open', i), get_data(df, 'Close', i)) < get_data(df, 'EMA', i)
    
    short_trade_signal = dwt and not upt
    short_trade_signal = short_trade_signal and max(get_data(df, 'Open', trigger_candle), get_data(df, 'Close', trigger_candle)) >= get_data(df, 'Bol_H', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'RSI', trigger_candle) > get_data(df, 'SMA_RSI', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'RSI', i) > get_data(df, 'SMA_RSI', i)
    short_trade_signal = short_trade_signal and get_data(df, 'Close', i) < get_data(df, 'Low', trigger_candle)
    short_trade_signal = short_trade_signal and get_data(df, 'Bol_L', i) > get_data(df, 'EMA', i)
    short_trade_signal = short_trade_signal and min(get_data(df, 'Open', i), get_data(df, 'Close', i)) > get_data(df, 'EMA', i)

    return long_trade_signal, short_trade_signal, exits

def rsi_macd_stoch_strat(df, i, profit_margin=1):
    exits = {'force_close_long' : False, 'force_close_short' : False}

    backcandles = 5
    ind.ATR(df, period=7)
    ind.stochastic(df, period=14, k_smoothing=3, d_smoothing=3)
    ind.MACD(df)
    ind.RSI(df, period=14)
    ind.SMA(df, 'SMA_RSI', period=14, column='RSI')
    ind.SMA(df, 'SMA_volume', period=30, column='Volume')
    ind.EMA(df, 'EMA_200', period=200)
    ind.strend(df, period=10, multiplier=2.5)
    ind.check_above_below_trend(df, MA =df['EMA_200'], window=8, column1='Open', column2='Close')
    ind.swing_L_H(df, period=15)
    df['stoch_signal'] = df['stochastic'].rolling(window=backcandles).min() < 20
    df['slow_stoch_signal'] = df['slow_stochastic'].rolling(window=backcandles).min() < 20
    # ind.multi_timeframe(df, prev_timeframe='5m', new_timeframe='1h', indicator=ind.strend, kwargs={'name' : 'strend_1h', 'period' : 30, 'multiplier' : 15})

    atr_multiplier = 1
    
    entry_price = get_data(df, 'Close', i)

    exits['long_stop_loss']  = get_data(df, 'Swing_L', i) - get_data(df, 'ATR', i) * atr_multiplier
    exits['long_take_profit']  = entry_price + (entry_price - exits['long_stop_loss']) * profit_margin

    long_trade_signal = get_data(df, 'stoch_signal', i) and get_data(df, 'slow_stoch_signal', i)
    long_trade_signal = long_trade_signal and get_data(df, 'RSI', i) > get_data(df, 'SMA_RSI', i)
    long_trade_signal = long_trade_signal and get_data(df, 'MACD', i) > get_data(df, 'MACD_signal', i)
    long_trade_signal = long_trade_signal and get_data(df, 'Volume', i) > get_data(df, 'SMA_volume', i)
    long_trade_signal = long_trade_signal and get_data(df, 'upt', i)
    long_trade_signal = long_trade_signal and get_data(df, 'strend', i)
    long_trade_signal = long_trade_signal and get_data(df, 'stochastic', i) > 20 and get_data(df, 'stochastic', i) < 80

    return long_trade_signal, False, exits

def fetch_crypto_data(code, strategies, interval, i=0, data=None, debug=False):
    global position, strat, shares, value, held_time, stop_loss, take_profit, entry_time, max_hold, prev, commisions, max_spend

    buy_signal = False
    sell_signal = False
    
    if debug:
        df = data.iloc[:i].copy()
    else:
        ohlcv = exchange.exchange.fetch_ohlcv(code, interval, limit=500)
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

    current = df.iloc[-1]
    current_time = list(df.index.values)[-1]
    balance = exchange.fetch_account_balance('USDT')
    money = min(max_spend, balance['free'])

    gains = 0
    price = 0
    margin = 0
    fee = 0

    stock_data = ''
    stock_data += f'Version: {VERSION}\n'
    stock_data += f'\nIn Position: {position}, {strat}\n'
    stock_data += f'Last Updated: {dt.datetime.now()}\n'
    stock_data += f'Candle Start Time: {df.index[-1]}\n'
    stock_data += f'Current Max Spend: ${max_spend}\n'
    stock_data += f'Current Total Money: ${round(balance["free"], 2)} USDT\n'

    df['Searched'] = current_time == prev

    entry_price = current['Close']
    for s in strategies:
        strat_name = s['strat'].__name__
        long_signal, short_signal, exits = s['strat'](df=df, i=-1, profit_margin= s['profit_margin'])

        stock_data += f'\nStrategy: {strat_name}\n'
        for i in df:
            stock_data += f'{i}: {df[i].iloc[-1]}\n'

        if current_time != prev:
            if not position == 'nan':
                held_time = calculate_minute_diff(current_time, entry_time) / 5

            if position == 'Long' and strat == strat_name and (stop_loss > current['Low'] or take_profit < current['High'] or held_time > max_hold or exits['force_close_long']):
                exit_price = 0
                exit_commision = 0
                if take_profit < current['High']:
                    exit_price = take_profit
                    exit_commision = commisions['take_profit']
                elif stop_loss > current['Low']:
                    exit_price = stop_loss
                    exit_commision = commisions['stop_loss']
                else:
                    if exits['force_close_long']:
                        exit_price = current['Close']
                    else:
                        exit_price = current['Open']
                    exit_commision = commisions['take_profit']

                price = exit_price
                exit_value = shares * exit_price
                fee = value * commisions['entry'] + exit_value * exit_commision
                gains = exit_value - value - fee
                value = exit_value
                sell_signal = True
                break
            
            if position == 'Short' and strat == strat_name and (stop_loss < current['High'] or take_profit > current['Low'] or held_time > max_hold or exits['force_close_short']):
                exit_price = 0
                if take_profit > current['Low']:
                    exit_price = take_profit
                    exit_commision = commisions['take_profit']
                elif stop_loss < current['High']:
                    exit_price = stop_loss
                    exit_commision = commisions['stop_loss']
                else:
                    if exits['force_close_short']:
                        exit_price = current['Close']
                    else:
                        exit_price = current['Open']
                    exit_commision = commisions['take_profit']
                    
                price = exit_price
                exit_value = shares * exit_price
                fee = value * commisions['entry'] + exit_value * exit_commision
                gains = value - exit_value - fee
                value = exit_value
                sell_signal = True
                break
                
            if long_signal and position == 'nan':
                position = 'Long'
                strat = strat_name
                stop_loss = exits['long_stop_loss']
                take_profit = exits['long_take_profit']
                shares = calculate_position_size(balance=money, entry_price=entry_price, stop_loss=stop_loss, risk=s['risk'])
                value = shares * entry_price
                margin = round(value / money, 2)
                price = entry_price
                entry_time = current_time
                buy_signal = True
                break

            if short_signal and position == 'nan':
                position = 'Short'
                strat = strat_name
                stop_loss = exits['short_stop_loss']
                take_profit = exits['short_take_profit']
                shares = calculate_position_size(balance=money, entry_price=entry_price, stop_loss=stop_loss, risk=s['risk'])
                value = shares * entry_price
                margin = round(value / money, 2)
                price = entry_price
                entry_time = current_time
                buy_signal = True
                break
    
    prev = current_time

    positions = {}
    positions['Time'] = current_time
    positions['Position'] = position
    positions['Shares'] = shares
    positions['Value'] = value
    positions['Margin'] = margin
    positions['Held Time'] = held_time
    positions['Stop Loss'] = stop_loss
    positions['Take Profit'] = take_profit
    positions['Buy Signal'] = buy_signal
    positions['Sell Signal'] = sell_signal
    positions['Gains'] = gains
    positions['Price'] = price
    positions['Fee'] = fee

    if sell_signal:
        position = 'nan'
        strat = 'nan'

    return stock_data, positions

def test():
    global position, strat
    # start_date = 3
    # days = 3
    # minutes = 12 * 24 * days
    candles = 1000
    strategies = [
        {'strat' : vwap_boll_strat, 'risk' : 5 / 100, 'profit_margin' : 2},
        {'strat' : rsi_macd_stoch_strat, 'risk' : 5 / 100, 'profit_margin' : 2},
    ]
    data = loader.get_historical_crypto_data_candles('BTC/USDT:USDT', interval='5m', candles=candles)
    for i in range(10, candles):
        stock_data, positions = fetch_crypto_data(code='BTC/USDT:USDT', strategies=strategies, interval='5m', i=i, data=data, debug=True)
        print(f"Computing Progress: %{round(i / (candles) * 100, 2)}")
        sys.stdout.write("\033[F") 
        if positions['Buy Signal']:
            m = f"Entering {positions['Position']} position, {strat}, at {positions['Time']}\n"
            m += f"Shares: {positions['Shares']}\n"
            m += f"Value: {positions['Value']}\n"
            m += f"Price: {positions['Price']}\n"
            m += f"Margin: {positions['Margin']}\n"
            m += f"Stop Loss: {positions['Stop Loss']}\n"
            m += f"Take Profit: {positions['Take Profit']}\n"
            print(m)
        elif positions['Sell Signal']:
            m += f"Exiting Position at {positions['Time']}\n"
            m += f"Value: {positions['Value']}\n"
            m += f"Gain: ${positions['Gains']}\n"
            m += f"Price: {positions['Price']}\n"
            m += f"Fee: {positions['Fee']}\n"
            m += f"Held time: {positions['Held Time']}\n"
            print(m)
