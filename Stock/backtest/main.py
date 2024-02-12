import ccxt
import numpy as np
import strategies
import plot as pl
import sys
import loader
import random

marker_size = 100
total_money = 0
total_wins = 0
total_loss = 0
ATR_data = {}

def calculate_position_size(balance, entry_price, stop_loss, risk, entry_commision, exit_commision):
    maximum_loss = abs(entry_price - stop_loss)
    exit_fee = stop_loss * exit_commision
    entry_fee = entry_price * entry_commision
    maximum_loss = maximum_loss + exit_fee + entry_fee
    shares = (balance * risk) / maximum_loss 
    return shares

def simulate_contracts(stock, df, strategy, starting_money, max_spend, max_hold=50, monthly_deposit=0, commisions={}, profit_panel=1, debug=False, display=True, display_ind=False):
    pl.plots.clear()
    data = df.copy()
    short_trading = False
    long_trading = False

    order_list = []

    money = starting_money
    money_start = starting_money
    shares = 0
    net_worth = starting_money
    contract = 0

    stop_loss_plot = [np.nan]
    take_profit_plot = [np.nan]

    exit_long_signals = [np.nan]
    enter_long_signals = [np.nan]
    exit_short_signals = [np.nan]
    enter_short_signals = [np.nan]

    total_equity = [np.nan]

    trending_up_plot = [np.nan]
    trending_down_plot = [np.nan]

    loss_signals = [np.nan]

    total_entry_long = 0
    total_exit_long = 0
    total_entry_short = 0
    total_exit_short = 0

    total_assets = [starting_money]

    max_networth = 0
    min_networth = 1e9

    win_count = 0
    loss_count = 0

    entry_time = 0
    average_hold = 0
    average_win = 0
    average_loss = 0
    longest_hold = 0
    longest_losing_streak = 0
    losing_streak = 0
    timed_out = 0

    post_only_miss = 0

    min_margin = 1e9
    max_margin = -1e9

    idle = 0
    max_idle = -1

    strat = 'nan'

    for s in strategy:              
        strategies.load_indicators(data, s['strat'], kwargs=s['kwargs'], display_ind=display_ind)

    for i in range(1, len(data.index)):
        if i % (loader.time_frame[interval] * 30) == 0: 
            money += monthly_deposit
            max_spend += monthly_deposit

        market_open_price = strategies.get_data(data, 'Open', i)
        market_close_price = strategies.get_data(data, 'Close', i)
        market_low_price = strategies.get_data(data, 'Low', i)
        market_high_price = strategies.get_data(data, 'High', i)
        
        enter_long_signals.append(np.nan)
        exit_long_signals.append(np.nan)
        enter_short_signals.append(np.nan)
        exit_short_signals.append(np.nan)    
        loss_signals.append(np.nan)
        trending_up_plot.append(np.nan)
        trending_down_plot.append(np.nan)

        total_equity.append(money)

        if not debug: 
            print(f"Computing Progress: %{int(i / len(data.index) * 100)}, ATR Multiplier: {s['atr_multiplier']}")
            sys.stdout.write("\033[F") 

        for s in strategy:
            orders, exits = s['strat'](data, i, profit_margin=s['profit_margin'], atr_multiplier=s['atr_multiplier'])

            order = {'Type' : None, 'Entry' : market_close_price}

            if not (long_trading or short_trading):
                stop_loss = np.nan
                take_profit = np.nan
                order_list += orders
                for j in order_list:
                    if j['Entry'] >= market_low_price and j['Entry'] <= market_high_price:
                        order = j
                        order_list.clear()
                        break
                idle += 1
                max_idle = max(max_idle, idle)
            else:
                idle = 0

            entry_price = order['Entry']

            #Long Exit
            if long_trading and s['strat'].__name__ == strat and (stop_loss > market_low_price or take_profit < market_high_price or i - entry_time > max_hold or exits['force_close_long']):
                exit_price = 0
                exit_commision = 0
                average_hold += i - entry_time
                if take_profit < market_high_price:
                    exit_price = take_profit + random.randint(0, post_only_miss)
                    exit_commision = commisions['take_profit']
                elif stop_loss > market_low_price:
                    exit_price = stop_loss - random.randint(0, post_only_miss)
                    exit_commision = commisions['stop_loss']
                else:
                    if debug: print(f"Long trade forced exit at: ${market_open_price}")
                    timed_out += 1
                    if exits['force_close_long']:
                        exit_price = market_close_price
                    else:
                        exit_price = market_open_price
                    exit_commision = commisions['take_profit']
                                
                gains = shares * exit_price
                fee = contract * commisions['entry'] + gains * exit_commision
                money += (gains - contract) - fee
                if debug: print(f"Time: {df.index[i]}, Exiting Long Trade at: ${exit_price} per share, Actual Price ${gains}, Buying Contract: {contract}, gained ${money - money_start}, Exchange took {fee}, Held time: {i - entry_time}.")
                if money > money_start: 
                    win_count += 1
                    losing_streak = 0
                    average_win += money - money_start
                else:
                    loss_count += 1
                    losing_streak += 1
                    longest_losing_streak = max(longest_losing_streak, losing_streak)
                    loss_signals[i] = market_high_price
                    average_loss += abs(money - money_start)
                shares = 0
                long_trading= False
                exit_long_signals[i] = market_high_price
                total_exit_long += 1
                longest_hold = max(longest_hold, i - entry_time)
                strat = 'nan'

            #Short Exit
            if short_trading and s['strat'].__name__ == strat and (stop_loss < market_high_price or take_profit > market_low_price or i - entry_time > max_hold or exits['force_close_short']):
                exit_price = 0
                exit_commision = 0
                average_hold += i - entry_time
                if take_profit > market_low_price:
                    exit_price = take_profit - random.randint(0, post_only_miss)
                    exit_commision = commisions['take_profit']
                elif stop_loss < market_high_price:
                    exit_price = stop_loss + random.randint(0, post_only_miss)
                    exit_commision = commisions['stop_loss']
                else:
                    if debug: print(f"Short trade forced exit at: ${market_open_price}")
                    timed_out += 1
                    if exits['force_close_short']:
                        exit_price = market_close_price
                    else:
                        exit_price = market_open_price
                    exit_commision = commisions['take_profit']
                gains = shares * exit_price
                fee = contract * commisions['entry'] + gains * exit_commision
                money += (contract - gains) - fee
                if debug: 
                    print(f"Time: {df.index[i]}, Exiting Short Trade at: ${exit_price} per share, Actual Price ${gains}, Selling Contract: {contract}, gained ${money - money_start}, Exchange took {fee}, Held time: {i - entry_time}")
                if money > money_start: 
                    win_count += 1
                    losing_streak = 0
                    average_win += money - money_start
                else: 
                    loss_count += 1 
                    losing_streak += 1
                    longest_losing_streak = max(longest_losing_streak, losing_streak)
                    loss_signals[i] = market_low_price
                    average_loss += abs(money - money_start)
                shares = 0
                short_trading = False
                exit_short_signals[i] = market_low_price
                total_exit_short += 1
                longest_hold = max(longest_hold, i - entry_time)
                strat = 'nan'

            #Long Entry
            if not (long_trading or short_trading) and order['Type'] == 'Long':
                money_start = money
                money_use = min(money, max_spend)
                shares = calculate_position_size(balance=money_use, entry_price=entry_price, stop_loss=order['stop_loss'], risk=s['risk'], entry_commision=commisions['entry'], exit_commision=commisions['stop_loss'])
                contract = shares * entry_price
                long_margin = round(max(1, contract / money_use), 2)
                min_margin = min(min_margin, long_margin)
                max_margin = max(max_margin, long_margin)
                if debug: 
                    print(f"Stop Loss: {order['stop_loss']}, Take Profit: {order['take_profit']}")
                    print(f"Time: {df.index[i]}, Entering Long Trade at: ${entry_price} per share, with ${money_start}, {shares} shares, Margin: {long_margin}, Contract: {contract}")
                long_trading = True
                stop_loss = order['stop_loss']
                take_profit = order['take_profit']
                enter_long_signals[i] = entry_price
                total_entry_long += 1
                entry_time = i
                strat = s['strat'].__name__

            #Short Entry
            if not (long_trading or short_trading) and order['Type'] == 'Short':
                money_start = money
                money_use = min(money, max_spend)
                shares = calculate_position_size(balance=money_use, entry_price=entry_price, stop_loss=order['stop_loss'], risk=s['risk'], entry_commision=commisions['entry'], exit_commision=commisions['stop_loss'])
                contract = shares * entry_price
                short_margin = round(max(1, contract / money_use), 2)
                min_margin = min(min_margin, short_margin)
                max_margin = max(max_margin, short_margin)
                if debug: 
                    print(f"Stop Loss: {order['stop_loss']}, Take Profit: {order['take_profit']}")
                    print(f"Time: {df.index[i]}, Entering Short Trade at: ${entry_price} per share, with {shares} shares and ${money}, Margin: {short_margin}, Contract: {contract}.")
                if shares > 0: short_trading = True
                stop_loss = order['stop_loss']
                take_profit = order['take_profit']
                enter_short_signals[i] = entry_price
                total_entry_short += 1
                entry_time = i
                strat = s['strat'].__name__

        net_worth = money
        max_networth = max(max_networth, net_worth)
        min_networth = min(min_networth, net_worth)
        total_assets.append(net_worth)

        if net_worth <= 0:
            print("Uh oh! You're broke :(")
            break
        
        stop_loss_plot.append(stop_loss)
        take_profit_plot.append(take_profit)

    
    if(short_trading):
        money += contract - (shares * market_close_price) - contract * commisions['entry'] - (shares * market_close_price) * commisions['take_profit']
        shares = 0

    if(long_trading):
        money += (shares * market_close_price) - contract - contract * commisions['entry'] - (shares * market_close_price) * commisions['take_profit']
        shares = 0

    print('\n')
    print(f"----{stock}----")
    print(f"You started with ${starting_money},and ended with ${money}.")
    print(f"That is a %{net_worth / starting_money * 100 - 100} increase.")
    print(f"Total transactions: {total_entry_long + total_entry_short}")
    if win_count + loss_count > 0: 
        print(f"Total Wins: {win_count}, Total Losses: {loss_count}, Win Rate: {win_count / (win_count + loss_count) * 100}%")
        if win_count > 0: print(f"Average Win: {average_win / win_count}")
        if loss_count > 0: print(f"Average Loss: {average_loss / loss_count}")
        print(f"Average Hold: {average_hold / (win_count + loss_count)}")
        print(f"Longest Hold: {longest_hold}")
        print(f"Longest Losing Streak: {longest_losing_streak}")
        print(f"Trade Forced Exits: {timed_out}")
    else: print("Total Wins: 0")
    print(f'Minimum Margin: {min_margin}')
    print(f'Maximum Margin: {max_margin}')
    print(f'Longest Time without Activity: {max_idle / loader.time_frame["5m"]} Days')
    deposit = ((period / 30) - 1) * monthly_deposit + starting_money
    print(f"You Deposited ${deposit}, You earned ${net_worth - deposit}")
    print(f"That is a %{(net_worth - deposit) / deposit * 100} increase.")
    print(f"Highest Net Worth: {max_networth}, Lowest Net Worth: {min_networth}")

    if display:
        if total_exit_long > 0: pl.add_marker(exit_long_signals, marker_size=marker_size, symbol='^', color='red')  
        if total_entry_long > 0: pl.add_marker(enter_long_signals, marker_size=marker_size, symbol='^', color='green')
        if total_exit_short > 0: pl.add_marker(exit_short_signals, marker_size=marker_size, symbol='v', color='red')
        if total_entry_short > 0: pl.add_marker(enter_short_signals, marker_size=marker_size, symbol='v', color='green')
        if loss_count > 0: pl.add_marker(loss_signals, marker_size=marker_size, symbol='*', color='purple')
        if total_entry_long + total_entry_short > 0: pl.add_plot(stop_loss_plot, '#fc7c7c')
        if total_entry_long + total_entry_short > 0: pl.add_plot(take_profit_plot, '#87fc7c')
        if data['long'].sum(): pl.add_marker(data['long'], color='yellow', marker_size=marker_size, symbol='.', panel=0)
        if data['short'].sum():pl.add_marker(data['short'], color='white', marker_size=marker_size, symbol='.', panel=0)

        pl.add_plot(total_equity,  color='white', panel=profit_panel)
        pl.plot_data(stock=stock, data=data, type='candle', volume=False)
        
    global total_money, total_wins, total_loss, ATR_data
    total_money += money
    total_wins += win_count
    total_loss += loss_count

    print(len(df))
    print(len(df.index))

period = 30
interval = '5m'
starting_money = 200
max_hold = 12 * 24
monthly_deposit = 200

commisions = {
    'entry' : 0.01 / 100,
    'stop_loss' : 0.01 / 100,
    'take_profit' : 0.01 / 100,
}

stocks = {
    'BTC/USDT:USDT' : True
}

'''
TODO
    - 1H timeframe EMA to determine Uptrend/Downtrend 
'''
for stock in stocks: 
    # data = loader.get_real_time_stock_data('TSLA', period=period, interval=interval)
    # data = loader.get_historical_crypto_data_candles(ccxt.phemex(), stock, interval=interval, candles=1000)
    data = loader.get_historical_crypto_data(ccxt.phemex(), stock, interval=interval, period=period, debug=False)
    simulate_contracts(
                stock,
                data, 
                max_spend=starting_money,
                strategy=[
                    {'strat' : strategies.vwap_bol_adf_strat_5m, 'risk' : 5 / 100, 'atr_multiplier' : 1, 'profit_margin' : 2, 'kwargs' : {'Bol_period' : 14}}, 
                    {'strat' : strategies.rsi_macd_stoch_strat, 'risk' : 5 / 100, 'atr_multiplier' : 1, 'profit_margin' : 2, 'kwargs' : {}}, 
                ],
                starting_money=starting_money, 
                max_hold=max_hold,
                monthly_deposit=monthly_deposit,
                commisions=commisions,
                profit_panel=1,
                debug=False,
                display=stocks[stock],
                display_ind=False
            )

for i, j in ATR_data.items():
    print(f"ATR Multiplier: {i}, Profit: {j}")

# pl.plot_bar_graph(ATR_data, list(ATR_data.values()), list(ATR_data.keys()))