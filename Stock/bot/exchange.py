import ccxt
import API

exchange = ccxt.phemex({
    'apiKey': API.API_KEY,
    'secret': API.API_SECRET,
    'enableRateLimit': True,
})

def get_positions(symbol):
    positions_raw = exchange.fetch_positions(symbols=[symbol], params={'hedged' : True})
    positions = [x for x in positions_raw if float(x['info']['size']) > 0]
    return positions

def open_position(symbol, side, max_spend, leverage=1, value=-1, price=-1):
    book = exchange.fetch_order_book(symbol)

    if price == -1: price = book['bids'][0][0] if side == 'Long' else book['asks'][0][0]

    balance = min(max_spend, round(exchange.fetch_balance(params={'type' : 'swap', 'code' : 'USDT'})['USDT']['free'], 2))

    if value == -1:
        value = balance
        amount = ((value * leverage) / price)
    else:
        value = value
        leverage = value / balance
        amount = (value / price)
    
    exchange.set_leverage(round(leverage, 2), "BTC/USDT:USDT", {"hedged" : True})

    params = {'timeInForce' : 'PostOnly', "hedged" : True, 'posSide' : side}
    return exchange.create_order(symbol=symbol, type='limit', side='buy' if side == 'Long' else 'sell', amount=amount, price=price, params=params)

def close_position(symbol, position, price=-1):
    book = exchange.fetch_order_book(symbol)

    side = 'Long' if position['info']['side'] == 'Buy' else 'Short'
    params = {'timeInForce' : 'PostOnly', "hedged" : True, 'posSide' : side, 'closePosition' : 'Close-All'}
    if price == -1: price = book['asks'][0][0] if side == 'Long' else book['bids'][0][0]

    amount = position['info']['size']    
    return exchange.create_order(symbol=symbol, type='limit', side='sell' if side == 'Long' else 'buy', amount=amount, price=price, params=params)

def create_stop_loss_take_profit(symbol, side, sl_price, tp_price):
    positions = exchange.fetch_positions(symbols=[symbol])[0]
    amount = positions['info']['size']    

    sl_params = {
        'timeInForce' : 'PostOnly', 
        "hedged" : True, 
        'posSide' : side,
        'triggerType' : 'ByLastPrice',
        'stopPx' : sl_price,
    }

    tp_params = {
        'timeInForce' : 'PostOnly', 
        "hedged" : True, 
        'posSide' : side,
    }

    sl = exchange.create_order(symbol, type='StopLimit', side='sell' if side == 'Long' else 'buy', amount=amount, price=sl_price, params=sl_params)
    tp = exchange.create_order(symbol, type='limit', side='sell' if side == 'Long' else 'buy', amount=amount, price=tp_price, params=tp_params)

    return sl, tp

def get_orders():
    orders_raw = exchange.fetch_orders(symbol='BTC/USDT:USDT')
    orders = [x for x in orders_raw if x['status'] == 'open']
    for i in orders:
        print(i)
    return orders

def cancel_all_orders(symbol):
    exchange.cancel_all_orders(symbol=symbol)

def cancel_order(order):
    for i in get_orders():
        if i['id'] == order['id']:
            print('found')
    side = 'Long' if order['info']['side'] == 'Buy' else 'Short'
    exchange.cancel_order(id=order['id'], symbol=order['symbol'], params={'hedged' : True, 'posSide' : side})

def fetch_account_balance(code):
    try:
        params = {'type' : 'swap', 'code' : code}
        balance = exchange.fetch_balance(params=params)[code]
        return balance
    except:
        return None

def get_order_book(symbol):
    book = exchange.fetch_order_book(symbol)
    asks = book['asks'][:10]
    bids = book['bids'][:10]
    return {'asks' : asks, 'bids' : bids}

def fetch_current_price(symbol):
    book = exchange.fetch_order_book(symbol)
    return {'bid' : book['bids'][0][0], 'ask' : book['asks'][0][0]}