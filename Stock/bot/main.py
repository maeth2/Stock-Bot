import discord
from discord.ext import tasks, commands
import exchange
import strategy
import asyncio

STOCK_CHANNEL = 1205165224462913609
GENERAL_CHANNEL = 1205164480842170371
POSITIONS_CHANNEL = 1205165155525201950
ERRORS_CHANNEL = 1205165136910745631

code = 'BTC/USDT:USDT'
interval = '5m'

client = commands.Bot(command_prefix='$', intents=discord.Intents.all())

async def delete_previous_message(channel):
    m = client.get_channel(int(channel)).last_message
    if m != None: await m.delete()

async def clear_channel(channel):
    await client.get_channel(int(channel)).purge()

async def send_message(message, channel, do_print=True):
    try:
        if do_print: print(f"Sending Message:\n{message}\nTo {client.get_channel(channel)}")
        await client.get_channel(channel).send(message)
    except Exception as e:
        print(e)

async def reply_message(ctx, message):
    try:
        await ctx.reply(message)
    except Exception as e:
        print(e)

async def open_position(symbol, side, leverage=1, value=-1, max_tries=10):
    order = None
    message = ''
    pos_count = len(exchange.get_positions(symbol=symbol))
    while max_tries > 0:
        book = exchange.get_order_book(symbol=symbol)
        ask = [i[0] for i in book['asks'][:5]]
        bid = [i[0] for i in book['bids'][:5]]
        print(f'Asks: {ask}, Bids: {bid}')
        if len(exchange.get_positions(symbol=symbol)) != pos_count: break
        if order == None or (side == 'Long' and not order['price'] in bid) or (side == 'Short' and not order['price'] in ask):
            exchange.cancel_all_orders(symbol=symbol)
            order = exchange.open_position(symbol=symbol, side=side, max_spend=strategy.max_spend, leverage=float(leverage), value=value)
            message = f"Entering {side} Position in {symbol} at ${order['info']['priceRp']}"
            max_tries -= 1
        await asyncio.sleep(5)
    
    if len(exchange.get_positions(symbol=symbol)) == pos_count: 
        exchange.cancel_all_orders(symbol=symbol)
        await send_message(message='Could not enter position', channel=POSITIONS_CHANNEL)
    else: 
        print(order)
        await send_message(message=message, channel=POSITIONS_CHANNEL)

async def close_position(symbol):
    message = ''
    pos_count = len(exchange.get_positions(symbol=symbol))
    if pos_count == 0:
        await send_message(message=f'Exited Position in {symbol}', channel=POSITIONS_CHANNEL)
        return
    position = exchange.get_positions(symbol=symbol)[0]
    order = None
    side = 'Long' if position['info']['side'] == 'Buy' else 'Short'
    while pos_count == len(exchange.get_positions(symbol=symbol)):
        book = exchange.get_order_book(symbol=symbol)
        ask = [i[0] for i in book['asks'][:5]]
        bid = [i[0] for i in book['bids'][:5]]
        print(f'Asks: {ask}, Bids: {bid}')
        if len(exchange.get_positions(symbol=symbol)) != pos_count: break
        if order == None or (side == 'Long' and not order['price'] in ask) or (side == 'Short' and not order['price'] in bid):
            exchange.cancel_all_orders(symbol=symbol)
            order = exchange.close_position(symbol=symbol, position=position)
            message = f"Exiting Position in {symbol} at ${order['info']['priceRp']} "
        await asyncio.sleep(5)
    print(order)
    await send_message(message=message, channel=POSITIONS_CHANNEL)

@client.command(name='spend')
async def spend(ctx, amount):
    await delete_previous_message(channel=ctx.message.channel.id)
    strategy.set_max_spend(float(amount))
    await send_message(f'Changed Max Spend to ${amount}.', channel=GENERAL_CHANNEL)

@client.command(name='balance')
async def balance(ctx, code):
    balance = exchange.fetch_account_balance(code=code)
    if balance == None:
        await reply_message(ctx, "Cannot find balance.")
    else:
        m = ''
        m += f"Current USDT Balance:\n"
        m += f"Total: {balance['total']}\n"
        m += f"Free: {balance['free']}\n"
        m += f"Used: {balance['used']}\n"
        await reply_message(ctx, m)

@client.command(name='clear')
async def clear(ctx):
    await clear_channel(channel=ctx.message.channel.id)

@client.command(name='stop')
async def stop(ctx):
    background_task.cancel()
    await send_message(message='Stopping Task...', channel=GENERAL_CHANNEL)

@client.command(name='open')
async def open(ctx, side, leverage):
    symbol = 'BTC/USDT:USDT'
    await open_position(symbol=symbol, side=side, leverage=leverage)
    await delete_previous_message(channel=ctx.message.channel.id)

@client.command(name='close')
async def close(ctx):
    symbol = 'BTC/USDT:USDT'
    await close_position(symbol=symbol)
    await delete_previous_message(channel=ctx.message.channel.id)

@client.command(name='resume')
async def resume(ctx):
    background_task.start()
    await send_message(message='Starting Task...', channel=GENERAL_CHANNEL)

@client.event
async def on_ready():
    background_task.start()
    print(f'{client.user} is now running!')
    await send_message(message=f'{client.user} is now running!', channel=GENERAL_CHANNEL)

@client.event
async def on_shard_resumed(shard_id):
    background_task.start()
    await send_message(message=f'Shard #{shard_id} is resuming', channel=GENERAL_CHANNEL)

@client.event
async def on_error(event, *args, **kwargs):
    await send_message(str(args[1]), channel=ERRORS_CHANNEL)

@client.event
async def on_command_error(msg, error):
    await send_message(str(error), channel=ERRORS_CHANNEL)

@tasks.loop(seconds=10)
async def background_task():
    symbol='BTC/USDT:USDT'
    strategies = [
        {'strat' : strategy.vwap_boll_strat, 'risk' : 15 / 100, 'profit_margin' : 2},
        {'strat' : strategy.rsi_macd_stoch_strat, 'risk' : 7 / 100, 'profit_margin' : 2},
    ]
    stock_data, positions = strategy.fetch_crypto_data(code=code, interval=interval, strategies=strategies)

    if positions['Buy Signal']:
        m = f"Entering {positions['Position']} position, {strategy.strat}, at {positions['Time']}\n"
        m += f"Shares: {positions['Shares']}\n"
        m += f"Value: {positions['Value']}\n"
        m += f"Price: {positions['Price']}\n"
        m += f"Margin: {positions['Margin']}\n"
        m += f"Stop Loss: {positions['Stop Loss']}\n"
        m += f"Take Profit: {positions['Take Profit']}\n"
        await send_message(m, channel=POSITIONS_CHANNEL)
        await open_position(symbol=symbol, side=positions['Position'], value=positions['Value'])
        exchange.create_stop_loss_take_profit(symbol=symbol, side=positions['Position'], sl_price=positions['Stop Loss'], tp_price=positions['Take Profit'])
            
    if positions['Sell Signal']:
        m = f"Exiting Position at {positions['Time']}\n"
        m += f"Value: {positions['Value']}\n"
        m += f"Gain: ${positions['Gains']}\n"
        m += f"Price: {positions['Price']}\n"
        m += f"Fee: {positions['Fee']}\n"
        m += f"Held time: {positions['Held Time']}\n"
        await send_message(m, channel=POSITIONS_CHANNEL)
        await close_position(symbol=symbol)

    await clear_channel(channel=STOCK_CHANNEL)
    await send_message(stock_data, channel=STOCK_CHANNEL, do_print=False)

def run_discord_bot():
    TOKEN = ''
    client.run(TOKEN)

if __name__ == '__main__':
    run_discord_bot()