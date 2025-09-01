'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
from copy import deepcopy

from cryptofeed import FeedHandler
from cryptofeed.callback import BookCallback
from cryptofeed.defines import L2_BOOK
from cryptofeed.exchanges import Coinbase,BinanceFutures, Binance


PREV = {}
counter = 0


async def book(feed, symbol, book, timestamp):
    global PREV
    global counter
    if book == PREV:
        print("Current")
        print(book)
        print("\n\n")
        print("Previous")
        print(PREV)
    assert book != PREV
    PREV = deepcopy(book)
    counter += 1
    if counter % 10 == 0:
        print(".", end='', flush=True)

async def handle_orderbook(client, timestamp, symbol, book, checksum,** kwargs):
    """处理订单簿数据的回调函数"""
    print(f"Timestamp: {timestamp} Symbol: {symbol} Book: {book} Checksum: {checksum}")

    

def main():
    f = FeedHandler()
    try:
        f.add_feed(BinanceFutures(max_depth=5, symbols=['ETH-USDT-PERP'], channels=[L2_BOOK], callbacks={L2_BOOK: BookCallback(book)}))
        f.run()
    except Exception as e:
        print("Connection reset by peer, retrying...")

def check_symbol(symbol, exchange):
    """检查交易对是否被交易所支持"""
    try:
        exchange_info = exchange.info()
        symbols = exchange_info['symbols']
        print(symbols)
        # supported = any(s['symbol'] == symbol for s in symbols)
        # return supported
    except Exception as e:
        print(f"获取信息出错: {e}")
        # return False
    
if __name__ == '__main__':
    # binance_futures = BinanceFutures()
    # check_symbol('ETHUSDT', binance_futures)
    main()