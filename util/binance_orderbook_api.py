import requests
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import websocket
import threading
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceFuturesOrderBookAPI:
    """
    Binance合约订单簿API工具类
    支持REST API和WebSocket实时数据获取
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        """
        初始化Binance合约API客户端
        
        Args:
            api_key: API密钥（可选，获取公开数据时可为空）
            api_secret: API密钥（可选，获取公开数据时可为空）
            testnet: 是否使用测试网络
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # 设置基础URL
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com/ws"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com/ws"
        
        # 设置请求头
        self.headers = {
            'Content-Type': 'application/json',
            'X-MBX-APIKEY': api_key
        }
        
        # WebSocket连接状态
        self.ws_connected = False
        self.ws = None
        
    def get_order_book_rest(self, symbol: str, limit: int = 100) -> Dict:
        """
        通过REST API获取订单簿数据
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            limit: 订单簿深度，默认100，最大1000
            
        Returns:
            订单簿数据字典
        """
        endpoint = f"{self.base_url}/fapi/v1/depth"
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取订单簿数据失败: {e}")
            return None
    
    def get_order_book_ticker_rest(self, symbol: str) -> Dict:
        """
        获取订单簿价格统计信息
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            
        Returns:
            价格统计信息
        """
        endpoint = f"{self.base_url}/fapi/v1/ticker/bookTicker"
        params = {'symbol': symbol}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取订单簿价格统计失败: {e}")
            return None
    
    def get_24hr_ticker_rest(self, symbol: str) -> Dict:
        """
        获取24小时价格变动统计
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            
        Returns:
            24小时统计信息
        """
        endpoint = f"{self.base_url}/fapi/v1/ticker/24hr"
        params = {'symbol': symbol}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取24小时统计失败: {e}")
            return None
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        获取资金费率
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            
        Returns:
            资金费率信息
        """
        endpoint = f"{self.base_url}/fapi/v1/fundingRate"
        params = {'symbol': symbol, 'limit': 1}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取资金费率失败: {e}")
            return None
    
    def get_open_interest(self, symbol: str) -> Dict:
        """
        获取未平仓量
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            
        Returns:
            未平仓量信息
        """
        endpoint = f"{self.base_url}/fapi/v1/openInterest"
        params = {'symbol': symbol}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取未平仓量失败: {e}")
            return None
    
    def parse_order_book_to_dataframe(self, order_book: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        将订单簿数据解析为DataFrame
        
        Args:
            order_book: 订单簿数据字典
            
        Returns:
            (买盘DataFrame, 卖盘DataFrame)
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return None, None
        
        # 解析买盘数据
        bids_df = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'])
        bids_df['price'] = bids_df['price'].astype(float)
        bids_df['quantity'] = bids_df['quantity'].astype(float)
        bids_df['cumulative_quantity'] = bids_df['quantity'].cumsum()
        bids_df['side'] = 'buy'
        
        # 解析卖盘数据
        asks_df = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'])
        asks_df['price'] = asks_df['price'].astype(float)
        asks_df['quantity'] = asks_df['quantity'].astype(float)
        asks_df['cumulative_quantity'] = asks_df['quantity'].cumsum()
        asks_df['side'] = 'sell'
        
        return bids_df, asks_df
    
    def calculate_order_book_metrics(self, bids_df: pd.DataFrame, asks_df: pd.DataFrame) -> Dict:
        """
        计算订单簿指标
        
        Args:
            bids_df: 买盘DataFrame
            asks_df: 卖盘DataFrame
            
        Returns:
            订单簿指标字典
        """
        if bids_df is None or asks_df is None or len(bids_df) == 0 or len(asks_df) == 0:
            return {}
        
        # 获取最优买卖价
        best_bid = bids_df.iloc[0]['price']
        best_ask = asks_df.iloc[0]['price']
        best_bid_qty = bids_df.iloc[0]['quantity']
        best_ask_qty = asks_df.iloc[0]['quantity']
        
        # 计算中间价和价差
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_percentage = (spread / mid_price) * 100
        
        # 计算订单簿不平衡
        total_bid_qty = bids_df['quantity'].sum()
        total_ask_qty = asks_df['quantity'].sum()
        order_imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
        
        # 计算前5档的加权平均价格
        bid_vwap = (bids_df.head(5)['price'] * bids_df.head(5)['quantity']).sum() / bids_df.head(5)['quantity'].sum()
        ask_vwap = (asks_df.head(5)['price'] * asks_df.head(5)['quantity']).sum() / asks_df.head(5)['quantity'].sum()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'best_bid': best_bid,
            'best_ask': best_ask,
            'best_bid_qty': best_bid_qty,
            'best_ask_qty': best_ask_qty,
            'mid_price': mid_price,
            'spread': spread,
            'spread_percentage': spread_percentage,
            'total_bid_qty': total_bid_qty,
            'total_ask_qty': total_ask_qty,
            'order_imbalance': order_imbalance,
            'bid_vwap_5': bid_vwap,
            'ask_vwap_5': ask_vwap,
            'bid_ask_ratio': total_bid_qty / total_ask_qty if total_ask_qty > 0 else float('inf')
        }
    
    def start_websocket_orderbook(self, symbol: str, callback=None):
        """
        启动WebSocket订单簿数据流
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            callback: 回调函数，接收订单簿数据
        """
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if callback:
                    callback(data)
                else:
                    print(f"订单簿数据: {data}")
            except json.JSONDecodeError as e:
                logger.error(f"解析WebSocket消息失败: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket错误: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket连接已关闭")
            self.ws_connected = False
        
        def on_open(ws):
            logger.info("WebSocket连接已建立")
            self.ws_connected = True
        
        # 创建WebSocket连接
        stream_name = f"{symbol.lower()}@depth20@100ms"  # 20档深度，100ms更新
        ws_url = f"{self.ws_url}/{stream_name}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # 在独立线程中运行WebSocket
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
    
    def stop_websocket(self):
        """停止WebSocket连接"""
        if self.ws:
            self.ws.close()
            self.ws_connected = False
    
    def get_comprehensive_orderbook_data(self, symbol: str) -> Dict:
        """
        获取综合订单簿数据（包括订单簿、价格统计、资金费率、未平仓量）
        
        Args:
            symbol: 交易对，如 'BTCUSDT'
            
        Returns:
            综合数据字典
        """
        try:
            # 获取订单簿数据
            order_book = self.get_order_book_rest(symbol, limit=100)
            if not order_book:
                return {}
            
            # 解析订单簿
            bids_df, asks_df = self.parse_order_book_to_dataframe(order_book)
            
            # 计算订单簿指标
            order_book_metrics = self.calculate_order_book_metrics(bids_df, asks_df)
            
            # 获取价格统计
            ticker_data = self.get_order_book_ticker_rest(symbol)
            
            # 获取24小时统计
            ticker_24hr = self.get_24hr_ticker_rest(symbol)
            
            # 获取资金费率
            funding_rate = self.get_funding_rate(symbol)
            
            # 获取未平仓量
            open_interest = self.get_open_interest(symbol)
            
            # 整合数据
            comprehensive_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'order_book_metrics': order_book_metrics,
                'ticker_data': ticker_data,
                'ticker_24hr': ticker_24hr,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'bids_df': bids_df.to_dict('records') if bids_df is not None else [],
                'asks_df': asks_df.to_dict('records') if asks_df is not None else []
            }
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"获取综合订单簿数据失败: {e}")
            return {}
    
    def save_orderbook_data(self, data: Dict, filename: str):
        """
        保存订单簿数据到文件
        
        Args:
            data: 订单簿数据
            filename: 文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def monitor_orderbook_continuously(self, symbol: str, interval: int = 5, duration: int = 3600):
        """
        持续监控订单簿数据
        
        Args:
            symbol: 交易对
            interval: 监控间隔（秒）
            duration: 监控时长（秒）
        """
        start_time = time.time()
        data_list = []
        
        logger.info(f"开始监控 {symbol} 订单簿数据，间隔 {interval} 秒，持续 {duration} 秒")
        
        while time.time() - start_time < duration:
            try:
                data = self.get_comprehensive_orderbook_data(symbol)
                if data:
                    data_list.append(data)
                    logger.info(f"获取数据成功: {data['order_book_metrics']['mid_price']:.2f}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("监控被用户中断")
                break
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
                time.sleep(interval)
        
        # 保存监控数据
        if data_list:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orderbook_data_{symbol}_{timestamp}.json"
            self.save_orderbook_data(data_list, filename)
        
        logger.info(f"监控结束，共获取 {len(data_list)} 条数据")


def demo_basic_usage():
    """演示基本使用方法"""
    print("=== Binance合约订单簿API演示 ===")
    
    # 创建API客户端
    api = BinanceFuturesOrderBookAPI()
    
    # 获取BTCUSDT订单簿数据
    symbol = "BTCUSDT"
    print(f"\n1. 获取 {symbol} 订单簿数据:")
    order_book = api.get_order_book_rest(symbol, limit=10)
    
    if order_book:
        bids_df, asks_df = api.parse_order_book_to_dataframe(order_book)
        
        print("\n买盘前5档:")
        print(bids_df.head())
        
        print("\n卖盘前5档:")
        print(asks_df.head())
        
        # 计算指标
        metrics = api.calculate_order_book_metrics(bids_df, asks_df)
        print("\n订单簿指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    
    # 获取价格统计
    print(f"\n2. 获取 {symbol} 价格统计:")
    ticker = api.get_order_book_ticker_rest(symbol)
    if ticker:
        print(json.dumps(ticker, indent=2))
    
    # 获取资金费率
    print(f"\n3. 获取 {symbol} 资金费率:")
    funding = api.get_funding_rate(symbol)
    if funding:
        print(json.dumps(funding, indent=2))
    
    # 获取未平仓量
    print(f"\n4. 获取 {symbol} 未平仓量:")
    oi = api.get_open_interest(symbol)
    if oi:
        print(json.dumps(oi, indent=2))


def demo_websocket_usage():
    """演示WebSocket使用方法"""
    print("\n=== WebSocket实时数据演示 ===")
    
    api = BinanceFuturesOrderBookAPI()
    symbol = "BTCUSDT"
    
    def orderbook_callback(data):
        print(f"实时订单簿数据: {data}")
    
    # 启动WebSocket连接
    api.start_websocket_orderbook(symbol, orderbook_callback)
    
    # 运行10秒后停止
    time.sleep(10)
    api.stop_websocket()


if __name__ == "__main__":
    # 运行演示
    demo_basic_usage()
    
    # 如果需要测试WebSocket，取消注释下面这行
    # demo_websocket_usage()

