import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import math
import os

# 设置中文显示
plt.rcParams["font.family"] = ['Heiti TC', 'Heiti TC']
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class MAStrategyAnalyzer:
    def __init__(self, data, short_window=5, long_window=20):
        """
        初始化均线策略分析器
        :param data: 包含OHLC数据的DataFrame，需包含'open_time', 'open', 'high', 'low', 'close'列
        :param short_window: 短期均线周期
        :param long_window: 长期均线周期
        """
        self.data = data.copy()
        # 确保时间列是datetime类型
        self.data['open_time'] = pd.to_datetime(self.data['open_time'])
        self.data['close_time'] = pd.to_datetime(self.data['close_time'])
        
        self.short_window = short_window
        self.long_window = long_window
        
        # 计算均线
        self.data['short_ma'] = self.data['close'].rolling(window=short_window).mean()
        self.data['long_ma'] = self.data['close'].rolling(window=long_window).mean()
        
        # 初始化交易相关列
        self.data['signal'] = 0  # 1表示买入，-1表示卖出
        self.data['position'] = 0  # 1表示持有多单，0表示空仓
        self.data['strategy_return'] = 0.0  # 策略收益
        self.data['cumulative_return'] = 1.0  # 累计收益
        
        self.trades = []  # 记录所有交易
        self.performance_metrics = {}  # 记录绩效指标
        
    def generate_signals(self):
        """生成交易信号：短期均线上穿长期均线买入，下穿卖出"""
        # 短期均线上穿长期均线，买入信号
        self.data.loc[self.data['short_ma'] > self.data['long_ma'], 'signal'] = 1
        # 短期均线下穿长期均线，卖出信号
        self.data.loc[self.data['short_ma'] < self.data['long_ma'], 'signal'] = -1
        
        # 确定持仓状态，避免重复信号
        for i in range(1, len(self.data)):
            if self.data['signal'].iloc[i] == 1 and self.data['position'].iloc[i-1] == 0:
                self.data.at[i, 'position'] = 1  # 买入，持有多单
                # 记录买入交易
                self.trades.append({
                    'type': 'buy',
                    'time': self.data['close_time'].iloc[i],  # 使用close_time
                    'price': self.data['open'].iloc[i],  # 下一根K线开盘价买入
                    'index': i
                })
            elif self.data['signal'].iloc[i] == -1 and self.data['position'].iloc[i-1] == 1:
                self.data.at[i, 'position'] = 0  # 卖出，空仓
                # 记录卖出交易，并与最近的买入匹配
                if self.trades and self.trades[-1]['type'] == 'buy':
                    buy_trade = self.trades[-1]
                    profit = self.data['open'].iloc[i] - buy_trade['price']
                    profit_ratio = profit / buy_trade['price']
                    self.trades.append({
                        'type': 'sell',
                        'time': self.data['close_time'].iloc[i],  # 使用close_time
                        'price': self.data['open'].iloc[i],  # 下一根K线开盘价卖出
                        'profit': profit,
                        'profit_ratio': profit_ratio,
                        'holding_period': i - buy_trade['index'],
                        'index': i
                    })
            else:
                self.data.at[i, 'position'] = self.data['position'].iloc[i-1]  # 保持原有持仓状态
        
        # 计算策略收益
        self.data['market_return'] = self.data['close'].pct_change()  # 市场收益率
        self.data['strategy_return'] = self.data['market_return'] * self.data['position'].shift(1)  # 策略收益率
        
        # 计算累计收益
        self.data['cumulative_market'] = (1 + self.data['market_return']).cumprod()
        self.data['cumulative_strategy'] = (1 + self.data['strategy_return']).cumprod()
    
    def calculate_performance(self):
        """计算策略绩效指标"""
        if not self.trades:
            print("没有交易记录，无法计算绩效指标")
            return
        
        # 筛选出所有完整交易（买入后卖出）
        profitable_trades = [t for t in self.trades if t['type'] == 'sell' and t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['type'] == 'sell' and t['profit'] <= 0]
        
        # 胜率
        total_sells = len([t for t in self.trades if t['type'] == 'sell'])
        win_rate = len(profitable_trades) / total_sells if total_sells > 0 else 0
        
        # 累计收益
        total_return = self.data['cumulative_strategy'].iloc[-1] - 1 if len(self.data) > 0 else 0
        
        # 夏普比率 (假设无风险利率为0)
        returns = self.data['strategy_return'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            # 假设一年252个交易日，每个交易日96个15分钟K线
            sharpe_ratio = math.sqrt(252 * 96) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative = self.data['cumulative_strategy'].dropna()
        if len(cumulative) > 0:
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative / peak) - 1
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # 平均盈亏比
        avg_profit = np.mean([t['profit_ratio'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([abs(t['profit_ratio']) for t in losing_trades]) if losing_trades else 1
        profit_factor = avg_profit / avg_loss if avg_loss != 0 else 0
        
        # 平均持仓周期
        avg_holding_period = np.mean([t['holding_period'] for t in self.trades if t['type'] == 'sell']) if total_sells > 0 else 0
        
        self.performance_metrics = {
            '累计收益率': total_return,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '盈亏比': profit_factor,
            '平均持仓周期(15分钟)': avg_holding_period,
            '总交易次数': total_sells
        }
        
        return self.performance_metrics
    
    def plot_results(self, save_path=None):
        """绘制策略结果，包括价格、均线、交易信号和净值曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        
        # 绘制价格和均线，使用close_time作为x轴
        ax1.plot(self.data['close_time'], self.data['close'], label='收盘价', linewidth=1.5)
        # ax1.plot(self.data['close_time'], self.data['short_ma'], label=f'{self.short_window}周期均线', linewidth=1.2)
        # ax1.plot(self.data['close_time'], self.data['long_ma'], label=f'{self.long_window}周期均线', linewidth=1.2)
        
        # 标记买入卖出点
        buy_signals = [t for t in self.trades if t['type'] == 'buy']
        sell_signals = [t for t in self.trades if t['type'] == 'sell']
        
        if buy_signals:
            buy_times = [t['time'] for t in buy_signals]
            buy_prices = [t['price'] for t in buy_signals]
            ax1.scatter(buy_times, buy_prices, marker='^', color='g', label='买入', s=100)
        
        if sell_signals:
            sell_times = [t['time'] for t in sell_signals]
            sell_prices = [t['price'] for t in sell_signals]
            ax1.scatter(sell_times, sell_prices, marker='v', color='r', label='卖出', s=100)
        
        ax1.set_title('价格走势与交易信号')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制累计收益曲线，使用close_time作为x轴
        ax2.plot(self.data['close_time'], self.data['cumulative_strategy'], label='策略累计收益', linewidth=1.5)
        ax2.plot(self.data['close_time'], self.data['cumulative_market'], label='市场累计收益', linewidth=1.5)
        ax2.set_title('策略与市场累计收益对比')
        ax2.set_xlabel('时间 (close_time)')
        ax2.set_ylabel('累计收益倍数')
        ax2.legend()
        ax2.grid(True)
        
        # 设置时间轴格式
        fig.autofmt_xdate()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        
        # 保存图表到指定路径
        if save_path:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
    
    def run_strategy(self, save_plot_path=None):
        """运行完整策略并输出结果"""
        self.generate_signals()
        metrics = self.calculate_performance()
        
        print("策略参数:")
        print(f"短期均线周期: {self.short_window} (15分钟K线)")
        print(f"长期均线周期: {self.long_window} (15分钟K线)")
        print("\n策略绩效指标:")
        for key, value in metrics.items():
            if key in ['累计收益率', '胜率']:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        
        self.plot_results(save_path=save_plot_path)

# 示例数据生成和使用
def generate_sample_data(n_periods=1000):
    """生成示例15分钟K线数据用于测试"""
    # 生成时间序列
    start_time = datetime(2023, 1, 1, 9, 30)
    times = pd.date_range(start=start_time, periods=n_periods, freq='15T')
    
    # 生成价格数据 (带趋势和波动)
    base_price = 100.0
    trend = np.linspace(0, 50, n_periods)  # 总体趋势
    noise = np.random.normal(0, 1, n_periods).cumsum()  # 随机波动
    
    close_prices = base_price + trend + noise
    
    # 生成开盘价、最高价、最低价
    open_prices = close_prices[:-1]
    # 最后一个开盘价用前一个收盘价
    open_prices = np.append(open_prices, close_prices[-1] * (1 + np.random.normal(0, 0.001)))
    
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 1, n_periods)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 1, n_periods)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'open_time': times,
        'close_time': times + pd.Timedelta(minutes=15),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    
    return data

def generate_date_range(start_date, end_date):    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list

def load_daily_data(start_date:str, end_date:str, interval:str, crypto:str = "BNBUSDT") -> pd.DataFrame:
    date_list = generate_date_range(start_date, end_date)
    crypto_date_data = []
    suffix = "2025-01-01_2025-07-01"
    file_path = f'/Volumes/Ext-Disk/data/futures/um/daily/klines'
    for date in date_list:
        crypto_date_data.append(pd.read_csv(f"{file_path}/{crypto}/{interval}/{suffix}/{crypto}-{interval}-{date}.zip"))
    
    z = pd.concat(crypto_date_data, axis=0, ignore_index=True)
    # 处理时间戳为 DatetimeIndex，便于后续按日/月/年分组和年化计算
    z['open_time'] = pd.to_datetime(z['open_time'], unit='ms')
    z['close_time'] = pd.to_datetime(z['close_time'], unit='ms')

    z = z.sort_values(by='close_time', ascending=True) # 注意这一步是非常必要的，要以timestamp作为排序基准
    z = z.drop_duplicates('close_time').reset_index(drop=True) # 注意这一步非常重要，以timestamp为基准进行去重处理
    # z = z.set_index('close_time')
    z['interval'] = interval  # 保存interval信息，供后续使用
    return z

if __name__ == "__main__":
    # 生成示例数据
    print("生成示例15分钟K线数据...")
    # sample_data = generate_sample_data(1000)
    
    # 初始化并运行策略，可指定保存路径，如"./strategy_results.png"
    print("运行均线交易策略...")

    start_date = "2025-05-01"
    end_date = "2025-05-20"
    df_price = load_daily_data(start_date, end_date, "15m")

    strategy = MAStrategyAnalyzer(df_price, short_window=5, long_window=20)
    # 如需保存图表，取消下面一行的注释并指定路径
    strategy.run_strategy(save_plot_path="/Users/aming/project/python/crypto-trade/output/trading_strategy_results.png")
    # strategy.run_strategy()  # 不保存图表，仅显示
