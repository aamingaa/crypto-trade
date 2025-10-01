import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MovingAverageStrategy:
    def __init__(self, data, short_window=5, long_window=20):
        """
        初始化均线交易策略
        
        参数:
        data: DataFrame，包含OHLC数据，需包含'open', 'high', 'low', 'close', 'open_time', 'close_time'列
        short_window: 短期均线窗口大小
        long_window: 长期均线窗口大小
        """
        self.data = data.copy()
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None
        self.portfolio = None
        
        # 确保时间列是 datetime 类型
        self.data['open_time'] = pd.to_datetime(self.data['open_time'])
        self.data['close_time'] = pd.to_datetime(self.data['close_time'])
        
    def calculate_indicators(self):
        """计算技术指标，包括短期和长期均线"""
        # 计算短期均线
        self.data['short_ma'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        # 计算长期均线
        self.data['long_ma'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()
        
    def generate_signals(self):
        """生成交易信号"""
        self.calculate_indicators()
        
        # 创建信号列，1表示买入，-1表示卖出，0表示无信号
        self.data['signal'] = 0
        
        # 短期均线上穿长期均线时，产生买入信号
        self.data.loc[self.data['short_ma'] > self.data['long_ma'], 'signal'] = 1
        
        # 短期均线下穿长期均线时，产生卖出信号
        self.data.loc[self.data['short_ma'] < self.data['long_ma'], 'signal'] = -1
        
        # 只保留状态变化的信号（从无持仓到买入，或从持仓到卖出）
        self.data['position'] = self.data['signal'].diff()
        
        # 保存信号数据
        self.signals = self.data[['open_time', 'close', 'short_ma', 'long_ma', 'position']]
        
    def backtest(self, initial_capital=100000):
        """回测策略表现"""
        if self.signals is None:
            self.generate_signals()
            
        # 初始化投资组合DataFrame
        self.portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        self.portfolio['close_price'] = self.data['close']
        
        # 记录持仓情况，1表示持有，0表示空仓
        self.portfolio['holdings'] = 0.0
        
        # 初始资金
        self.portfolio['cash'] = initial_capital
        self.portfolio['total'] = initial_capital
        
        # 执行交易
        for i in range(1, len(self.portfolio)):
            # 继承前一天的持仓状态
            self.portfolio['holdings'].iloc[i] = self.portfolio['holdings'].iloc[i-1]
            self.portfolio['cash'].iloc[i] = self.portfolio['cash'].iloc[i-1]
            
            # 买入信号
            if self.signals['position'].iloc[i] == 2:  # 从-1到1的变化
                # 全部资金买入
                price = self.portfolio['close_price'].iloc[i]
                shares = self.portfolio['cash'].iloc[i] / price
                self.portfolio['holdings'].iloc[i] = shares
                self.portfolio['cash'].iloc[i] = 0
                
            # 卖出信号
            elif self.signals['position'].iloc[i] == -2:  # 从1到-1的变化
                # 全部卖出
                price = self.portfolio['close_price'].iloc[i]
                cash = self.portfolio['holdings'].iloc[i] * price
                self.portfolio['holdings'].iloc[i] = 0
                self.portfolio['cash'].iloc[i] = cash
            
            # 计算总资产
            self.portfolio['total'].iloc[i] = (self.portfolio['holdings'].iloc[i] * 
                                              self.portfolio['close_price'].iloc[i] + 
                                              self.portfolio['cash'].iloc[i])
        
        # 计算策略收益率
        self.portfolio['return'] = self.portfolio['total'].pct_change()
        self.portfolio['cumulative_return'] = (1 + self.portfolio['return']).cumprod()
        
        # 计算基准收益率（买入并持有）
        self.portfolio['benchmark_return'] = self.portfolio['close_price'].pct_change()
        self.portfolio['benchmark_cumulative'] = (1 + self.portfolio['benchmark_return']).cumprod()
        
    def plot_results(self):
        """绘制策略结果图表"""
        if self.portfolio is None:
            self.backtest()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        
        # 绘制价格和均线
        ax1.plot(self.data['open_time'], self.data['close'], label='收盘价', alpha=0.5)
        ax1.plot(self.data['open_time'], self.data['short_ma'], label=f'{self.short_window}期均线')
        ax1.plot(self.data['open_time'], self.data['long_ma'], label=f'{self.long_window}期均线')
        
        # 标记买入和卖出信号
        buy_signals = self.signals[self.signals['position'] == 2]
        sell_signals = self.signals[self.signals['position'] == -2]
        
        ax1.scatter(buy_signals['open_time'], buy_signals['close'], 
                   marker='^', color='g', label='买入信号', zorder=3)
        ax1.scatter(sell_signals['open_time'], sell_signals['close'], 
                   marker='v', color='r', label='卖出信号', zorder=3)
        
        ax1.set_title('价格与均线及交易信号')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制收益曲线
        ax2.plot(self.data['open_time'], self.portfolio['cumulative_return'], 
                label='策略累计收益')
        ax2.plot(self.data['open_time'], self.portfolio['benchmark_cumulative'], 
                label='基准累计收益（买入并持有）')
        
        ax2.set_title('策略与基准收益对比')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def get_performance_metrics(self):
        """计算并返回策略表现指标"""
        if self.portfolio is None:
            self.backtest()
            
        # 总收益率
        total_return = (self.portfolio['total'].iloc[-1] / self.portfolio['total'].iloc[0] - 1) * 100
        
        # 交易次数
        buy_count = len(self.signals[self.signals['position'] == 2])
        sell_count = len(self.signals[self.signals['position'] == -2])
        total_trades = buy_count + sell_count
        
        # 夏普比率 (假设无风险利率为0)
        returns = self.portfolio['return'].dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
        
        # 最大回撤
        rolling_max = self.portfolio['total'].cummax()
        daily_drawdown = self.portfolio['total'] / rolling_max - 1.0
        max_drawdown = daily_drawdown.min() * 100
        
        return {
            '总收益率(%)': round(total_return, 2),
            '交易次数': total_trades,
            '买入次数': buy_count,
            '卖出次数': sell_count,
            '夏普比率': round(sharpe_ratio, 2),
            '最大回撤(%)': round(max_drawdown, 2)
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据（实际使用时替换为你的K线数据）
    def generate_sample_data(days=365):
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        data = {
            'open_time': dates,
            'close_time': dates + pd.Timedelta(days=1),
            'open': np.random.normal(100, 5, days).cumprod() / 100,
            'high': np.random.normal(100, 5, days).cumprod() / 100 + np.random.uniform(0, 2, days),
            'low': np.random.normal(100, 5, days).cumprod() / 100 - np.random.uniform(0, 2, days),
            'close': np.random.normal(100, 5, days).cumprod() / 100
        }
        return pd.DataFrame(data)
    
    # 生成示例数据
    sample_data = generate_sample_data(365)
    
    # 初始化并运行策略
    strategy = MovingAverageStrategy(sample_data, short_window=10, long_window=50)
    strategy.generate_signals()
    strategy.backtest()
    
    # 打印绩效指标
    print("策略绩效指标:")
    metrics = strategy.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # 绘制结果
    strategy.plot_results()
