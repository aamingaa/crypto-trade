import pandas as pd
import numpy as np
import talib as ta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

from crypto_process import load_data, resample_data, load_daily_data
from plot_trading import plot_trading, plot_strategy_performance

def preprocess_data(z_:pd.DataFrame) -> pd.DataFrame:
    """陈述这个函数所要达到的目的
    数据预处理部分,在原始数据基础上增加指标计算、仓位和买卖标记"""
    z = z_.copy()  # copy的作用是避免在原来的dataframe上进行修改
    # 对列名修改 rename
    # 计算指标，计算了sma short moving average lma long moving average
    # open + close
    z['sma'] = ta.MA(z['close'], timeperiod = 10, matype = 1)  # 0为SMA 1为EMA
    z['lma'] = ta.MA(z['close'], timeperiod = 30, matype = 1)
    # 增加了2个列，仓位列和记录买卖的列
    z['position'] = 0.0 # 记录仓位
    z['flag'] = 0.0 # 记录买卖，对买卖的情况进行记录
    return z


def run_strategy(z: pd.DataFrame) -> tuple:
    """策略执行：短期均线上穿长期均线做多，短期均线下穿长期均线平仓。
    同时加入盈利回撤5%平仓和止损5%的逻辑。(对于强势资产，谨慎做空)
    """
    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    max_profit = 0  # 跟踪开仓后的最大浮动利润

    # 对每一行进行遍历
    for i in range(2, z.shape[0]):
        # 情形一: 当前无仓位且短期均线上穿长期均线(金叉)开多仓
        if (z['position'][i - 1] == 0) and (z['sma'][i - 2] < z['lma'][i - 2]) and (
                z['sma'][i - 1] > z['lma'][i - 1]):
            z['flag'][i] = 1  # 记录买入信号
            z['position'][i] = 1  # 仓位记录为1
            date_in = z.index[i]  # 记录买入的时间
            price_in = z['open'][i]  # 记录买入的价格
            max_profit = 0  # 初始化最大浮动利润
            print(z.index[i], f'=========短期均线上穿长期均线买入，此时仓位为{z["position"][i]}', price_in)
            Buy.append([date_in, price_in, '短期均线上穿长期均线买入'])  # 保存买入记录

        # 情形二：当前持仓且短期均线下穿长期均线(死叉)平仓
        elif (z['position'][i - 1] == 1) and (z['sma'][i - 2] > z['lma'][i - 2]) and (
                z['sma'][i - 1] < z['lma'][i - 1]):
            z['flag'][i] = -1  # 记录卖出信号
            z['position'][i] = 0  # 仓位清零
            date_out = z.index[i]  # 记录卖出的时间
            price_out = z['open'][i]  # 记录卖出的价格
            print(z.index[i], '=========短期均线下穿长期均线平仓')
            Sell.append([date_out, price_out, '短期均线下穿长期均线平仓'])  # 保存卖出记录

        # 情形三：持仓时，止盈回撤10%或止损5%
        elif z['position'][i - 1] == 1:
            # 计算当前浮动收益率
            floating_profit = (z['close'][i - 1] - price_in) / price_in

            # 更新最大浮动利润
            max_profit = max(max_profit, floating_profit)

            # 止损条件
            if floating_profit < -0.1:  # 浮动亏损超过10%
                z['flag'][i] = -1  # 卖出信号
                z['position'][i] = 0  # 仓位清零
                date_out = z.index[i]
                price_out = z['open'][i]
                print(z.index[i], '=========止损平仓')
                Sell.append([date_out, price_out, '止损平仓'])

            # 止盈回撤条件
            elif floating_profit < max_profit - 0.05:  # 盈利回撤超过5%
                z['flag'][i] = -1  # 卖出信号
                z['position'][i] = 0  # 仓位清零
                date_out = z.index[i]
                price_out = z['open'][i]
                print(z.index[i], '=========回撤平仓')
                Sell.append([date_out, price_out, '回撤平仓'])

            else:
                z['position'][i] = z['position'][i - 1]  # 继续持仓
                print(z.index[i], f'============没有平仓，继续持仓，此时的仓位为{z["position"][i]}')

        # 其他情况：保持仓位不变
        else:
            z['position'][i] = z['position'][i - 1]
            print(z.index[i], f'============没有开仓，仓位保持为{z["position"][i]}')

    # 将买卖记录转为 DataFrame
    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transaction = pd.concat([p1, p2], axis=1)  # 合并买卖记录

    # 假设数据如下：
    # 日期    收盘价    收益率    持仓    策略收益    净值
    # Day1    100      0        1      0         1.0
    # Day2    110      0.1      1      0.1       1.1
    # Day3    99      -0.1      0      0         1.1
    # Day4    108.9    0.1      1      0.1       1.2
    
    # ===========================重点1 单利和复利=============================
    # 计算收益率与净值曲线
    z['ret'] = z.close.pct_change(1).fillna(0)  # 计算每日收益率
    z['nav'] = 1 + (z.ret * z.position).cumsum()  # 计算净值曲线（单利方式）
    # z['nav'] = (1 + z.ret * z.position).cumprod()
    z['benchmark'] = z.close / z.close[0]  # 持有不动的基准收益曲线

    return z,transaction



def calculate_performance_metrics(data_price:pd.DataFrame,transactions:pd.DataFrame) -> pd.DataFrame:
    '''计算绩效指标'''
    N = 365  # 一年的交易天数,国内的期货和股票都是252个交易日左右，加密货币365天
    rf = 0.02 # risk free rate，无风险收益率

    # 年化收益率
    rety = data_price.nav.iloc[-1]**(N/data_price.shape[0]) - 1

    # ========================重点2===========================
    # 夏普比率
    strategy_returns = data_price.ret * data_price.position  # 下一节课详细讲解
    Sharpe = (strategy_returns.mean()*np.sqrt(N)-rf/N)/ strategy_returns.std()  # 衡量了年化的夏普比率

    # 胜率
    VictoryRatio = ((transactions['卖出价格'] - transactions['买入价格']) > 0).mean()

    # 最大回撤
    DD = 1 - data_price.nav / data_price.nav.cummax()  # drawdown
    MDD = DD.max()  # maximum drawdown

    # 月均交易次数
    trade_count = data_price.flag.abs().sum() / data_price.shape[0] * 20

    # 将结果整理成字典
    result = {
        'Sharpe': Sharpe, # 夏普比率
        'Annual_Return': rety, # 年化收益率
        'MDD': MDD,  # 最大回撤
        'Winning_Rate': VictoryRatio,  # 胜率
        'Trading_Num': round(trade_count, 1) # 月均交易次数
    }
    
    return pd.DataFrame(result, index=[0])

def plot_strategy(z: pd.DataFrame):
    """绘制K线图和净值曲线"""
    z.index = pd.to_datetime(z.index)

    # 创建子图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("K线图", "净值曲线"),
                        vertical_spacing=0.1)

    # 添加K线图
    fig.add_trace(go.Candlestick(x=z.index,
                                  open=z['open'],
                                  high=z['high'],
                                  low=z['low'],
                                  close=z['close'],
                                  name='K线'),
                  row=1, col=1)

    # 添加买入标记
    buy_signals = z[z['flag'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=buy_signals['low'] * 0.99,  # 在低点下方显示买入标记
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='green', size=10),
                             name='买入信号'),
                  row=1, col=1)

    # 添加卖出标记
    sell_signals = z[z['flag'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index,
                             y=sell_signals['high'] * 1.01,  # 在高点上方显示卖出标记
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='red', size=10),
                             name='卖出信号'),
                  row=1, col=1)

    # 添加净值曲线
    fig.add_trace(go.Scatter(x=z.index, y=z['nav'], mode='lines', name='净值', line=dict(color='blue')),
                  row=2, col=1)

    # 更新布局
    fig.update_layout(title='K线图与净值曲线',
                      xaxis_title='日期',
                      yaxis_title='价格',
                      xaxis_rangeslider_visible=False)

    # 保存为HTML文件
    fig.write_html('strategy_plot.html')

class MA_strategy:
    def __init__(self, start_date: str, end_date: str, freq: str = '1h'):
        """
        初始化MA策略
        
        Parameters:
        -----------
        start_date : str
            开始日期，格式：'YYYY-MM'
        end_date : str
            结束日期，格式：'YYYY-MM'
        freq : str
            数据频率，如'1h', '4h', '1d'等
        """
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.data = None
        self.position = None
        self.nav = None
        
    def load_and_process_data(self):
        """加载并处理数据"""
        # 加载数据
        self.data = load_daily_data(self.start_date, self.end_date, self.freq)
        # 重采样
        # self.data = resample_data(self.data, self.freq)
        return self.data
    
    def calculate_signals(self, short_window: int = 5, long_window: int = 20):
        """
        计算交易信号
        
        Parameters:
        -----------
        short_window : int
            短期均线周期
        long_window : int
            长期均线周期
        """
        if self.data is None:
            self.load_and_process_data()
            
        # 计算移动平均
        self.data['MA_short'] = self.data['close'].rolling(window=short_window).mean()
        self.data['MA_long'] = self.data['close'].rolling(window=long_window).mean()
        
        # 计算交易信号
        self.data['position'] = 0
        # 金叉：短期均线上穿长期均线
        self.data.loc[self.data['MA_short'] > self.data['MA_long'], 'position'] = 1
        # 死叉：短期均线下穿长期均线
        self.data.loc[self.data['MA_short'] < self.data['MA_long'], 'position'] = -1
        
        # 计算收益率和净值
        self.data['ret'] = self.data['close'].pct_change()
        self.data['nav'] = (1 + self.data['ret'] * self.data['position']).cumprod()
        
        return self.data
    
    def plot_strategy(self, title: str = None, save_path: str = "/Users/aming/project/python/crypto-trade/util"):
        """
        绘制策略图表
        
        Parameters:
        -----------
        title : str
            图表标题，如果为None则使用默认标题
        save_path : str
            保存图表的路径，如果为None则显示图表
        """
        if self.data is None or 'position' not in self.data.columns:
            raise ValueError("请先运行calculate_signals()计算交易信号")
        
        if title is None:
            title = f"MA策略 ({self.freq}) - {self.start_date} to {self.end_date}"
        
        # 绘制K线图和交易信号
        plot_trading(
            self.data,
            title=title,
            save_path=save_path,
            show_volume=True,
            show_signals=True
        )
        
        # 绘制策略表现
        if save_path:
            # 如果提供了保存路径，为策略表现图添加后缀
            performance_save_path = save_path.replace('.png', '_performance.png')
        else:
            performance_save_path = None
            
        plot_strategy_performance(
            self.data,
            title=f"{title} - 策略表现",
            save_path=performance_save_path
        )
    
    def get_performance_metrics(self):
        """
        计算策略表现指标
        
        Returns:
        --------
        dict
            包含各项表现指标的字典
        """
        if self.data is None or 'nav' not in self.data.columns:
            raise ValueError("请先运行calculate_signals()计算交易信号")
        
        # 计算各项指标
        total_return = self.data['nav'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(self.data)) - 1
        daily_returns = self.data['nav'].pct_change()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        max_drawdown = (self.data['nav'] / self.data['nav'].cummax() - 1).min()
        
        return {
            '总收益率': f"{total_return:.2%}",
            '年化收益率': f"{annual_return:.2%}",
            '夏普比率': f"{sharpe_ratio:.2f}",
            '最大回撤': f"{max_drawdown:.2%}",
            '交易次数': abs(self.data['position'].diff()).sum() / 2
        }

if __name__ == '__main__':
    # 第一部分 读取数据，并对数据做预处理
    # import crypto_process
    # start_month = '2023-10'
    # end_month = '2024-10'
    # start_date = "2025-06-01"
    # end_date = "2025-06-30"
    # freq = '1d'
    # z_original = crypto_process.load_daily_data(start_date, end_date, "15m")
    
    # z = preprocess_data(z_original)
    # # 第二部分 运行策略
    # data_price,transaction = run_strategy(z) 
    # # # 第三部分 计算绩效和作图
    # result = calculate_performance_metrics(data_price,transaction)
    # print(result)

    # 示例使用
    strategy = MA_strategy("2025-06-01", "2025-06-30", freq="15m")
    strategy.load_and_process_data()
    strategy.calculate_signals(short_window=5, long_window=20)
    
    # 打印策略表现指标
    print("\n策略表现指标:")
    for metric, value in strategy.get_performance_metrics().items():
        print(f"{metric}: {value}")
    
    # # 绘制策略图表
    strategy.plot_strategy()