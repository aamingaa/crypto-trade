# 策略可视化工具使用说明

## 新增功能：综合策略分析图

参考 `util/MA_strategy_v2.py` 的绘图风格，新增了 `plot_strategy_comprehensive()` 方法，可以一次性绘制完整的策略分析图表。

## 功能特点

### 📊 图表包含以下内容：

1. **价格走势与交易信号**
   - 收盘价曲线
   - 买入点标记（红色三角 ▲）
   - 卖出点标记（绿色倒三角 ▼）

2. **策略净值 vs 基准净值**
   - 策略净值曲线
   - 基准净值曲线（Buy & Hold）
   - 收益率统计信息（策略收益、基准收益、超额收益）

3. **持仓变化**
   - 持仓状态的时间序列图

4. **回撤曲线**
   - 策略回撤曲线
   - 最大回撤点标注

5. **累计收益对比**
   - 策略累计收益
   - 基准累计收益
   - 超额收益（双Y轴）

## 使用方法

### 基础用法

```python
from utils.visualization import TradingVisualizer

# 创建可视化工具
visualizer = TradingVisualizer()

# 绘制策略综合分析图
visualizer.plot_strategy_comprehensive(
    strategy_data=strategy_df,      # 策略数据
    transactions=transactions_df,    # 交易记录（可选）
    title="我的策略分析",
    save_path="output/my_strategy.png"
)
```

### 数据格式要求

#### 1. strategy_data (必需)

必须是 pandas DataFrame，索引为时间，包含以下列：

| 列名 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `close` | float | ✅ | 收盘价 |
| `nav` | float | ✅ | 策略净值（初始值=1） |
| `benchmark` | float | ✅ | 基准净值（初始值=1） |
| `position` | int/float | 推荐 | 持仓状态（1=持仓，0=空仓，-1=做空） |
| `flag` | int | 可选 | 交易信号（1=买入，-1=卖出） |

#### 2. transactions (可选)

如果提供，应包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `买入日期` | datetime | 买入时间 |
| `买入价格` | float | 买入价格 |
| `卖出日期` | datetime | 卖出时间 |
| `卖出价格` | float | 卖出价格 |

**注意**：如果不提供 transactions，工具会自动从 `position` 列推断买卖点。

## 示例

### 示例1：使用 MA_strategy_v2 的结果

```python
from util.MA_strategy_v2 import MA_Strategy, load_daily_data
from utils.visualization import TradingVisualizer

# 加载数据并运行策略
df_price = load_daily_data("2025-05-01", "2025-06-30", "15m", crypto="BNBUSDT")
transactions, strategy_data = MA_Strategy(df_price, window_short=5, 
                                          window_median=12, window_long=30)

# 可视化
visualizer = TradingVisualizer()
visualizer.plot_strategy_comprehensive(
    strategy_data=strategy_data,
    transactions=transactions,
    title="MA策略分析 - BNBUSDT",
    save_path="output/ma_strategy_analysis.png"
)
```

### 示例2：自定义策略

```python
import pandas as pd
from utils.visualization import TradingVisualizer

# 假设你有策略数据
strategy_data = pd.DataFrame({
    'close': [100, 102, 101, 103, 105],
    'nav': [1.0, 1.02, 1.01, 1.03, 1.05],
    'benchmark': [1.0, 1.01, 1.00, 1.02, 1.03],
    'position': [0, 1, 1, 1, 0]
}, index=pd.date_range('2025-01-01', periods=5, freq='D'))

visualizer = TradingVisualizer()
visualizer.plot_strategy_comprehensive(
    strategy_data=strategy_data,
    transactions=None,  # 自动从position推断
    title="我的策略",
    save_path="output/my_strategy.png"
)
```

### 示例3：完整的工作流

```python
# 运行示例脚本
python example_strategy_visualization.py

# 选择示例:
# 1 = MA策略（需要数据文件）
# 2 = 模拟策略
# 3 = 持仓推断示例
# 0 = 运行所有示例
```

## 图表说明

### 颜色方案

- **红色**：买入信号、策略净值、回撤
- **绿色**：卖出信号
- **蓝色**：价格、超额收益
- **灰色**：基准净值、持仓

### 交互提示

生成的是静态PNG图片，如需交互式图表，可以考虑：
- 使用 `save_path=None` 直接显示图表（matplotlib交互模式）
- 修改代码使用 plotly 或 pyecharts（如 MA_strategy_v2.py）

## 与 MA_strategy_v2 的对比

| 特性 | MA_strategy_v2 | 新的visualization |
|------|----------------|-------------------|
| 图表库 | pyecharts (交互式HTML) | matplotlib (静态PNG) |
| K线图 | ✅ | ❌（仅收盘价线） |
| 买卖点 | ❌ | ✅ |
| 净值对比 | ✅ | ✅ |
| 回撤曲线 | ❌ | ✅ |
| 持仓变化 | ❌ | ✅ |
| 超额收益 | ✅ | ✅ |
| 自定义 | 较难 | 容易 |

## 常见问题

### Q: 图片中文显示为方框？

A: 需要配置中文字体，visualization.py 已经尝试使用常见字体，如果还有问题：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# 或
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
```

### Q: 如何添加更多指标？

A: 可以扩展 `strategy_data`，然后修改 `_plot_xxx` 辅助方法。例如添加成交量：

```python
# 在 _plot_price_and_signals 中添加：
ax2 = ax.twinx()
ax2.bar(strategy_data.index, strategy_data['volume'], alpha=0.3, color='gray')
```

### Q: 支持做空策略吗？

A: 支持！`position` 列可以是负数（-1表示做空）。

## 进阶定制

如果需要更精细的控制，可以单独调用各个子图方法：

```python
visualizer = TradingVisualizer()

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

visualizer._plot_price_and_signals(axes[0], strategy_data, transactions)
visualizer._plot_nav_comparison(axes[1], strategy_data)
visualizer._plot_drawdown(axes[2], strategy_data)

plt.tight_layout()
plt.savefig('custom_plot.png')
```

## 更新日志

- **2025-01-03**: 新增 `plot_strategy_comprehensive()` 方法
- 参考 `MA_strategy_v2.py` 的设计理念
- 支持自动从持仓推断买卖点
- 增加回撤曲线和持仓变化子图

