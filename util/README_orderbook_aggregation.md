# 订单簿数据聚合工具

## 概述

这个工具用于将ns（纳秒）级别的五档订单簿数据聚合到ms（毫秒）级别。适用于高频交易数据分析、市场微观结构研究等场景。

## 功能特性

- **多种聚合方法**：支持基本聚合、成交量加权平均价格(VWAP)、OHLC聚合
- **灵活的时间窗口**：可自定义聚合时间窗口（1ms, 5ms, 10ms等）
- **订单簿快照**：生成标准化的五档订单簿快照
- **数据完整性**：保持原始数据的完整性和一致性
- **高性能**：优化的pandas操作，支持大规模数据处理

## 安装依赖

```bash
pip install pandas numpy
```

## 使用方法

### 基本使用

```python
from util.orderbook_aggregation import OrderBookAggregator
import pandas as pd

# 加载数据
df = pd.read_csv('your_orderbook_data.csv.gz')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')

# 创建聚合器（1ms时间窗口）
aggregator = OrderBookAggregator(time_window_ms=1)

# 基本聚合
aggregated_data = aggregator.aggregate_orderbook_data(df)
```

### 聚合方法

#### 1. 基本聚合（Last Value）
使用时间窗口内的最后一个值作为代表值：

```python
aggregated = aggregator.aggregate_orderbook_data(df)
```

#### 2. 成交量加权平均价格（VWAP）
根据成交量计算加权平均价格：

```python
aggregated_vwap = aggregator.aggregate_with_volume_weighted_price(df)
```

#### 3. OHLC聚合
计算时间窗口内的开高低收价格：

```python
aggregated_ohlc = aggregator.aggregate_with_ohlc(df)
```

### 订单簿快照

生成标准化的五档订单簿快照：

```python
snapshot = aggregator.get_orderbook_snapshot(df, method='last')
```

快照包含以下字段：
- `timestamp_ms`: 毫秒时间戳
- `bid_price_1` 到 `bid_price_5`: 买盘价格（1-5档）
- `bid_amount_1` 到 `bid_amount_5`: 买盘数量（1-5档）
- `ask_price_1` 到 `ask_price_5`: 卖盘价格（1-5档）
- `ask_amount_1` 到 `ask_amount_5`: 卖盘数量（1-5档）

## 数据格式要求

输入数据必须包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| timestamp | datetime | 时间戳（纳秒精度） |
| side | string | 买卖方向（'buy'/'sell'） |
| price | float | 价格 |
| amount | float | 数量 |
| symbol | string | 交易对 |
| exchange | string | 交易所 |

可选列：
- `local_timestamp`: 本地时间戳
- `exchange_timestamp`: 交易所时间戳
- `sequence_number`: 序列号
- `channel`: 数据通道
- `id`: 记录ID

## 聚合策略说明

### 时间窗口划分
- 使用 `pd.Timestamp.floor()` 将纳秒时间戳向下取整到指定的毫秒级别
- 例如：`1575158400123456789ns` → `1575158400123000000ns` (1ms窗口)

### 聚合规则
- **时间戳**: 使用窗口内第一个时间戳
- **序列号**: 使用窗口内最新的序列号
- **价格**: 根据聚合方法确定（平均值、VWAP、OHLC等）
- **数量**: 窗口内数量求和
- **其他字段**: 使用窗口内第一个值

### 买卖盘处理
- 分别对买盘和卖盘进行聚合
- 保持买卖盘的价格排序（买盘降序，卖盘升序）
- 生成标准化的五档订单簿结构

## 性能优化建议

1. **内存管理**：
   ```python
   # 对于大文件，使用分块读取
   chunks = pd.read_csv('large_file.csv.gz', chunksize=100000)
   for chunk in chunks:
       aggregated_chunk = aggregator.aggregate_orderbook_data(chunk)
   ```

2. **数据类型优化**：
   ```python
   # 使用适当的数据类型减少内存使用
   df['price'] = df['price'].astype('float32')
   df['amount'] = df['amount'].astype('float32')
   ```

3. **并行处理**：
   ```python
   # 对于多个文件，可以使用多进程
   from multiprocessing import Pool
   
   def process_file(filename):
       df = pd.read_csv(filename)
       return aggregator.aggregate_orderbook_data(df)
   
   with Pool(4) as p:
       results = p.map(process_file, file_list)
   ```

## 示例脚本

运行示例脚本：

```bash
# 运行演示
python util/orderbook_aggregation.py

# 运行实际数据处理
python tardis/aggregate_orderbook_example.py
```

## 输出文件

聚合工具会生成以下输出文件：

1. `aggregated_basic_1ms.csv`: 基本聚合结果
2. `aggregated_vwap_1ms.csv`: VWAP聚合结果
3. `aggregated_ohlc_1ms.csv`: OHLC聚合结果
4. `orderbook_snapshot_1ms.csv`: 订单簿快照

## 注意事项

1. **数据质量**：确保输入数据的时间戳是连续的，没有重复或缺失
2. **内存使用**：处理大文件时注意内存使用情况
3. **时间精度**：聚合会损失纳秒级精度，请根据实际需求选择合适的时间窗口
4. **数据一致性**：不同聚合方法可能产生不同的结果，请根据应用场景选择合适的方法

## 常见问题

### Q: 如何处理缺失数据？
A: 聚合器会自动跳过缺失的时间窗口，确保输出数据的连续性。

### Q: 聚合后的数据量会减少多少？
A: 数据压缩比取决于原始数据的更新频率和时间窗口大小。通常1ms窗口可以减少50-90%的数据量。

### Q: 如何选择合适的时间窗口？
A: 根据分析需求选择：
- 1ms: 高频交易分析
- 5-10ms: 市场微观结构研究
- 50-100ms: 一般市场分析

## 扩展功能

可以根据需要扩展以下功能：

1. **更多聚合方法**：中位数、众数等
2. **自定义聚合函数**：支持用户定义的聚合逻辑
3. **实时聚合**：支持流式数据的实时聚合
4. **多维度聚合**：按价格档位、交易所等维度聚合

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。

