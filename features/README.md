# 微观结构特征提取模块

该模块提供了完整的微观结构特征提取功能，将原来的单一特征提取器拆分为多个专门化的子模块，便于维护和扩展。

## 模块结构

```
features/
├── __init__.py                      # 模块入口
├── base_extractor.py               # 特征提取器基类
├── microstructure_extractor.py     # 主特征提取器
├── basic_features.py               # 基础特征提取器
├── volatility_features.py          # 波动率特征提取器
├── momentum_features.py            # 动量与反转特征提取器
├── orderflow_features.py           # 订单流特征提取器
├── impact_features.py              # 价格冲击特征提取器
├── tail_features.py                # 大单特征提取器
├── path_shape_features.py          # 路径形状特征提取器
└── README.md                       # 本文档
```

## 特征分类

### 1. 基础特征 (BasicFeatureExtractor)
- `int_trade_vwap`: 成交量加权平均价
- `int_trade_volume_sum`: 总成交量
- `int_trade_dollar_sum`: 总成交金额
- `int_trade_signed_volume`: 有向成交量
- `int_trade_buy_ratio`: 买量占比
- `int_trade_intensity`: 交易强度
- `int_trade_rv`: 已实现波动率

### 2. 波动率特征 (VolatilityFeatureExtractor)
- `rv`: 已实现波动率
- `bpv`: 双幂变差
- `jump_rv_bpv`: 跳跃组件 (RV - BPV)
- `micro_dp_short`: 微动量（短期价格变化）
- `micro_dp_zscore`: 微动量Z-score
- `hl_amplitude_ratio`: 高低幅度比

### 3. 动量与反转特征 (MomentumFeatureExtractor)
- `mr_rho1`: 一阶自相关系数
- `mr_strength`: 均值回复强度

### 4. 订单流特征 (OrderFlowFeatureExtractor)
- `ofi_signed_qty_sum`: 有向成交量和（OFI相关）
- `ofi_signed_quote_sum`: 有向成交金额和（OFI相关）
- `gof_by_count`: 按交易次数的Garman订单流
- `gof_by_volume`: 按成交量的Garman订单流

### 5. 价格冲击特征 (PriceImpactFeatureExtractor)
- `kyle_lambda`: Kyle Lambda (价格冲击系数)
- `amihud_lambda`: Amihud Lambda (非流动性指标)
- `hasbrouck_lambda`: Hasbrouck Lambda (信息冲击系数)
- `impact_half_life`: 价格冲击半衰期
- `impact_perm_share`: 永久冲击占比
- `impact_transient_share`: 暂时冲击占比

### 6. 大单特征 (TailFeatureExtractor)
- `large_q90_buy_dollar_sum`: 90分位大单买方金额
- `large_q90_sell_dollar_sum`: 90分位大单卖方金额
- `large_q90_lti`: 90分位大单流动性接受不平衡指标
- `large_q95_buy_dollar_sum`: 95分位大单买方金额
- `large_q95_sell_dollar_sum`: 95分位大单卖方金额
- `large_q95_lti`: 95分位大单流动性接受不平衡指标

### 7. 路径形状特征 (PathShapeFeatureExtractor)
- `signed_vwap_deviation`: 有向VWAP偏离
- `vwap_deviation`: VWAP偏离
- `corr_cumsum_signed_qty_logp`: 累计有向成交量与对数价格相关性
- `corr_cumsum_signed_dollar_logp`: 累计有向成交金额与对数价格相关性

## 使用方法

### 基本使用

```python
from features import MicrostructureFeatureExtractor
from data.trades_processor import TradesContext

# 创建特征提取器
extractor = MicrostructureFeatureExtractor()

# 从交易上下文中提取特征
features = extractor.extract_from_context(ctx, start_ts, end_ts)
```

### 配置特征提取

```python
# 自定义配置
config = {
    'basic': True,          # 启用基础特征
    'volatility': True,     # 启用波动率特征
    'momentum': True,       # 启用动量特征
    'orderflow': False,     # 禁用订单流特征
    'impact': True,         # 启用价格冲击特征
    'tail': {               # 大单特征配置
        'enabled': True,
        'quantiles': [0.9, 0.95, 0.99]  # 自定义分位数
    },
    'path_shape': True,     # 启用路径形状特征
}

extractor = MicrostructureFeatureExtractor(config)
```

### 使用单个特征提取器

```python
from features import VolatilityFeatureExtractor

# 只使用波动率特征
volatility_extractor = VolatilityFeatureExtractor()
volatility_features = volatility_extractor.extract_from_context(ctx, start_ts, end_ts)
```

### 动态控制特征组

```python
# 运行时启用/禁用特征组
extractor.enable_feature_group('orderflow', False)  # 禁用订单流特征
extractor.enable_feature_group('impact', True)      # 启用价格冲击特征

# 获取所有特征组名称
groups = extractor.get_feature_groups()
print(groups)  # ['basic', 'volatility', 'momentum', ...]
```

## 扩展新特征

如需添加新的特征类型，请按以下步骤：

1. 继承 `MicrostructureBaseExtractor` 创建新的特征提取器
2. 实现 `get_feature_names()` 和 `_extract_features()` 方法
3. 在 `MicrostructureFeatureExtractor` 中注册新的提取器
4. 更新 `__init__.py` 导出新的提取器

## 注意事项

1. 所有特征提取器都需要 `TradesContext` 对象作为输入
2. 特征提取过程中如果出错，会打印警告但不会中断整个流程
3. 可以通过配置文件精细控制每个特征组的参数
4. 建议在生产环境中根据实际需求选择性启用特征组以提高性能
