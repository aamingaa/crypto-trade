# Label (方向 / 波动率 / 极值)
#  ├─ 特征因子
#  │   ├─ 微观结构 (OB, VWAP偏离, 主动买卖, spread)
#  │   ├─ 技术因子 (动量, 波动率, K线形态)
#  │   ├─ 跨周期因子 (日线趋势, 开盘效应, HAR)
#  │   ├─ 流动性因子 (成交量, Amihud, spread*volume)
#  │   └─ 衍生因子 (交互项, PCA/AE embedding)
#  ├─ 模型层
#  │   ├─ 线性类 (ARIMA, HAR, GARCH)
#  │   ├─ ML类 (RF, XGBoost, LightGBM)
#  │   ├─ DL类 (LSTM, Transformer)
#  │   └─ 混合 (HAR-GARCH + ML残差)
#  └─ 评估
#      ├─ 方向准确率
#      ├─ 收益指标
#      ├─ 风险指标
#      └─ 稳定性评估

