"""
Time Bars 使用示例
演示如何使用时间切分功能对tick数据进行重采样
"""
import pandas as pd
from pipeline.trading_pipeline import TradingPipeline

# ===== 示例1: 使用15分钟时间条 =====
def example_15min_bars():
    """使用15分钟时间条"""
    pipeline = TradingPipeline()
    
    # 配置参数
    config = {
        'bar_type': 'time',           # 使用时间条
        'time_interval': '15min',     # 15分钟间隔
        'data_config': {
            'trades_zip_path': 'path/to/your/trades.zip'
        },
        'bar_zip_path': 'output/bars_15min.zip',
        'feature_window_bars': 10,
        'model_type': 'ridge',
        'target_horizon': 5,
        'n_splits': 5,
        'embargo_bars': 3,
        'save_plots': True,
        'plot_save_dir': 'output/plots_15min'
    }
    
    # 运行完整管道
    results = pipeline.run_full_pipeline(**config)
    print("15分钟bars分析完成！")
    return results


# ===== 示例2: 使用1小时时间条 =====
def example_1hour_bars():
    """使用1小时时间条"""
    pipeline = TradingPipeline()
    
    config = {
        'bar_type': 'time',           # 使用时间条
        'time_interval': '1h',        # 1小时间隔
        'data_config': {
            'trades_zip_path': 'path/to/your/trades.zip'
        },
        'bar_zip_path': 'output/bars_1h.zip',
        'feature_window_bars': 10,
        'model_type': 'ridge',
        'target_horizon': 5,
    }
    
    results = pipeline.run_full_pipeline(**config)
    print("1小时bars分析完成！")
    return results


# ===== 示例3: 比较Dollar Bars和Time Bars =====
def compare_dollar_vs_time_bars():
    """比较Dollar Bars和Time Bars的性能"""
    
    # Dollar Bars
    print("=" * 50)
    print("运行 Dollar Bars 分析...")
    print("=" * 50)
    pipeline_dollar = TradingPipeline()
    results_dollar = pipeline_dollar.run_full_pipeline(
        bar_type='dollar',
        dollar_threshold=60000000,
        data_config={'trades_zip_path': 'path/to/your/trades.zip'},
        bar_zip_path='output/bars_dollar.zip',
        feature_window_bars=10,
        model_type='ridge',
    )
    
    # Time Bars (15分钟)
    print("\n" + "=" * 50)
    print("运行 Time Bars (15分钟) 分析...")
    print("=" * 50)
    pipeline_time = TradingPipeline()
    results_time = pipeline_time.run_full_pipeline(
        bar_type='time',
        time_interval='15min',
        data_config={'trades_zip_path': 'path/to/your/trades.zip'},
        bar_zip_path='output/bars_15min.zip',
        feature_window_bars=10,
        model_type='ridge',
    )
    
    # 比较结果
    print("\n" + "=" * 50)
    print("结果比较")
    print("=" * 50)
    print(f"Dollar Bars - 平均IC: {results_dollar['evaluation']['summary']['pearson_ic_mean']:.4f}")
    print(f"Time Bars   - 平均IC: {results_time['evaluation']['summary']['pearson_ic_mean']:.4f}")
    print(f"\nDollar Bars - bars数量: {len(results_dollar['bars'])}")
    print(f"Time Bars   - bars数量: {len(results_time['bars'])}")
    
    return results_dollar, results_time


# ===== 示例4: 只构建Time Bars（不运行完整管道）=====
def example_build_time_bars_only():
    """只构建Time Bars，不进行特征提取和模型训练"""
    pipeline = TradingPipeline()
    
    # 加载数据
    trades_df = pipeline.load_data(trades_zip_path='path/to/your/trades.zip')
    print(f"加载了 {len(trades_df)} 条交易记录")
    
    # 构建30分钟Time Bars
    bars = pipeline.build_bars(
        trades_df,
        bar_type='time',
        time_interval='30min',
        bar_zip_path='output/bars_30min.zip'
    )
    
    print(f"构建了 {len(bars)} 个30分钟bars")
    print("\nBars结构预览:")
    print(bars.head())
    print("\nBars列:")
    print(bars.columns.tolist())
    
    return bars


# ===== 时间间隔支持说明 =====
"""
支持的时间间隔格式（基于pandas的频率字符串）：
- '1min', '5min', '15min', '30min' - 分钟级别
- '1h', '2h', '4h', '6h', '12h'    - 小时级别
- '1D'                              - 日级别
- '1W'                              - 周级别

示例：
- '15min' -> 15分钟K线
- '1h'    -> 1小时K线
- '4h'    -> 4小时K线
- '1D'    -> 日K线
"""


if __name__ == '__main__':
    # 运行示例（请先修改数据路径）
    print("Time Bars 使用示例")
    print("请先修改代码中的数据路径，然后取消注释下面的代码运行")
    
    # 示例1: 15分钟bars
    # example_15min_bars()
    
    # 示例2: 1小时bars
    # example_1hour_bars()
    
    # 示例3: 比较Dollar Bars和Time Bars
    # compare_dollar_vs_time_bars()
    
    # 示例4: 只构建bars
    # example_build_time_bars_only()

