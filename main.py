"""
主程序入口
"""
import os
from pipeline.trading_pipeline import TradingPipeline


def main():
    """主函数"""
    # 配置参数
    config = {
        'features': {
            'base': True,
            'order_flow': True,
            'price_impact': True,
            'volatility_noise': True,
            'path_shape': True,
            'tail_directional': True,
        },
        'model': {
            'random_state': 42
        }
    }
    
    # 数据配置
    start_date = '2025-01-01'
    end_date = '2025-01-10'
    dollar_threshold = 10000 * 60000
    dollar_threshold_str = str(dollar_threshold).replace("*", "_")
    
    # 文件路径
    base_path = '/Users/aming/project/python/crypto-trade/output'
    trades_zip_path = f'{base_path}/trades-{start_date}-{end_date}-{dollar_threshold_str}.zip'
    bar_zip_path = f'{base_path}/bars-{start_date}-{end_date}-{dollar_threshold_str}.zip'
    plot_save_dir = '/Users/aming/project/python/crypto-trade/strategy/fusion/pic'
    
    # 数据路径模板
    data_path_template = '/Volumes/Ext-Disk/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-{date}.zip'
    
    # 创建管道
    pipeline = TradingPipeline(config)
    
    # 运行参数
    run_config = {
        'data_config': {
            'trades_zip_path': trades_zip_path,
            'date_range': (start_date, end_date),
            'data_path_template': data_path_template
        },
        'dollar_threshold': dollar_threshold,
        'bar_zip_path': bar_zip_path,
        'feature_window_bars': 10,
        'model_type': 'linear',
        'target_horizon': 5,
        'n_splits': 5,
        'embargo_bars': 3,
        'save_plots': True,
        'plot_save_dir': plot_save_dir
    }
    
    try:
        # 运行完整管道
        results = pipeline.run_full_pipeline(**run_config)
        
        # 输出结果摘要
        print("\n" + "="*50)
        print("分析结果摘要")
        print("="*50)
        summary = results['evaluation']['summary']
        print(f"平均Pearson IC: {summary['pearson_ic_mean']:.4f}")
        print(f"平均Spearman IC: {summary['spearman_ic_mean']:.4f}")
        print(f"平均RMSE: {summary['rmse_mean']:.4f}")
        print(f"平均方向准确率: {summary['dir_acc_mean']:.4f}")
        print(f"平均夏普比率: {summary['sharpe_net_mean']:.4f}")
        print(f"有效折数: {summary['n_splits_effective']}")
        
        # 特征重要性前10
        if hasattr(results['model'], 'get_feature_importance'):
            importance = results['model'].get_feature_importance()
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n前10重要特征:")
            for i, (feature, score) in enumerate(top_features, 1):
                print(f"{i:2d}. {feature}: {score:.6f}")
        
        print(f"\n数据统计:")
        print(f"样本数量: {len(results['features'])}")
        print(f"特征数量: {len(results['features'].columns)}")
        print(f"Bar数量: {len(results['bars'])}")
        
        return results
        
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
