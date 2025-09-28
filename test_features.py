"""
测试新的微观结构特征提取器
"""
from features import MicrostructureFeatureExtractor

def test_feature_extractor():
    """测试特征提取器的初始化"""
    
    # 测试1: 默认配置
    print("测试1: 默认配置")
    try:
        extractor1 = MicrostructureFeatureExtractor()
        print("✓ 默认配置初始化成功")
        print(f"特征组: {extractor1.get_feature_groups()}")
    except Exception as e:
        print(f"✗ 默认配置初始化失败: {e}")
    
    # 测试2: 布尔值配置
    print("\n测试2: 布尔值配置")
    try:
        config_bool = {
            'basic': True,
            'volatility': True,
            'momentum': False,
            'orderflow': True,
            'impact': False,
            'tail': True,
            'path_shape': False,
        }
        extractor2 = MicrostructureFeatureExtractor(config_bool)
        print("✓ 布尔值配置初始化成功")
        print(f"特征配置: {extractor2.feature_config}")
    except Exception as e:
        print(f"✗ 布尔值配置初始化失败: {e}")
    
    # 测试3: 混合配置（布尔值 + 字典）
    print("\n测试3: 混合配置")
    try:
        config_mixed = {
            'basic': True,
            'volatility': {'enabled': True, 'micro_momentum_window': 15},
            'momentum': False,
            'tail': {'enabled': True, 'quantiles': [0.9, 0.95, 0.99]},
            'impact': {'enabled': True, 'min_trades': 5},
        }
        extractor3 = MicrostructureFeatureExtractor(config_mixed)
        print("✓ 混合配置初始化成功")
        
        # 测试获取特征名称
        feature_names = extractor3.get_feature_names()
        print(f"总特征数: {len(feature_names)}")
        print(f"前10个特征: {feature_names[:10]}")
        
    except Exception as e:
        print(f"✗ 混合配置初始化失败: {e}")

if __name__ == "__main__":
    test_feature_extractor()
