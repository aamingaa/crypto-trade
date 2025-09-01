"""
多模型量化交易策略类
重构版本 - 面向对象设计

主要功能：
1. 技术指标因子生成
2. 因子筛选和预处理
3. 多模型训练（OLS, Ridge, Lasso, XGBoost, LightGBM）
4. 回测和风险指标计算
5. 结果可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import math
import warnings
import os  # 添加 os 模块导入
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体支持（Mac系统）
import platform

def setup_chinese_font_for_mac():
    """
    为Mac系统设置中文字体支持
    """
    if platform.system() == 'Darwin':  # Mac系统
        # 检查系统可用字体
        available_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
        
        # Mac系统常见的中文字体列表（按优先级排序）
        mac_chinese_fonts = [
            'PingFang SC',      # 苹果默认中文字体
            'Songti SC',        # 宋体
            'STSong',          # 华文宋体
            'Arial Unicode MS', # 支持中文的Arial
            'SimHei',          # 黑体
            'Hiragino Sans GB', # 冬青黑体
            'STHeiti'          # 华文黑体
        ]
        
        # 寻找可用的中文字体
        selected_font = None
        for font in mac_chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已设置中文字体: {selected_font}")
            return True
        else:
            # 如果找不到中文字体，提供解决方案
            print("⚠️  未检测到可用的中文字体")
            print("📝 解决方案：")
            print("1. 使用Homebrew安装中文字体包：")
            print("   brew install --cask font-source-han-sans")
            print("   brew install --cask font-source-han-serif")
            print("2. 或者手动下载安装思源黑体：")
            print("   https://github.com/adobe-fonts/source-han-sans")
            print("3. 重启Python内核后重新运行")
            
            # 设置基本配置避免负号显示问题
            plt.rcParams['axes.unicode_minus'] = False
            return False
    return True

# 调用字体设置函数
setup_chinese_font_for_mac()

if platform.system() == 'Darwin':  # Mac系统
    # 尝试设置常见的Mac中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 如果上述字体不可用，使用系统默认字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("警告：未找到中文字体，图表中的中文可能显示异常")

import talib as ta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import pickle

class QuantTradingStrategy:
    """
    多模型量化交易策略类
    """
    
    def __init__(self, data_path, config=None):
        """
        初始化策略
        
        Args:
            data_path (str): 数据文件路径
            config (dict): 配置参数
        """
        self.data_path = data_path
        self.config = config or self._get_default_config()
        
        # 数据相关
        self.raw_data = None
        self.factor_data = None
        self.feed_data = None
        
        # 训练数据
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_set_end_index = None
        
        # 模型
        self.models = {}
        self.predictions = {}
        
        # 回测结果
        self.backtest_results = {}
        self.performance_metrics = None
        
        # 设置中文字体
        self._setup_chinese_font()
        
        print(f"策略初始化完成")
        # print(f"XGBoost版本: {xgb.__version__}")
        # print(f"LightGBM版本: {lgb.__version__}")
    
    def _setup_chinese_font(self):
        """设置中文字体支持"""
        # 字体设置已在文件开头完成，这里只做简单检查
        if platform.system() == 'Darwin':
            current_font = plt.rcParams['font.sans-serif'][0] if plt.rcParams['font.sans-serif'] else 'default'
            print(f"当前使用字体: {current_font}")
        pass
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'return_period': 1,  # 收益率计算周期
            'corr_threshold': 0.3,  # 相关性筛选阈值
            'sharpe_threshold': 0.2,  # 夏普比率筛选阈值
            'train_end_date': '2016-12-30',  # 训练集结束日期
            'position_size': 1.0,  # 仓位大小
            'clip_num': 2.0,  # 仓位限制
            'fixed_return': 0.0,  # 无风险收益率
            'fees_rate': 0.004,  # 手续费率
            'model_save_path': './'  # 改为当前目录
        }
    
    def load_data(self):
        """加载和预处理原始数据"""
        print("正在加载数据...")
        
        # 读取数据
        self.raw_data = pd.read_pickle(self.data_path).reset_index()
        self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
        self.raw_data = self.raw_data.sort_values(by='timestamp', ascending=True)
        self.raw_data = self.raw_data.set_index('timestamp')
        
        # 计算收益率
        t = self.config['return_period']
        self.raw_data['return'] = (self.raw_data['close'].shift(-t) / self.raw_data['close'] - 1)
        self.raw_data = self.raw_data.replace([np.nan], 0.0)
        
        print(f"数据加载完成，共 {len(self.raw_data)} 条记录")
        return self
    
    def generate_factors(self):
        """生成技术指标因子"""
        print("正在生成技术指标因子...")
        
        fct_value = pd.DataFrame()
        close = self.raw_data['close']
        high = self.raw_data['high']
        low = self.raw_data['low']
        volume = self.raw_data['volume']
        
        # 1. MA类指标
        fct_value['ma5'] = ta.MA(close, timeperiod=5, matype=0)
        fct_value['ma10'] = ta.MA(close, timeperiod=10, matype=0)
        fct_value['ma20'] = ta.MA(close, timeperiod=20, matype=0)
        fct_value['ma5diff'] = fct_value['ma5'] / close - 1
        fct_value['ma10diff'] = fct_value['ma10'] / close - 1
        fct_value['ma20diff'] = fct_value['ma20'] / close - 1
        
        # 2. 布林带指标
        fct_value['h_line'], fct_value['m_line'], fct_value['l_line'] = ta.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        fct_value['stdevrate'] = (fct_value['h_line'] - fct_value['l_line']) / (close * 4)
        
        # 3. SAR指标
        fct_value['sar_index'] = ta.SAR(high, low)
        fct_value['sar_close'] = (fct_value['sar_index'] - close) / close
        
        # 4. Aroon指标
        fct_value['aroon_index'] = ta.AROONOSC(high, low, timeperiod=14)
        
        # 5. CCI指标
        fct_value['cci_14'] = ta.CCI(close, high, low, timeperiod=14)
        fct_value['cci_25'] = ta.CCI(close, high, low, timeperiod=25)
        fct_value['cci_55'] = ta.CCI(close, high, low, timeperiod=55)
        
        # 6. CMO指标
        fct_value['cmo_14'] = ta.CMO(close, timeperiod=14)
        fct_value['cmo_25'] = ta.CMO(close, timeperiod=25)
        
        # 7. MFI指标
        fct_value['mfi_index'] = ta.MFI(high, low, close, volume)
        
        # 8. 动量指标
        fct_value['mom_14'] = ta.MOM(close, timeperiod=14)
        fct_value['mom_25'] = ta.MOM(close, timeperiod=25)
        
        # 9. PPO指标
        fct_value['ppo_index'] = ta.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        
        # 10. AD指标
        fct_value['ad_index'] = ta.AD(high, low, close, volume)
        fct_value['ad_real'] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        # 11. OBV指标
        fct_value['obv_index'] = ta.OBV(close, volume)
        
        # 12. ATR指标
        fct_value['atr_14'] = ta.ATR(high, low, close, timeperiod=14)
        fct_value['atr_25'] = ta.ATR(high, low, close, timeperiod=25)
        fct_value['atr_60'] = ta.ATR(high, low, close, timeperiod=60)
        fct_value['tr_index'] = ta.TRANGE(high, low, close)
        fct_value['tr_ma5'] = ta.MA(fct_value['tr_index'], timeperiod=5, matype=0) / close
        fct_value['tr_ma10'] = ta.MA(fct_value['tr_index'], timeperiod=10, matype=0) / close
        fct_value['tr_ma20'] = ta.MA(fct_value['tr_index'], timeperiod=20, matype=0) / close
        
        # 13. KDJ指标
        fct_value['kdj_k'], fct_value['kdj_d'] = ta.STOCH(
            high, low, close, fastk_period=9, slowk_period=5, slowk_matype=1,
            slowd_period=5, slowd_matype=1)
        fct_value['kdj_j'] = fct_value['kdj_k'] - fct_value['kdj_d']
        
        # 14. MACD指标
        fct_value['macd_dif'], fct_value['macd_dea'], fct_value['macd_hist'] = ta.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # 15. RSI指标
        fct_value['rsi_6'] = ta.RSI(close, timeperiod=6)
        fct_value['rsi_12'] = ta.RSI(close, timeperiod=12)
        fct_value['rsi_25'] = ta.RSI(close, timeperiod=25)
        
        # 处理缺失值
        fct_value = fct_value.replace([np.nan], 0.0)
        
        self.factor_data = fct_value
        print(f"因子生成完成，共生成 {len(fct_value.columns)} 个因子")
        return self
    
    def normalize_factors(self):
        """因子标准化"""
        print("正在进行因子标准化...")
        
        # 扩展窗口标准化
        factors_mean = self.factor_data.cumsum() / np.arange(1, self.factor_data.shape[0] + 1)[:, np.newaxis]
        factors_std = self.factor_data.expanding().std()
        factor_value = (self.factor_data - factors_mean) / factors_std
        factor_value = factor_value.replace([np.nan], 0.0)
        factor_value = factor_value.clip(-6, 6)  # 异常值处理
        
        self.factor_data = factor_value
        print("因子标准化完成")
        return self
    
    def factor_selection_by_correlation(self, column_list, corr_threshold):
        """基于相关性筛选因子"""
        fac_columns = column_list.copy()
        if 'return' in column_list:
            fac_columns = column_list[:-1]
        
        # 创建临时数据框用于筛选
        temp_data = self.factor_data[fac_columns].copy()
        temp_data['return'] = self.raw_data['return'].values
        
        X = temp_data[fac_columns]
        y = temp_data['return']
        
        # 计算相关性矩阵
        X_corr_matrix = X.corr()
        
        factor_list_1 = [i for i in X_corr_matrix.columns]
        factor_list_2 = [i for i in X_corr_matrix.columns]
        
        for i in range(0, len(factor_list_1), 1):
            fct_1 = factor_list_1[i]
            for j in range(0, i, 1):
                fct_2 = factor_list_1[j]
                corr_value = X_corr_matrix.iloc[i, j]
                if abs(corr_value) > corr_threshold:
                    corr_1 = np.corrcoef(X[fct_1], y)[0, 1]
                    corr_2 = np.corrcoef(X[fct_2], y)[0, 1]
                    if (abs(corr_1) < abs(corr_2)) and (fct_1 in factor_list_2):
                        factor_list_2.remove(fct_1)
        
        return factor_list_2
    
    def select_factors(self):
        """因子筛选"""
        print("正在进行因子筛选...")
        
        # 添加收益率到因子数据
        factor_data_with_return = self.factor_data.copy()
        factor_data_with_return['return'] = self.raw_data['return'].values
        
        # 获取所有因子列名
        column_list = list(self.factor_data.columns)
        
        # 基于相关性筛选
        selected_factors = self.factor_selection_by_correlation(
            column_list, self.config['corr_threshold'])
        
        print(f"相关性筛选后剩余 {len(selected_factors)} 个因子")
        
        # 这里可以添加更多筛选逻辑，比如基于单因子表现等
        # 目前使用相关性筛选的结果
        self.selected_factors = selected_factors
        
        return self
    
    def prepare_training_data(self):
        """准备训练数据"""
        print("正在准备训练数据...")
        
        # 准备feed_data
        self.feed_data = self.factor_data[self.selected_factors].copy()
        self.feed_data['y'] = self.raw_data['return'].values
        self.feed_data = self.feed_data.reset_index()
        
        # 确定训练集结束索引
        train_end_date = pd.to_datetime(self.config['train_end_date'])
        mask = ((self.feed_data['timestamp'].dt.year == train_end_date.year) & 
                (self.feed_data['timestamp'].dt.month == train_end_date.month) & 
                (self.feed_data['timestamp'].dt.day == train_end_date.day))
        
        if mask.any():
            self.train_set_end_index = self.feed_data[mask].index[-1]
        else:
            # 如果找不到确切日期，使用最接近的日期
            self.train_set_end_index = int(len(self.feed_data) * 0.7)  # 70%作为训练集
        
        # 划分训练集和测试集
        self.X_train = self.feed_data[self.selected_factors][:self.train_set_end_index].values
        self.y_train = self.feed_data['y'][:self.train_set_end_index].values.reshape(-1, 1)
        self.X_test = self.feed_data[self.selected_factors][self.train_set_end_index:].values
        self.y_test = self.feed_data['y'][self.train_set_end_index:].values.reshape(-1, 1)
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        
        return self
    
    def train_models(self):
        """训练多个模型"""
        print("正在训练模型...")
        
        # 1. 线性回归
        print("训练线性回归模型...")
        lr_model = LinearRegression(fit_intercept=True)
        lr_model.fit(self.X_train, self.y_train)
        self.models['LinearRegression'] = lr_model
        
        # 2. Ridge回归
        print("训练Ridge回归模型...")
        ridge_model = Ridge(alpha=0.2, fit_intercept=True)
        ridge_model.fit(self.X_train, self.y_train)
        self.models['Ridge'] = ridge_model
        
        # 3. Lasso回归
        print("训练Lasso回归模型...")
        lasso_model = LassoCV(fit_intercept=True)
        lasso_model.fit(self.X_train, self.y_train.ravel())
        self.models['Lasso'] = lasso_model
        
        # 4. XGBoost
        print("训练XGBoost模型...")
        X_train_df = pd.DataFrame(self.X_train, columns=self.selected_factors)
        y_train_series = pd.Series(self.y_train.ravel())
        X_test_df = pd.DataFrame(self.X_test, columns=self.selected_factors)
        y_test_series = pd.Series(self.y_test.ravel())
        
        xgb_model = XGBRegressor(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            objective='reg:squarederror',
            random_state=0,
            early_stopping_rounds=20
        )
        
        xgb_model.fit(
            X_train_df, y_train_series,
            eval_set=[(X_test_df, y_test_series)],
            verbose=False
        )
        
        self.models['XGBoost'] = xgb_model
        
        # 5. LightGBM
        print("训练LightGBM模型...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'learning_rate': 0.054,
            'max_depth': 3,
            'num_leaves': 32,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'lambda_l1': 0.05,
            'lambda_l2': 120,
            'verbose': -1
        }
        
        lgb_train = lgb.Dataset(X_train_df, y_train_series)
        lgb_val = lgb.Dataset(X_test_df, y_test_series, reference=lgb_train)
        
        lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_val,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        self.models['LightGBM'] = lgb_model
        
        print("所有模型训练完成")
        return self
    
    def make_predictions(self, weight_method='equal'):
        """生成预测
        
        Args:
            weight_method (str): 权重计算方法，可选 'equal'（等权重）或 'sharpe'（基于夏普比率）
        """
        print("正在生成预测...")
        
        # 存储所有模型的预测结果
        all_train_predictions = []
        all_test_predictions = []
        model_names = []
        
        for model_name, model in self.models.items():
            if model_name == 'LightGBM':
                # LightGBM需要特殊处理
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
            else:
                train_pred = model.predict(self.X_train).flatten()
                test_pred = model.predict(self.X_test).flatten()
            
            # 存储单个模型的预测结果
            self.predictions[model_name] = {
                'train': train_pred,
                'test': test_pred
            }
            
            # 收集所有模型的预测结果用于组合
            all_train_predictions.append(train_pred)
            all_test_predictions.append(test_pred)
            model_names.append(model_name)
        
        # 计算组合权重
        if weight_method == 'equal':
            # 等权重组合
            weights = {name: 1.0/len(model_names) for name in model_names}
        elif weight_method == 'sharpe':
            # 基于夏普比率的权重
            # 先对每个模型进行回测以获取夏普比率
            sharpe_ratios = {}
            for model_name in model_names:
                # 获取时间索引
                train_time = self.feed_data['timestamp'][:self.train_set_end_index].values
                test_time = self.feed_data['timestamp'][self.train_set_end_index:].values
                
                # 计算训练集表现
                train_ret_frame = self.calculate_portfolio_performance(
                    self.predictions[model_name]['train'], train_time, 'train')
                
                # 计算测试集表现
                test_ret_frame = self.calculate_portfolio_performance(
                    self.predictions[model_name]['test'], test_time, 'test')
                
                # 计算夏普比率
                train_metrics = self.calculate_performance_metrics(train_ret_frame)
                test_metrics = self.calculate_performance_metrics(test_ret_frame)
                
                # 使用测试集夏普比率的绝对值作为权重
                sharpe_ratios[model_name] = abs(test_metrics['夏普比率'])
            
            # 计算权重（确保权重和为1）
            total_sharpe = sum(sharpe_ratios.values())
            if total_sharpe > 0:
                weights = {name: sharpe/total_sharpe for name, sharpe in sharpe_ratios.items()}
            else:
                # 如果所有夏普比率都为0，使用等权重
                weights = {name: 1.0/len(model_names) for name in model_names}
        else:
            raise ValueError(f"不支持的权重计算方法: {weight_method}")
        
        # 存储权重信息
        self.ensemble_weights = weights
        
        # 计算加权组合预测
        ensemble_train_pred = np.zeros_like(all_train_predictions[0])
        ensemble_test_pred = np.zeros_like(all_test_predictions[0])
        
        for i, model_name in enumerate(model_names):
            ensemble_train_pred += weights[model_name] * all_train_predictions[i]
            ensemble_test_pred += weights[model_name] * all_test_predictions[i]
        
        # 添加组合模型预测结果
        self.predictions['Ensemble'] = {
            'train': ensemble_train_pred,
            'test': ensemble_test_pred
        }
        
        print("预测生成完成")
        print(f"已生成 {len(self.predictions)} 个模型的预测结果，包括{weight_method}权重组合模型")
        return self
    
    def get_ensemble_weights(self):
        """获取各个模型在组合中的权重"""
        if 'Ensemble' not in self.predictions:
            print("请先运行make_predictions生成预测结果")
            return None
        
        if not hasattr(self, 'ensemble_weights'):
            print("未找到权重信息")
            return None
        
        # 打印权重信息
        print("\n模型组合权重:")
        for model_name, weight in self.ensemble_weights.items():
            print(f"{model_name}: {weight:.2%}")
        
        return self.ensemble_weights
    
    def save_models(self):
        """保存模型"""
        print("正在保存模型...")
        
        save_path = self.config['model_save_path']
        for model_name, model in self.models.items():
            if model_name == 'LightGBM':
                model.save_model(f"{save_path}/lgb_model.txt")
            else:
                joblib.dump(model, f"{save_path}/{model_name.lower()}_model.pkl")
        
        print("模型保存完成")
        return self
    
    def generate_etime_close_data(self, bgn_date, end_date, index_code='510050', frequency='15'):
        """生成时间-价格数据"""
        # 这里简化处理，直接使用原始数据
        mask = ((self.raw_data.index >= pd.to_datetime(bgn_date)) & 
                (self.raw_data.index <= pd.to_datetime(end_date)))
        
        result = self.raw_data[mask][['close']].reset_index()
        result.columns = ['etime', 'close']
        result['tdate'] = result['etime'].dt.date
        
        return result
    
    def calculate_portfolio_performance(self, predictions, time_index, data_type='test'):
        """计算组合表现"""
        position_size = self.config['position_size']
        clip_num = self.config['clip_num']
        
        # 获取对应时间段的价格数据
        if data_type == 'train':
            begin_date = str(time_index[0])
            end_date = str(time_index[-1])
        else:
            begin_date = str(time_index[0])
            end_date = str(time_index[-1])
        
        ret_frame = self.generate_etime_close_data(begin_date, end_date)
        
        # 映射仓位
        ret_frame['position'] = [(i / 0.0005) * position_size for i in predictions]
        ret_frame['position'] = ret_frame['position'].clip(-clip_num, clip_num)
        
        # 计算持仓净值
        ret_frame.loc[0, '持仓净值'] = 1
        
        for i in range(0, len(ret_frame), 1):
            if i == 0 or ret_frame.loc[i-1, 'position'] == 0:
                ret_frame.loc[i, '持仓净值'] = 1
            else:
                close_2 = ret_frame.loc[i, 'close']
                close_1 = ret_frame.loc[i-1, 'close']
                position = abs(ret_frame.loc[i-1, 'position'])
                
                if ret_frame.loc[i-1, 'position'] > 0:  # 多仓
                    ret_frame.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
                elif ret_frame.loc[i-1, 'position'] < 0:  # 空仓
                    ret_frame.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)
        
        # 计算累计净值
        ret_frame.loc[0, '持仓净值（累计）'] = 1
        for i in range(1, len(ret_frame)):
            ret_frame.loc[i, '持仓净值（累计）'] = (ret_frame.loc[i-1, '持仓净值（累计）'] * 
                                                ret_frame.loc[i, '持仓净值'])
        
        return ret_frame
    
    def calculate_performance_metrics(self, ret_frame):
        """计算绩效指标"""
        fixed_return = self.config['fixed_return']
        
        # 基本收益指标
        start_index = ret_frame.index[0]
        end_index = ret_frame.index[-1]
        
        net_value_start = ret_frame.loc[start_index, '持仓净值（累计）']
        net_value_end = ret_frame.loc[end_index, '持仓净值（累计）']
        total_return = net_value_end / net_value_start - 1
        
        # 年化收益率
        date_list = ret_frame['tdate'].unique()
        run_days = len(date_list)
        annual_return = math.pow(1 + total_return, 252 / run_days) - 1
        
        # 计算日度收益率用于波动率和夏普比率
        daily_nav = ret_frame.groupby('tdate')['持仓净值（累计）'].last()
        daily_returns = daily_nav.pct_change().dropna()
        
        # 年化波动率
        annual_volatility = math.sqrt(252) * daily_returns.std()
        
        # 夏普比率
        sharpe_ratio = (annual_return - fixed_return) / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative_nav = daily_nav.values
        running_max = np.maximum.accumulate(cumulative_nav)
        drawdown = (running_max - cumulative_nav) / running_max
        max_drawdown = np.max(drawdown)
        
        # 卡尔玛比率
        calmar_ratio = (annual_return - fixed_return) / max_drawdown if max_drawdown > 0 else 0
        
        # 胜率
        win_rate = (daily_returns > 0).mean()
        
        return {
            '总收益': total_return,
            '年化收益': annual_return,
            '年化波动率': annual_volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤率': max_drawdown,
            '卡尔玛比率': calmar_ratio,
            '胜率': win_rate,
            '交易次数': len(ret_frame)
        }
    
    def backtest(self, model_name='LightGBM'):
        """回测指定模型"""
        print(f"正在回测 {model_name} 模型...")
        
        if model_name not in self.predictions:
            raise ValueError(f"模型 {model_name} 的预测结果不存在")
        
        # 获取时间索引
        train_time = self.feed_data['timestamp'][:self.train_set_end_index].values
        test_time = self.feed_data['timestamp'][self.train_set_end_index:].values
        
        # 计算训练集表现
        train_ret_frame = self.calculate_portfolio_performance(
            self.predictions[model_name]['train'], train_time, 'train')
        
        # 计算测试集表现
        test_ret_frame = self.calculate_portfolio_performance(
            self.predictions[model_name]['test'], test_time, 'test')
        
        # 计算绩效指标
        train_metrics = self.calculate_performance_metrics(train_ret_frame)
        test_metrics = self.calculate_performance_metrics(test_ret_frame)
        
        self.backtest_results[model_name] = {
            'train_frame': train_ret_frame,
            'test_frame': test_ret_frame,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        print(f"{model_name} 回测完成")
        print(f"样本内夏普比率: {train_metrics['夏普比率']:.4f}")
        print(f"样本外夏普比率: {test_metrics['夏普比率']:.4f}")
        
        return self
    
    def backtest_all_models(self):
        """回测所有模型"""
        print("正在回测所有模型...")
        
        for model_name in self.predictions.keys():
            self.backtest(model_name)
        
        return self
    
    def plot_results(self, model_name='Ensemble', close_fig=False):
        """绘制回测结果
        
        Args:
            model_name (str): 模型名称，默认为 'Ensemble'
            close_fig (bool): 是否关闭图形，默认为 False
        """
        if model_name not in self.backtest_results:
            print(f"模型 {model_name} 的回测结果不存在")
            return
        
        test_frame = self.backtest_results[model_name]['test_frame']
        test_metrics = self.backtest_results[model_name]['test_metrics']
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 设置颜色和线型
        if model_name == 'Ensemble':
            color = 'r'  # 组合模型用红色
            linewidth = 2.5
            linestyle = '-'
            title_prefix = '等权重组合模型'
        else:
            color = 'b'  # 其他模型用蓝色
            linewidth = 2
            linestyle = '-'
            title_prefix = model_name
        
        # 绘制净值曲线
        plt.plot(test_frame['etime'], test_frame['持仓净值（累计）'], 
                color=color, linestyle=linestyle, linewidth=linewidth,
                label=f'{title_prefix} 测试集净值曲线')
        
        # 设置图表标题和标签
        title = f'{title_prefix} 回测结果'
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('累计净值', fontsize=12)
        
        # 添加性能指标到图例
        if model_name == 'Ensemble':
            # 获取组合模型的权重信息
            weights = self.get_ensemble_weights()
            weight_info = "\n".join([f"{name}: {w:.1%}" for name, w in weights.items()])
            
            legend_text = (f'{title_prefix} 测试集净值曲线\n'
                          f'年化收益: {test_metrics["年化收益"]:.2%}\n'
                          f'夏普比率: {test_metrics["夏普比率"]:.3f}\n'
                          f'最大回撤: {test_metrics["最大回撤率"]:.2%}\n'
                          f'模型权重:\n{weight_info}')
        else:
            legend_text = (f'{title_prefix} 测试集净值曲线\n'
                          f'年化收益: {test_metrics["年化收益"]:.2%}\n'
                          f'夏普比率: {test_metrics["夏普比率"]:.3f}\n'
                          f'最大回撤: {test_metrics["最大回撤率"]:.2%}')
        
        plt.plot([], [], ' ', label=legend_text)
        plt.legend(loc='upper left', fontsize=10)
        
        # 设置网格和样式
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        
        # 设置y轴格式
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        
        # 调整布局
        plt.tight_layout()
        
        # 如果是Mac系统且字体有问题，显示提示
        if platform.system() == 'Darwin':
            try:
                # 测试中文显示
                test_fig = plt.figure(figsize=(1, 1))
                test_fig.text(0.5, 0.5, '测试中文', fontsize=1)
                plt.close(test_fig)
            except:
                print("提示：如果图表中的中文显示为方框，请安装中文字体")
        
        if close_fig:
            plt.show()
        else:
            plt.show(block=False)  # 非阻塞模式显示图形
        
        return self
    
    def get_performance_summary(self):
        """获取所有模型的绩效汇总"""
        if not self.backtest_results:
            print("请先运行回测")
            return None
        
        summary_data = []
        
        for model_name, results in self.backtest_results.items():
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']
            
            summary_data.append({
                '模型': model_name,
                '样本内年化收益': train_metrics['年化收益'],
                '样本内夏普比率': train_metrics['夏普比率'],
                '样本内最大回撤': train_metrics['最大回撤率'],
                '样本外年化收益': test_metrics['年化收益'],
                '样本外夏普比率': test_metrics['夏普比率'],
                '样本外最大回撤': test_metrics['最大回撤率']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        print("模型绩效汇总:")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def run_full_pipeline(self):
        """运行完整的策略流程"""
        print("开始运行完整的量化策略流程...")
        
        # 执行完整流程
        (self.load_data()
         .generate_factors()
         .normalize_factors()
         .select_factors()
        #  .prepare_training_data()
        #  .train_models()
        #  .make_predictions(weight_method=self.config['weight_method'])
        #  .save_models()
        #  .backtest_all_models()
         )
        
        # 显示模型组合权重
        # self.get_ensemble_weights()
        
        # 显示所有模型的绩效汇总
        # summary_df = self.get_performance_summary()
        
        print("策略流程执行完成！")
        return self

# 使用示例
if __name__ == "__main__":
    # 获取当前运行目录
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 获取工程根目录
    data_dir = os.path.join(current_dir, 'data')  # 指向工程根目录下的 data 目录
    models_dir = os.path.join(data_dir, 'models')  # 指向 data/models 目录

    # 确保 models 目录存在
    # os.makedirs(models_dir, exist_ok=True)

    # 配置参数
    config = {
        'return_period': 1,
        'corr_threshold': 0.3,
        'sharpe_threshold': 0.2,
        'train_end_date': '2016-12-30',
        'position_size': 1.0,
        'clip_num': 2.0,
        'fixed_return': 0.0,
        'fees_rate': 0.004,
        'weight_method': 'sharpe', # 权重方法，可选 'equal' 或 'sharpe' 
        'model_save_path': models_dir  # 使用 data/models 目录作为模型保存路径
    }
    
    # 创建策略实例
    strategy = QuantTradingStrategy(
        data_path=os.path.join(data_dir, 'courses', '510050.SH_15.pkl'),  # 使用 data/courses 目录下的数据文件
        config=config
    )
    
    # 运行完整流程，使用基于夏普比率的权重
    strategy.run_full_pipeline()

    # 绘制组合模型的结果
    strategy.plot_results('Ensemble')
    
    # 保持图形窗口显示，直到用户手动关闭
    plt.show(block=True)