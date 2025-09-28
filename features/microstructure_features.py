# """
# 市场微观结构特征提取模块
# """
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Optional, Any
# from core.base import BaseFeatureExtractor
# from data.trades_processor import TradesContext


# class MicrostructureFeatureExtractor(BaseFeatureExtractor):
#     """市场微观结构特征提取器"""
    
#     def __init__(self, config: Optional[Dict[str, Any]] = None):
#         super().__init__(config)
#         self.feature_config = self._get_default_config()
#         if config:
#             self.feature_config.update(config)
    
#     def _get_default_config(self) -> Dict[str, bool]:
#         """获取默认特征配置"""
#         return {
#             'base': True,                    # 基础汇总/VWAP/强度/买量占比
#             'order_flow': True,              # GOF/签名不平衡
#             'price_impact': True,            # Kyle/Amihud/Hasbrouck/半衰期/占比
#             'volatility_noise': True,        # RV/BPV/Jump/微动量/均值回复/高低幅比
#             'arrival_stats': False,          # 到达间隔统计
#             'run_markov': False,             # run-length/Markov/翻转率
#             'rolling_ofi': False,            # 滚动OFI
#             'hawkes': False,                 # 聚簇（Hawkes近似）
#             'path_shape': True,              # 协动相关/VWAP偏离
#             'tail': False,                   # 大单尾部比例
#             'tail_directional': True,        # 大单买卖方向性（分位阈值法）
#             'bar_large_abs': False,          # 纯 bar 级绝对阈值大单特征
#         }
    
#     def get_feature_names(self) -> List[str]:
#         """获取特征名称列表"""
#         names = []
#         if self.feature_config['base']:
#             names.extend(['int_trade_vwap', 'int_trade_volume_sum', 'int_trade_dollar_sum',
#                          'int_trade_signed_volume', 'int_trade_buy_ratio', 'int_trade_intensity', 'int_trade_rv'])
        
#         if self.feature_config['order_flow']:
#             names.extend(['ofi_signed_qty_sum', 'ofi_signed_quote_sum', 'gof_by_count', 'gof_by_volume'])
        
#         if self.feature_config['price_impact']:
#             names.extend(['kyle_lambda', 'amihud_lambda', 'hasbrouck_lambda', 
#                          'impact_half_life', 'impact_perm_share', 'impact_transient_share'])
        
#         if self.feature_config['volatility_noise']:
#             names.extend(['rv', 'bpv', 'jump_rv_bpv', 'micro_dp_short', 'micro_dp_zscore',
#                          'mr_rho1', 'mr_strength', 'hl_amplitude_ratio'])
        
#         if self.feature_config['path_shape']:
#             names.extend(['signed_vwap_deviation', 'vwap_deviation',
#                          'corr_cumsum_signed_qty_logp', 'corr_cumsum_signed_dollar_logp'])
        
#         if self.feature_config['tail_directional']:
#             names.extend(['large_q90_buy_dollar_sum', 'large_q90_sell_dollar_sum', 'large_q90_lti',
#                          'large_q95_buy_dollar_sum', 'large_q95_sell_dollar_sum', 'large_q95_lti'])
        
#         return names
    
#     def extract(self, data: pd.DataFrame) -> Dict[str, float]:
#         """从数据中提取特征（这里需要TradesContext）"""
#         raise NotImplementedError("请使用 extract_from_context 方法")
    
#     def extract_from_context(self, ctx: TradesContext, start_ts: pd.Timestamp, 
#                            end_ts: pd.Timestamp, bars: Optional[pd.DataFrame] = None,
#                            bar_window_start_idx: Optional[int] = None,
#                            bar_window_end_idx: Optional[int] = None) -> Dict[str, float]:
#         """从交易上下文中提取特征"""
#         if bars is not None and bar_window_start_idx is not None and bar_window_end_idx is not None:
#             s, e = bars.loc[bar_window_start_idx, 'start_trade_idx'], bars.loc[bar_window_end_idx, 'end_trade_idx'] + 1
#         else:
#             s, e = ctx.locate(start_ts, end_ts)
        
#         if e - s <= 0:
#             return {}
        
#         features = {}
        
#         # 基础特征
#         if self.feature_config['base']:
#             features.update(self._extract_base_features(ctx, s, e, start_ts, end_ts))
        
#         # 订单流特征
#         if self.feature_config['order_flow']:
#             features.update(self._extract_order_flow_features(ctx, s, e))
        
#         # 价格冲击特征
#         if self.feature_config['price_impact']:
#             features.update(self._extract_price_impact_features(ctx, s, e))
        
#         # 波动性特征
#         if self.feature_config['volatility_noise']:
#             features.update(self._extract_volatility_features(ctx, s, e))
        
#         # 路径形状特征
#         if self.feature_config['path_shape']:
#             features.update(self._extract_path_shape_features(ctx, s, e))
        
#         # 方向性大单特征
#         if self.feature_config['tail_directional']:
#             features.update(self._extract_directional_tail_features(ctx, s, e))
        
#         return features
    
#     def _extract_base_features(self, ctx: TradesContext, s: int, e: int,
#                               start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
#         """提取基础特征"""
#         sum_qty = self._sum_range(ctx.csum_qty, s, e)
#         sum_quote = self._sum_range(ctx.csum_quote, s, e)
#         sum_signed_qty = self._sum_range(ctx.csum_signed_qty, s, e)
#         sum_pxqty = self._sum_range(ctx.csum_pxqty, s, e)

#         vwap = sum_pxqty / sum_qty if sum_qty > 0 else np.nan
#         duration = max(1.0, (end_ts - start_ts).total_seconds())
#         intensity = (e - s) / duration

#         # 买量占比
#         buy_mask = ctx.sign[s:e] > 0
#         buy_qty = float(ctx.qty[s:e][buy_mask].sum()) if (e - s) > 0 else 0.0
#         trade_buy_ratio = (buy_qty / sum_qty) if sum_qty > 0 else np.nan

#         # RV
#         rv = self._sum_range(ctx.csum_ret2, s, e)

#         return {
#             'int_trade_vwap': vwap,
#             'int_trade_volume_sum': sum_qty,
#             'int_trade_dollar_sum': sum_quote,
#             'int_trade_signed_volume': sum_signed_qty,
#             'int_trade_buy_ratio': trade_buy_ratio,
#             'int_trade_intensity': intensity,
#             'int_trade_rv': rv,
#         }
    
#     def _extract_order_flow_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
#         """提取订单流特征"""
#         sum_qty = self._sum_range(ctx.csum_qty, s, e)
#         sum_signed_qty = self._sum_range(ctx.csum_signed_qty, s, e)
#         sum_signed_quote = self._sum_range(ctx.csum_signed_quote, s, e)

#         # Garman OF
#         n = float(e - s)
#         gof_by_count = float(np.mean(np.sign(ctx.sign[s:e]))) if n > 0 else np.nan
#         gof_by_volume = (sum_signed_qty / sum_qty) if sum_qty > 0 else np.nan

#         return {
#             'ofi_signed_qty_sum': sum_signed_qty,
#             'ofi_signed_quote_sum': sum_signed_quote,
#             'gof_by_count': gof_by_count,
#             'gof_by_volume': gof_by_volume,
#         }
    
#     def _extract_price_impact_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
#         """提取价格冲击特征"""
#         if e - s < 3:
#             return {
#                 'kyle_lambda': np.nan, 'amihud_lambda': np.nan, 'hasbrouck_lambda': np.nan,
#                 'impact_half_life': np.nan, 'impact_perm_share': np.nan, 'impact_transient_share': np.nan
#             }
        
#         r = np.diff(ctx.logp[s:e])
#         sdollar = ctx.sign[s:e] * ctx.quote[s:e]
#         x = sdollar[1:]
        
#         # Kyle Lambda
#         varx = float(np.var(x))
#         kyle = float(np.cov(x, r, ddof=1)[0, 1] / varx) if varx > 0 else np.nan
        
#         # Amihud Lambda
#         amihud = float((np.abs(r) / (ctx.quote[s+1:e])).mean()) if np.all(ctx.quote[s+1:e] > 0) else np.nan
        
#         # Hasbrouck Lambda
#         xh = np.sign(r) * np.sqrt(ctx.quote[s+1:e])
#         varxh = float(np.var(xh))
#         hasb = float(np.cov(xh, r, ddof=1)[0, 1] / varxh) if varxh > 0 and len(r) > 1 else np.nan

#         # 半衰期
#         r0 = r[:-1] - np.mean(r[:-1])
#         r1 = r[1:] - np.mean(r[1:])
#         denom = np.sqrt(np.sum(r0**2) * np.sum(r1**2))
#         if denom > 0:
#             rho = float(np.sum(r0 * r1) / denom)
#             t_half = float(np.log(2.0) / (-np.log(rho))) if (0 < rho < 1) else np.nan
#         else:
#             t_half = np.nan

#         # 冲击占比
#         dp = np.diff(ctx.price[s:e])
#         denom2 = float(np.sum(np.abs(dp)))
#         if denom2 > 0:
#             perm = float(np.abs(ctx.price[e-1] - ctx.price[s]) / denom2)
#             perm = float(np.clip(perm, 0.0, 1.0))
#             trans = float(1.0 - perm)
#         else:
#             perm = np.nan
#             trans = np.nan

#         return {
#             'kyle_lambda': kyle,
#             'amihud_lambda': amihud,
#             'hasbrouck_lambda': hasb,
#             'impact_half_life': t_half,
#             'impact_perm_share': perm,
#             'impact_transient_share': trans,
#         }
    
#     def _extract_volatility_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
#         """提取波动性特征"""
#         # RV/BPV/Jump
#         rv = self._sum_range(ctx.csum_ret2, s, e)
#         bpv = self._sum_range(ctx.csum_bpv, s, e)
#         jump = max(rv - bpv, 0.0) if (np.isfinite(rv) and np.isfinite(bpv)) else np.nan

#         # 微动量
#         W = min(20, e - s)
#         if W >= 2:
#             lp = ctx.logp[max(s, e - W):e]
#             dp_short = float(lp[-1] - lp[0])
#             mu = float(np.mean(lp))
#             sd = float(np.std(lp))
#             z = (float(lp[-1]) - mu) / sd if sd > 0 else np.nan
#         else:
#             dp_short = np.nan
#             z = np.nan

#         # 均值回复强度
#         r_slice = np.diff(ctx.logp[s:e])
#         if r_slice.size >= 2:
#             x0 = r_slice[:-1] - np.mean(r_slice[:-1])
#             x1 = r_slice[1:] - np.mean(r_slice[1:])
#             denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
#             mr_rho1 = float(np.sum(x0 * x1) / denom) if denom > 0 else np.nan
#             mr_strength = -mr_rho1 if pd.notna(mr_rho1) else np.nan
#         else:
#             mr_rho1 = np.nan
#             mr_strength = np.nan

#         # 高低幅度占比
#         if (e - s) > 0:
#             hi = float(np.max(ctx.price[s:e]))
#             lo = float(np.min(ctx.price[s:e]))
#             mid = (hi + lo) / 2.0
#             hl_amplitude_ratio = float((hi - lo) / mid) if mid != 0 else np.nan
#         else:
#             hl_amplitude_ratio = np.nan

#         return {
#             'rv': rv,
#             'bpv': bpv,
#             'jump_rv_bpv': jump,
#             'micro_dp_short': dp_short,
#             'micro_dp_zscore': z,
#             'mr_rho1': mr_rho1,
#             'mr_strength': mr_strength,
#             'hl_amplitude_ratio': hl_amplitude_ratio,
#         }
    
#     def _extract_path_shape_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
#         """提取路径形状特征"""
#         sum_qty = self._sum_range(ctx.csum_qty, s, e)
#         sum_signed_qty = self._sum_range(ctx.csum_signed_qty, s, e)
#         sum_pxqty = self._sum_range(ctx.csum_pxqty, s, e)

#         vwap = sum_pxqty / sum_qty if sum_qty > 0 else np.nan
#         p_last = ctx.price[e - 1]

#         # VWAP偏离
#         dev = (p_last - vwap) / vwap if vwap != 0 and np.isfinite(vwap) else np.nan
#         signed_dev = dev * (1.0 if sum_signed_qty > 0 else (-1.0 if sum_signed_qty < 0 else 0.0)) if pd.notna(dev) else np.nan

#         # 累计流量与价格相关性
#         if e - s <= 2:
#             corr_qty = np.nan
#             corr_dollar = np.nan
#         else:
#             logp_rel = ctx.logp[s:e] - ctx.logp[s]
#             cs_signed_qty = (ctx.csum_signed_qty[s:e] - (ctx.csum_signed_qty[s] if s < ctx.csum_signed_qty.size else 0.0))
#             cs_signed_dollar = (ctx.csum_signed_quote[s:e] - (ctx.csum_signed_quote[s] if s < ctx.csum_signed_quote.size else 0.0))
            
#             corr_qty = self._compute_correlation(cs_signed_qty, logp_rel)
#             corr_dollar = self._compute_correlation(cs_signed_dollar, logp_rel)

#         return {
#             'signed_vwap_deviation': signed_dev,
#             'vwap_deviation': dev,
#             'corr_cumsum_signed_qty_logp': corr_qty,
#             'corr_cumsum_signed_dollar_logp': corr_dollar,
#         }
    
#     def _extract_directional_tail_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
#         """提取方向性大单特征"""
#         if e - s <= 0:
#             return {}
        
#         dv = ctx.quote[s:e]
#         if dv.size == 0:
#             return {}
        
#         sign = ctx.sign[s:e]
#         eps = 1e-12
#         total_dollar = float(dv.sum()) if np.isfinite(dv.sum()) else 0.0
        
#         q_list = [0.9, 0.95]
#         features = {}
        
#         for q in q_list:
#             thr = float(np.quantile(dv, q)) if np.isfinite(dv).all() else np.nan
#             if not np.isfinite(thr) or thr <= 0:
#                 continue
            
#             mask = dv >= thr
#             if not mask.any():
#                 continue
            
#             buy_dollar = float(dv[mask][sign[mask] > 0].sum())
#             sell_dollar = float(dv[mask][sign[mask] < 0].sum())
#             lti = (buy_dollar - sell_dollar) / (buy_dollar + sell_dollar + eps)
            
#             tag = f"q{int(round(q * 100))}"
#             features.update({
#                 f'large_{tag}_buy_dollar_sum': buy_dollar,
#                 f'large_{tag}_sell_dollar_sum': sell_dollar,
#                 f'large_{tag}_lti': lti,
#             })
        
#         return features
    
#     def _sum_range(self, prefix: np.ndarray, s: int, e: int) -> float:
#         """计算前缀和的区间和"""
#         if e <= s:
#             return 0.0
#         return float(prefix[e - 1] - (prefix[s - 1] if s > 0 else 0.0))
    
#     def _compute_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
#         """计算相关系数"""
#         if a.size != b.size or a.size < 3:
#             return np.nan
#         sa = np.std(a)
#         sb = np.std(b)
#         if sa == 0 or sb == 0:
#             return np.nan
#         c = np.corrcoef(a, b)[0, 1]
#         return float(c) if np.isfinite(c) else np.nan
