"""
PAInpaint任务特定的注意力工具函数
包含mask感知、质量评估、自适应融合等功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class MaskAwareAttentionHelper:
    """Mask感知注意力辅助类"""
    
    @staticmethod
    def extract_mask_info(latents: torch.Tensor, mask_channel_idx: int = 4) -> Optional[torch.Tensor]:
        """从latents中提取mask信息"""
        if latents.shape[1] > mask_channel_idx:
            return latents[:, mask_channel_idx:mask_channel_idx+1]
        return None
    
    @staticmethod
    def compute_mask_statistics(mask: torch.Tensor) -> Dict[str, float]:
        """计算mask的统计信息"""
        if mask is None:
            return {"ratio": 0.5, "complexity": 0.5}
        
        mask_ratio = mask.mean().item()
        
        # 计算mask复杂度 (边缘密度)
        mask_binary = (mask > 0.5).float()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(mask_binary, sobel_x, padding=1)
        edges_y = F.conv2d(mask_binary, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        complexity = edge_magnitude.mean().item()
        
        return {
            "ratio": mask_ratio,
            "complexity": min(complexity, 1.0)  # 归一化到[0,1]
        }


class AdaptiveFusionScheduler:
    """自适应融合调度器"""
    
    def __init__(self, total_timesteps: int = 50, schedule_type: str = "cosine"):
        self.total_timesteps = total_timesteps
        self.schedule_type = schedule_type
    
    def get_fusion_weights(self, timestep: int, mask_stats: Dict[str, float], layer_type: str = "mid") -> torch.Tensor:
        """根据时间步、mask信息和层类型计算融合权重"""
        progress = timestep / self.total_timesteps
        
        # 基础权重根据层类型确定
        if layer_type == "mid":
            base_weights = torch.tensor([0.3, 0.3, 0.4])  # 更注重交叉注意力
        elif layer_type == "early_up":
            base_weights = torch.tensor([0.4, 0.4, 0.2])  # 平衡
        else:  # late_up
            base_weights = torch.tensor([0.5, 0.4, 0.1])  # 更注重自注意力
        
        # 时间步调整
        if self.schedule_type == "cosine":
            # 早期更依赖参考，后期更保持目标
            time_factor = 0.5 + 0.5 * np.cos(progress * np.pi)
            base_weights[2] *= (1 + time_factor)  # 交叉注意力
            base_weights[1] *= (2 - time_factor)  # 目标自注意力
        
        # Mask感知调整
        mask_ratio = mask_stats.get("ratio", 0.5)
        mask_complexity = mask_stats.get("complexity", 0.5)
        
        if mask_ratio > 0.7:  # 大面积mask
            base_weights[2] *= 1.5  # 更依赖参考
            base_weights[1] *= 0.8
        elif mask_ratio < 0.3:  # 小面积mask
            base_weights[1] *= 1.3  # 更保持目标
            base_weights[2] *= 0.7
        
        # 复杂mask需要更多参考信息
        if mask_complexity > 0.6:
            base_weights[2] *= (1 + mask_complexity * 0.3)
        
        # 归一化
        return base_weights / base_weights.sum()


class FeatureQualityAssessor:
    """特征质量评估器"""
    
    @staticmethod
    def compute_feature_diversity(features: torch.Tensor) -> float:
        """计算特征多样性"""
        # 使用特征向量间的余弦相似度衡量多样性
        features_flat = features.view(features.shape[0], -1)
        features_norm = F.normalize(features_flat, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # 多样性 = 1 - 平均相似度
        mask = torch.eye(similarity_matrix.shape[0], device=features.device).bool()
        off_diagonal = similarity_matrix[~mask]
        diversity = 1.0 - off_diagonal.mean().item()
        
        return max(0.0, min(1.0, diversity))
    
    @staticmethod
    def compute_feature_stability(features: torch.Tensor) -> float:
        """计算特征稳定性"""
        # 使用特征的方差作为稳定性指标
        variance = torch.var(features, dim=-1).mean().item()
        # 将方差映射到[0,1]范围，较小的方差表示更高的稳定性
        stability = 1.0 / (1.0 + variance)
        return stability
    
    @staticmethod
    def compute_semantic_coherence(features: torch.Tensor) -> float:
        """计算语义一致性"""
        # 使用特征的空间平滑性作为语义一致性指标
        if len(features.shape) == 4:  # [B, C, H, W]
            # 计算空间梯度
            grad_x = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
            grad_y = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
            
            smoothness = 1.0 / (1.0 + grad_x.mean().item() + grad_y.mean().item())
            return smoothness
        else:
            # 对于非空间特征，使用相邻token的一致性
            if features.shape[1] > 1:
                diff = torch.abs(features[:, 1:] - features[:, :-1])
                consistency = 1.0 / (1.0 + diff.mean().item())
                return consistency
            return 1.0


class ProgressiveAttentionController:
    """渐进式注意力控制器"""
    
    def __init__(self, total_steps: int = 50):
        self.total_steps = total_steps
        self.current_step = 0
    
    def update_step(self, step: int):
        """更新当前步骤"""
        self.current_step = step
    
    def get_attention_temperature(self) -> float:
        """获取注意力温度参数"""
        progress = self.current_step / self.total_steps
        
        # 早期高温度(探索)，后期低温度(利用)
        if progress < 0.3:
            return 2.0  # 高温度，更均匀的注意力分布
        elif progress < 0.7:
            return 1.5 - progress  # 逐渐降低
        else:
            return 0.5  # 低温度，更集中的注意力
    
    def get_fusion_strategy(self) -> str:
        """获取当前阶段的融合策略"""
        progress = self.current_step / self.total_steps
        
        if progress < 0.2:
            return "exploration"  # 探索阶段，更多依赖参考
        elif progress < 0.8:
            return "balanced"     # 平衡阶段
        else:
            return "refinement"   # 精化阶段，更多保持目标


def create_layer_specific_config(layer_name: str, total_layers: int, layer_idx: int) -> Dict[str, Any]:
    """为特定层创建配置"""
    config = {
        "layer_name": layer_name,
        "layer_idx": layer_idx,
        "total_layers": total_layers,
        "relative_depth": layer_idx / total_layers
    }
    
    # 根据层的位置确定特性
    if "mid_block" in layer_name:
        config.update({
            "layer_type": "mid",
            "semantic_weight": 1.0,
            "detail_weight": 0.5,
            "reference_dependency": 0.8
        })
    elif "up_blocks" in layer_name:
        # 解析up_block的索引
        if "up_blocks.0" in layer_name or "up_blocks.1" in layer_name:
            config.update({
                "layer_type": "early_up",
                "semantic_weight": 0.8,
                "detail_weight": 0.7,
                "reference_dependency": 0.6
            })
        else:
            config.update({
                "layer_type": "late_up",
                "semantic_weight": 0.4,
                "detail_weight": 1.0,
                "reference_dependency": 0.3
            })
    else:
        config.update({
            "layer_type": "down",
            "semantic_weight": 0.6,
            "detail_weight": 0.8,
            "reference_dependency": 0.5
        })
    
    return config
