# 多模态融合算法
import numpy as np


class MultiModalFusion:
    """ 多模态数据融合算法 """

    def __init__(self, fusion_type: str = 'dynamic_weight'):
        """
        :param fusion_type: 融合策略类型（dynamic_weight, attention）
        """
        self.fusion_type = fusion_type

    def fuse(self, tactile_feat: np.ndarray, visual_feat: np.ndarray) -> np.ndarray:
        """
        融合触觉与视觉特征
        :return: 融合后的特征向量
        """
        if self.fusion_type == 'dynamic_weight':
            return self._dynamic_weight_fusion(tactile_feat, visual_feat)
        elif self.fusion_type == 'attention':
            return self._attention_fusion(tactile_feat, visual_feat)
        else:
            raise ValueError("未知的融合类型")

    def _dynamic_weight_fusion(self, t_feat, v_feat):
        """ 动态权重融合 """
        energy_t = np.linalg.norm(t_feat)
        energy_v = np.linalg.norm(v_feat)
        w_t = energy_t / (energy_t + energy_v + 1e-6)
        return w_t * t_feat + (1 - w_t) * v_feat

    def _attention_fusion(self, t_feat, v_feat):
        """ 基于注意力的融合 """
        # 计算注意力权重
        combined = np.concatenate([t_feat, v_feat])
        attention = np.exp(combined) / np.sum(np.exp(combined))
        t_weight, v_weight = attention[:len(t_feat)], attention[len(t_feat):]
        return t_weight * t_feat + v_weight * v_feat