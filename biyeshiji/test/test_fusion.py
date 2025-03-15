import unittest
import numpy as np
from src.processing.fusion_algorithms import MultiModalFusion


class TestFusionAlgorithms(unittest.TestCase):
    """ 多模态数据融合算法测试套件 """

    def setUp(self):
        self.fusion = MultiModalFusion(fusion_type='dynamic_weight')
        self.tactile_feat = np.array([1.2, 0.8, 2.1])
        self.visual_feat = np.array([0.5, 1.5, 0.3])

    def test_dynamic_weight_fusion(self):
        """ 动态权重融合算法验证 """
        fused = self.fusion.fuse(self.tactile_feat, self.visual_feat)

        # 验证输出维度
        self.assertEqual(fused.shape, (3,), "融合特征维度错误")

        # 验证权重计算正确性
        energy_t = np.linalg.norm(self.tactile_feat)
        energy_v = np.linalg.norm(self.visual_feat)
        expected_weight = energy_t / (energy_t + energy_v)
        expected_result = expected_weight * self.tactile_feat + (1 - expected_weight) * self.visual_feat

        np.testing.assert_allclose(fused, expected_result, rtol=1e-6)

    def test_invalid_fusion_type(self):
        """ 无效融合类型异常处理测试 """
        with self.assertRaises(ValueError):
            invalid_fusion = MultiModalFusion(fusion_type='invalid')
            invalid_fusion.fuse(self.tactile_feat, self.visual_feat)


if __name__ == '__main__':
    unittest.main()