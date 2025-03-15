import unittest
import numpy as np
from src.processing import TactileProcessor


class TestTactileProcessing(unittest.TestCase):
    """ 触觉数据处理算法测试 """

    def setUp(self):
        self.processor = TactileProcessor(E=0.3, nu=0.48)
        self.test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    def test_gradient_computation(self):
        """ 梯度场计算验证 """
        gx, gy = self.processor.compute_gradients(self.test_image)
        self.assertEqual(gx.shape, (240, 320), "水平梯度尺寸错误")
        self.assertFalse(np.all(gx == 0), "梯度场全零错误")

    def test_poisson_reconstruction(self):
        """ 泊松重建验证 """
        gx, gy = self.processor.compute_gradients(self.test_image)
        depth = self.processor.poisson_reconstruct(gx, gy)
        self.assertTrue(depth.min() < depth.max(), "深度图数值范围异常")