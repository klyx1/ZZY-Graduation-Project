import unittest
import numpy as np
from src.sensors import GelSightCamera, RealSenseCamera


class SensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gelsight = GelSightCamera()
        cls.realsense = RealSenseCamera()

    def test_gelsight_init(self):
        frame = self.gelsight.get_calibrated_frame()
        self.assertEqual(frame.shape, (240, 320, 3), "触觉图像尺寸错误")

    def test_realsense_depth(self):
        rgb, depth = self.realsense.get_aligned_frames()
        self.assertFalse(np.all(depth == 0), "深度图全零错误")

    @classmethod
    def tearDownClass(cls):
        cls.gelsight.close()
        cls.realsense.stop()


if __name__ == '__main__':
    unittest.main()