import unittest
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.tactile.sensor import GelSightMini
from src.sensors.tactile.processor import TactileProcessor

class TestTactileSensor(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.config_path = "config/config.yaml"
        self.sensor = GelSightMini(self.config_path)
        self.processor = TactileProcessor(self.config_path)
        
    def tearDown(self):
        """测试后清理"""
        self.sensor.disconnect()
        
    def test_sensor_initialization(self):
        """测试传感器初始化"""
        self.assertIsNotNone(self.sensor.config)
        self.assertIsNotNone(self.sensor.serial)
        
    def test_sensor_connection(self):
        """测试传感器连接"""
        self.assertTrue(self.sensor.connect())
        self.assertTrue(self.sensor.is_connected)
        
    def test_get_image(self):
        """测试获取触觉图像"""
        self.sensor.connect()
        image = self.sensor.get_image()
        
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[2], 3)
        
    def test_get_force(self):
        """测试获取力信息"""
        self.sensor.connect()
        force = self.sensor.get_force()
        
        self.assertIsNotNone(force)
        self.assertIsInstance(force, float)
        
    def test_get_shear(self):
        """测试获取剪切力"""
        self.sensor.connect()
        shear = self.sensor.get_shear()
        
        self.assertIsNotNone(shear)
        self.assertEqual(len(shear), 2)
        
    def test_tactile_processor(self):
        """测试触觉处理器"""
        self.sensor.connect()
        image = self.sensor.get_image()
        
        # 测试图像处理
        processed_image = self.processor.process_image(image)
        self.assertIsNotNone(processed_image)
        
        # 测试特征提取
        features = self.processor.extract_features(processed_image)
        self.assertIsNotNone(features)
        self.assertIn('pressure_distribution', features)
        self.assertIn('contact_area', features)
        self.assertIn('deformation', features)
        
    def test_slip_detection(self):
        """测试滑动检测"""
        self.sensor.connect()
        image1 = self.sensor.get_image()
        image2 = self.sensor.get_image()
        
        slip = self.processor.detect_slip(image1, image2)
        self.assertIsNotNone(slip)
        self.assertIn('magnitude', slip)
        self.assertIn('direction', slip)
        
    def test_sensor_calibration(self):
        """测试传感器标定"""
        self.sensor.connect()
        calibration_data = self.sensor.calibrate()
        
        self.assertIsNotNone(calibration_data)
        self.assertIn('force_scale', calibration_data)
        self.assertIn('shear_scale', calibration_data)
        
    def test_sensor_configuration(self):
        """测试传感器配置"""
        self.sensor.connect()
        config = self.sensor.get_configuration()
        
        self.assertIsNotNone(config)
        self.assertIn('exposure', config)
        self.assertIn('gain', config)
        self.assertIn('fps', config)

if __name__ == '__main__':
    unittest.main() 