import unittest
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.vision.camera import RealSenseCamera
from src.sensors.vision.processor import VisionProcessor

class TestVisionSensor(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.config_path = "config/config.yaml"
        self.camera = RealSenseCamera(self.config_path)
        self.processor = VisionProcessor(self.config_path)
        
    def tearDown(self):
        """测试后清理"""
        self.camera.disconnect()
        
    def test_camera_initialization(self):
        """测试相机初始化"""
        self.assertIsNotNone(self.camera.config)
        self.assertIsNotNone(self.camera.pipeline)
        
    def test_camera_connection(self):
        """测试相机连接"""
        self.assertTrue(self.camera.connect())
        self.assertTrue(self.camera.is_connected)
        
    def test_get_frames(self):
        """测试获取图像帧"""
        self.camera.connect()
        depth_image, color_image = self.camera.get_frames()
        
        self.assertIsNotNone(depth_image)
        self.assertIsNotNone(color_image)
        self.assertEqual(depth_image.shape[:2], color_image.shape[:2])
        
    def test_get_pointcloud(self):
        """测试获取点云数据"""
        self.camera.connect()
        depth_image, _ = self.camera.get_frames()
        pointcloud = self.camera.get_pointcloud(depth_image)
        
        self.assertIsNotNone(pointcloud)
        self.assertEqual(pointcloud.shape[1], 3)
        
    def test_vision_processor(self):
        """测试视觉处理器"""
        self.camera.connect()
        _, color_image = self.camera.get_frames()
        
        # 测试图像处理
        processed_image = self.processor.process_image(color_image)
        self.assertIsNotNone(processed_image)
        
        # 测试特征提取
        features = self.processor.extract_features(processed_image)
        self.assertIsNotNone(features)
        self.assertIn('edges', features)
        self.assertIn('segments', features)
        
    def test_camera_intrinsics(self):
        """测试相机内参"""
        self.camera.connect()
        intrinsics = self.camera.get_intrinsics()
        
        self.assertIsNotNone(intrinsics)
        self.assertIn('width', intrinsics)
        self.assertIn('height', intrinsics)
        self.assertIn('fx', intrinsics)
        self.assertIn('fy', intrinsics)
        self.assertIn('cx', intrinsics)
        self.assertIn('cy', intrinsics)
        
    def test_camera_extrinsics(self):
        """测试相机外参"""
        self.camera.connect()
        extrinsics = self.camera.get_extrinsics()
        
        self.assertIsNotNone(extrinsics)
        self.assertEqual(extrinsics.shape, (4, 4))

if __name__ == '__main__':
    unittest.main() 