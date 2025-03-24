import unittest
import numpy as np
import cv2
import os
import sys
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.camera import RealSenseCamera
from src.tactile.sensor import GelSightSensor
from src.fusion.fusion import SensorFusion
from src.modeling.object_model import FlexibleObjectModel
from src.control.gripper import GripperController

class TestGraspingSystem(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.config_path = "config/config.yaml"
        self.vision_sensor = RealSenseCamera(self.config_path)
        self.tactile_sensor = GelSightSensor(self.config_path)
        self.fusion_module = SensorFusion(self.config_path)
        self.object_model = FlexibleObjectModel(self.config_path)
        self.gripper = GripperController(self.config_path)
    
    def tearDown(self):
        """测试后的清理工作"""
        self.vision_sensor.release()
        self.tactile_sensor.stop()
        self.gripper.disconnect()
    
    def test_vision_sensor(self):
        """测试视觉传感器"""
        # 获取深度图和彩色图
        depth_image, color_image = self.vision_sensor.get_frames()
        self.assertIsNotNone(depth_image)
        self.assertIsNotNone(color_image)
        
        # 检查图像尺寸
        self.assertEqual(depth_image.shape[:2], color_image.shape[:2])
        
        # 获取点云数据
        pointcloud = self.vision_sensor.get_pointcloud()
        self.assertIsNotNone(pointcloud)
        self.assertEqual(pointcloud.shape[1], 3)
    
    def test_tactile_sensor(self):
        """测试触觉传感器"""
        # 连接传感器
        self.assertTrue(self.tactile_sensor.connect())
        
        # 启动数据采集
        self.tactile_sensor.start()
        
        # 获取触觉图像
        tactile_image = self.tactile_sensor.get_image()
        self.assertIsNotNone(tactile_image)
        
        # 获取力信息
        force = self.tactile_sensor.get_force()
        shear = self.tactile_sensor.get_shear()
        self.assertIsNotNone(force)
        self.assertIsNotNone(shear)
    
    def test_fusion_module(self):
        """测试数据融合模块"""
        # 启动融合模块
        self.fusion_module.start()
        
        # 获取融合数据
        fused_data = self.fusion_module.get_fused_data()
        self.assertIsNotNone(fused_data)
        
        # 检查数据完整性
        self.assertIn('vision', fused_data)
        self.assertIn('tactile', fused_data)
        self.assertIn('fused', fused_data)
    
    def test_object_model(self):
        """测试物体建模模块"""
        # 创建测试数据
        pointcloud = np.random.rand(100, 3)
        tactile_data = {
            'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'force': 50.0,
            'shear': (10.0, 10.0)
        }
        force_data = {
            'total_force': 50.0,
            'direction': 0.0,
            'components': {
                'normal': 50.0,
                'shear_x': 10.0,
                'shear_y': 10.0
            }
        }
        
        # 更新模型
        self.assertTrue(self.object_model.update_model(
            pointcloud, tactile_data, force_data
        ))
        
        # 获取模型状态
        model_state = self.object_model.get_model_state()
        self.assertIsNotNone(model_state)
        
        # 获取抓取点
        grasp_points = self.object_model.get_grasp_points()
        self.assertIsNotNone(grasp_points)
    
    def test_gripper_control(self):
        """测试夹爪控制"""
        # 连接夹爪
        self.assertTrue(self.gripper.connect())
        
        # 设置位置
        self.assertTrue(self.gripper.set_position(80.0))
        
        # 设置力
        self.assertTrue(self.gripper.set_force(50.0))
        
        # 获取状态
        status = self.gripper.get_status()
        self.assertIsNotNone(status)
        
        # 执行抓取
        self.assertTrue(self.gripper.grasp(80.0, 50.0))
        
        # 释放
        self.assertTrue(self.gripper.release())
    
    def test_system_integration(self):
        """测试系统集成"""
        # 启动融合模块
        self.fusion_module.start()
        
        # 获取融合数据
        fused_data = self.fusion_module.get_fused_data()
        
        # 更新物体模型
        self.object_model.update_model(
            fused_data['vision']['pointcloud'],
            fused_data['tactile'],
            fused_data['fused'].get('force')
        )
        
        # 获取抓取点
        grasp_points = self.object_model.get_grasp_points()
        
        if grasp_points:
            # 选择最佳抓取点
            best_point, best_normal = grasp_points[0]
            
            # 执行抓取
            self.gripper.connect()
            self.assertTrue(self.gripper.grasp(80.0, 50.0))
            self.assertTrue(self.gripper.release())

if __name__ == '__main__':
    unittest.main() 