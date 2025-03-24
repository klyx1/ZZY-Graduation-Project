import unittest
import numpy as np
import cv2
import os
import sys
import time
from typing import Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.simulation.data_generator import SensorDataGenerator
from tests.simulation.fusion_sim import SensorFusionSim

class TestSensorFusionSim(unittest.TestCase):
    """传感器融合模块仿真测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建数据生成器
        self.data_generator = SensorDataGenerator(
            image_size=(640, 480),
            noise_level=0.1
        )
        
        # 创建融合模块
        self.fusion_module = SensorFusionSim()
        
    def tearDown(self):
        """测试后的清理工作"""
        self.fusion_module.stop()
        
    def test_data_generation(self):
        """测试数据生成"""
        # 生成视觉数据
        vision_data = self.data_generator.generate_vision_data()
        
        # 验证视觉数据
        self.assertIn('depth', vision_data)
        self.assertIn('color', vision_data)
        self.assertIn('pointcloud', vision_data)
        self.assertEqual(vision_data['depth'].shape, (640, 480))
        self.assertEqual(vision_data['color'].shape, (640, 480, 3))
        self.assertEqual(vision_data['pointcloud'].shape, (640, 480, 3))
        
        # 生成触觉数据
        tactile_data = self.data_generator.generate_tactile_data()
        
        # 验证触觉数据
        self.assertIn('image', tactile_data)
        self.assertIn('force', tactile_data)
        self.assertIn('shear', tactile_data)
        self.assertEqual(tactile_data['image'].shape, (640, 480, 3))
        self.assertIsInstance(tactile_data['force'], float)
        self.assertEqual(len(tactile_data['shear']), 2)
        
    def test_fusion_process(self):
        """测试融合过程"""
        # 启动融合模块
        self.fusion_module.start()
        
        # 生成测试数据
        vision_data = self.data_generator.generate_vision_data()
        tactile_data = self.data_generator.generate_tactile_data()
        
        # 获取融合数据
        fused_data = self.fusion_module.get_fused_data(vision_data, tactile_data)
        
        # 验证融合数据
        self.assertIn('vision', fused_data)
        self.assertIn('tactile', fused_data)
        self.assertIn('fused', fused_data)
        
        # 验证融合结果
        fused = fused_data['fused']
        self.assertIn('image', fused)
        self.assertIn('force', fused)
        self.assertIn('features', fused)
        
        # 验证图像融合
        self.assertEqual(fused['image'].shape, (640, 480, 3))
        
        # 验证力信息融合
        force_info = fused['force']
        self.assertIn('total_force', force_info)
        self.assertIn('direction', force_info)
        self.assertIn('components', force_info)
        
        # 验证特征融合
        features = fused['features']
        self.assertIn('vision', features)
        self.assertIn('tactile', features)
        self.assertIn('combined', features)
        
    def test_fusion_quality(self):
        """测试融合质量"""
        # 启动融合模块
        self.fusion_module.start()
        
        # 生成多帧测试数据
        num_frames = 10
        fused_data_list = []
        
        for _ in range(num_frames):
            vision_data = self.data_generator.generate_vision_data()
            tactile_data = self.data_generator.generate_tactile_data()
            fused_data = self.fusion_module.get_fused_data(vision_data, tactile_data)
            fused_data_list.append(fused_data)
            
        # 验证数据一致性
        for i in range(1, num_frames):
            # 验证图像尺寸一致性
            self.assertEqual(
                fused_data_list[i]['fused']['image'].shape,
                fused_data_list[0]['fused']['image'].shape
            )
            
            # 验证力信息格式一致性
            self.assertEqual(
                set(fused_data_list[i]['fused']['force'].keys()),
                set(fused_data_list[0]['fused']['force'].keys())
            )
            
            # 验证特征格式一致性
            self.assertEqual(
                set(fused_data_list[i]['fused']['features'].keys()),
                set(fused_data_list[0]['fused']['features'].keys())
            )
            
    def test_fusion_performance(self):
        """测试融合性能"""
        # 启动融合模块
        self.fusion_module.start()
        
        # 生成测试数据
        vision_data = self.data_generator.generate_vision_data()
        tactile_data = self.data_generator.generate_tactile_data()
        
        # 测量处理时间
        start_time = time.time()
        fused_data = self.fusion_module.get_fused_data(vision_data, tactile_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 验证处理时间是否满足仿真环境的要求(5fps)
        self.assertLess(processing_time, 1.0/5.0)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 启动融合模块
        self.fusion_module.start()
        
        # 测试无效数据
        invalid_vision_data = {'depth': None, 'color': None, 'pointcloud': None}
        invalid_tactile_data = {'image': None, 'force': None, 'shear': None}
        
        # 验证处理无效数据
        fused_data = self.fusion_module.get_fused_data(
            invalid_vision_data, invalid_tactile_data
        )
        
        # 验证融合结果
        self.assertIn('fused', fused_data)
        fused = fused_data['fused']
        
        # 验证图像融合结果为空
        self.assertNotIn('image', fused)
        
        # 验证力信息融合结果为空
        self.assertNotIn('force', fused)
        
        # 验证特征融合结果为空
        self.assertNotIn('features', fused)
        
    def test_noise_robustness(self):
        """测试噪声鲁棒性"""
        # 创建不同噪声水平的数据生成器
        noise_levels = [0.1, 0.2, 0.3]
        fused_data_list = []
        
        for noise_level in noise_levels:
            # 创建数据生成器
            data_generator = SensorDataGenerator(
                image_size=(640, 480),
                noise_level=noise_level
            )
            
            # 生成测试数据
            vision_data = data_generator.generate_vision_data()
            tactile_data = data_generator.generate_tactile_data()
            
            # 获取融合数据
            fused_data = self.fusion_module.get_fused_data(vision_data, tactile_data)
            fused_data_list.append(fused_data)
            
        # 验证不同噪声水平下的融合结果
        for i in range(1, len(noise_levels)):
            # 验证图像尺寸一致性
            self.assertEqual(
                fused_data_list[i]['fused']['image'].shape,
                fused_data_list[0]['fused']['image'].shape
            )
            
            # 验证力信息格式一致性
            self.assertEqual(
                set(fused_data_list[i]['fused']['force'].keys()),
                set(fused_data_list[0]['fused']['force'].keys())
            )
            
            # 验证特征格式一致性
            self.assertEqual(
                set(fused_data_list[i]['fused']['features'].keys()),
                set(fused_data_list[0]['fused']['features'].keys())
            )
            
            # 验证特征置信度随噪声增加而降低
            confidence_prev = fused_data_list[i-1]['fused']['features']['combined']['confidence']
            confidence_curr = fused_data_list[i]['fused']['features']['combined']['confidence']
            self.assertLessEqual(confidence_curr, confidence_prev)

if __name__ == '__main__':
    unittest.main() 