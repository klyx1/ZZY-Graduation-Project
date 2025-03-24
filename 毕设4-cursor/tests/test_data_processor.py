import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.processor = DataProcessor()
        
    def tearDown(self):
        """测试后清理"""
        self.processor.cleanup()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.processor.config)
        self.assertIsNotNone(self.processor.processing_config)
        self.assertEqual(self.processor.cache_size, 10)
        
    def test_data_validation(self):
        """测试数据验证"""
        # 有效数据
        valid_data = {
            'point_cloud': np.random.rand(100, 3),
            'tactile': np.random.rand(640, 480)
        }
        self.assertTrue(self.processor._validate_data(valid_data))
        
        # 无效数据
        invalid_data = {
            'point_cloud': np.random.rand(100, 2),  # 维度错误
            'tactile': np.random.rand(100)  # 维度错误
        }
        self.assertFalse(self.processor._validate_data(invalid_data))
        
    def test_point_cloud_processing(self):
        """测试点云处理"""
        # 生成测试数据
        point_cloud = np.random.rand(1000, 3)
        
        # 处理点云
        processed_cloud = self.processor._process_point_cloud(point_cloud)
        
        # 验证结果
        self.assertIsInstance(processed_cloud, np.ndarray)
        self.assertEqual(processed_cloud.shape[1], 3)
        
    def test_tactile_processing(self):
        """测试触觉数据处理"""
        # 生成测试数据
        tactile_data = np.random.rand(640, 480)
        
        # 处理触觉数据
        processed_tactile = self.processor._process_tactile_data(tactile_data)
        
        # 验证结果
        self.assertIsInstance(processed_tactile, np.ndarray)
        self.assertEqual(len(processed_tactile.shape), 3)  # 特征维度
        
    def test_data_augmentation(self):
        """测试数据增强"""
        # 生成测试数据
        data = {
            'point_cloud': np.random.rand(100, 3),
            'tactile': np.random.rand(640, 480)
        }
        
        # 数据增强
        augmented_data = self.processor._augment_data(data)
        
        # 验证结果
        self.assertIn('point_cloud', augmented_data)
        self.assertIn('tactile', augmented_data)
        self.assertEqual(augmented_data['point_cloud'].shape, (100, 3))
        self.assertEqual(augmented_data['tactile'].shape, (640, 480))
        
    def test_queue_operations(self):
        """测试队列操作"""
        # 启动处理线程
        self.processor.start_processing()
        
        # 添加数据
        data = {
            'point_cloud': np.random.rand(100, 3),
            'tactile': np.random.rand(640, 480)
        }
        self.processor.data_queue.put(data)
        
        # 等待处理
        time.sleep(0.1)
        
        # 获取处理后的数据
        processed_data = self.processor.processed_queue.get()
        
        # 验证结果
        self.assertIsInstance(processed_data, dict)
        self.assertIn('point_cloud', processed_data)
        self.assertIn('tactile', processed_data)
        
        # 停止处理线程
        self.processor.stop_processing()
        
    def test_performance_monitoring(self):
        """测试性能监控"""
        # 启动处理线程
        self.processor.start_processing()
        
        # 添加一些数据
        for _ in range(10):
            data = {
                'point_cloud': np.random.rand(100, 3),
                'tactile': np.random.rand(640, 480)
            }
            self.processor.data_queue.put(data)
            
        # 等待处理
        time.sleep(0.5)
        
        # 获取状态
        status = self.processor.get_queue_status()
        
        # 验证状态
        self.assertIsInstance(status, dict)
        self.assertIn('data_queue_size', status)
        self.assertIn('processed_queue_size', status)
        self.assertIn('avg_processing_time', status)
        
        # 停止处理线程
        self.processor.stop_processing()

if __name__ == '__main__':
    unittest.main() 