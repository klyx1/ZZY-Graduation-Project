import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from scipy import ndimage
import yaml
import os
import threading
import queue
import time
import logging
import open3d as o3d
import psutil
import gc
import zlib
import pickle
from numba import jit

class DataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self._setup_logging()
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.processing_config = self.config.get('data_processing', {})
        
        # 验证配置
        self._validate_config()
        
        # 数据队列
        self.data_queue = queue.Queue(maxsize=100)
        self.processed_queue = queue.Queue(maxsize=100)
        
        # 处理线程
        self.processing_thread = None
        self.monitor_thread = None
        self.memory_monitor_thread = None
        self.is_running = False
        
        # 缓存
        self.cache_size = self.processing_config.get('cache_size', 10)
        self.point_cloud_cache = []
        self.tactile_cache = []
        
        # 性能监控
        self.processing_times = []
        self.max_processing_time = 0.1  # 最大处理时间(秒)
        
        # 内存管理
        self.memory_threshold = self.processing_config.get('memory_threshold', 0.8)  # 内存使用阈值
        self.max_cache_size = self.processing_config.get('max_cache_size', 1000)  # 最大缓存大小
        self.compression_level = self.processing_config.get('compression_level', 6)  # 压缩级别
        self.last_cleanup_time = time.time()
        self.cleanup_interval = self.processing_config.get('cleanup_interval', 300)  # 清理间隔(秒)
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/data_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataProcessor')
        
    def _validate_config(self):
        """验证配置"""
        required_fields = ['cache_size', 'enable_augmentation', 'memory_threshold']
        for field in required_fields:
            if field not in self.processing_config:
                self.logger.warning(f"配置缺少字段: {field}, 使用默认值")
                if field == 'cache_size':
                    self.processing_config[field] = 10
                elif field == 'enable_augmentation':
                    self.processing_config[field] = True
                elif field == 'memory_threshold':
                    self.processing_config[field] = 0.8
                    
    def start_processing(self):
        """启动数据处理线程"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self.processing_thread.start()
        self.monitor_thread.start()
        self.memory_monitor_thread.start()
        self.logger.info("数据处理线程已启动")
        
    def stop_processing(self):
        """停止数据处理线程"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.memory_monitor_thread:
            self.memory_monitor_thread.join()
        self.logger.info("数据处理线程已停止")
        
    def _processing_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                # 从队列获取数据
                data = self.data_queue.get(timeout=0.1)
                
                # 验证数据
                if not self._validate_data(data):
                    self.logger.error("数据验证失败")
                    continue
                    
                # 处理数据
                start_time = time.time()
                processed_data = self._process_data(data)
                processing_time = time.time() - start_time
                
                # 记录处理时间
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                    
                # 放入处理后的队列
                self.processed_queue.put(processed_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"数据处理错误: {str(e)}")
                continue
                
    def _monitor_performance(self):
        """监控性能"""
        while self.is_running:
            try:
                # 检查队列大小
                data_queue_size = self.data_queue.qsize()
                processed_queue_size = self.processed_queue.qsize()
                
                if data_queue_size > 80:
                    self.logger.warning(f"数据队列接近满: {data_queue_size}")
                if processed_queue_size > 80:
                    self.logger.warning(f"处理后队列接近满: {processed_queue_size}")
                    
                # 检查处理时间
                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    if avg_time > self.max_processing_time:
                        self.logger.warning(f"平均处理时间过长: {avg_time:.3f}秒")
                        
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {str(e)}")
                time.sleep(1)
                
    def _monitor_memory(self):
        """监控内存使用"""
        while self.is_running:
            try:
                # 获取当前进程
                process = psutil.Process()
                
                # 获取内存使用情况
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # 检查内存使用是否超过阈值
                if memory_percent > self.memory_threshold:
                    self.logger.warning(f"内存使用率过高: {memory_percent:.2f}%")
                    self._cleanup_memory()
                    
                # 定期清理
                current_time = time.time()
                if current_time - self.last_cleanup_time > self.cleanup_interval:
                    self._cleanup_memory()
                    self.last_cleanup_time = current_time
                    
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"内存监控错误: {str(e)}")
                time.sleep(1)
                
    def _cleanup_memory(self):
        """清理内存"""
        try:
            # 清理缓存
            if len(self.point_cloud_cache) > self.max_cache_size:
                self.point_cloud_cache = self.point_cloud_cache[-self.max_cache_size:]
            if len(self.tactile_cache) > self.max_cache_size:
                self.tactile_cache = self.tactile_cache[-self.max_cache_size:]
                
            # 清理处理时间记录
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
                
            # 强制垃圾回收
            gc.collect()
            
            # 压缩数据
            self._compress_data()
            
            self.logger.info("内存已清理")
            
        except Exception as e:
            self.logger.error(f"内存清理错误: {str(e)}")
            
    def _compress_data(self):
        """压缩数据"""
        try:
            # 压缩点云缓存
            for i in range(len(self.point_cloud_cache)):
                if isinstance(self.point_cloud_cache[i], np.ndarray):
                    compressed = zlib.compress(
                        pickle.dumps(self.point_cloud_cache[i]),
                        level=self.compression_level
                    )
                    self.point_cloud_cache[i] = compressed
                    
            # 压缩触觉数据缓存
            for i in range(len(self.tactile_cache)):
                if isinstance(self.tactile_cache[i], np.ndarray):
                    compressed = zlib.compress(
                        pickle.dumps(self.tactile_cache[i]),
                        level=self.compression_level
                    )
                    self.tactile_cache[i] = compressed
                    
        except Exception as e:
            self.logger.error(f"数据压缩错误: {str(e)}")
            
    def _decompress_data(self, compressed_data):
        """解压数据"""
        try:
            if isinstance(compressed_data, bytes):
                decompressed = pickle.loads(zlib.decompress(compressed_data))
                return decompressed
            return compressed_data
        except Exception as e:
            self.logger.error(f"数据解压错误: {str(e)}")
            return None
            
    def _validate_data(self, data: Dict) -> bool:
        """验证数据
        
        Args:
            data: 输入数据
            
        Returns:
            是否有效
        """
        if not isinstance(data, dict):
            return False
            
        if 'point_cloud' in data:
            if not isinstance(data['point_cloud'], np.ndarray):
                return False
            if data['point_cloud'].shape[1] != 3:
                return False
                
        if 'tactile' in data:
            if not isinstance(data['tactile'], np.ndarray):
                return False
            if len(data['tactile'].shape) != 2:
                return False
                
        return True
        
    def cleanup(self):
        """清理资源"""
        self.stop_processing()
        self.data_queue.queue.clear()
        self.processed_queue.queue.clear()
        self.point_cloud_cache.clear()
        self.tactile_cache.clear()
        self.processing_times.clear()
        gc.collect()
        self.logger.info("资源已清理")
        
    def get_queue_status(self) -> Dict:
        """获取队列状态
        
        Returns:
            状态字典
        """
        return {
            'data_queue_size': self.data_queue.qsize(),
            'processed_queue_size': self.processed_queue.qsize(),
            'point_cloud_cache_size': len(self.point_cloud_cache),
            'tactile_cache_size': len(self.tactile_cache),
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'memory_usage': psutil.Process().memory_percent()
        }
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _process_data(self, data: Dict) -> Dict:
        """处理数据
        
        Args:
            data: 输入数据字典
            
        Returns:
            处理后的数据字典
        """
        try:
            # 使用numba加速计算
            
            @jit(nopython=True)
            def fast_process_point_cloud(point_cloud):
                # 快速处理点云数据
                # 1. 移除离群点
                mean = np.mean(point_cloud, axis=0)
                std = np.std(point_cloud, axis=0)
                mask = np.all(np.abs(point_cloud - mean) <= 3 * std, axis=1)
                point_cloud = point_cloud[mask]
                
                # 2. 降采样
                if len(point_cloud) > self.max_points:
                    indices = np.random.choice(len(point_cloud), self.max_points, replace=False)
                    point_cloud = point_cloud[indices]
                
                return point_cloud
            
            @jit(nopython=True)
            def fast_process_tactile(tactile_data):
                # 快速处理触觉数据
                # 1. 归一化
                tactile_data = (tactile_data - np.min(tactile_data)) / (np.max(tactile_data) - np.min(tactile_data))
                
                # 2. 高斯滤波
                tactile_data = cv2.GaussianBlur(tactile_data, (3, 3), 0)
                
                return tactile_data
            
            # 处理数据
            processed_data = {
                'point_cloud': fast_process_point_cloud(data['point_cloud']),
                'tactile_data': fast_process_tactile(data['tactile_data'])
            }
            
            # 更新缓存
            self._update_cache(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"数据处理错误: {str(e)}")
            return None
            
    def _update_cache(self, data: Dict):
        """更新缓存
        
        Args:
            data: 处理后的数据
        """
        try:
            # 更新点云缓存
            self.point_cloud_cache.append(data['point_cloud'])
            if len(self.point_cloud_cache) > self.cache_size:
                self.point_cloud_cache.pop(0)
                
            # 更新触觉数据缓存
            self.tactile_cache.append(data['tactile_data'])
            if len(self.tactile_cache) > self.cache_size:
                self.tactile_cache.pop(0)
                
            # 检查内存使用
            self._check_memory_usage()
            
        except Exception as e:
            self.logger.error(f"缓存更新错误: {str(e)}")
            
    def _check_memory_usage(self):
        """检查内存使用情况"""
        try:
            current_time = time.time()
            if current_time - self.last_cleanup_time > self.cleanup_interval:
                memory_usage = psutil.Process().memory_percent()
                if memory_usage > self.memory_threshold:
                    self._cleanup_memory()
                self.last_cleanup_time = current_time
                
        except Exception as e:
            self.logger.error(f"内存检查错误: {str(e)}")
            
    def _cleanup_memory(self):
        """清理内存"""
        try:
            # 清理缓存
            self.point_cloud_cache = self.point_cloud_cache[-self.cache_size:]
            self.tactile_cache = self.tactile_cache[-self.cache_size:]
            
            # 强制垃圾回收
            gc.collect()
            
            self.logger.info("内存清理完成")
            
        except Exception as e:
            self.logger.error(f"内存清理错误: {str(e)}")
        
    def _process_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """处理点云数据
        
        Args:
            point_cloud: 输入点云
            
        Returns:
            处理后的点云
        """
        # 更新缓存
        self.point_cloud_cache.append(point_cloud)
        if len(self.point_cloud_cache) > self.cache_size:
            self.point_cloud_cache.pop(0)
            
        # 降噪
        point_cloud = self._denoise_point_cloud(point_cloud)
        
        # 补全
        point_cloud = self._complete_point_cloud(point_cloud)
        
        return point_cloud
        
    def _process_tactile_data(self, tactile_data: np.ndarray) -> np.ndarray:
        """处理触觉数据
        
        Args:
            tactile_data: 输入触觉数据
            
        Returns:
            处理后的触觉数据
        """
        # 更新缓存
        self.tactile_cache.append(tactile_data)
        if len(self.tactile_cache) > self.cache_size:
            self.tactile_cache.pop(0)
            
        # 图像增强
        tactile_data = self._enhance_tactile_image(tactile_data)
        
        # 特征提取
        tactile_data = self._extract_tactile_features(tactile_data)
        
        return tactile_data
        
    def _denoise_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """点云降噪
        
        Args:
            point_cloud: 输入点云
            
        Returns:
            降噪后的点云
        """
        # 统计滤波
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
        
        # 半径滤波
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=16,
            radius=0.05
        )
        
        return np.asarray(pcd.points)
        
    def _enhance_tactile_image(self, tactile_data: np.ndarray) -> np.ndarray:
        """增强触觉图像
        
        Args:
            tactile_data: 输入触觉数据
            
        Returns:
            增强后的触觉数据
        """
        # 对比度增强
        tactile_data = cv2.convertScaleAbs(
            tactile_data,
            alpha=1.2,
            beta=10
        )
        
        # 高斯滤波
        tactile_data = cv2.GaussianBlur(
            tactile_data,
            (5, 5),
            0
        )
        
        # 边缘增强
        tactile_data = cv2.Laplacian(
            tactile_data,
            cv2.CV_64F
        )
        
        return tactile_data
        
    def _extract_tactile_features(self, tactile_data: np.ndarray) -> np.ndarray:
        """提取触觉特征
        
        Args:
            tactile_data: 输入触觉数据
            
        Returns:
            触觉特征
        """
        # 计算梯度
        gradient_x = cv2.Sobel(
            tactile_data,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )
        gradient_y = cv2.Sobel(
            tactile_data,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # 计算局部统计特征
        local_mean = ndimage.uniform_filter(
            tactile_data,
            size=5
        )
        local_std = ndimage.generic_filter(
            tactile_data,
            np.std,
            size=5
        )
        
        # 组合特征
        features = np.stack([
            tactile_data,
            magnitude,
            direction,
            local_mean,
            local_std
        ])
        
        return features
        
    def _augment_data(self, data: Dict) -> Dict:
        """数据增强
        
        Args:
            data: 输入数据
            
        Returns:
            增强后的数据
        """
        augmented_data = {}
        
        # 点云增强
        if 'point_cloud' in data:
            augmented_data['point_cloud'] = self._augment_point_cloud(
                data['point_cloud']
            )
            
        # 触觉数据增强
        if 'tactile' in data:
            augmented_data['tactile'] = self._augment_tactile_data(
                data['tactile']
            )
            
        return augmented_data
        
    def _augment_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """点云增强
        
        Args:
            point_cloud: 输入点云
            
        Returns:
            增强后的点云
        """
        # 随机旋转
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        point_cloud = np.dot(point_cloud, rotation_matrix)
        
        # 随机平移
        translation = np.random.uniform(-0.1, 0.1, 3)
        point_cloud += translation
        
        # 随机噪声
        noise = np.random.normal(0, 0.01, point_cloud.shape)
        point_cloud += noise
        
        return point_cloud
        
    def _augment_tactile_data(self, tactile_data: np.ndarray) -> np.ndarray:
        """触觉数据增强
        
        Args:
            tactile_data: 输入触觉数据
            
        Returns:
            增强后的触觉数据
        """
        # 随机亮度调整
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-10, 10)
        tactile_data = cv2.convertScaleAbs(
            tactile_data,
            alpha=alpha,
            beta=beta
        )
        
        # 随机旋转
        angle = np.random.uniform(-10, 10)
        center = (tactile_data.shape[1]/2, tactile_data.shape[0]/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        tactile_data = cv2.warpAffine(
            tactile_data,
            rotation_matrix,
            (tactile_data.shape[1], tactile_data.shape[0])
        )
        
        # 随机噪声
        noise = np.random.normal(0, 0.05, tactile_data.shape)
        tactile_data += noise
        
        return tactile_data
        
    def _compute_transform(self,
                          source: np.ndarray,
                          target: np.ndarray) -> np.ndarray:
        """计算变换矩阵
        
        Args:
            source: 源点云
            target: 目标点云
            
        Returns:
            变换矩阵
        """
        # 计算质心
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        
        # 去中心化
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        
        # 计算协方差矩阵
        cov_matrix = np.dot(source_centered.T, target_centered)
        
        # SVD分解
        U, _, Vh = np.linalg.svd(cov_matrix)
        
        # 计算旋转矩阵
        rotation = np.dot(Vh.T, U.T)
        
        # 计算平移向量
        translation = target_centroid - np.dot(rotation, source_centroid)
        
        # 组合变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        return transform
        
    def _apply_transform(self,
                        point_cloud: np.ndarray,
                        transform: np.ndarray) -> np.ndarray:
        """应用变换
        
        Args:
            point_cloud: 输入点云
            transform: 变换矩阵
            
        Returns:
            变换后的点云
        """
        # 添加齐次坐标
        points_homo = np.hstack([
            point_cloud,
            np.ones((point_cloud.shape[0], 1))
        ])
        
        # 应用变换
        transformed_points = np.dot(points_homo, transform.T)
        
        # 移除齐次坐标
        return transformed_points[:, :3] 