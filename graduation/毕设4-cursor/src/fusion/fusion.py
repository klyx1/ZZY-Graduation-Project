import numpy as np
import cv2
import yaml
from typing import Optional, Tuple, Dict
from src.vision.camera import RealSenseCamera
from src.tactile.sensor import GelSightSensor

class SensorFusion:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化传感器融合模块
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.vision_sensor = RealSenseCamera(config_path)
        self.tactile_sensor = GelSightSensor(config_path)
        self.fusion_config = self.config['fusion']
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def start(self):
        """启动传感器和数据融合"""
        self.tactile_sensor.start()
    
    def stop(self):
        """停止传感器和数据融合"""
        self.tactile_sensor.stop()
        self.vision_sensor.release()
    
    def get_fused_data(self) -> Dict:
        """获取融合后的数据
        
        Returns:
            包含融合数据的字典
        """
        # 获取视觉数据
        depth_image, color_image = self.vision_sensor.get_aligned_frames()
        pointcloud = self.vision_sensor.get_pointcloud()
        
        # 获取触觉数据
        tactile_image = self.tactile_sensor.get_image()
        force = self.tactile_sensor.get_force()
        shear = self.tactile_sensor.get_shear()
        
        # 融合数据
        fused_data = {
            'vision': {
                'depth': depth_image,
                'color': color_image,
                'pointcloud': pointcloud
            },
            'tactile': {
                'image': tactile_image,
                'force': force,
                'shear': shear
            },
            'fused': self._fuse_data(depth_image, color_image, tactile_image, force, shear)
        }
        
        return fused_data
    
    def _fuse_data(self, 
                  depth_image: Optional[np.ndarray],
                  color_image: Optional[np.ndarray],
                  tactile_image: Optional[np.ndarray],
                  force: Optional[float],
                  shear: Optional[Tuple[float, float]]) -> Dict:
        """融合视觉和触觉数据
        
        Args:
            depth_image: 深度图
            color_image: 彩色图
            tactile_image: 触觉图像
            force: 压力值
            shear: 剪切力
            
        Returns:
            融合后的数据字典
        """
        fused_data = {}
        
        # 1. 图像融合
        if depth_image is not None and tactile_image is not None:
            fused_data['image'] = self._fuse_images(depth_image, tactile_image)
        
        # 2. 力信息融合
        if force is not None and shear is not None:
            fused_data['force'] = self._fuse_force(force, shear)
        
        # 3. 特征融合
        if color_image is not None and tactile_image is not None:
            fused_data['features'] = self._fuse_features(color_image, tactile_image)
        
        return fused_data
    
    def _fuse_images(self, depth_image: np.ndarray, tactile_image: np.ndarray) -> np.ndarray:
        """融合深度图和触觉图像
        
        Args:
            depth_image: 深度图
            tactile_image: 触觉图像
            
        Returns:
            融合后的图像
        """
        # 调整触觉图像大小以匹配深度图
        tactile_resized = cv2.resize(tactile_image, (depth_image.shape[1], depth_image.shape[0]))
        
        # 归一化深度图
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 加权融合
        weight_vision = self.fusion_config['weight_vision']
        weight_tactile = self.fusion_config['weight_tactile']
        
        fused_image = cv2.addWeighted(depth_normalized, weight_vision,
                                    tactile_resized, weight_tactile, 0)
        
        return fused_image
    
    def _fuse_force(self, force: float, shear: Tuple[float, float]) -> Dict:
        """融合压力值和剪切力
        
        Args:
            force: 压力值
            shear: 剪切力
            
        Returns:
            融合后的力信息
        """
        # 计算合力
        total_force = np.sqrt(force**2 + shear[0]**2 + shear[1]**2)
        
        # 计算力的方向
        direction = np.arctan2(shear[1], shear[0])
        
        return {
            'total_force': total_force,
            'direction': direction,
            'components': {
                'normal': force,
                'shear_x': shear[0],
                'shear_y': shear[1]
            }
        }
    
    def _fuse_features(self, color_image: np.ndarray, tactile_image: np.ndarray) -> Dict:
        """融合视觉和触觉特征
        
        Args:
            color_image: 彩色图
            tactile_image: 触觉图像
            
        Returns:
            融合后的特征字典
        """
        # 提取视觉特征
        vision_features = self._extract_vision_features(color_image)
        
        # 提取触觉特征
        tactile_features = self._extract_tactile_features(tactile_image)
        
        # 特征融合
        fused_features = {
            'vision': vision_features,
            'tactile': tactile_features,
            'combined': self._combine_features(vision_features, tactile_features)
        }
        
        return fused_features
    
    def _extract_vision_features(self, color_image: np.ndarray) -> Dict:
        """提取视觉特征
        
        Args:
            color_image: 彩色图
            
        Returns:
            视觉特征字典
        """
        # 转换为灰度图
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # 提取SIFT特征
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
    
    def _extract_tactile_features(self, tactile_image: np.ndarray) -> Dict:
        """提取触觉特征
        
        Args:
            tactile_image: 触觉图像
            
        Returns:
            触觉特征字典
        """
        # 转换为灰度图
        gray = cv2.cvtColor(tactile_image, cv2.COLOR_BGR2GRAY)
        
        # 计算压力分布
        pressure_distribution = self._compute_pressure_distribution(gray)
        
        # 计算变形场
        deformation = self._compute_deformation_field(gray)
        
        # 计算接触区域
        contact_area = self._compute_contact_area(gray)
        
        # 计算局部特征
        local_features = self._compute_local_features(gray)
        
        return {
            'pressure_distribution': pressure_distribution,
            'deformation': deformation,
            'contact_area': contact_area,
            'local_features': local_features
        }
        
    def _compute_pressure_distribution(self, image: np.ndarray) -> np.ndarray:
        """计算压力分布
        
        Args:
            image: 灰度图像
            
        Returns:
            压力分布图
        """
        # 归一化图像
        normalized = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        
        # 应用高斯滤波平滑压力分布
        smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
        
        return smoothed
        
    def _compute_deformation_field(self, image: np.ndarray) -> np.ndarray:
        """计算变形场
        
        Args:
            image: 灰度图像
            
        Returns:
            变形场
        """
        if not hasattr(self, '_prev_tactile_image'):
            self._prev_tactile_image = image
            return np.zeros((image.shape[0], image.shape[1], 2))
            
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_tactile_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        self._prev_tactile_image = image
        return flow
        
    def _compute_contact_area(self, image: np.ndarray) -> float:
        """计算接触区域
        
        Args:
            image: 灰度图像
            
        Returns:
            接触区域比例
        """
        # 二值化
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算接触区域比例
        contact_area = np.sum(binary > 0) / binary.size
        
        return contact_area
        
    def _compute_local_features(self, image: np.ndarray) -> Dict:
        """计算局部特征
        
        Args:
            image: 灰度图像
            
        Returns:
            局部特征字典
        """
        # 计算梯度
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # 计算局部统计特征
        local_mean = cv2.blur(image, (5, 5))
        local_std = cv2.blur(image**2, (5, 5)) - local_mean**2
        
        return {
            'gradient_magnitude': gradient_magnitude,
            'gradient_direction': gradient_direction,
            'local_mean': local_mean,
            'local_std': local_std
        }
        
    def _combine_features(self, vision_features: Dict, tactile_features: Dict) -> Dict:
        """组合视觉和触觉特征
        
        Args:
            vision_features: 视觉特征
            tactile_features: 触觉特征
            
        Returns:
            组合后的特征字典
        """
        # 提取关键特征
        vision_keypoints = vision_features['keypoints']
        vision_descriptors = vision_features['descriptors']
        
        # 提取触觉特征
        pressure_dist = tactile_features['pressure_distribution']
        deformation = tactile_features['deformation']
        contact_area = tactile_features['contact_area']
        local_features = tactile_features['local_features']
        
        # 计算特征权重
        vision_weight = self.fusion_config['weight_vision']
        tactile_weight = self.fusion_config['weight_tactile']
        
        # 组合特征
        combined_features = {
            'keypoints': vision_keypoints,
            'descriptors': vision_descriptors,
            'pressure': pressure_dist,
            'deformation': deformation,
            'contact_area': contact_area,
            'local_features': local_features,
            'weights': {
                'vision': vision_weight,
                'tactile': tactile_weight
            }
        }
        
        # 计算特征置信度
        confidence = self._compute_feature_confidence(
            vision_features, tactile_features
        )
        
        return {
            'combined_features': combined_features,
            'confidence': confidence
        }
        
    def _compute_feature_confidence(self,
                                  vision_features: Dict,
                                  tactile_features: Dict) -> float:
        """计算特征置信度
        
        Args:
            vision_features: 视觉特征
            tactile_features: 触觉特征
            
        Returns:
            置信度值
        """
        # 计算视觉特征置信度
        vision_confidence = len(vision_features['keypoints']) / 1000.0  # 归一化关键点数量
        
        # 计算触觉特征置信度
        pressure_confidence = np.mean(tactile_features['pressure_distribution'])
        contact_confidence = tactile_features['contact_area']
        
        # 综合置信度
        confidence = (
            self.fusion_config['weight_vision'] * vision_confidence +
            self.fusion_config['weight_tactile'] * (pressure_confidence + contact_confidence) / 2
        )
        
        return min(confidence, 1.0)  # 限制在[0,1]范围内 