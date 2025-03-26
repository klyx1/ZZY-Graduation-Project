import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import yaml
import logging

class TactileProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化触觉处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self.logger = logging.getLogger('TactileProcessor')
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.tactile_config = self.config['tactile']
        
        # 图像参数
        self.width = self.tactile_config.get('image_width', 640)
        self.height = self.tactile_config.get('image_height', 480)
        
        # 处理参数
        self.blur_kernel = (3, 3)
        self.threshold = 127
        self.max_value = 255
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """处理触觉图像
        
        Args:
            image: 原始触觉图像
            
        Returns:
            处理后的图像
        """
        if image is None:
            return None
            
        try:
            # 图像增强
            enhanced = self._enhance_image(image)
            
            # 降噪
            denoised = self._denoise_image(enhanced)
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {str(e)}")
            return image
            
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """增强图像
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        # 对比度增强
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
        
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """降噪
        
        Args:
            image: 输入图像
            
        Returns:
            降噪后的图像
        """
        # 高斯滤波
        denoised = cv2.GaussianBlur(image, self.blur_kernel, 0)
        
        return denoised
        
    def extract_features(self, image: np.ndarray) -> Dict:
        """提取触觉特征
        
        Args:
            image: 触觉图像
            
        Returns:
            特征字典
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 计算压力分布
            pressure_dist = self._compute_pressure_distribution(gray)
            
            # 计算接触区域
            contact_area = self._compute_contact_area(gray)
            
            # 计算变形
            deformation = self._compute_deformation(gray)
            
            return {
                'pressure_distribution': pressure_dist,
                'contact_area': contact_area,
                'deformation': deformation
            }
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            return {}
            
    def _compute_pressure_distribution(self, image: np.ndarray) -> Dict:
        """计算压力分布
        
        Args:
            image: 灰度图像
            
        Returns:
            压力分布字典
        """
        # 二值化
        _, binary = cv2.threshold(image, self.threshold, self.max_value, cv2.THRESH_BINARY)
        
        # 计算压力分布
        pressure = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        
        return {
            'binary': binary,
            'pressure': pressure,
            'mean_pressure': np.mean(pressure),
            'max_pressure': np.max(pressure)
        }
        
    def _compute_contact_area(self, image: np.ndarray) -> Dict:
        """计算接触区域
        
        Args:
            image: 灰度图像
            
        Returns:
            接触区域字典
        """
        # 二值化
        _, binary = cv2.threshold(image, self.threshold, self.max_value, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算面积
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = sum(areas)
        
        return {
            'contours': contours,
            'areas': areas,
            'total_area': total_area,
            'num_contacts': len(contours)
        }
        
    def _compute_deformation(self, image: np.ndarray) -> Dict:
        """计算变形
        
        Args:
            image: 灰度图像
            
        Returns:
            变形字典
        """
        # 计算梯度
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'mean_magnitude': np.mean(magnitude),
            'max_magnitude': np.max(magnitude)
        }
        
    def detect_slip(self, prev_image: np.ndarray, curr_image: np.ndarray) -> Dict:
        """检测滑动
        
        Args:
            prev_image: 前一帧图像
            curr_image: 当前帧图像
            
        Returns:
            滑动检测结果
        """
        try:
            # 转换为灰度图
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # 计算滑动特征
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            direction = np.arctan2(flow[..., 1], flow[..., 0])
            
            return {
                'flow': flow,
                'magnitude': magnitude,
                'direction': direction,
                'mean_magnitude': np.mean(magnitude),
                'max_magnitude': np.max(magnitude)
            }
            
        except Exception as e:
            self.logger.error(f"滑动检测失败: {str(e)}")
            return {} 