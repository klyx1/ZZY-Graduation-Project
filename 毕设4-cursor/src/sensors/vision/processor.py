import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import yaml

class VisionProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化视觉处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.vision_config = self.config['vision']
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def process_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """处理深度图像
        
        Args:
            depth_image: 原始深度图像
            
        Returns:
            处理后的深度图像
        """
        if depth_image is None:
            return None
            
        # 应用深度比例因子
        depth_scale = self.vision_config.get('depth_scale', 0.001)
        depth_scaled = depth_image * depth_scale
        
        # 限制深度范围
        min_depth = self.vision_config.get('min_depth', 0.1)
        max_depth = self.vision_config.get('max_depth', 2.0)
        depth_scaled = np.clip(depth_scaled, min_depth, max_depth)
        
        return depth_scaled
        
    def process_color(self, color_image: np.ndarray) -> np.ndarray:
        """处理彩色图像
        
        Args:
            color_image: 原始彩色图像
            
        Returns:
            处理后的彩色图像
        """
        if color_image is None:
            return None
            
        # 图像增强
        enhanced = cv2.convertScaleAbs(color_image, alpha=1.2, beta=10)
        
        # 降噪
        denoised = cv2.fastNlMeansDenoisingColored(enhanced)
        
        return denoised
        
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """检测图像边缘
        
        Args:
            image: 输入图像
            
        Returns:
            边缘图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        return edges
        
    def segment_object(self, color_image: np.ndarray, depth_image: np.ndarray) -> Dict:
        """分割目标物体
        
        Args:
            color_image: 彩色图像
            depth_image: 深度图像
            
        Returns:
            分割结果字典
        """
        if color_image is None or depth_image is None:
            return None
            
        # 创建掩码
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        
        # 使用GrabCut算法分割
        rect = self._get_roi(depth_image)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(color_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # 提取前景
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        segmented = color_image * mask2[:, :, np.newaxis]
        
        return {
            'mask': mask2,
            'segmented': segmented,
            'bbox': rect
        }
        
    def _get_roi(self, depth_image: np.ndarray) -> Tuple[int, int, int, int]:
        """获取感兴趣区域
        
        Args:
            depth_image: 深度图像
            
        Returns:
            矩形区域(x, y, width, height)
        """
        # 对深度图进行二值化
        _, binary = cv2.threshold(depth_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (0, 0, depth_image.shape[1], depth_image.shape[0])
            
        # 获取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        return (x, y, w, h)
        
    def extract_features(self, image: np.ndarray) -> Dict:
        """提取图像特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # SIFT特征
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # ORB特征
        orb = cv2.ORB_create()
        kp_orb, des_orb = orb.detectAndCompute(gray, None)
        
        return {
            'sift': {
                'keypoints': keypoints,
                'descriptors': descriptors
            },
            'orb': {
                'keypoints': kp_orb,
                'descriptors': des_orb
            }
        }