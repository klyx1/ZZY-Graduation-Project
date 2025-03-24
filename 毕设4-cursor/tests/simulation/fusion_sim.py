import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List

class SensorFusionSim:
    """传感器融合模块的仿真版本"""
    
    def __init__(self):
        """初始化传感器融合模块"""
        # 硬编码配置
        self.fusion_config = {
            'weight_vision': 0.7,  # 增加视觉权重
            'weight_tactile': 0.3,  # 减少触觉权重
            'image_fusion': {
                'gaussian_kernel_size': 3,
                'gaussian_sigma': 1.5
            },
            'feature_extraction': {
                'sift': {
                    'nfeatures': 1000,
                    'nOctaveLayers': 3,
                    'contrastThreshold': 0.04,
                    'edgeThreshold': 10,
                    'sigma': 1.6
                },
                'tactile': {
                    'pressure': {
                        'kernel_size': 5,
                        'sigma': 1.5
                    },
                    'deformation': {
                        'kernel_size': 3,
                        'sigma': 1.0
                    }
                }
            },
            'feature_fusion': {
                'matching': {
                    'ratio_threshold': 0.8,
                    'max_distance': 50
                },
                'weights': {
                    'vision': 0.6,
                    'tactile': 0.4
                },
                'confidence': {
                    'vision_weight': 0.6,
                    'tactile_weight': 0.4,
                    'min_confidence': 0.3
                }
            },
            'performance': {
                'max_processing_time': 0.033,  # 30fps
                'batch_size': 1,
                'use_multithreading': False,
                'num_threads': 4
            }
        }
    
    def start(self):
        """启动融合模块"""
        pass
    
    def stop(self):
        """停止融合模块"""
        pass
    
    def get_fused_data(self, vision_data: Dict, tactile_data: Dict) -> Dict:
        """获取融合数据
        
        Args:
            vision_data: 视觉数据
            tactile_data: 触觉数据
            
        Returns:
            融合后的数据字典
        """
        # 生成模拟的触觉数据
        tactile_data = self._generate_tactile_data()
        
        # 从触觉数据中提取所需信息
        tactile_image = tactile_data['image']
        force = [tactile_data['total_force'], 0.0, 0.0]  # 添加y和z方向的力
        shear = tactile_data['shear_force']
        
        # 从视觉数据中提取所需信息
        depth_image = vision_data.get('depth_map')
        color_image = vision_data.get('color_image')
        
        # 融合数据
        fused_data = self._fuse_data(
            depth_image=depth_image,
            color_image=color_image,
            tactile_image=tactile_image,
            force=force,
            shear=shear
        )
        
        return {
            'vision': vision_data,
            'tactile': tactile_data,
            'fused': fused_data
        }
    
    def _fuse_data(self, 
                  depth_image: Optional[np.ndarray],
                  color_image: Optional[np.ndarray],
                  tactile_image: Optional[np.ndarray],
                  force: Optional[List[float]],
                  shear: Optional[List[float]]) -> Dict:
        """融合视觉和触觉数据
        
        Args:
            depth_image: 深度图
            color_image: 彩色图
            tactile_image: 触觉图像
            force: 压力值列表 [fx, fy, fz]
            shear: 剪切力列表 [sx, sy]
            
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
        # 确保深度图为float32类型
        depth_image = depth_image.astype(np.float32)
        
        # 打印调试信息
        print(f"深度图范围: [{np.min(depth_image)}, {np.max(depth_image)}]")
        print(f"触觉图范围: [{np.min(tactile_image)}, {np.max(tactile_image)}]")
        
        # 深度图归一化到[0, 1]范围
        depth_min = np.min(depth_image)
        depth_max = np.max(depth_image)
        if depth_max > depth_min:
            depth_normalized = (depth_image - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_image)
            
        # 应用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # 增加对比度限制
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        depth_enhanced = clahe.apply(depth_normalized)
        
        # 如果触觉图像是3通道的，转换为单通道
        if len(tactile_image.shape) == 3:
            tactile_image = np.mean(tactile_image, axis=2)
            
        # 触觉图像归一化
        tactile_min = np.min(tactile_image)
        tactile_max = np.max(tactile_image)
        if tactile_max > tactile_min:
            tactile_normalized = (tactile_image - tactile_min) / (tactile_max - tactile_min)
        else:
            tactile_normalized = np.zeros_like(tactile_image)
            
        # 应用CLAHE增强触觉图像对比度
        tactile_normalized = (tactile_normalized * 255).astype(np.uint8)
        tactile_enhanced = clahe.apply(tactile_normalized)
        
        # 打印归一化后的范围
        print(f"归一化后深度图范围: [{np.min(depth_enhanced)}, {np.max(depth_enhanced)}]")
        print(f"归一化后触觉图范围: [{np.min(tactile_enhanced)}, {np.max(tactile_enhanced)}]")
        
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_enhanced, cv2.COLORMAP_RAINBOW)
        tactile_colored = cv2.applyColorMap(tactile_enhanced, cv2.COLORMAP_JET)
        
        # 转换回灰度图并归一化
        depth_gray = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2GRAY)
        tactile_gray = cv2.cvtColor(tactile_colored, cv2.COLOR_BGR2GRAY)
        
        depth_gray = depth_gray.astype(np.float32) / 255.0
        tactile_gray = tactile_gray.astype(np.float32) / 255.0
        
        # 打印融合前的范围
        print(f"颜色映射后深度图范围: [{np.min(depth_gray)}, {np.max(depth_gray)}]")
        print(f"颜色映射后触觉图范围: [{np.min(tactile_gray)}, {np.max(tactile_gray)}]")
        
        # 加权融合
        fused_image = (
            self.fusion_config['weight_vision'] * depth_gray +
            self.fusion_config['weight_tactile'] * tactile_gray
        )
        
        # 打印融合后的范围
        print(f"融合后图像范围: [{np.min(fused_image)}, {np.max(fused_image)}]")
        
        # 增强亮度和对比度
        fused_image = (fused_image * 255).astype(np.uint8)
        fused_image = cv2.convertScaleAbs(fused_image, alpha=1.5, beta=50)  # 增加亮度和对比度
        
        # 应用高斯模糊减少噪声
        fused_image = cv2.GaussianBlur(fused_image, (3,3), 0)
        
        # 打印最终范围
        print(f"最终图像范围: [{np.min(fused_image)}, {np.max(fused_image)}]")
        
        return fused_image
    
    def _fuse_force(self, force: List[float], shear: List[float]) -> Dict:
        """融合压力值和剪切力
        
        Args:
            force: 压力值列表 [fx, fy, fz]
            shear: 剪切力列表 [sx, sy]
            
        Returns:
            融合后的力信息
        """
        # 计算合力
        total_force = np.sqrt(sum(f**2 for f in force) + sum(s**2 for s in shear))
        
        # 计算力的方向
        direction = np.arctan2(shear[1], shear[0])
        
        return {
            'total_force': float(total_force),
            'direction': float(direction),
            'components': {
                'normal_x': force[0],
                'normal_y': force[1],
                'normal_z': force[2],
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
        combined_features = self._combine_features(vision_features, tactile_features)
        
        return {
            'vision': vision_features,
            'tactile': tactile_features,
            'confidence': combined_features['confidence']
        }
    
    def _extract_vision_features(self, color_image: np.ndarray) -> Dict:
        """提取视觉特征
        
        Args:
            color_image: 彩色图
            
        Returns:
            视觉特征字典
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # 尝试使用SIFT
            try:
                sift = cv2.SIFT_create()
            except AttributeError:
                # 如果SIFT不可用，尝试使用ORB
                sift = cv2.ORB_create()
                
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if descriptors is None:
                descriptors = np.array([])
                
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'feature_type': 'SIFT' if isinstance(sift, cv2.SIFT) else 'ORB'
            }
            
        except Exception as e:
            print(f"特征提取错误: {str(e)}")
            return {
                'keypoints': [],
                'descriptors': np.array([]),
                'feature_type': 'none'
            }
    
    def _extract_tactile_features(self, tactile_image: np.ndarray) -> Dict:
        """提取触觉特征
        
        Args:
            tactile_image: 触觉图像
            
        Returns:
            触觉特征字典
        """
        # 确保图像是正确的格式
        if tactile_image.dtype != np.uint8:
            tactile_image = (tactile_image * 255).astype(np.uint8)
        
        # 转换为灰度图
        if len(tactile_image.shape) == 3:
            gray = cv2.cvtColor(tactile_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = tactile_image
        
        # 计算压力分布
        pressure_distribution = self._compute_pressure_distribution(gray)
        
        # 计算变形场
        deformation = self._compute_deformation_field(gray)
        
        # 计算接触面积
        contact_area = self._compute_contact_area(gray)
        
        # 计算局部特征
        local_features = self._compute_local_features(gray)
        
        return {
            'pressure': pressure_distribution,
            'deformation': deformation,
            'contact_area': contact_area,
            'local': local_features
        }
    
    def _compute_pressure_distribution(self, image: np.ndarray) -> np.ndarray:
        """计算压力分布
        
        Args:
            image: 灰度图像
            
        Returns:
            压力分布图
        """
        # 使用高斯滤波平滑压力分布
        kernel_size = self.fusion_config['feature_extraction']['tactile']['pressure']['kernel_size']
        sigma = self.fusion_config['feature_extraction']['tactile']['pressure']['sigma']
        
        pressure = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return pressure
    
    def _compute_deformation_field(self, image: np.ndarray) -> np.ndarray:
        """计算变形场
        
        Args:
            image: 灰度图像
            
        Returns:
            变形场
        """
        # 确保图像是正确的格式
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 使用Sobel算子计算梯度
        gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        # 创建变形场
        deformation = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
        deformation[..., 0] = gradient_x
        deformation[..., 1] = gradient_y
        
        return deformation
    
    def _compute_contact_area(self, image: np.ndarray) -> float:
        """计算接触面积
        
        Args:
            image: 灰度图像
            
        Returns:
            接触面积
        """
        config = self.fusion_config['feature_extraction']['tactile']['contact']
        
        # 二值化图像
        _, binary = cv2.threshold(image, config['threshold'] * 255, 255, cv2.THRESH_BINARY)
        
        # 计算白色区域的面积
        area = np.sum(binary > 0) / (image.shape[0] * image.shape[1])
        return area
    
    def _compute_local_features(self, image: np.ndarray) -> Dict:
        """计算局部特征
        
        Args:
            image: 灰度图像
            
        Returns:
            局部特征字典
        """
        # 计算局部统计特征
        mean = np.mean(image)
        std = np.std(image)
        max_val = np.max(image)
        min_val = np.min(image)
        
        # 计算梯度特征
        gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        return {
            'statistics': {
                'mean': mean,
                'std': std,
                'max': max_val,
                'min': min_val
            },
            'gradient': {
                'magnitude': gradient_magnitude,
                'direction': gradient_direction
            }
        }
    
    def _combine_features(self, vision_features: Dict, tactile_features: Dict) -> Dict:
        """组合视觉和触觉特征
        
        Args:
            vision_features: 视觉特征
            tactile_features: 触觉特征
            
        Returns:
            组合后的特征字典
        """
        # 计算特征置信度
        confidence = self._compute_feature_confidence(vision_features, tactile_features)
        
        return {
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
            特征置信度
        """
        config = self.fusion_config['feature_fusion']['confidence']
        
        # 计算视觉特征的置信度
        vision_confidence = 0.0
        if vision_features['descriptors'] is not None:
            vision_confidence = len(vision_features['keypoints']) / 1000.0
            vision_confidence = min(1.0, vision_confidence)
        
        # 计算触觉特征的置信度
        tactile_confidence = tactile_features['contact_area']
        
        # 加权组合
        confidence = (config['vision_weight'] * vision_confidence +
                     config['tactile_weight'] * tactile_confidence)
        
        # 确保置信度在有效范围内
        confidence = max(config['min_confidence'], min(1.0, confidence))
        
        return confidence
    
    def _generate_tactile_data(self) -> Dict:
        """生成模拟的 GelSight Mini 触觉数据
        
        Returns:
            触觉数据字典
        """
        # 生成基础触觉图像 (640x480)
        tactile_image = np.ones((480, 640, 3), dtype=np.uint8) * 180  # 浅灰色背景
        
        # 添加细微的纹理
        for i in range(3):
            noise = np.random.normal(0, 5, (480, 640))  # 减小噪声强度
            tactile_image[:, :, i] = np.clip(tactile_image[:, :, i] + noise, 0, 255).astype(np.uint8)
        
        # 添加多个接触区域，模拟真实接触
        num_contacts = np.random.randint(1, 4)
        for _ in range(num_contacts):
            # 随机选择接触点位置
            center_x = np.random.randint(100, 540)
            center_y = np.random.randint(100, 380)
            
            # 生成椭圆形接触区域
            axes = (np.random.randint(20, 40), np.random.randint(15, 30))
            angle = np.random.randint(0, 180)
            
            # 创建接触区域的渐变效果
            mask = np.zeros((480, 640), dtype=np.uint8)
            cv2.ellipse(mask, (center_x, center_y), axes, angle, 0, 360, 255, -1)
            
            # 添加高斯模糊使边缘更自然
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            
            # 在接触区域添加深色渐变
            for i in range(3):
                tactile_image[:, :, i] = cv2.subtract(
                    tactile_image[:, :, i],
                    (mask * 0.7).astype(np.uint8)  # 减小接触区域的强度
                )
        
        # 生成压力分布图
        pressure_map = np.zeros((480, 640), dtype=np.float32)
        for i in range(num_contacts):
            y, x = np.ogrid[0:480, 0:640]
            pressure = np.exp(-((x - center_x)**2 + (y - center_y)**2)/(2*axes[0]*axes[1])) * 100
            pressure_map = np.maximum(pressure_map, pressure)
        
        # 生成法向量图
        normal_map = np.zeros((480, 640, 3), dtype=np.float32)
        # 将压力图转换为8位格式进行梯度计算
        pressure_uint8 = cv2.normalize(pressure_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gradient_x = cv2.Sobel(pressure_uint8, cv2.CV_32F, 1, 0)
        gradient_y = cv2.Sobel(pressure_uint8, cv2.CV_32F, 0, 1)
        
        # 计算法向量
        normal_map[:, :, 0] = gradient_x
        normal_map[:, :, 1] = gradient_y
        normal_map[:, :, 2] = np.ones_like(gradient_x) * 100  # z方向为主
        
        # 归一化法向量
        norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
        normal_map = normal_map / (norm + 1e-6)
        
        # 生成剪切力信息
        shear_force = [
            np.random.uniform(-5, 5),  # x方向剪切力
            np.random.uniform(-5, 5)   # y方向剪切力
        ]
        
        # 计算总压力
        total_force = np.sum(pressure_map) * 0.01  # 缩放到合理范围
        
        return {
            'image': tactile_image,
            'pressure_map': pressure_map,
            'normal_map': normal_map,
            'shear_force': shear_force,
            'total_force': float(total_force)
        } 