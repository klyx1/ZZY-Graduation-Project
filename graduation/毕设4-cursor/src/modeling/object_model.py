import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import yaml
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

class FlexibleObjectModel:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化柔性物体模型
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.modeling_config = self.config['modeling']
        self.points = None
        self.normals = None
        self.deformation = None
        self.stiffness = None
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def update_model(self, 
                    pointcloud: np.ndarray,
                    tactile_data: Dict,
                    force_data: Dict) -> bool:
        """更新物体模型
        
        Args:
            pointcloud: 点云数据
            tactile_data: 触觉数据
            force_data: 力数据
            
        Returns:
            是否更新成功
        """
        try:
            # 1. 更新点云数据
            self._update_pointcloud(pointcloud)
            
            # 2. 计算法向量
            self._compute_normals()
            
            # 3. 更新变形信息
            self._update_deformation(tactile_data, force_data)
            
            # 4. 更新刚度信息
            self._update_stiffness(force_data)
            
            return True
        except Exception as e:
            print(f"更新模型失败: {str(e)}")
            return False
    
    def _update_pointcloud(self, pointcloud: np.ndarray):
        """更新点云数据
        
        Args:
            pointcloud: 点云数据
        """
        if pointcloud is None or len(pointcloud) == 0:
            return
            
        # 降采样
        if len(pointcloud) > self.modeling_config['max_points']:
            indices = np.random.choice(len(pointcloud), 
                                    self.modeling_config['max_points'],
                                    replace=False)
            pointcloud = pointcloud[indices]
            
        self.points = pointcloud
    
    def _compute_normals(self):
        """计算点云法向量"""
        if self.points is None or len(self.points) < 3:
            return
            
        # 构建KD树
        tree = cKDTree(self.points)
        
        # 对每个点计算法向量
        normals = np.zeros_like(self.points)
        for i in range(len(self.points)):
            # 找到邻近点
            indices = tree.query(self.points[i], k=10)[1]
            neighbors = self.points[indices]
            
            # 计算协方差矩阵
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = np.dot(centered.T, centered)
            
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 最小特征值对应的特征向量即为法向量
            normal = eigenvectors[:, 0]
            if normal[2] < 0:  # 确保法向量朝外
                normal = -normal
                
            normals[i] = normal
            
        self.normals = normals
    
    def _update_deformation(self, tactile_data: Dict, force_data: Dict):
        """更新变形信息
        
        Args:
            tactile_data: 触觉数据
            force_data: 力数据
        """
        if tactile_data is None or force_data is None:
            return
            
        # 从触觉图像中提取变形信息
        tactile_image = tactile_data.get('image')
        if tactile_image is not None:
            # 转换为灰度图
            gray = cv2.cvtColor(tactile_image, cv2.COLOR_BGR2GRAY)
            
            # 计算变形场
            deformation_field = self._compute_deformation_field(gray)
            
            # 更新变形信息
            self.deformation = {
                'field': deformation_field,
                'magnitude': np.max(np.abs(deformation_field)),
                'direction': np.mean(deformation_field, axis=0)
            }
    
    def _compute_deformation_field(self, image: np.ndarray) -> np.ndarray:
        """计算变形场
        
        Args:
            image: 灰度图像
            
        Returns:
            变形场
        """
        # 使用光流法计算变形场
        if not hasattr(self, '_prev_image'):
            self._prev_image = image
            return np.zeros((image.shape[0], image.shape[1], 2))
            
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        self._prev_image = image
        return flow
    
    def _update_stiffness(self, force_data: Dict):
        """更新刚度信息
        
        Args:
            force_data: 力数据
        """
        if force_data is None:
            return
            
        # 计算局部刚度
        force = force_data.get('total_force', 0)
        deformation = self.deformation.get('magnitude', 0) if self.deformation else 0
        
        if deformation > 0:
            stiffness = force / deformation
        else:
            stiffness = 0
            
        self.stiffness = {
            'value': stiffness,
            'distribution': self._compute_stiffness_distribution()
        }
    
    def _compute_stiffness_distribution(self) -> np.ndarray:
        """计算刚度分布
        
        Returns:
            刚度分布图
        """
        if self.points is None or self.deformation is None:
            return None
            
        # 创建网格
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        
        # 计算刚度分布
        points = np.column_stack((X.ravel(), Y.ravel()))
        values = self.stiffness['value'] * np.ones(len(points))
        
        # 插值
        grid_z = griddata(
            points, values, (X, Y),
            method='cubic',
            fill_value=0
        )
        
        return grid_z
    
    def get_model_state(self) -> Dict:
        """获取模型状态
        
        Returns:
            模型状态字典
        """
        return {
            'points': self.points,
            'normals': self.normals,
            'deformation': self.deformation,
            'stiffness': self.stiffness
        }
    
    def predict_deformation(self, force: float) -> np.ndarray:
        """预测给定力下的变形
        
        Args:
            force: 施加的力
            
        Returns:
            预测的变形场
        """
        if self.stiffness is None or self.deformation is None:
            return None
            
        # 使用线性模型预测变形
        deformation_scale = force / self.stiffness['value']
        predicted_deformation = self.deformation['field'] * deformation_scale
        
        return predicted_deformation
    
    def get_grasp_points(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取推荐抓取点
        
        Returns:
            抓取点列表，每个元素为(位置, 法向量)的元组
        """
        if self.points is None or self.normals is None:
            return []
            
        # 基于曲率和法向量选择抓取点
        grasp_points = []
        
        # 计算曲率
        curvature = self._compute_curvature()
        
        # 选择曲率较小的点作为候选抓取点
        threshold = np.percentile(curvature, 80)
        candidate_indices = np.where(curvature < threshold)[0]
        
        # 选择法向量合适的点
        for idx in candidate_indices:
            normal = self.normals[idx]
            if abs(normal[2]) > 0.7:  # 法向量接近垂直
                grasp_points.append((self.points[idx], normal))
                
        return grasp_points
    
    def _compute_curvature(self) -> np.ndarray:
        """计算曲率
        
        Returns:
            曲率数组
        """
        if self.points is None or self.normals is None:
            return None
            
        # 构建KD树
        tree = cKDTree(self.points)
        
        # 计算曲率
        curvature = np.zeros(len(self.points))
        for i in range(len(self.points)):
            # 找到邻近点
            indices = tree.query(self.points[i], k=10)[1]
            neighbors = self.points[indices]
            
            # 计算局部曲率
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = np.dot(centered.T, centered)
            eigenvalues = np.linalg.eigvals(cov)
            
            # 使用最小特征值作为曲率估计
            curvature[i] = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
            
        return curvature 