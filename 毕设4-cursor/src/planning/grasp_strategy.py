import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

class GraspStrategy:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化抓取策略
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.strategy_config = self.config.get('grasp_strategy', {})
        
        # 加载或初始化模型
        self.model_path = os.path.join(
            self.config['system']['data_path'],
            'models',
            'grasp_model.pkl'
        )
        self.model = self._load_or_init_model()
        
        # 数据标准化器
        self.scaler = StandardScaler()
        
        # 历史数据
        self.grasp_history = []
        self.success_history = []
        
        # 自适应参数
        self.force_learning_rate = self.strategy_config.get('force_learning_rate', 0.1)
        self.min_force = self.config['control']['min_force']
        self.max_force = self.config['control']['max_force']
        self.current_force = self.min_force
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_or_init_model(self) -> RandomForestClassifier:
        """加载或初始化模型
        
        Returns:
            随机森林分类器
        """
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
    def select_grasp_point(self, 
                          point_cloud: np.ndarray,
                          normals: np.ndarray,
                          object_model: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """选择最佳抓取点
        
        Args:
            point_cloud: 点云数据
            normals: 法向量数据
            object_model: 物体模型状态
            
        Returns:
            抓取点和法向量
        """
        # 提取特征
        features = self._extract_features(point_cloud, normals, object_model)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 预测抓取点得分
        scores = self.model.predict_proba(features_scaled)[:, 1]
        
        # 选择得分最高的点
        best_idx = np.argmax(scores)
        grasp_point = point_cloud[best_idx]
        grasp_normal = normals[best_idx]
        
        return grasp_point, grasp_normal
        
    def _extract_features(self,
                         point_cloud: np.ndarray,
                         normals: np.ndarray,
                         object_model: Dict) -> np.ndarray:
        """提取特征
        
        Args:
            point_cloud: 点云数据
            normals: 法向量数据
            object_model: 物体模型状态
            
        Returns:
            特征矩阵
        """
        features = []
        
        for i in range(len(point_cloud)):
            # 几何特征
            point = point_cloud[i]
            normal = normals[i]
            
            # 计算局部曲率
            neighbors = self._find_neighbors(point_cloud, point)
            curvature = self._calculate_curvature(neighbors, normal)
            
            # 计算局部密度
            density = self._calculate_density(neighbors)
            
            # 计算到物体中心的距离
            center = np.mean(point_cloud, axis=0)
            center_dist = np.linalg.norm(point - center)
            
            # 计算到物体表面的距离
            surface_dist = self._calculate_surface_distance(
                point, normal, object_model
            )
            
            # 组合特征
            feature = np.array([
                curvature,
                density,
                center_dist,
                surface_dist,
                *normal,  # 法向量
                *point    # 位置
            ])
            
            features.append(feature)
            
        return np.array(features)
        
    def _find_neighbors(self,
                       point_cloud: np.ndarray,
                       point: np.ndarray,
                       radius: float = 0.03) -> np.ndarray:
        """查找邻近点
        
        Args:
            point_cloud: 点云数据
            point: 目标点
            radius: 搜索半径
            
        Returns:
            邻近点索引
        """
        distances = np.linalg.norm(point_cloud - point, axis=1)
        return np.where(distances <= radius)[0]
        
    def _calculate_curvature(self,
                           neighbors: np.ndarray,
                           normal: np.ndarray) -> float:
        """计算局部曲率
        
        Args:
            neighbors: 邻近点索引
            normal: 法向量
            
        Returns:
            曲率值
        """
        if len(neighbors) < 3:
            return 0.0
            
        # 使用PCA计算曲率
        cov_matrix = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        return eigenvalues[2] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
        
    def _calculate_density(self, neighbors: np.ndarray) -> float:
        """计算局部密度
        
        Args:
            neighbors: 邻近点索引
            
        Returns:
            密度值
        """
        return len(neighbors)
        
    def _calculate_surface_distance(self,
                                  point: np.ndarray,
                                  normal: np.ndarray,
                                  object_model: Dict) -> float:
        """计算到物体表面的距离
        
        Args:
            point: 目标点
            normal: 法向量
            object_model: 物体模型状态
            
        Returns:
            距离值
        """
        # 这里需要根据实际的物体模型实现
        # 示例实现
        return 0.0
        
    def update_force(self,
                    current_force: float,
                    deformation: float,
                    success: bool):
        """更新夹持力
        
        Args:
            current_force: 当前夹持力
            deformation: 变形量
            success: 是否成功
        """
        # 记录历史数据
        self.grasp_history.append({
            'force': current_force,
            'deformation': deformation
        })
        self.success_history.append(success)
        
        # 根据历史数据调整力
        if len(self.success_history) >= 5:
            recent_success_rate = np.mean(self.success_history[-5:])
            
            if recent_success_rate < 0.6:  # 成功率过低
                self.current_force = min(
                    self.current_force * (1 + self.force_learning_rate),
                    self.max_force
                )
            elif recent_success_rate > 0.8:  # 成功率过高
                self.current_force = max(
                    self.current_force * (1 - self.force_learning_rate),
                    self.min_force
                )
                
    def get_force(self) -> float:
        """获取当前夹持力
        
        Returns:
            夹持力值
        """
        return self.current_force
        
    def train_model(self, training_data: List[Dict]):
        """训练模型
        
        Args:
            training_data: 训练数据
        """
        # 提取特征和标签
        features = []
        labels = []
        
        for data in training_data:
            point_cloud = data['point_cloud']
            normals = data['normals']
            object_model = data['object_model']
            success = data['success']
            
            # 提取特征
            data_features = self._extract_features(
                point_cloud, normals, object_model
            )
            
            features.extend(data_features)
            labels.extend([success] * len(data_features))
            
        # 标准化特征
        features = np.array(features)
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        # 训练模型
        self.model.fit(features_scaled, labels)
        
        # 保存模型
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
    def save_history(self):
        """保存历史数据"""
        history_path = os.path.join(
            self.config['system']['data_path'],
            'history',
            'grasp_history.json'
        )
        
        history = {
            'grasp_history': self.grasp_history,
            'success_history': self.success_history
        }
        
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history, f) 