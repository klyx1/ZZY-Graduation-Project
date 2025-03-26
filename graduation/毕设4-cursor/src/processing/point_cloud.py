import numpy as np
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from sklearn.neighbors import KDTree
import yaml
import os

class PointCloudProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化点云处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.processing_config = self.config.get('point_cloud_processing', {})
        
        # 创建点云对象
        self.pcd = o3d.geometry.PointCloud()
        
        # 创建KD树
        self.kdtree = None
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def load_point_cloud(self, points: np.ndarray):
        """加载点云数据
        
        Args:
            points: 点云数据
        """
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.kdtree = KDTree(points)
        
    def preprocess(self) -> np.ndarray:
        """预处理点云
        
        Returns:
            处理后的点云数据
        """
        # 移除离群点
        self.pcd, _ = self.pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
        
        # 降采样
        self.pcd = self.pcd.voxel_down_sample(
            voxel_size=self.processing_config.get('voxel_size', 0.01)
        )
        
        # 估计法向量
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.03,
                max_nn=30
            )
        )
        
        return np.asarray(self.pcd.points)
        
    def register_point_clouds(self,
                            source_points: np.ndarray,
                            target_points: np.ndarray) -> np.ndarray:
        """配准点云
        
        Args:
            source_points: 源点云
            target_points: 目标点云
            
        Returns:
            变换矩阵
        """
        # 创建点云对象
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        # 估计法向量
        source_pcd.estimate_normals()
        target_pcd.estimate_normals()
        
        # 计算FPFH特征
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        
        # 全局配准
        distance_threshold = 0.05
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh,
            True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        
        # ICP精配准
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            distance_threshold,
            result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        return result.transformation
        
    def extract_features(self,
                        point: np.ndarray,
                        radius: float = 0.03) -> Dict:
        """提取点特征
        
        Args:
            point: 目标点
            radius: 搜索半径
            
        Returns:
            特征字典
        """
        # 查找邻近点
        indices = self.kdtree.query_radius(
            point.reshape(1, -1),
            r=radius
        )[0]
        
        if len(indices) < 3:
            return {}
            
        # 获取邻近点
        neighbors = np.asarray(self.pcd.points)[indices]
        
        # 计算协方差矩阵
        cov_matrix = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 计算曲率
        curvature = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
        
        # 计算法向量
        normal = eigenvectors[:, 0]
        
        # 计算局部密度
        density = len(indices) / (4/3 * np.pi * radius**3)
        
        # 计算局部平面度
        planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
        
        # 计算局部线性度
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        
        return {
            'curvature': curvature,
            'normal': normal,
            'density': density,
            'planarity': planarity,
            'linearity': linearity,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        }
        
    def segment_point_cloud(self,
                          num_clusters: int = 5) -> List[np.ndarray]:
        """分割点云
        
        Args:
            num_clusters: 聚类数量
            
        Returns:
            分割后的点云列表
        """
        # 计算点云法向量
        self.pcd.estimate_normals()
        
        # 计算FPFH特征
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        
        # 聚类
        labels = np.array(self.pcd.cluster_dbscan(
            eps=0.05,
            min_points=10
        ))
        
        # 分割点云
        segments = []
        for i in range(num_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                segment = np.asarray(self.pcd.points)[mask]
                segments.append(segment)
                
        return segments
        
    def track_object(self,
                    prev_points: np.ndarray,
                    curr_points: np.ndarray) -> np.ndarray:
        """跟踪物体
        
        Args:
            prev_points: 上一帧点云
            curr_points: 当前帧点云
            
        Returns:
            变换矩阵
        """
        # 创建点云对象
        prev_pcd = o3d.geometry.PointCloud()
        curr_pcd = o3d.geometry.PointCloud()
        
        prev_pcd.points = o3d.utility.Vector3dVector(prev_points)
        curr_pcd.points = o3d.utility.Vector3dVector(curr_points)
        
        # 估计法向量
        prev_pcd.estimate_normals()
        curr_pcd.estimate_normals()
        
        # 计算FPFH特征
        prev_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            prev_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        curr_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            curr_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        
        # 特征匹配
        distance_threshold = 0.05
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            prev_pcd, curr_pcd, prev_fpfh, curr_fpfh,
            True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        
        return result.transformation
        
    def save_point_cloud(self, file_path: str):
        """保存点云
        
        Args:
            file_path: 保存路径
        """
        o3d.io.write_point_cloud(file_path, self.pcd)
        
    def load_point_cloud_from_file(self, file_path: str):
        """从文件加载点云
        
        Args:
            file_path: 文件路径
        """
        self.pcd = o3d.io.read_point_cloud(file_path)
        self.kdtree = KDTree(np.asarray(self.pcd.points)) 