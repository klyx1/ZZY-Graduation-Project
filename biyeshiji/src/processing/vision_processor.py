# 视觉点云/YOLO检测
import cv2
import numpy as np
from sklearn.decomposition import PCA


class VisionProcessor:
    """ 视觉特征提取与处理 """

    def __init__(self, pca_components: int = 3):
        self.pca = PCA(n_components=pca_components)
        self.is_fitted = False

    def depth_to_pointcloud(self, depth_map: np.ndarray, intrinsics) -> np.ndarray:
        """
        将深度图转换为点云
        :param depth_map: 深度图（单位：毫米）
        :param intrinsics: 相机内参（fx, fy, cx, cy）
        :return: (N,3)点云坐标（单位：米）
        """
        fx, fy, cx, cy = intrinsics
        height, width = depth_map.shape

        # 生成像素网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_map.astype(float) / 1000.0  # 转换为米

        # 计算点云坐标
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        return points[~np.isinf(z).flatten()]  # 移除无效点

    def extract_features(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        从点云中提取PCA特征
        :param pointcloud: (N,3)点云数据
        :return: (3,)主成分特征向量
        """
        if not self.is_fitted:
            self.pca.fit(pointcloud)
            self.is_fitted = True

        return self.pca.transform(pointcloud.mean(0).reshape(1, -1)).flatten()