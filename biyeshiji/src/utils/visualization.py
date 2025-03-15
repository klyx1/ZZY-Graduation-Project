# Open3D/热力图可视化
import cv2
import numpy as np
import open3d as o3d


class TactileVisualizer:
    """ 触觉数据可视化工具类 """

    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("3D形变可视化", width=800, height=600)
        self.mesh = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)

    def update_3d_mesh(self, depth_map: np.ndarray, mmpp: float = 0.1) -> None:
        """
        根据深度图更新3D网格显示
        :param depth_map: 深度图（单位: mm）
        :param mmpp: 每像素对应的物理尺寸（毫米/像素）
        """
        h, w = depth_map.shape
        x = np.arange(0, w * mmpp, mmpp)
        y = np.arange(0, h * mmpp, mmpp)
        xx, yy = np.meshgrid(x, y)

        # 创建点云
        points = np.stack([xx, yy, depth_map], axis=-1).reshape(-1, 3)
        self.mesh.vertices = o3d.utility.Vector3dVector(points)

        # 生成三角面片
        triangles = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                triangles.append([idx, idx + 1, idx + w])
                triangles.append([idx + 1, idx + w + 1, idx + w])
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)

        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    @staticmethod
    def overlay_heatmap(rgb_img: np.ndarray, force_map: np.ndarray, alpha=0.6) -> np.ndarray:
        """
        将力场热力图叠加到RGB图像
        :param rgb_img: 原始RGB图像 (H,W,3)
        :param force_map: 力场图 (H',W')
        :param alpha: 叠加透明度
        :return: 叠加后的图像
        """
        # 归一化并生成热力图
        heatmap = cv2.normalize(force_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        # 调整尺寸匹配
        h, w = rgb_img.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))

        # Alpha混合
        return cv2.addWeighted(rgb_img, 1 - alpha, heatmap, alpha, 0)