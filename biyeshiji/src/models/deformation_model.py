# 柔性物体形变模型
import numpy as np
import cv2


class MassSpringDeformation:
    """ 基于质量-弹簧模型的柔性物体形变模拟 """

    def __init__(self, grid_size=(10, 10), stiffness=100.0, damping=0.1):
        """
        :param grid_size: 网格尺寸（rows, cols）
        :param stiffness: 弹簧刚度系数（N/m）
        :param damping: 阻尼系数（Ns/m）
        """
        self.stiffness = stiffness
        self.damping = damping

        # 初始化网格顶点
        x, y = np.meshgrid(np.linspace(0, 1, grid_size[1]),
                           np.linspace(0, 1, grid_size[0]))
        self.positions = np.stack([x.flatten(), y.flatten(), np.zeros_like(x.flatten())], axis=1)
        self.velocities = np.zeros_like(self.positions)

        # 构建弹簧连接（水平和垂直方向）
        self.springs = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if j < grid_size[1] - 1:  # 水平弹簧
                    self.springs.append((i * grid_size[1] + j, i * grid_size[1] + j + 1))
                if i < grid_size[0] - 1:  # 垂直弹簧
                    self.springs.append((i * grid_size[1] + j, (i + 1) * grid_size[1] + j))

    def apply_force(self, force_map: np.ndarray):
        """
        应用外部力场
        :param force_map: (H,W,3)力场图（单位：N）
        """
        # 将力场下采样到网格尺寸
        h, w = self.positions[:, :2].max(0) + 1
        sampled_forces = cv2.resize(force_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # 应用力到每个顶点
        self.velocities += sampled_forces.reshape(-1, 3) * 0.1  # 时间步长0.1s

    def update(self):
        """ 更新物理模拟 """
        # 计算弹簧力
        for i, j in self.springs:
            delta = self.positions[j] - self.positions[i]
            length = np.linalg.norm(delta)
            if length == 0:
                continue

            # 胡克定律：F = -k * (x - x0)
            force = self.stiffness * (delta / length) * (length - 0.1)  # 初始长度0.1m

            # 更新速度
            self.velocities[i] += force * 0.1  # 时间步长0.1s
            self.velocities[j] -= force * 0.1

        # 应用阻尼
        self.velocities *= (1 - self.damping)

        # 更新位置
        self.positions += self.velocities * 0.1