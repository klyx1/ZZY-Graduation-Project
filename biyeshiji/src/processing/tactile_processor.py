# 触觉梯度/力场计算
import numpy as np
import cv2
from scipy import fftpack


class TactileProcessor:
    """ 触觉数据预处理与力场计算 """

    def __init__(self, E: float = 0.3, nu: float = 0.48):
        """
        :param E: 弹性体杨氏模量（单位: MPa）
        :param nu: 泊松比（硅胶材料典型值）
        """
        self.E = E
        self.nu = nu

    def compute_gradients(self, tactile_img: np.ndarray) -> tuple:
        """
        计算触觉图像的梯度场
        :param tactile_img: (H,W,3)的触觉图像
        :return: (gx, gy) 梯度场，形状(H,W)
        """
        gray = cv2.cvtColor(tactile_img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 水平梯度
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 垂直梯度
        return gx, gy

    def poisson_reconstruct(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        基于泊松方程重建深度图
        :param gx: 水平梯度场
        :param gy: 垂直梯度场
        :return: 重建的深度图（单位: mm）
        """
        # 计算散度场 ∇·G = ∂gx/∂x + ∂gy/∂y
        gyy, gxx = np.gradient(gy, axis=0), np.gradient(gx, axis=1)
        divergence = gxx + gyy

        # 使用DCT求解泊松方程（Neumann边界条件）
        f_dct = fftpack.dct(fftpack.dct(divergence.T, norm='ortho').T, norm='ortho')
        denom = (np.arange(1, f_dct.shape[0] + 1)[:, None] ** 2 +
                 np.arange(1, f_dct.shape[1] + 1) ** 2)
        f_dct /= -denom
        f_dct[0, 0] = 0  # 去除直流分量

        # 反DCT得到深度图
        depth = fftpack.idct(fftpack.idct(f_dct, norm='ortho').T, norm='ortho')
        return depth * 1000  # 转换为毫米

    def depth_to_force(self, depth_map: np.ndarray) -> np.ndarray:
        """
        将深度图转换为法向力场
        :param depth_map: 深度图（单位: mm）
        :return: 法向力场（单位: N）
        """
        strain = depth_map / depth_map.max()  # 标准化应变
        return self.E * strain / (1 - self.nu ** 2)  # 平面应力模型