import numpy as np
import logging
import random
from typing import Dict, Union, List

logger = logging.getLogger(__name__)

class SensorDataGenerator:
    def __init__(self, image_size=(480, 480), noise_level=0.1):
        """初始化数据生成器
        
        Args:
            image_size (tuple): 图像尺寸，格式为(宽度, 高度)
            noise_level (float): 噪声水平，范围[0, 1]
        """
        self.image_size = image_size
        self.noise_level = noise_level
        self.max_depth = 2.0
        self.orange_radius = 0.05
        self.orange_center = np.array([0, 0, 0.5])
        self.camera_pos = np.array([0, 0, 0])
        self.camera_focal = 500
        self.camera_center = np.array([image_size[0]/2, image_size[1]/2])
        
        # 生成不规则变形参数
        self.deform_points = np.random.rand(10, 3) * 2 - 1  # 10个随机变形点
        self.deform_strengths = np.random.rand(10) * 0.02  # 变形强度
        self.surface_noise = np.random.rand(32, 32, 3) * 0.01  # 表面噪声
        
    def _get_deformed_radius(self, point):
        """计算变形后的半径
        
        Args:
            point: 表面点
            
        Returns:
            变形后的半径
        """
        # 基础半径
        radius = self.orange_radius
        
        # 添加全局不规则变形
        for i in range(len(self.deform_points)):
            dist = np.linalg.norm(point - self.deform_points[i])
            if dist < self.orange_radius * 2:
                # 使用平滑的高斯变形
                deform = self.deform_strengths[i] * np.exp(-dist**2 / (2 * (self.orange_radius/2)**2))
                radius += deform
                
        # 添加局部表面噪声
        u = np.arctan2(point[0], point[2]) / (2 * np.pi) + 0.5
        v = np.arcsin(np.clip(point[1], -1, 1)) / np.pi + 0.5
        
        # 双线性插值获取表面噪声
        x = u * (self.surface_noise.shape[0] - 1)
        y = v * (self.surface_noise.shape[1] - 1)
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.surface_noise.shape[0] - 1), min(y0 + 1, self.surface_noise.shape[1] - 1)
        
        # 计算插值权重
        wx = x - x0
        wy = y - y0
        
        # 双线性插值
        noise = (self.surface_noise[x0, y0] * (1 - wx) * (1 - wy) +
                self.surface_noise[x1, y0] * wx * (1 - wy) +
                self.surface_noise[x0, y1] * (1 - wx) * wy +
                self.surface_noise[x1, y1] * wx * wy)
        
        radius += np.mean(noise)
        
        return radius
        
    def _ray_sphere_intersection(self, ray):
        """计算光线与变形球体的交点"""
        try:
            # 使用解析方法求解光线与球体的交点
            oc = self.camera_pos - self.orange_center
            a = np.dot(ray, ray)
            b = 2.0 * np.dot(oc, ray)
            c = np.dot(oc, oc) - self.orange_radius * self.orange_radius
            
            discriminant = b * b - 4 * a * c
            
            if discriminant < 0:
                return None
                
            t = (-b - np.sqrt(discriminant)) / (2.0 * a)
            
            if t < 0:
                return None
                
            return t
            
        except Exception as e:
            logger.error(f"光线求交计算错误: {str(e)}")
            return None
        
    def _generate_orange_texture(self):
        """生成橘子表面纹理"""
        texture_size = (256, 256)
        texture = np.zeros((texture_size[0], texture_size[1], 3), dtype=np.uint8)
        
        # 基础颜色（橙色）
        base_color = np.array([255, 165, 0])
        
        # 生成简单的随机纹理
        noise = np.random.rand(texture_size[0], texture_size[1])
        
        # 添加纹理变化
        for y in range(texture_size[0]):
            for x in range(texture_size[1]):
                # 使用噪声生成颜色变化
                color_var = (noise[y, x] * 50 - 25)
                color = np.clip(base_color + color_var, 0, 255)
                texture[y, x] = color
                
        return texture
        
    def _perlin_noise(self, x, y):
        """生成Perlin噪声
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            噪声值
        """
        # 获取整数坐标
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1
        
        # 计算小数部分
        sx = x - x0
        sy = y - y0
        
        # 平滑插值
        sx = sx * sx * (3 - 2 * sx)
        sy = sy * sy * (3 - 2 * sy)
        
        # 生成随机梯度
        def get_gradient(x, y):
            # 使用确定性哈希函数代替随机数生成
            h = x * 374761393 + y * 668265263
            h = (h ^ (h >> 13)) * 1274126177
            h = h ^ (h >> 16)
            angle = (h % 360) * np.pi / 180
            return np.array([np.cos(angle), np.sin(angle)])
            
        # 计算四个角的贡献
        n0 = np.dot(get_gradient(x0, y0), [x - x0, y - y0])
        n1 = np.dot(get_gradient(x1, y0), [x - x1, y - y0])
        ix0 = n0 + sx * (n1 - n0)
        
        n0 = np.dot(get_gradient(x0, y1), [x - x0, y - y1])
        n1 = np.dot(get_gradient(x1, y1), [x - x1, y - y1])
        ix1 = n0 + sx * (n1 - n0)
        
        return ix0 + sy * (ix1 - ix0)
        
    def _get_ray_direction(self, x, y):
        """计算从相机到像素点的光线方向"""
        # 将像素坐标转换为相机坐标系下的方向向量
        direction = np.array([
            (x - self.camera_center[0]) / self.camera_focal,
            (y - self.camera_center[1]) / self.camera_focal,
            1.0
        ])
        # 归一化方向向量
        return direction / np.linalg.norm(direction)
        
    def _depth_to_point_cloud(self, depth_map):
        """将深度图转换为点云"""
        points = []
        for y in range(depth_map.shape[0]):
            for x in range(depth_map.shape[1]):
                if depth_map[y, x] < self.max_depth:
                    # 计算3D点坐标
                    ray = self._get_ray_direction(x, y)
                    point = self.camera_pos + depth_map[y, x] * ray
                    points.append(point)
        return np.array(points)
        
    def generate_vision_data(self) -> Dict[str, np.ndarray]:
        """生成视觉数据
        
        Returns:
            Dict[str, np.ndarray]: 包含深度图和彩色图像的字典
        """
        logger.debug("开始生成深度图")
        
        # 生成所有像素的坐标
        x = np.arange(self.image_size[0])
        y = np.arange(self.image_size[1])
        X, Y = np.meshgrid(x, y)
        
        # 计算所有像素的光线方向
        directions = np.stack([
            (X - self.camera_center[0]) / self.camera_focal,
            (Y - self.camera_center[1]) / self.camera_focal,
            np.ones_like(X)
        ], axis=-1)
        
        # 归一化方向向量
        norms = np.sqrt(np.sum(directions ** 2, axis=-1, keepdims=True))
        directions = directions / norms
        
        # 计算所有光线与球体的交点
        oc = self.camera_pos - self.orange_center
        a = np.sum(directions ** 2, axis=-1)
        b = 2.0 * np.sum(oc * directions, axis=-1)
        c = np.dot(oc, oc) - self.orange_radius * self.orange_radius
        
        discriminant = b * b - 4 * a * c
        valid_rays = discriminant >= 0
        
        # 初始化深度图
        depth_map = np.full(self.image_size[::-1], self.max_depth, dtype=np.float32)
        
        # 计算有效光线的交点
        t = np.zeros_like(depth_map)
        t[valid_rays] = (-b[valid_rays] - np.sqrt(discriminant[valid_rays])) / (2.0 * a[valid_rays])
        valid_t = (t > 0) & (t < self.max_depth)
        
        # 计算交点的深度
        intersection_points = self.camera_pos + t[..., np.newaxis] * directions
        depth = np.linalg.norm(intersection_points - self.orange_center, axis=-1)
        depth_map[valid_t] = depth[valid_t]
        
        logger.debug("添加深度图噪声")
        # 添加噪声
        depth_noise = np.random.normal(0, 0.01, depth_map.shape)
        depth_map = np.clip(depth_map + depth_noise, 0, self.max_depth)
        
        logger.debug("开始生成彩色图像")
        # 生成彩色图
        color_image = np.zeros((*self.image_size[::-1], 3), dtype=np.uint8)
        
        logger.debug("生成纹理")
        texture = self._generate_orange_texture()
        
        # 计算有效交点的纹理坐标
        valid_points = intersection_points[valid_t]
        point_dirs = valid_points - self.orange_center
        point_dirs = point_dirs / np.linalg.norm(point_dirs, axis=-1, keepdims=True)
        
        u = np.arctan2(point_dirs[:, 0], point_dirs[:, 2]) / (2 * np.pi) + 0.5
        v = np.arcsin(np.clip(point_dirs[:, 1], -1, 1)) / np.pi + 0.5
        
        # 获取纹理颜色
        tx = (u * (texture.shape[1] - 1)).astype(np.int32)
        ty = (v * (texture.shape[0] - 1)).astype(np.int32)
        
        # 将颜色应用到有效像素
        valid_coords = np.where(valid_t)
        color_image[valid_coords] = texture[ty, tx]
        
        logger.debug("视觉数据生成完成")
        return {
            'depth_map': depth_map,
            'color_image': color_image,
            'point_cloud': self._depth_to_point_cloud(depth_map)
        }
        
    def generate_tactile_data(self) -> Dict[str, Union[np.ndarray, List[float]]]:
        """生成触觉数据
        
        Returns:
            Dict[str, Union[np.ndarray, List[float]]]: 包含触觉图像和力/剪切信息的字典
        """
        # 生成640x480的触觉图像，使用3通道RGB格式
        tactile_image = np.ones((480, 640, 3), dtype=np.uint8) * 180  # 浅灰色背景
        
        # 添加基础纹理，模拟橘子表面的纹理
        texture = np.random.normal(0, 3, size=(480, 640, 3))  # 减小噪声强度
        tactile_image = np.clip(tactile_image + texture, 0, 255).astype(np.uint8)
        
        # 生成随机接触点
        num_contacts = random.randint(1, 2)  # 减少接触点数量
        contact_points = []
        
        for _ in range(num_contacts):
            # 随机选择接触点位置
            center_x = random.randint(100, 540)  # 避免边缘
            center_y = random.randint(100, 380)
            
            # 生成椭圆形接触区域，模拟橘子表面的凹凸
            radius_x = random.randint(15, 25)  # 减小接触区域大小
            radius_y = random.randint(15, 25)
            angle = random.uniform(0, np.pi)
            
            # 创建网格
            y, x = np.ogrid[-center_y:480-center_y, -center_x:640-center_x]
            
            # 旋转坐标
            x_rot = x * np.cos(angle) + y * np.sin(angle)
            y_rot = -x * np.sin(angle) + y * np.cos(angle)
            
            # 创建椭圆形mask
            mask = (x_rot*x_rot)/(radius_x*radius_x) + (y_rot*y_rot)/(radius_y*radius_y) <= 1
            
            # 计算压力分布，使用更小的压力值
            pressure = np.exp(-((x_rot*x_rot)/(2*radius_x*radius_x) + (y_rot*y_rot)/(2*radius_y*radius_y)))
            pressure = pressure * 0.3  # 减小压力值
            
            # 根据压力生成颜色变化
            intensity = random.uniform(0.2, 0.4)  # 减小颜色变化强度
            shadow_offset = random.randint(2, 4)  # 减小阴影偏移
            
            # 在接触区域添加颜色变化
            for c in range(3):
                # 主要接触区域
                tactile_image[mask, c] = np.clip(
                    tactile_image[mask, c] - pressure[mask] * 255 * intensity * (0.9 + c * 0.1),
                    0, 255
                ).astype(np.uint8)
                
                # 添加阴影效果
                shadow_mask = np.roll(mask, shadow_offset, axis=0)
                shadow_pressure = np.roll(pressure, shadow_offset, axis=0)
                tactile_image[shadow_mask, c] = np.clip(
                    tactile_image[shadow_mask, c] + shadow_pressure[shadow_mask] * 30,  # 减小阴影强度
                    0, 255
                ).astype(np.uint8)
            
            contact_points.append({
                'center': (center_x, center_y),
                'radius': (radius_x, radius_y),
                'angle': angle,
                'intensity': intensity
            })
        
        # 添加全局照明效果
        gradient = np.linspace(0, 15, 480).reshape(-1, 1, 1)  # 减小梯度强度
        gradient = np.repeat(gradient, 640, axis=1)
        gradient = np.repeat(gradient, 3, axis=2)
        tactile_image = np.clip(tactile_image + gradient, 0, 255).astype(np.uint8)
        
        # 计算力和剪切信息，使用更小的力值
        total_pressure = sum(point['intensity'] * np.pi * point['radius'][0] * point['radius'][1]
                           for point in contact_points)
        
        # 力的计算，使用更小的力值范围
        force_scale = 2.0  # 减小力的缩放因子
        force_z = total_pressure * force_scale
        
        # 根据接触点的分布计算力的方向
        if contact_points:
            center_of_pressure_x = np.mean([p['center'][0] for p in contact_points])
            center_of_pressure_y = np.mean([p['center'][1] for p in contact_points])
            
            # 将中心点坐标归一化到[-1, 1]范围
            norm_x = (center_of_pressure_x - 320) / 320
            norm_y = (center_of_pressure_y - 240) / 240
            
            force_x = norm_x * force_z * 0.2  # 减小水平力
            force_y = norm_y * force_z * 0.2
        else:
            force_x = force_y = 0
            
        force = [force_x, force_y, force_z]
        
        # 剪切力计算，使用更小的剪切力
        shear_scale = 0.1  # 减小剪切力比例
        shear = [
            force_x * shear_scale,
            force_y * shear_scale
        ]
        
        logger.info(f"触觉数据生成完成: tactile_image shape={tactile_image.shape}")
        
        return {
            'tactile_image': tactile_image,
            'force': force,
            'shear': shear
        }

    def _add_noise(self, depth_map: np.ndarray) -> np.ndarray:
        """为深度图添加噪声
        
        Args:
            depth_map (np.ndarray): 原始深度图
            
        Returns:
            np.ndarray: 添加噪声后的深度图
        """
        noise = np.random.normal(0, 0.05, depth_map.shape)
        return depth_map + noise

    def _generate_texture(self, image: np.ndarray) -> np.ndarray:
        """生成纹理
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 生成纹理后的图像
        """
        # 生成随机纹理
        texture = np.random.randint(0, 255, size=image.shape, dtype=np.uint8)
        return texture 