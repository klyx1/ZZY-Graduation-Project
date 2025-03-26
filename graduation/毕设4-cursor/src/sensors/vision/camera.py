import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import yaml
import logging

class RealSenseCamera:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化RealSense相机
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self.logger = logging.getLogger('RealSenseCamera')
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.camera_config = self.config['vision']
        
        # 相机参数
        self.width = self.camera_config.get('width', 640)
        self.height = self.camera_config.get('height', 480)
        self.fps = self.camera_config.get('fps', 30)
        self.min_depth = self.camera_config.get('min_depth', 0.1)
        self.max_depth = self.camera_config.get('max_depth', 4.0)
        
        # 初始化变量
        self.pipeline = None
        self.device = None
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.depth_to_color_extrinsics = None
        self.is_connected = False
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def connect(self) -> bool:
        """连接相机
        
        Returns:
            是否连接成功
        """
        try:
            # 创建管道
            self.pipeline = rs.pipeline()
            
            # 配置流
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # 启动管道
            self.pipeline.start(config)
            
            # 获取设备
            self.device = self.pipeline.get_active_profile().get_device()
            
            # 获取深度比例
            depth_sensor = self.device.first_depth_sensor()
            if depth_sensor:
                self.depth_scale = depth_sensor.get_depth_scale()
            
            # 获取内参和外参
            self._get_intrinsics()
            
            self.logger.info("相机连接成功")
            self.is_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"相机连接失败: {str(e)}")
            return False
            
    def disconnect(self):
        """断开相机连接"""
        if self.pipeline and self.is_connected:
            try:
                self.pipeline.stop()
                self.logger.info("相机已断开连接")
            except Exception as e:
                self.logger.error(f"断开相机连接失败: {str(e)}")
        self.is_connected = False
            
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取深度图和彩色图
        
        Returns:
            深度图和彩色图的元组
        """
        if not self.is_connected:
            self.logger.error("相机未连接")
            return None, None
            
        try:
            # 等待一帧数据
            frames = self.pipeline.wait_for_frames()
            
            # 获取深度图和彩色图
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 限制深度范围
            depth_meters = depth_image * self.depth_scale
            depth_meters = np.clip(depth_meters, self.min_depth, self.max_depth)
            depth_image = (depth_meters / self.depth_scale).astype(np.uint16)
            
            return depth_image, color_image
            
        except Exception as e:
            self.logger.error(f"获取图像失败: {str(e)}")
            return None, None
            
    def get_pointcloud(self, depth_frame: rs.frame) -> np.ndarray:
        """从深度图生成点云
        
        Args:
            depth_frame: 深度图帧
            
        Returns:
            点云数据 (N, 3)
        """
        if not self.is_connected:
            self.logger.error("相机未连接")
            return None
            
        try:
            # 获取当前帧
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
                
            # 创建点云处理器
            pc = rs.pointcloud()
            points = rs.points()
            
            # 处理深度图
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            
            # 转换为numpy数组
            vertices = np.asanyarray(points.get_vertices())
            vertices_array = vertices.view(np.float32).reshape(-1, 3)
            
            # 过滤无效点
            valid_mask = np.all(np.isfinite(vertices_array), axis=1)
            valid_points = vertices_array[valid_mask]
            
            # 限制深度范围
            z_coords = valid_points[:, 2]
            valid_mask = (z_coords >= self.min_depth) & (z_coords <= self.max_depth)
            valid_points = valid_points[valid_mask]
            
            return valid_points
            
        except Exception as e:
            self.logger.error(f"生成点云失败: {str(e)}")
            return None
            
    def _get_intrinsics(self):
        """获取相机内参和外参"""
        try:
            # 获取深度流和彩色流
            depth_stream = self.pipeline.get_active_profile().get_stream(rs.stream.depth)
            color_stream = self.pipeline.get_active_profile().get_stream(rs.stream.color)
            
            # 获取内参
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # 获取外参
            self.depth_to_color_extrinsics = depth_stream.as_video_stream_profile().get_extrinsics_to(color_stream)
            
        except Exception as e:
            self.logger.error(f"获取相机参数失败: {str(e)}")
            
    def get_intrinsics(self) -> Dict:
        """获取相机内参
        
        Returns:
            内参字典
        """
        return {
            'depth': {
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy,
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'model': self.depth_intrinsics.model,
                'coeffs': self.depth_intrinsics.coeffs
            },
            'color': {
                'width': self.color_intrinsics.width,
                'height': self.color_intrinsics.height,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy,
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'model': self.color_intrinsics.model,
                'coeffs': self.color_intrinsics.coeffs
            }
        }
        
    def get_extrinsics(self) -> Dict:
        """获取相机外参
        
        Returns:
            外参字典
        """
        return {
            'rotation': self.depth_to_color_extrinsics.rotation,
            'translation': self.depth_to_color_extrinsics.translation
        }