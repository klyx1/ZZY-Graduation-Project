# RealSense视觉驱动
import pyrealsense2 as rs
import numpy as np
from typing import Tuple, Optional


class RealSenseCamera:
    """ Intel RealSense D435i 深度摄像头驱动（Windows兼容版） """

    def __init__(self, preset: str = 'high_accuracy'):
        """
        初始化深度摄像头
        :param preset: 预设配置（high_accuracy, high_density等）
        """
        # 配置深度和彩色流参数
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # 启动设备并应用预设
        self.profile: Optional[rs.pipeline_profile] = None
        self.align = rs.align(rs.stream.color)  # 对齐到彩色帧
        self._start_pipeline(config, preset)

    def _start_pipeline(self, config: rs.config, preset: str) -> None:
        """ 启动设备流并配置深度传感器参数 """
        self.profile = self.pipeline.start(config)
        depth_sensor = self.profile.get_device().first_depth_sensor()

        # 设置预设模式（需转换为RealSense内部枚举值）
        presets = {
            'high_accuracy': rs.rs400_visual_preset.high_accuracy,
            'high_density': rs.rs400_visual_preset.high_density
        }
        depth_sensor.set_option(rs.option.visual_preset, presets[preset])

        # 启用自动曝光（增强图像稳定性）
        color_sensor = self.profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)

    def get_aligned_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取对齐的RGB和深度帧
        返回:
            - rgb_frame: (720, 1280, 3) BGR图像
            - depth_frame: (480, 848) 单位毫米的深度图
        """
        if not self.profile:
            raise RuntimeError("摄像头未初始化")

        frames = self.pipeline.wait_for_frames(5000)  # 5秒超时
        aligned_frames = self.align.process(frames)

        # 提取对齐后的帧数据
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("无法获取对齐的帧数据")

        # 转换为Numpy数组并调整尺寸
        rgb_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(float)
        return rgb_image, depth_image * depth_frame.get_units() * 1000  # 转换为毫米

    def stop(self) -> None:
        """ 停止摄像头流并释放资源 """
        if self.pipeline:
            self.pipeline.stop()