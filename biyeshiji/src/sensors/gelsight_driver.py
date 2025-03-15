# GelSight触觉传感器驱动
import cv2
import numpy as np
from typing import Optional


class GelSightCamera:
    """ GelSight Mini触觉传感器驱动类 """

    def __init__(self, device_name: str = "GelSight Mini"):
        """
        初始化触觉摄像头
        :param device_name: 设备名称（用于自动查找摄像头ID）
        """
        self.device_name = device_name
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_camera()

    def _init_camera(self) -> None:
        """ 根据设备名称查找并初始化摄像头 """
        # 遍历所有视频设备，查找匹配名称的摄像头
        for i in range(5):  # 假设最多检查前5个设备
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                # 通过摄像头属性获取设备名称（需Linux V4L2支持）
                device_info = cap.get(cv2.CAP_PROP_HW_DEVICE)
                if device_info and self.device_name in device_info:
                    self.cap = cap
                    print(f"找到GelSight摄像头：/dev/video{i}")
                    self._setup_camera_params()
                    return
                cap.release()
        raise RuntimeError(f"未找到设备名为 {self.device_name} 的摄像头")

    def _setup_camera_params(self) -> None:
        """ 配置摄像头参数（分辨率、帧率等） """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 100)  # 目标帧率100Hz

    def get_calibrated_frame(self) -> np.ndarray:
        """
        获取经过校准的触觉图像
        返回: (320,240,3)的BGR图像，已应用暗场校正
        """
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("无法从GelSight读取帧")
        # 应用暗场校正（需提前标定）
        calibrated = cv2.subtract(frame, self.dark_frame)
        return cv2.resize(calibrated, (320, 240))

    def close(self) -> None:
        """ 释放摄像头资源 """
        if self.cap:
            self.cap.release()