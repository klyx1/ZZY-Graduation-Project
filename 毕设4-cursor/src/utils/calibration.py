import numpy as np
import cv2
import open3d as o3d
from typing import Dict, List, Tuple, Optional
import yaml
import logging
import os

class Calibration:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化标定工具
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self.logger = logging.getLogger('Calibration')
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 标定参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_to_robot = None
        self.tactile_to_gripper = None
        
        # 标定数据
        self.calibration_data = {
            'camera': [],
            'tactile': [],
            'robot': []
        }
        
    def calibrate_camera(self, 
                        images: List[np.ndarray],
                        pattern_size: Tuple[int, int] = (9, 6),
                        square_size: float = 0.025) -> bool:
        """相机标定
        
        Args:
            images: 标定板图像列表
            pattern_size: 标定板角点数量
            square_size: 标定板方格大小(m)
            
        Returns:
            是否标定成功
        """
        try:
            # 准备标定数据
            obj_points = []
            img_points = []
            obj_p = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            obj_p[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            
            # 提取角点
            for img in images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
                
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    obj_points.append(obj_p)
                    img_points.append(corners2)
                    
            if not obj_points:
                self.logger.error("未找到足够的角点")
                return False
                
            # 相机标定
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, gray.shape[::-1], None, None
            )
            
            if not ret:
                self.logger.error("相机标定失败")
                return False
                
            # 保存标定结果
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            
            # 保存标定数据
            self.calibration_data['camera'] = {
                'matrix': mtx.tolist(),
                'dist_coeffs': dist.tolist(),
                'rvecs': [r.tolist() for r in rvecs],
                'tvecs': [t.tolist() for t in tvecs]
            }
            
            self.logger.info("相机标定成功")
            return True
            
        except Exception as e:
            self.logger.error(f"相机标定错误: {str(e)}")
            return False
            
    def calibrate_camera_to_robot(self,
                                 camera_poses: List[np.ndarray],
                                 robot_poses: List[np.ndarray]) -> bool:
        """相机到机器人的标定
        
        Args:
            camera_poses: 相机位姿列表
            robot_poses: 机器人位姿列表
            
        Returns:
            是否标定成功
        """
        try:
            # 计算变换矩阵
            camera_poses = np.array(camera_poses)
            robot_poses = np.array(robot_poses)
            
            # 使用SVD求解AX=YB问题
            A = np.zeros((4, 4))
            B = np.zeros((4, 4))
            
            for i in range(len(camera_poses)):
                A += np.outer(camera_poses[i], robot_poses[i])
                B += np.outer(robot_poses[i], robot_poses[i])
                
            # 求解变换矩阵
            U, _, Vh = np.linalg.svd(A)
            X = np.dot(Vh.T, U.T)
            
            # 保存变换矩阵
            self.camera_to_robot = X
            
            # 保存标定数据
            self.calibration_data['camera_to_robot'] = {
                'matrix': X.tolist(),
                'camera_poses': [p.tolist() for p in camera_poses],
                'robot_poses': [p.tolist() for p in robot_poses]
            }
            
            self.logger.info("相机到机器人标定成功")
            return True
            
        except Exception as e:
            self.logger.error(f"相机到机器人标定错误: {str(e)}")
            return False
            
    def calibrate_tactile_to_gripper(self,
                                   tactile_poses: List[np.ndarray],
                                   gripper_poses: List[np.ndarray]) -> bool:
        """触觉传感器到夹爪的标定
        
        Args:
            tactile_poses: 触觉传感器位姿列表
            gripper_poses: 夹爪位姿列表
            
        Returns:
            是否标定成功
        """
        try:
            # 计算变换矩阵
            tactile_poses = np.array(tactile_poses)
            gripper_poses = np.array(gripper_poses)
            
            # 使用SVD求解AX=YB问题
            A = np.zeros((4, 4))
            B = np.zeros((4, 4))
            
            for i in range(len(tactile_poses)):
                A += np.outer(tactile_poses[i], gripper_poses[i])
                B += np.outer(gripper_poses[i], gripper_poses[i])
                
            # 求解变换矩阵
            U, _, Vh = np.linalg.svd(A)
            X = np.dot(Vh.T, U.T)
            
            # 保存变换矩阵
            self.tactile_to_gripper = X
            
            # 保存标定数据
            self.calibration_data['tactile_to_gripper'] = {
                'matrix': X.tolist(),
                'tactile_poses': [p.tolist() for p in tactile_poses],
                'gripper_poses': [p.tolist() for p in gripper_poses]
            }
            
            self.logger.info("触觉传感器到夹爪标定成功")
            return True
            
        except Exception as e:
            self.logger.error(f"触觉传感器到夹爪标定错误: {str(e)}")
            return False
            
    def transform_point(self,
                       point: np.ndarray,
                       from_frame: str,
                       to_frame: str) -> np.ndarray:
        """坐标转换
        
        Args:
            point: 输入点
            from_frame: 源坐标系
            to_frame: 目标坐标系
            
        Returns:
            转换后的点
        """
        try:
            # 添加齐次坐标
            point_homo = np.append(point, 1)
            
            # 根据坐标系选择变换矩阵
            if from_frame == 'camera' and to_frame == 'robot':
                transform = self.camera_to_robot
            elif from_frame == 'tactile' and to_frame == 'gripper':
                transform = self.tactile_to_gripper
            else:
                self.logger.error(f"不支持的坐标系转换: {from_frame} -> {to_frame}")
                return point
                
            # 应用变换
            transformed_point = np.dot(transform, point_homo)
            
            # 移除齐次坐标
            return transformed_point[:3]
            
        except Exception as e:
            self.logger.error(f"坐标转换错误: {str(e)}")
            return point
            
    def save_calibration(self, file_path: str):
        """保存标定数据
        
        Args:
            file_path: 保存路径
        """
        try:
            with open(file_path, 'w') as f:
                yaml.dump(self.calibration_data, f)
            self.logger.info(f"标定数据已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存标定数据错误: {str(e)}")
            
    def load_calibration(self, file_path: str) -> bool:
        """加载标定数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否加载成功
        """
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # 恢复标定参数
            if 'camera' in data:
                self.camera_matrix = np.array(data['camera']['matrix'])
                self.dist_coeffs = np.array(data['camera']['dist_coeffs'])
                
            if 'camera_to_robot' in data:
                self.camera_to_robot = np.array(data['camera_to_robot']['matrix'])
                
            if 'tactile_to_gripper' in data:
                self.tactile_to_gripper = np.array(data['tactile_to_gripper']['matrix'])
                
            self.calibration_data = data
            self.logger.info(f"标定数据已从 {file_path} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载标定数据错误: {str(e)}")
            return False
            
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)