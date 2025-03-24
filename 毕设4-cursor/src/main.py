import cv2
import numpy as np
import yaml
import time
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from utils.common import ConfigLoader, Logger, DataCache, PerformanceMonitor
from sensors.vision.camera import RealSenseCamera
from sensors.vision.processor import VisionProcessor
from sensors.tactile.sensor import GelSightMini
from sensors.tactile.processor import TactileProcessor
from processing.data_processor import DataProcessor
from modeling.object_model import FlexibleObjectModel
from control.gripper import GripperController
from utils.calibration import Calibration
from utils.data_logger import DataLogger
from ml.model import GraspNet
from ml.predictor import GraspPredictor
from ml.trainer import ModelTrainer

class GraspingSystem:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化抓取系统
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self.logger = Logger.setup_logger('GraspingSystem')
        
        # 加载配置
        self.config = ConfigLoader.load_config(config_path)
        
        # 初始化性能监控
        self.monitor = PerformanceMonitor()
        
        # 初始化数据缓存
        self.cache = DataCache(max_size=self.config['data_processing']['cache_size'])
        
        # 初始化标定工具
        self.calibration = Calibration(config_path)
        
        # 初始化传感器
        self.camera = RealSenseCamera(config_path)
        self.tactile = GelSightMini(config_path)
        
        # 初始化处理器
        self.vision_processor = VisionProcessor(config_path)
        self.tactile_processor = TactileProcessor(config_path)
        self.data_processor = DataProcessor(config_path)
        
        # 初始化对象模型
        self.object_model = FlexibleObjectModel(config_path)
        
        # 初始化夹爪控制器
        self.gripper = GripperController(config_path)
        
        # 初始化数据记录器
        self.data_logger = DataLogger(config_path)
        
        # 初始化机器学习模型
        self._init_ml_model()
        
        # 系统状态
        self.is_running = False
        self.is_calibrated = False
        
    def _init_ml_model(self):
        """初始化机器学习模型"""
        try:
            # 创建模型
            self.model = GraspNet(
                input_size=tuple(self.config['ml']['model']['input_size'])
            )
            
            # 创建预测器
            self.predictor = GraspPredictor(
                model=self.model,
                config=self.config
            )
            
            # 创建训练器
            self.trainer = ModelTrainer(
                model=self.model,
                config=self.config
            )
            
            self.logger.info("机器学习模型初始化成功")
            
        except Exception as e:
            self.logger.error(f"机器学习模型初始化失败: {str(e)}")
            raise
            
    def start(self) -> bool:
        """启动系统
        
        Returns:
            是否启动成功
        """
        try:
            # 连接传感器
            if not self.camera.connect():
                self.logger.error("相机连接失败")
                return False
                
            if not self.tactile.connect():
                self.logger.error("触觉传感器连接失败")
                return False
                
            # 连接夹爪
            if not self.gripper.connect():
                self.logger.error("夹爪连接失败")
                return False
                
            # 启动数据处理器
            self.data_processor.start_processing()
            
            # 启动数据记录
            self.data_logger.start()
            
            self.is_running = True
            self.logger.info("系统启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"系统启动错误: {str(e)}")
            return False
            
    def stop(self):
        """停止系统"""
        try:
            # 停止数据处理器
            self.data_processor.stop_processing()
            
            # 停止数据记录
            self.data_logger.stop()
            
            # 断开连接
            self.camera.disconnect()
            self.tactile.disconnect()
            self.gripper.disconnect()
            
            self.is_running = False
            self.logger.info("系统已停止")
            
        except Exception as e:
            self.logger.error(f"系统停止错误: {str(e)}")
            
    def calibrate(self) -> bool:
        """系统标定
        
        Returns:
            是否标定成功
        """
        try:
            self.monitor.start_timer('calibration')
            
            # 相机标定
            self.logger.info("开始相机标定...")
            images = []
            for _ in range(10):  # 采集10张标定板图像
                frame = self.camera.get_frame()
                if frame is not None:
                    images.append(frame)
                time.sleep(0.5)
                
            if not self.calibration.calibrate_camera(images):
                self.logger.error("相机标定失败")
                return False
                
            # 相机到机器人标定
            self.logger.info("开始相机到机器人标定...")
            camera_poses = []
            robot_poses = []
            
            # 采集多组位姿数据
            for _ in range(5):
                # 获取相机位姿
                camera_pose = self.camera.get_pose()
                if camera_pose is not None:
                    camera_poses.append(camera_pose)
                    
                # 获取机器人位姿
                robot_pose = self.gripper.get_pose()
                if robot_pose is not None:
                    robot_poses.append(robot_pose)
                    
                time.sleep(1.0)
                
            if not self.calibration.calibrate_camera_to_robot(camera_poses, robot_poses):
                self.logger.error("相机到机器人标定失败")
                return False
                
            # 触觉传感器到夹爪标定
            self.logger.info("开始触觉传感器到夹爪标定...")
            tactile_poses = []
            gripper_poses = []
            
            # 采集多组位姿数据
            for _ in range(5):
                # 获取触觉传感器位姿
                tactile_pose = self.tactile.get_pose()
                if tactile_pose is not None:
                    tactile_poses.append(tactile_pose)
                    
                # 获取夹爪位姿
                gripper_pose = self.gripper.get_pose()
                if gripper_pose is not None:
                    gripper_poses.append(gripper_pose)
                    
                time.sleep(1.0)
                
            if not self.calibration.calibrate_tactile_to_gripper(tactile_poses, gripper_poses):
                self.logger.error("触觉传感器到夹爪标定失败")
                return False
                
            # 保存标定数据
            self.calibration.save_calibration('data/calibration.yaml')
            
            self.is_calibrated = True
            self.logger.info("系统标定完成")
            
            self.monitor.stop_timer('calibration')
            calibration_time = self.monitor.get_average_time('calibration')
            self.logger.info(f"标定耗时: {calibration_time:.2f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"系统标定错误: {str(e)}")
            return False
            
    def run(self):
        """运行系统"""
        if not self.is_running:
            self.logger.error("系统未启动")
            return
            
        if not self.is_calibrated:
            self.logger.error("系统未标定")
            return
            
        try:
            while self.is_running:
                self.monitor.start_timer('grasp_cycle')
                
                # 获取传感器数据
                depth_image, color_image = self.camera.get_frames()
                tactile_image = self.tactile.get_image()
                
                if depth_image is None or color_image is None or tactile_image is None:
                    self.logger.warning("获取传感器数据失败")
                    continue
                    
                # 处理视觉数据
                processed_vision = self.vision_processor.process_image(color_image)
                
                # 处理触觉数据
                processed_tactile = self.tactile_processor.process_image(tactile_image)
                
                if processed_vision is None or processed_tactile is None:
                    self.logger.warning("数据处理失败")
                    continue
                    
                # 提取特征
                vision_features = self.vision_processor.extract_features(processed_vision)
                tactile_features = self.tactile_processor.extract_features(processed_tactile)
                
                # 更新对象模型
                self.object_model.update_model(
                    pointcloud=self.camera.get_pointcloud(depth_image),
                    tactile_data={
                        'image': processed_tactile,
                        'features': tactile_features
                    },
                    force_data=tactile_features.get('pressure_distribution', {})
                )
                
                # 预测抓取参数
                grasp_params, quality_score = self.predictor.predict(
                    processed_vision,
                    processed_tactile
                )
                
                if grasp_params is None:
                    self.logger.warning("未找到合适的抓取点")
                    continue
                    
                # 转换坐标
                grasp_point_robot = self.calibration.transform_point(
                    grasp_params['position'],
                    'camera',
                    'robot'
                )
                
                # 计算夹爪参数
                gripper_params = self._calculate_gripper_params(
                    grasp_point_robot,
                    grasp_params['direction'],
                    quality_score
                )
                
                # 执行抓取
                success = self._execute_grasp(gripper_params)
                
                # 记录数据
                self.data_logger.log_data({
                    'vision': {
                        'depth': depth_image,
                        'color': color_image,
                        'processed': processed_vision,
                        'features': vision_features
                    },
                    'tactile': {
                        'image': tactile_image,
                        'processed': processed_tactile,
                        'features': tactile_features
                    },
                    'grasp_point': grasp_params,
                    'grasp_result': success
                })
                
                self.monitor.stop_timer('grasp_cycle')
                cycle_time = self.monitor.get_average_time('grasp_cycle')
                
                if cycle_time > self.config['data_processing']['max_processing_time']:
                    self.logger.warning(f"抓取周期过长: {cycle_time:.2f}秒")
                    
                time.sleep(1.0 / self.config['fusion']['update_rate'])
                
        except Exception as e:
            self.logger.error(f"系统运行错误: {str(e)}")
            self.stop()
            
    def _calculate_gripper_params(self,
                                position: np.ndarray,
                                direction: np.ndarray,
                                quality_score: float) -> Dict:
        """计算夹爪参数
        
        Args:
            position: 抓取位置
            direction: 抓取方向
            quality_score: 质量分数
            
        Returns:
            夹爪参数字典
        """
        # 根据质量分数调整夹持力
        force = self.config['control']['min_force'] + \
                (self.config['control']['max_force'] - self.config['control']['min_force']) * \
                quality_score
                
        # 根据方向计算夹爪角度
        angle = np.arctan2(direction[1], direction[0])
        
        return {
            'position': position,
            'angle': angle,
            'force': force,
            'width': self.config['control']['max_width'] * 0.8
        }
        
    def _execute_grasp(self, params: Dict) -> bool:
        """执行抓取动作
        
        Args:
            params: 夹爪参数
            
        Returns:
            是否抓取成功
        """
        try:
            # 移动到抓取位置
            if not self.gripper.move_to(params['position']):
                return False
                
            # 设置夹爪角度
            if not self.gripper.set_angle(params['angle']):
                return False
                
            # 设置夹持力
            if not self.gripper.set_force(params['force']):
                return False
                
            # 执行抓取
            if not self.gripper.grasp(params['width']):
                return False
                
            # 等待抓取完成
            time.sleep(0.5)
            
            # 检查抓取状态
            status = self.gripper.get_status()
            return status['is_grasped']
            
        except Exception as e:
            self.logger.error(f"抓取执行错误: {str(e)}")
            return False
            
    def train_model(self,
                   train_data: Dict,
                   val_data: Dict,
                   num_epochs: Optional[int] = None) -> Dict:
        """训练模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            num_epochs: 训练轮数
            
        Returns:
            训练历史
        """
        try:
            self.monitor.start_timer('training')
            
            # 训练模型
            history = self.trainer.train(
                train_data,
                val_data,
                num_epochs
            )
            
            self.monitor.stop_timer('training')
            training_time = self.monitor.get_average_time('training')
            self.logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            
            return history
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            raise

if __name__ == "__main__":
    # 创建系统实例
    system = GraspingSystem()
    
    # 启动系统
    if system.start():
        # 系统标定
        if system.calibrate():
            # 运行系统
            system.run()
        else:
            print("系统标定失败")
    else:
        print("系统启动失败")
        
    # 停止系统
    system.stop() 