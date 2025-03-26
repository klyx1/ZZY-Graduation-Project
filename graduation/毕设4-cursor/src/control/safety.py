import numpy as np
from typing import Dict, Optional, Tuple
import yaml
import threading
import queue
import time

class SafetyController:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化安全控制器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.control_config = self.config['control']
        self.safety_config = self.config.get('safety', {})
        
        # 创建数据队列
        self.data_queue = queue.Queue()
        
        # 状态变量
        self.is_running = False
        self.emergency_stop = False
        self.last_check_time = 0
        
        # 力限制
        self.max_force = self.control_config['max_force']
        self.min_force = self.control_config['min_force']
        self.force_threshold = self.safety_config.get('force_threshold', 0.8)
        
        # 碰撞检测参数
        self.collision_threshold = self.safety_config.get('collision_threshold', 0.5)
        self.velocity_threshold = self.safety_config.get('velocity_threshold', 0.1)
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def start(self):
        """启动安全控制器"""
        self.is_running = True
        self.monitor_thread.start()
        
    def stop(self):
        """停止安全控制器"""
        self.is_running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def emergency_stop(self):
        """紧急停止"""
        self.emergency_stop = True
        self.is_running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def update_data(self, data: Dict):
        """更新数据
        
        Args:
            data: 传感器数据
        """
        self.data_queue.put(data)
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 从队列获取数据
                data = self.data_queue.get(timeout=1.0)
                
                # 检查安全状态
                safety_status = self._check_safety(data)
                
                # 如果检测到不安全状态,触发紧急停止
                if not safety_status['is_safe']:
                    self.emergency_stop()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"安全监控错误: {str(e)}")
                self.emergency_stop()
                break
                
    def _check_safety(self, data: Dict) -> Dict:
        """检查安全状态
        
        Args:
            data: 传感器数据
            
        Returns:
            安全状态字典
        """
        safety_status = {
            'is_safe': True,
            'warnings': [],
            'errors': []
        }
        
        # 检查力限制
        force_status = self._check_force_limits(data)
        if not force_status['is_safe']:
            safety_status['is_safe'] = False
            safety_status['warnings'].extend(force_status['warnings'])
            safety_status['errors'].extend(force_status['errors'])
            
        # 检查碰撞
        collision_status = self._check_collision(data)
        if not collision_status['is_safe']:
            safety_status['is_safe'] = False
            safety_status['warnings'].extend(collision_status['warnings'])
            safety_status['errors'].extend(collision_status['errors'])
            
        # 检查速度限制
        velocity_status = self._check_velocity(data)
        if not velocity_status['is_safe']:
            safety_status['is_safe'] = False
            safety_status['warnings'].extend(velocity_status['warnings'])
            safety_status['errors'].extend(velocity_status['errors'])
            
        return safety_status
        
    def _check_force_limits(self, data: Dict) -> Dict:
        """检查力限制
        
        Args:
            data: 传感器数据
            
        Returns:
            力检查状态
        """
        status = {
            'is_safe': True,
            'warnings': [],
            'errors': []
        }
        
        # 获取当前力
        current_force = data.get('force', 0.0)
        
        # 检查是否超过最大力
        if current_force > self.max_force * self.force_threshold:
            status['is_safe'] = False
            status['warnings'].append(f"力接近最大限制: {current_force:.1f}N")
            
        # 检查是否低于最小力
        if current_force < self.min_force:
            status['is_safe'] = False
            status['warnings'].append(f"力低于最小限制: {current_force:.1f}N")
            
        # 检查力的变化率
        if 'force_history' in data:
            force_history = data['force_history']
            if len(force_history) > 1:
                force_change = abs(force_history[-1] - force_history[-2])
                if force_change > self.max_force * 0.1:  # 力变化超过10%
                    status['is_safe'] = False
                    status['warnings'].append(f"力变化过大: {force_change:.1f}N")
                    
        return status
        
    def _check_collision(self, data: Dict) -> Dict:
        """检查碰撞
        
        Args:
            data: 传感器数据
            
        Returns:
            碰撞检查状态
        """
        status = {
            'is_safe': True,
            'warnings': [],
            'errors': []
        }
        
        # 获取位置和速度数据
        position = data.get('position', None)
        velocity = data.get('velocity', None)
        
        if position is not None and velocity is not None:
            # 检查位置是否在安全范围内
            if not self._is_position_safe(position):
                status['is_safe'] = False
                status['errors'].append("位置超出安全范围")
                
            # 检查速度是否在安全范围内
            if not self._is_velocity_safe(velocity):
                status['is_safe'] = False
                status['errors'].append("速度超出安全范围")
                
        return status
        
    def _check_velocity(self, data: Dict) -> Dict:
        """检查速度限制
        
        Args:
            data: 传感器数据
            
        Returns:
            速度检查状态
        """
        status = {
            'is_safe': True,
            'warnings': [],
            'errors': []
        }
        
        # 获取速度数据
        velocity = data.get('velocity', None)
        
        if velocity is not None:
            # 计算速度大小
            velocity_magnitude = np.linalg.norm(velocity)
            
            # 检查是否超过速度阈值
            if velocity_magnitude > self.velocity_threshold:
                status['is_safe'] = False
                status['warnings'].append(f"速度过大: {velocity_magnitude:.2f}m/s")
                
        return status
        
    def _is_position_safe(self, position: np.ndarray) -> bool:
        """检查位置是否安全
        
        Args:
            position: 位置数据
            
        Returns:
            是否安全
        """
        # 这里需要根据实际工作空间定义安全范围
        # 示例实现
        safe_range = {
            'x': (-0.5, 0.5),
            'y': (-0.5, 0.5),
            'z': (0.0, 0.5)
        }
        
        return (
            safe_range['x'][0] <= position[0] <= safe_range['x'][1] and
            safe_range['y'][0] <= position[1] <= safe_range['y'][1] and
            safe_range['z'][0] <= position[2] <= safe_range['z'][1]
        )
        
    def _is_velocity_safe(self, velocity: np.ndarray) -> bool:
        """检查速度是否安全
        
        Args:
            velocity: 速度数据
            
        Returns:
            是否安全
        """
        # 计算速度大小
        velocity_magnitude = np.linalg.norm(velocity)
        
        # 检查是否超过速度阈值
        return velocity_magnitude <= self.velocity_threshold
        
    def get_safety_status(self) -> Dict:
        """获取安全状态
        
        Returns:
            安全状态字典
        """
        return {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'last_check_time': self.last_check_time
        }
        
    def reset(self):
        """重置安全控制器"""
        self.emergency_stop = False
        self.last_check_time = time.time()
        
    def set_force_limits(self, min_force: float, max_force: float):
        """设置力限制
        
        Args:
            min_force: 最小力
            max_force: 最大力
        """
        self.min_force = min_force
        self.max_force = max_force
        
    def set_collision_threshold(self, threshold: float):
        """设置碰撞检测阈值
        
        Args:
            threshold: 碰撞阈值
        """
        self.collision_threshold = threshold
        
    def set_velocity_threshold(self, threshold: float):
        """设置速度阈值
        
        Args:
            threshold: 速度阈值
        """
        self.velocity_threshold = threshold 