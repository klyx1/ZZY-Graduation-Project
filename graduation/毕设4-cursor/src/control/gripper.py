import serial
import time
import logging
import yaml
from typing import Optional, Dict, Tuple
import numpy as np

class GripperController:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化夹爪控制器
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self.logger = logging.getLogger('GripperController')
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.gripper_config = self.config.get('control', {})
        
        # 初始化参数
        self.port = self.gripper_config.get('gripper_port', 'COM4')
        self.baud_rate = self.gripper_config.get('baud_rate', 115200)
        self.max_force = self.gripper_config.get('max_force', 95)
        self.min_force = self.gripper_config.get('min_force', 10)
        self.max_width = self.gripper_config.get('max_width', 160)
        self.min_width = self.gripper_config.get('min_width', 0)
        
        # 通信参数
        self.serial = None
        self.is_connected = False
        
        # 命令定义
        self.CMD_INIT = 0x01
        self.CMD_SET_POSITION = 0x02
        self.CMD_SET_FORCE = 0x03
        self.CMD_GET_STATUS = 0x04
        self.CMD_EMERGENCY_STOP = 0x05
        
        # 状态变量
        self.current_position = 0
        self.current_force = 0
        self.is_moving = False
        self.is_grasping = False
        
    def connect(self) -> bool:
        """连接夹爪
        
        Returns:
            是否连接成功
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            
            # 等待夹爪初始化
            time.sleep(1.0)
            
            # 发送初始化命令
            self._send_command(self.CMD_INIT)
            
            # 等待响应
            response = self._read_response()
            if response and response[0] == 0x01:
                self.is_connected = True
                self.logger.info("夹爪连接成功")
                return True
            else:
                self.logger.error("夹爪初始化失败")
                return False
                
        except Exception as e:
            self.logger.error(f"夹爪连接错误: {str(e)}")
            return False
            
    def disconnect(self):
        """断开连接"""
        if self.serial and self.serial.is_open:
            self._send_command(self.CMD_EMERGENCY_STOP)  # 发送紧急停止命令
            self.serial.close()
            self.is_connected = False
            self.logger.info("夹爪已断开连接")
            
    def set_position(self, position: int) -> bool:
        """设置夹爪位置
        
        Args:
            position: 目标位置(mm)
            
        Returns:
            是否设置成功
        """
        if not self.is_connected:
            self.logger.error("夹爪未连接")
            return False
            
        # 检查位置范围
        position = max(self.min_width, min(position, self.max_width))
        
        try:
            # 构建命令
            cmd = bytearray([0xAA, self.CMD_SET_POSITION])
            cmd.extend(position.to_bytes(2, byteorder='little'))
            cmd.append(0x55)
            
            # 发送命令
            self.serial.write(cmd)
            
            # 等待响应
            response = self._read_response()
            if response and response[0] == 0x01:
                self.current_position = position
                self.is_moving = True
                return True
            else:
                self.logger.error("设置位置失败")
                return False
                
        except Exception as e:
            self.logger.error(f"设置位置错误: {str(e)}")
            return False
            
    def set_force(self, force: int) -> bool:
        """设置夹持力
        
        Args:
            force: 目标力值(N)
            
        Returns:
            是否设置成功
        """
        if not self.is_connected:
            self.logger.error("夹爪未连接")
            return False
            
        # 检查力值范围
        force = max(self.min_force, min(force, self.max_force))
        
        try:
            # 构建命令
            cmd = bytearray([0xAA, self.CMD_SET_FORCE])
            cmd.extend(force.to_bytes(2, byteorder='little'))
            cmd.append(0x55)
            
            # 发送命令
            self.serial.write(cmd)
            
            # 等待响应
            response = self._read_response()
            if response and response[0] == 0x01:
                self.current_force = force
                return True
            else:
                self.logger.error("设置力值失败")
                return False
                
        except Exception as e:
            self.logger.error(f"设置力值错误: {str(e)}")
            return False
            
    def get_status(self) -> Optional[Dict]:
        """获取夹爪状态
        
        Returns:
            状态字典
        """
        if not self.is_connected:
            self.logger.error("夹爪未连接")
            return None
            
        try:
            # 发送状态查询命令
            self._send_command(self.CMD_GET_STATUS)
            
            # 读取响应
            response = self._read_response()
            if response and len(response) >= 6:
                # 解析状态数据
                position = int.from_bytes(response[0:2], byteorder='little')
                force = int.from_bytes(response[2:4], byteorder='little')
                status = response[4]
                
                # 更新状态
                self.current_position = position
                self.current_force = force
                self.is_moving = bool(status & 0x01)
                self.is_grasping = bool(status & 0x02)
                
                return {
                    'position': position,
                    'force': force,
                    'is_moving': self.is_moving,
                    'is_grasping': self.is_grasping
                }
            else:
                self.logger.error("获取状态失败")
                return None
                
        except Exception as e:
            self.logger.error(f"获取状态错误: {str(e)}")
            return None
            
    def grasp(self, position: int, force: int) -> bool:
        """执行抓取动作
        
        Args:
            position: 目标位置(mm)
            force: 目标力值(N)
            
        Returns:
            是否抓取成功
        """
        if not self.is_connected:
            self.logger.error("夹爪未连接")
            return False
            
        try:
            # 设置位置
            if not self.set_position(position):
                return False
                
            # 等待到位
            while self.is_moving:
                time.sleep(0.1)
                status = self.get_status()
                if status is None:
                    return False
                    
            # 设置力值
            if not self.set_force(force):
                return False
                
            # 等待稳定
            time.sleep(0.5)
            
            # 检查抓取状态
            status = self.get_status()
            if status is None:
                return False
                
            return status['is_grasping']
            
        except Exception as e:
            self.logger.error(f"抓取错误: {str(e)}")
            return False
            
    def release(self) -> bool:
        """释放物体
        
        Returns:
            是否释放成功
        """
        if not self.is_connected:
            self.logger.error("夹爪未连接")
            return False
            
        try:
            # 设置最大开口
            if not self.set_position(self.max_width):
                return False
                
            # 等待到位
            while self.is_moving:
                time.sleep(0.1)
                status = self.get_status()
                if status is None:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"释放错误: {str(e)}")
            return False
            
    def emergency_stop(self) -> bool:
        """紧急停止
        
        Returns:
            是否停止成功
        """
        if not self.is_connected:
            self.logger.error("夹爪未连接")
            return False
            
        try:
            # 发送紧急停止命令
            self._send_command(self.CMD_EMERGENCY_STOP)
            
            # 等待响应
            response = self._read_response()
            if response and response[0] == 0x01:
                self.is_moving = False
                self.is_grasping = False
                return True
            else:
                self.logger.error("紧急停止失败")
                return False
                
        except Exception as e:
            self.logger.error(f"紧急停止错误: {str(e)}")
            return False
            
    def _send_command(self, command: int):
        """发送命令
        
        Args:
            command: 命令字节
        """
        packet = bytearray([0xAA, command, 0x55])
        self.serial.write(packet)
        
    def _read_response(self) -> Optional[bytearray]:
        """读取响应
        
        Returns:
            响应数据
        """
        if not self.serial.in_waiting:
            return None
            
        response = self.serial.read(self.serial.in_waiting)
        if len(response) < 3:
            return None
            
        if response[0] != 0xAA or response[-1] != 0x55:
            return None
            
        return response[1:-1]
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def adaptive_grasp(self,
                      initial_position: float,
                      max_force: float,
                      min_force: float,
                      force_step: float = 5.0,
                      timeout: float = 5.0) -> bool:
        """自适应抓取
        
        Args:
            initial_position: 初始位置
            max_force: 最大夹持力
            min_force: 最小夹持力
            force_step: 力调整步长
            timeout: 超时时间
            
        Returns:
            是否抓取成功
        """
        if not self.is_connected:
            return False
            
        # 从最小力开始尝试
        current_force = min_force
        while current_force <= max_force:
            # 尝试抓取
            if self.grasp(initial_position, current_force, timeout):
                # 检查是否稳定
                if self._check_stability():
                    return True
                    
            # 增加力
            current_force += force_step
            
        return False
    
    def _check_stability(self, duration: float = 1.0) -> bool:
        """检查抓取稳定性
        
        Args:
            duration: 检查持续时间
            
        Returns:
            是否稳定
        """
        if not self.is_connected:
            return False
            
        # 记录初始状态
        initial_status = self.get_status()
        if initial_status is None:
            return False
            
        # 等待一段时间
        time.sleep(duration)
        
        # 获取当前状态
        current_status = self.get_status()
        if current_status is None:
            return False
            
        # 检查位置和力的变化
        position_diff = abs(current_status['position'] - initial_status['position'])
        force_diff = abs(current_status['force'] - initial_status['force'])
        
        # 如果变化很小，认为稳定
        return position_diff < 0.1 and force_diff < 1.0 