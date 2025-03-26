import serial
import numpy as np
import cv2
import time
import logging
from typing import Optional, Tuple, Dict
import os

class GelSightMini:
    def __init__(self, port: str = "COM3", baud_rate: int = 115200):
        """初始化GelSight Mini传感器
        
        Args:
            port: 串口名称
            baud_rate: 波特率
        """
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None
        self.is_connected = False
        self.image_width = 640
        self.image_height = 480
        self.frame_rate = 30
        
        # 设置日志
        self.logger = logging.getLogger('GelSightMini')
        
        # 通信协议参数
        self.START_BYTE = 0xAA
        self.END_BYTE = 0x55
        self.PACKET_SIZE = self.image_width * self.image_height * 2 + 8  # 图像数据 + 头部和尾部
        
    def connect(self) -> bool:
        """连接传感器
        
        Returns:
            是否连接成功
        """
        try:
            if not os.path.exists(self.port):
                self.logger.error(f"串口 {self.port} 不存在")
                return False
                
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            
            # 等待传感器初始化
            time.sleep(1.0)
            
            # 发送初始化命令
            self._send_command(0x01)
            
            # 等待响应
            response = self._read_response()
            if response and response[0] == 0x01:
                self.is_connected = True
                self.logger.info("GelSight Mini连接成功")
                return True
            else:
                self.logger.error("GelSight Mini初始化失败")
                return False
                
        except serial.SerialException as e:
            self.logger.error(f"串口连接错误: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"GelSight Mini连接错误: {str(e)}")
            return False
        finally:
            if not self.is_connected and self.serial and self.serial.is_open:
                self.serial.close()
            
    def disconnect(self):
        """断开连接"""
        if self.serial and self.serial.is_open:
            self._send_command(0x02)  # 发送关闭命令
            self.serial.close()
            self.is_connected = False
            self.logger.info("GelSight Mini已断开连接")
            
    def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧数据
        
        Returns:
            触觉图像数据
        """
        if not self.is_connected:
            self.logger.error("GelSight Mini未连接")
            return None
            
        try:
            # 发送获取数据命令
            self._send_command(0x03)
            
            # 读取数据包
            packet = self._read_packet()
            if packet is None:
                return None
                
            # 解析数据
            image_data = self._parse_packet(packet)
            if image_data is None:
                return None
                
            # 转换为图像
            image = np.frombuffer(image_data, dtype=np.uint16)
            image = image.reshape((self.image_height, self.image_width))
            
            # 转换为8位图像
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return image
            
        except Exception as e:
            self.logger.error(f"获取GelSight Mini数据错误: {str(e)}")
            return None
            
    def _send_command(self, command: int):
        """发送命令
        
        Args:
            command: 命令字节
        """
        packet = bytearray([self.START_BYTE, command, self.END_BYTE])
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
            
        if response[0] != self.START_BYTE or response[-1] != self.END_BYTE:
            return None
            
        return response[1:-1]
        
    def _read_packet(self) -> Optional[bytearray]:
        """读取数据包
        
        Returns:
            数据包
        """
        # 等待起始字节
        while True:
            if not self.serial.in_waiting:
                return None
            if self.serial.read(1)[0] == self.START_BYTE:
                break
                
        # 读取数据
        packet = bytearray([self.START_BYTE])
        while len(packet) < self.PACKET_SIZE:
            if not self.serial.in_waiting:
                return None
            packet.extend(self.serial.read(1))
            
        # 检查结束字节
        if packet[-1] != self.END_BYTE:
            return None
            
        return packet
        
    def _parse_packet(self, packet: bytearray) -> Optional[bytearray]:
        """解析数据包
        
        Args:
            packet: 数据包
            
        Returns:
            图像数据
        """
        if len(packet) != self.PACKET_SIZE:
            return None
            
        # 提取图像数据
        image_data = packet[2:-6]  # 跳过头部和校验和
        
        # 验证校验和
        checksum = sum(packet[:-4]) & 0xFFFFFFFF
        received_checksum = int.from_bytes(packet[-4:], byteorder='little')
        
        if checksum != received_checksum:
            self.logger.warning("数据包校验和错误")
            return None
            
        return image_data
        
    def get_status(self) -> Dict:
        """获取传感器状态
        
        Returns:
            状态字典
        """
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baud_rate': self.baud_rate,
            'image_size': (self.image_width, self.image_height),
            'frame_rate': self.frame_rate
        } 