# 自适应抓取算法
from pymodbus.client import ModbusSerialClient
import numpy as np


class AGGripperController:
    """ AG-95夹爪Modbus控制类 """

    def __init__(self, port: str = '/dev/ttyUSB0', slave_id: int = 1):
        """
        :param port: 串口路径（Linux: /dev/ttyUSB0, Windows: COM3）
        :param slave_id: Modbus从站ID
        """
        self.client = ModbusSerialClient(
            method='rtu',
            port=port,
            baudrate=115200,
            timeout=0.1
        )
        self.slave_id = slave_id
        self.connect()

    def connect(self) -> None:
        """ 连接夹爪设备 """
        if not self.client.connect():
            raise ConnectionError("无法连接夹爪，请检查端口和波特率")

    def set_force(self, force: float) -> None:
        """
        设置夹爪夹持力
        :param force: 目标力值（单位: N，范围0-100）
        """
        # 力值映射到Modbus寄存器（根据AG-95协议）
        reg_value = int(force * 100)  # 假设0-100N对应0-10000
        response = self.client.write_register(
            address=0x1001,
            value=reg_value,
            slave=self.slave_id
        )
        if response.isError():
            print("夹爪力设置错误:", response)

    def get_position(self) -> float:
        """ 读取夹爪当前位置（单位: mm） """
        response = self.client.read_holding_registers(
            address=0x2000,
            count=1,
            slave=self.slave_id
        )
        if not response.isError():
            return response.registers[0] / 10.0  # 寄存器值单位为0.1mm
        else:
            return -1.0