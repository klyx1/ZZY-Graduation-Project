# AG夹爪Modbus控制
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
import logging


class AGGripper:
    """ AG-95电动夹爪高级控制类 """

    def __init__(self, port: str = 'COM3', baudrate: int = 115200, slave_id: int = 1):
        """
        :param port: 串口路径（Windows: COM3, Linux: /dev/ttyUSB0）
        :param baudrate: 波特率（默认115200）
        :param slave_id: Modbus从站ID
        """
        self.client = ModbusSerialClient(
            method='rtu',
            port=port,
            baudrate=baudrate,
            timeout=0.2
        )
        self.slave_id = slave_id
        self.logger = logging.getLogger('gripper')
        self.connect()

    def connect(self) -> bool:
        """ 连接夹爪并验证通信 """
        if not self.client.connect():
            self.logger.error("夹爪连接失败")
            return False

        # 发送测试指令（读取设备信息）
        try:
            response = self.client.read_holding_registers(0x1000, 2, slave=self.slave_id)
            if response.isError():
                raise ModbusException("夹爪响应错误")
            self.logger.info(f"夹爪连接成功，设备ID: {response.registers[0]:04X}")
            return True
        except ModbusException as e:
            self.logger.error(f"夹爪通信异常: {str(e)}")
            return False

    def set_position(self, position: float, speed: float = 50.0) -> None:
        """
        设置夹爪目标位置（开度）
        :param position: 目标位置（0-100，0为全闭）
        :param speed: 运动速度（0-100）
        """
        if not 0 <= position <= 100 or not 0 <= speed <= 100:
            raise ValueError("参数超出有效范围")

        # 位置映射到寄存器值（单位0.1%）
        pos_reg = int(position * 10)
        speed_reg = int(speed * 10)

        # 写入目标位置和速度
        self.client.write_registers(
            address=0x2000,
            values=[pos_reg, speed_reg],
            slave=self.slave_id
        )

    def get_status(self) -> dict:
        """ 读取夹爪当前状态 """
        response = self.client.read_holding_registers(0x3000, 3, slave=self.slave_id)
        if response.isError():
            raise ModbusException("状态读取失败")

        return {
            'position': response.registers[0] / 10.0,  # 当前开度（%）
            'current': response.registers[1] / 1000.0,  # 电流（A）
            'temperature': response.registers[2]  # 温度（℃）
        }