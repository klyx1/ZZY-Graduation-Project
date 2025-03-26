# 多传感器融合柔性物体抓取系统

## 项目简介
本项目是一个基于多传感器融合的柔性物体抓取系统，集成了视觉传感器（RealSense D435i）、触觉传感器（GelSight Mini）和机械夹爪（AG-160-95）。系统通过机器学习方法实现智能抓取决策，能够处理柔性物体如布料、软管等。

## 主要功能
- 多传感器数据采集与融合
- 实时数据处理与分析
- 智能抓取决策
- 自适应夹持力控制
- 系统标定与坐标转换
- 数据记录与分析

## 系统架构
```
project/
├── config/                 # 配置文件目录
│   └── config.yaml        # 系统配置文件
├── src/                   # 源代码目录
│   ├── sensors/          # 传感器模块
│   │   ├── vision/       # 视觉传感器
│   │   │   ├── __init__.py
│   │   │   ├── camera.py # 相机控制
│   │   │   └── processor.py # 图像处理
│   │   └── tactile/      # 触觉传感器
│   │       ├── __init__.py
│   │       ├── sensor.py # 传感器控制
│   │       └── processor.py # 触觉数据处理
│   ├── processing/       # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_processor.py # 数据融合处理
│   │   └── point_cloud.py # 点云处理
│   ├── modeling/        # 对象建模模块
│   │   ├── __init__.py
│   │   └── object_model.py # 物体状态建模
│   ├── control/         # 控制模块
│   │   ├── __init__.py
│   │   ├── gripper.py   # 夹爪控制
│   │   └── safety.py    # 安全控制
│   ├── planning/        # 规划模块
│   │   ├── __init__.py
│   │   └── grasp_strategy.py # 抓取策略
│   ├── fusion/          # 数据融合模块
│   │   ├── __init__.py
│   │   └── fusion.py    # 传感器数据融合
│   ├── visualization/   # 可视化模块
│   │   ├── __init__.py
│   │   └── gui.py       # 图形界面
│   ├── ml/             # 机器学习模块
│   │   ├── __init__.py
│   │   ├── model.py     # 模型定义
│   │   ├── trainer.py   # 模型训练
│   │   └── predictor.py # 模型预测
│   ├── utils/          # 工具模块
│   │   ├── __init__.py
│   │   ├── common.py    # 通用工具
│   │   ├── calibration.py # 标定工具
│   │   └── data_logger.py # 数据记录
│   └── main.py         # 主程序入口
├── tests/              # 测试代码目录
│   ├── __init__.py
│   ├── test_system.py
│   ├── test_vision.py
│   ├── test_tactile.py
│   └── test_gripper.py
├── data/               # 数据存储目录
│   ├── logs/          # 日志文件
│   ├── models/        # 模型文件
│   ├── calibration/   # 标定数据
│   └── training/      # 训练数据
├── requirements.txt   # 项目依赖
└── README.md         # 项目说明文档
```

## 环境要求
- Python 3.8+
- CUDA 11.0+ (用于GPU加速)
- 操作系统：Windows 10/11, Linux
- 硬件要求：
  - RealSense D457相机
  - GelSight Mini触觉传感器
  - AG-160-95机械夹爪
  - USB 3.0接口

## 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/flexible-grasping.git
cd flexible-grasping
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置系统
- 复制`config/config.yaml.example`为`config/config.yaml`
- 根据实际硬件配置修改参数

## 使用说明

1. 启动系统
```bash
python src/main.py
```

2. 系统标定
- 系统首次运行时会自动进行标定
- 标定过程包括：
  - 相机标定
  - 相机到机器人标定
  - 触觉传感器到夹爪标定

3. 模型训练
```python
from src.main import GraspingSystem

system = GraspingSystem()
history = system.train_model(train_data, val_data)
```

4. 数据记录
- 系统运行时会自动记录：
  - 传感器数据
  - 处理结果
  - 抓取参数
  - 性能指标

## 配置说明
主要配置参数（config/config.yaml）：
```yaml
system:
  name: "柔性物体抓取系统"
  version: "1.0.0"
  debug: false

sensors:
  camera:
    type: "RealSense D435i"
    width: 1280
    height: 720
    fps: 30
    
  tactile:
    type: "GelSight Mini"
    width: 640
    height: 480
    fps: 30

ml:
  model:
    type: "GraspNet"
    input_size: [224, 224]
    batch_size: 32
    learning_rate: 0.001
```

## 常见问题

1. 硬件连接问题
- 检查USB连接
- 确认设备驱动安装
- 验证设备权限

2. 标定失败
- 确保标定板位置正确
- 检查标定数据质量
- 验证标定参数设置

3. 性能问题
- 检查GPU使用情况
- 优化数据处理流程
- 调整批处理大小

## 开发计划
- [ ] 添加更多传感器支持
- [ ] 优化机器学习模型
- [ ] 改进抓取策略
- [ ] 添加可视化界面

## 贡献指南
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证
MIT License 