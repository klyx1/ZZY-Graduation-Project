import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
from typing import Dict, Optional
import os
import sys
import threading
import queue
import logging
import traceback

# 设置日志
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.simulation.data_generator import SensorDataGenerator
from tests.simulation.fusion_sim import SensorFusionSim

class FusionVisualization:
    def __init__(self):
        """初始化可视化界面"""
        try:
            logger.info("初始化可视化界面")
            self.root = tk.Tk()
            self.root.title("传感器融合仿真可视化")
            self.root.geometry("1280x640")
            
            # 创建数据生成器和融合模块
            logger.info("创建数据生成器和融合模块")
            self.data_generator = SensorDataGenerator(
                image_size=(640, 480),
                noise_level=0.1
            )
            self.fusion_module = SensorFusionSim()
            self.fusion_module.start()
            
            # 创建数据队列
            self.data_queue = queue.Queue(maxsize=5)
            self.running = False
            
            # 创建界面组件
            self._create_widgets()
            
            # 创建数据生成线程
            self.data_thread = None
            logger.info("初始化完成")
            
        except Exception as e:
            logger.error(f"初始化错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _create_widgets(self):
        """创建界面组件"""
        try:
            logger.info("创建界面组件")
            # 创建主框架
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 创建图像显示区域
            self.image_frame = ttk.Frame(self.main_frame)
            self.image_frame.pack(fill=tk.BOTH, expand=True)
            
            # 创建深度图显示区域
            self.depth_frame = ttk.LabelFrame(self.image_frame, text="深度图")
            self.depth_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
            self.depth_label = tk.Label(self.depth_frame)
            self.depth_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 创建彩色图显示区域
            self.color_frame = ttk.LabelFrame(self.image_frame, text="彩色图")
            self.color_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
            self.color_label = tk.Label(self.color_frame)
            self.color_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 创建触觉图显示区域
            self.tactile_frame = ttk.LabelFrame(self.image_frame, text="触觉图")
            self.tactile_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
            self.tactile_label = tk.Label(self.tactile_frame)
            self.tactile_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 创建融合图显示区域
            self.fused_frame = ttk.LabelFrame(self.image_frame, text="融合图")
            self.fused_frame.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
            self.fused_label = tk.Label(self.fused_frame)
            self.fused_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 配置网格权重
            self.image_frame.grid_columnconfigure(0, weight=1)
            self.image_frame.grid_columnconfigure(1, weight=1)
            self.image_frame.grid_rowconfigure(0, weight=1)
            self.image_frame.grid_rowconfigure(1, weight=1)
            
            # 创建空白图像
            blank_image = np.zeros((300, 400, 3), dtype=np.uint8)
            blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
            blank_image = Image.fromarray(blank_image)
            blank_photo = ImageTk.PhotoImage(image=blank_image)
            
            # 设置所有图像标签的大小
            self.depth_label.config(image=blank_photo)
            self.color_label.config(image=blank_photo)
            self.tactile_label.config(image=blank_photo)
            self.fused_label.config(image=blank_photo)
            self.depth_label.image = blank_photo  # 保持引用
            self.color_label.image = blank_photo
            self.tactile_label.image = blank_photo
            self.fused_label.image = blank_photo
            
            # 创建信息显示区域
            self.info_frame = ttk.Frame(self.main_frame)
            self.info_frame.pack(fill=tk.X, pady=5)
            
            # 创建力信息显示
            self.force_label = ttk.Label(self.info_frame, text="力: 0.0 N")
            self.force_label.pack(side=tk.LEFT, padx=10)
            self.shear_label = ttk.Label(self.info_frame, text="剪切力: (0.0, 0.0) N")
            self.shear_label.pack(side=tk.LEFT, padx=10)
            
            # 创建特征信息显示
            self.features_label = ttk.Label(self.info_frame, text="特征信息: 无")
            self.features_label.pack(side=tk.LEFT, padx=10)
            
            # 创建控制按钮框架
            self.control_frame = ttk.Frame(self.root)
            self.control_frame.pack(fill=tk.X, pady=5)
            
            # 创建按钮样式
            style = ttk.Style()
            style.configure('Start.TButton', font=('Arial', 12, 'bold'))
            style.configure('Stop.TButton', font=('Arial', 12, 'bold'))
            
            # 创建启动按钮
            self.start_button = ttk.Button(
                self.control_frame,
                text="开始可视化",
                command=self._start_visualization,
                style='Start.TButton',
                width=20
            )
            self.start_button.pack(side=tk.LEFT, padx=10)
            
            # 创建停止按钮
            self.stop_button = ttk.Button(
                self.control_frame,
                text="停止可视化",
                command=self._stop_visualization,
                style='Stop.TButton',
                width=20,
                state=tk.DISABLED
            )
            self.stop_button.pack(side=tk.LEFT, padx=10)
            
        except Exception as e:
            logger.error(f"创建界面组件错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _start_visualization(self):
        """开始可视化"""
        try:
            logger.info("开始可视化")
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # 创建并启动数据生成线程
            self.data_thread = threading.Thread(target=self._data_generation_loop)
            self.data_thread.daemon = True
            self.data_thread.start()
            logger.info("数据生成线程已启动")
            
            # 启动更新循环
            self._update_loop()
            
        except Exception as e:
            logger.error(f"启动可视化错误: {str(e)}")
            logger.error(traceback.format_exc())
            self._stop_visualization()
            
    def _stop_visualization(self):
        """停止可视化"""
        try:
            logger.info("停止可视化")
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # 等待数据生成线程结束
            if self.data_thread and self.data_thread.is_alive():
                logger.info("等待数据生成线程结束")
                self.data_thread.join(timeout=1.0)
                if self.data_thread.is_alive():
                    logger.warning("数据生成线程未能正常结束")
                else:
                    logger.info("数据生成线程已结束")
                    
        except Exception as e:
            logger.error(f"停止可视化错误: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _data_generation_loop(self):
        """数据生成循环"""
        while self.running:
            try:
                # 生成视觉数据
                vision_data = self.data_generator.generate_vision_data()
                
                # 生成触觉数据
                tactile_data = self.data_generator.generate_tactile_data()
                
                # 获取融合数据
                logger.info("正在处理融合数据")
                fused_data = self.fusion_module.get_fused_data(vision_data, tactile_data)
                
                # 将数据放入队列
                data = {
                    'vision': vision_data,
                    'tactile': tactile_data,
                    'fused': fused_data
                }
                self.data_queue.put(data)
                logger.info("数据处理完成并加入队列")
                
                time.sleep(0.1)  # 控制数据生成速率
                
            except Exception as e:
                logger.error(f"数据生成错误: {str(e)}")
                logger.error(traceback.format_exc())
                break
                
        logger.info("数据生成线程结束")
            
    def _update_loop(self):
        """更新显示的循环"""
        try:
            data = self.data_queue.get_nowait()
            logger.info("成功获取数据，开始更新显示")
            self._update_display(data)
            logger.info("显示更新完成")
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"更新显示时发生错误: {e}")
            
        # 继续更新
        if self.running:
            self.root.after(50, self._update_loop)
            
    def _update_display(self, data):
        """更新显示
        
        Args:
            data: 要显示的数据
        """
        try:
            # 获取各类数据
            depth_map = data['vision']['depth_map']
            color_image = data['vision']['color_image']
            tactile_image = data['tactile']['tactile_image']
            force = data['tactile']['force']
            shear = data['tactile']['shear']
            fused_data = data['fused']
            
            # 更新图像显示
            if depth_map is not None:
                depth_display = cv2.resize(depth_map, (400, 300))
                depth_display = (depth_display * 255).astype(np.uint8)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                depth_display = cv2.cvtColor(depth_display, cv2.COLOR_BGR2RGB)
                depth_image = ImageTk.PhotoImage(Image.fromarray(depth_display))
                self.depth_label.configure(image=depth_image)
                self.depth_label.image = depth_image
            
            if color_image is not None:
                color_display = cv2.resize(color_image, (400, 300))
                color_display = cv2.cvtColor(color_display, cv2.COLOR_BGR2RGB)
                color_image = ImageTk.PhotoImage(Image.fromarray(color_display))
                self.color_label.configure(image=color_image)
                self.color_label.image = color_image
            
            if tactile_image is not None:
                tactile_display = cv2.resize(tactile_image, (400, 300))
                tactile_display = (tactile_display * 255).astype(np.uint8)
                tactile_display = cv2.applyColorMap(tactile_display, cv2.COLORMAP_VIRIDIS)
                tactile_display = cv2.cvtColor(tactile_display, cv2.COLOR_BGR2RGB)
                tactile_image = ImageTk.PhotoImage(Image.fromarray(tactile_display))
                self.tactile_label.configure(image=tactile_image)
                self.tactile_label.image = tactile_image
            
            if 'image' in fused_data:
                fused_display = cv2.resize(fused_data['image'], (400, 300))
                fused_display = cv2.cvtColor(fused_display, cv2.COLOR_BGR2RGB)
                fused_image = ImageTk.PhotoImage(Image.fromarray(fused_display))
                self.fused_label.configure(image=fused_image)
                self.fused_label.image = fused_image
            
            # 更新力信息显示
            total_force = np.sqrt(sum(f**2 for f in force))
            total_shear = np.sqrt(sum(s**2 for s in shear))
            self.force_label.configure(text=f"压力: {total_force:.1f}N")
            self.shear_label.configure(text=f"剪切力: {total_shear:.1f}N")
            
            # 更新特征信息显示
            if 'features' in fused_data:
                features = fused_data['features']
                if 'confidence' in features:
                    self.features_label.configure(text=f"特征置信度: {features['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"更新显示错误: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """归一化图像
        
        Args:
            image: 输入图像
            
        Returns:
            归一化后的图像
        """
        try:
            if image is None:
                return np.zeros((100, 100), dtype=np.uint8)
                
            # 归一化到0-255范围
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
            image = (image * 255).astype(np.uint8)
            return image
            
        except Exception as e:
            logger.error(f"图像归一化错误: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros((100, 100), dtype=np.uint8)
            
    def _display_image(self, image: np.ndarray, label: ttk.Label):
        """显示图像
        
        Args:
            image: 图像数据
            label: 显示标签
        """
        try:
            if image is None:
                return
                
            # 调整图像大小
            height, width = image.shape[:2]
            max_size = 300
            scale = min(max_size/width, max_size/height)
            new_size = (int(width*scale), int(height*scale))
            image = cv2.resize(image, new_size)
            
            # 转换为PIL图像
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image=image)
            
            # 更新标签
            label.config(image=photo)
            label.image = photo  # 保持引用
            
        except Exception as e:
            logger.error(f"图像显示错误: {str(e)}")
            logger.error(traceback.format_exc())
            
    def run(self):
        """运行可视化界面"""
        try:
            logger.info("运行可视化界面")
            self.root.mainloop()
            logger.info("可视化界面已关闭")
            
        except Exception as e:
            logger.error(f"运行错误: {str(e)}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        logger.info("启动程序")
        app = FusionVisualization()
        logger.info("运行可视化界面")
        app.root.mainloop()
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}")
        logger.error(traceback.format_exc()) 