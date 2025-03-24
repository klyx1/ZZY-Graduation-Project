import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time

class GraspingGUI:
    def __init__(self, system):
        """初始化GUI界面
        
        Args:
            system: 抓取系统实例
        """
        self.system = system
        self.root = tk.Tk()
        self.root.title("柔性物体抓取系统")
        self.root.geometry("1200x800")
        
        # 创建数据队列
        self.data_queue = queue.Queue()
        self.image_queue = queue.Queue()
        
        # 创建界面组件
        self._create_widgets()
        
        # 状态变量
        self.is_running = False
        self.emergency_stop = False
        
        # 启动更新线程
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        
    def _create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建左侧面板
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 创建右侧面板
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建图像显示区域
        self._create_image_display()
        
        # 创建状态显示区域
        self._create_status_display()
        
        # 创建控制按钮
        self._create_control_buttons()
        
        # 创建数据图表
        self._create_data_plots()
        
    def _create_image_display(self):
        """创建图像显示区域"""
        # 视觉图像显示
        self.vision_frame = ttk.LabelFrame(self.left_frame, text="视觉图像")
        self.vision_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.vision_label = ttk.Label(self.vision_frame)
        self.vision_label.pack(fill=tk.BOTH, expand=True)
        
        # 触觉图像显示
        self.tactile_frame = ttk.LabelFrame(self.left_frame, text="触觉图像")
        self.tactile_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tactile_label = ttk.Label(self.tactile_frame)
        self.tactile_label.pack(fill=tk.BOTH, expand=True)
        
    def _create_status_display(self):
        """创建状态显示区域"""
        # 系统状态
        self.status_frame = ttk.LabelFrame(self.right_frame, text="系统状态")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="状态: 待机")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # 传感器状态
        self.sensor_frame = ttk.LabelFrame(self.right_frame, text="传感器状态")
        self.sensor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.vision_status = ttk.Label(self.sensor_frame, text="视觉传感器: 未连接")
        self.vision_status.pack(fill=tk.X, padx=5, pady=2)
        
        self.tactile_status = ttk.Label(self.sensor_frame, text="触觉传感器: 未连接")
        self.tactile_status.pack(fill=tk.X, padx=5, pady=2)
        
        self.gripper_status = ttk.Label(self.sensor_frame, text="夹爪: 未连接")
        self.gripper_status.pack(fill=tk.X, padx=5, pady=2)
        
        # 力信息显示
        self.force_frame = ttk.LabelFrame(self.right_frame, text="力信息")
        self.force_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.force_label = ttk.Label(self.force_frame, text="当前力: 0.0 N")
        self.force_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.shear_label = ttk.Label(self.force_frame, text="剪切力: (0.0, 0.0) N")
        self.shear_label.pack(fill=tk.X, padx=5, pady=2)
        
    def _create_control_buttons(self):
        """创建控制按钮"""
        self.control_frame = ttk.LabelFrame(self.right_frame, text="控制")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 启动按钮
        self.start_button = ttk.Button(
            self.control_frame,
            text="启动系统",
            command=self._start_system
        )
        self.start_button.pack(fill=tk.X, padx=5, pady=2)
        
        # 停止按钮
        self.stop_button = ttk.Button(
            self.control_frame,
            text="停止系统",
            command=self._stop_system,
            state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, padx=5, pady=2)
        
        # 紧急停止按钮
        self.emergency_button = ttk.Button(
            self.control_frame,
            text="紧急停止",
            command=self._emergency_stop,
            style="Emergency.TButton"
        )
        self.emergency_button.pack(fill=tk.X, padx=5, pady=2)
        
        # 创建紧急停止按钮样式
        style = ttk.Style()
        style.configure("Emergency.TButton", foreground="red")
        
    def _create_data_plots(self):
        """创建数据图表"""
        self.plot_frame = ttk.LabelFrame(self.right_frame, text="数据记录")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建力数据图表
        self.force_fig, self.force_ax = plt.subplots(figsize=(6, 3))
        self.force_canvas = FigureCanvasTkAgg(self.force_fig, master=self.plot_frame)
        self.force_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建变形数据图表
        self.deformation_fig, self.deformation_ax = plt.subplots(figsize=(6, 3))
        self.deformation_canvas = FigureCanvasTkAgg(self.deformation_fig, master=self.plot_frame)
        self.deformation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _start_system(self):
        """启动系统"""
        if self.system.start():
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="状态: 运行中")
            self.update_thread.start()
        else:
            self.status_label.config(text="状态: 启动失败")
            
    def _stop_system(self):
        """停止系统"""
        self.is_running = False
        self.system.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 已停止")
        
    def _emergency_stop(self):
        """紧急停止"""
        self.emergency_stop = True
        self.is_running = False
        self.system.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 紧急停止")
        
    def _update_loop(self):
        """更新循环"""
        while self.is_running:
            try:
                # 更新图像显示
                self._update_images()
                
                # 更新状态显示
                self._update_status()
                
                # 更新数据图表
                self._update_plots()
                
                time.sleep(0.1)  # 控制更新频率
                
            except Exception as e:
                print(f"更新错误: {str(e)}")
                self._stop_system()
                break
                
    def _update_images(self):
        """更新图像显示"""
        # 获取视觉图像
        depth_image, color_image = self.system.vision_sensor.get_frames()
        if color_image is not None:
            self._display_image(color_image, self.vision_label)
            
        # 获取触觉图像
        tactile_image = self.system.tactile_sensor.get_image()
        if tactile_image is not None:
            self._display_image(tactile_image, self.tactile_label)
            
    def _update_status(self):
        """更新状态显示"""
        # 更新传感器状态
        self.vision_status.config(
            text=f"视觉传感器: {'已连接' if self.system.vision_sensor else '未连接'}"
        )
        self.tactile_status.config(
            text=f"触觉传感器: {'已连接' if self.system.tactile_sensor else '未连接'}"
        )
        self.gripper_status.config(
            text=f"夹爪: {'已连接' if self.system.gripper.is_connected else '未连接'}"
        )
        
        # 更新力信息
        if self.system.gripper.is_connected:
            status = self.system.gripper.get_status()
            if status:
                self.force_label.config(text=f"当前力: {status['force']:.1f} N")
                
    def _update_plots(self):
        """更新数据图表"""
        # 更新力数据图表
        self.force_ax.clear()
        self.force_ax.plot(self.system.force_history, label='力')
        self.force_ax.set_title('力数据记录')
        self.force_ax.set_xlabel('时间')
        self.force_ax.set_ylabel('力 (N)')
        self.force_ax.legend()
        self.force_canvas.draw()
        
        # 更新变形数据图表
        self.deformation_ax.clear()
        self.deformation_ax.plot(self.system.deformation_history, label='变形')
        self.deformation_ax.set_title('变形数据记录')
        self.deformation_ax.set_xlabel('时间')
        self.deformation_ax.set_ylabel('变形量')
        self.deformation_ax.legend()
        self.deformation_canvas.draw()
        
    def _display_image(self, image: np.ndarray, label: ttk.Label):
        """显示图像
        
        Args:
            image: 图像数据
            label: 显示标签
        """
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
        
    def run(self):
        """运行GUI"""
        self.root.mainloop() 