import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from scipy import signal
import threading
import queue

class DataLogger:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化数据记录器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # 创建数据队列
        self.data_queue = queue.Queue()
        
        # 数据缓冲区
        self.force_history = []
        self.deformation_history = []
        self.grasp_points_history = []
        self.sensor_data_history = []
        
        # 统计信息
        self.stats = {
            'grasp_success_rate': 0.0,
            'avg_grasp_force': 0.0,
            'avg_deformation': 0.0,
            'max_force': 0.0,
            'min_force': float('inf')
        }
        
        # 启动记录线程
        self.is_running = False
        self.record_thread = threading.Thread(target=self._record_loop)
        self.record_thread.daemon = True
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """设置日志记录"""
        # 创建日志目录
        log_dir = os.path.join(self.config['system']['data_path'], 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 创建数据目录
        data_dir = os.path.join(self.config['system']['data_path'], 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # 设置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"data_{timestamp}.log")
        self.data_file = os.path.join(data_dir, f"data_{timestamp}.csv")
        
        # 创建CSV文件头
        self._create_csv_header()
        
    def _create_csv_header(self):
        """创建CSV文件头"""
        headers = [
            'timestamp',
            'force',
            'deformation',
            'grasp_point_x',
            'grasp_point_y',
            'grasp_point_z',
            'grasp_normal_x',
            'grasp_normal_y',
            'grasp_normal_z',
            'vision_data',
            'tactile_data'
        ]
        
        with open(self.data_file, 'w') as f:
            f.write(','.join(headers) + '\n')
            
    def start(self):
        """启动数据记录"""
        self.is_running = True
        self.record_thread.start()
        
    def stop(self):
        """停止数据记录"""
        self.is_running = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
            
    def log_data(self, data: Dict):
        """记录数据
        
        Args:
            data: 要记录的数据字典
        """
        self.data_queue.put(data)
        
    def _record_loop(self):
        """记录循环"""
        while self.is_running:
            try:
                # 从队列获取数据
                data = self.data_queue.get(timeout=1.0)
                
                # 记录数据
                self._record_data(data)
                
                # 更新统计信息
                self._update_stats(data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"数据记录错误: {str(e)}")
                break
                
    def _record_data(self, data: Dict):
        """记录数据到文件
        
        Args:
            data: 要记录的数据
        """
        # 准备CSV行数据
        row_data = [
            datetime.now().isoformat(),
            data.get('force', 0.0),
            data.get('deformation', 0.0),
            data.get('grasp_point', [0.0, 0.0, 0.0])[0],
            data.get('grasp_point', [0.0, 0.0, 0.0])[1],
            data.get('grasp_point', [0.0, 0.0, 0.0])[2],
            data.get('grasp_normal', [0.0, 0.0, 0.0])[0],
            data.get('grasp_normal', [0.0, 0.0, 0.0])[1],
            data.get('grasp_normal', [0.0, 0.0, 0.0])[2],
            json.dumps(data.get('vision_data', {})),
            json.dumps(data.get('tactile_data', {}))
        ]
        
        # 写入CSV文件
        with open(self.data_file, 'a') as f:
            f.write(','.join(map(str, row_data)) + '\n')
            
        # 更新历史数据
        self.force_history.append(data.get('force', 0.0))
        self.deformation_history.append(data.get('deformation', 0.0))
        self.grasp_points_history.append(data.get('grasp_point', [0.0, 0.0, 0.0]))
        self.sensor_data_history.append({
            'vision': data.get('vision_data', {}),
            'tactile': data.get('tactile_data', {})
        })
        
    def _update_stats(self, data: Dict):
        """更新统计信息
        
        Args:
            data: 数据字典
        """
        # 更新力统计
        force = data.get('force', 0.0)
        self.stats['max_force'] = max(self.stats['max_force'], force)
        self.stats['min_force'] = min(self.stats['min_force'], force)
        self.stats['avg_grasp_force'] = np.mean(self.force_history)
        
        # 更新变形统计
        self.stats['avg_deformation'] = np.mean(self.deformation_history)
        
        # 更新抓取成功率
        if data.get('grasp_success', False):
            self.stats['grasp_success_rate'] = (
                self.stats['grasp_success_rate'] * 0.9 + 0.1
            )
        else:
            self.stats['grasp_success_rate'] *= 0.9
            
    def analyze_data(self) -> Dict:
        """分析数据
        
        Returns:
            分析结果字典
        """
        analysis = {
            'basic_stats': self._analyze_basic_stats(),
            'force_analysis': self._analyze_force_data(),
            'deformation_analysis': self._analyze_deformation_data(),
            'grasp_analysis': self._analyze_grasp_data()
        }
        
        return analysis
        
    def _analyze_basic_stats(self) -> Dict:
        """分析基本统计信息
        
        Returns:
            基本统计信息字典
        """
        return {
            'force_mean': np.mean(self.force_history),
            'force_std': np.std(self.force_history),
            'deformation_mean': np.mean(self.deformation_history),
            'deformation_std': np.std(self.deformation_history),
            'grasp_success_rate': self.stats['grasp_success_rate']
        }
        
    def _analyze_force_data(self) -> Dict:
        """分析力数据
        
        Returns:
            力数据分析结果
        """
        if not self.force_history:
            return {}
            
        # 计算力频谱
        force_fft = np.fft.fft(self.force_history)
        freqs = np.fft.fftfreq(len(self.force_history))
        
        # 计算力变化率
        force_changes = np.diff(self.force_history)
        
        return {
            'force_spectrum': {
                'frequencies': freqs.tolist(),
                'amplitudes': np.abs(force_fft).tolist()
            },
            'force_changes': {
                'mean': np.mean(force_changes),
                'std': np.std(force_changes),
                'max': np.max(force_changes),
                'min': np.min(force_changes)
            }
        }
        
    def _analyze_deformation_data(self) -> Dict:
        """分析变形数据
        
        Returns:
            变形数据分析结果
        """
        if not self.deformation_history:
            return {}
            
        # 计算变形频谱
        deformation_fft = np.fft.fft(self.deformation_history)
        freqs = np.fft.fftfreq(len(self.deformation_history))
        
        # 计算变形变化率
        deformation_changes = np.diff(self.deformation_history)
        
        return {
            'deformation_spectrum': {
                'frequencies': freqs.tolist(),
                'amplitudes': np.abs(deformation_fft).tolist()
            },
            'deformation_changes': {
                'mean': np.mean(deformation_changes),
                'std': np.std(deformation_changes),
                'max': np.max(deformation_changes),
                'min': np.min(deformation_changes)
            }
        }
        
    def _analyze_grasp_data(self) -> Dict:
        """分析抓取数据
        
        Returns:
            抓取数据分析结果
        """
        if not self.grasp_points_history:
            return {}
            
        # 计算抓取点分布
        grasp_points = np.array(self.grasp_points_history)
        grasp_centers = np.mean(grasp_points, axis=0)
        grasp_std = np.std(grasp_points, axis=0)
        
        return {
            'grasp_point_distribution': {
                'mean': grasp_centers.tolist(),
                'std': grasp_std.tolist()
            },
            'grasp_success_rate': self.stats['grasp_success_rate']
        }
        
    def plot_analysis(self, save_path: Optional[str] = None):
        """绘制分析图表
        
        Args:
            save_path: 保存路径
        """
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 绘制力数据
        axes[0, 0].plot(self.force_history)
        axes[0, 0].set_title('力数据')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('力 (N)')
        
        # 绘制变形数据
        axes[0, 1].plot(self.deformation_history)
        axes[0, 1].set_title('变形数据')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('变形量')
        
        # 绘制力频谱
        force_fft = np.fft.fft(self.force_history)
        freqs = np.fft.fftfreq(len(self.force_history))
        axes[1, 0].plot(freqs, np.abs(force_fft))
        axes[1, 0].set_title('力频谱')
        axes[1, 0].set_xlabel('频率')
        axes[1, 0].set_ylabel('幅值')
        
        # 绘制抓取点分布
        grasp_points = np.array(self.grasp_points_history)
        axes[1, 1].scatter(grasp_points[:, 0], grasp_points[:, 1])
        axes[1, 1].set_title('抓取点分布')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def export_data(self, format: str = 'csv') -> str:
        """导出数据
        
        Args:
            format: 导出格式
            
        Returns:
            导出文件路径
        """
        if format == 'csv':
            return self.data_file
        elif format == 'json':
            json_file = self.data_file.replace('.csv', '.json')
            data = {
                'force_history': self.force_history,
                'deformation_history': self.deformation_history,
                'grasp_points_history': self.grasp_points_history,
                'sensor_data_history': self.sensor_data_history,
                'stats': self.stats
            }
            with open(json_file, 'w') as f:
                json.dump(data, f)
            return json_file
        else:
            raise ValueError(f"不支持的导出格式: {format}") 