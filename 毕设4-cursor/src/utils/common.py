import yaml
import logging
import os
from typing import Dict, Any
from datetime import datetime

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"加载配置文件失败: {str(e)}")
            
class Logger:
    @staticmethod
    def setup_logger(name: str, 
                    log_dir: str = 'data/logs',
                    level: int = logging.INFO) -> logging.Logger:
        """设置日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            level: 日志级别
            
        Returns:
            日志记录器
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 如果已经有处理器，不重复添加
        if logger.handlers:
            return logger
            
        # 创建文件处理器
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
class DataCache:
    def __init__(self, max_size: int = 1000):
        """初始化数据缓存
        
        Args:
            max_size: 最大缓存大小
        """
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key: str) -> Any:
        """获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存数据
        """
        return self.cache.get(key)
        
    def set(self, key: str, value: Any):
        """设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存数据
        """
        if len(self.cache) >= self.max_size:
            # 删除最早的缓存
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
        
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        
class PerformanceMonitor:
    def __init__(self):
        """初始化性能监控器"""
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, name: str):
        """开始计时
        
        Args:
            name: 计时器名称
        """
        self.start_times[name] = time.time()
        
    def stop_timer(self, name: str):
        """停止计时
        
        Args:
            name: 计时器名称
        """
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(elapsed)
            
    def get_average_time(self, name: str) -> float:
        """获取平均时间
        
        Args:
            name: 计时器名称
            
        Returns:
            平均时间
        """
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0
        
    def clear(self):
        """清空所有指标"""
        self.metrics.clear()
        self.start_times.clear() 