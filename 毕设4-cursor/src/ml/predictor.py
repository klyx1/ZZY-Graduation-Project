import torch
import numpy as np
import logging
from typing import Dict, Tuple, Optional
import cv2

from .model import GraspNet

class GraspPredictor:
    def __init__(self, 
                 model: GraspNet,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化预测器
        
        Args:
            model: 模型实例
            config: 配置字典
            device: 预测设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 设置日志
        self.logger = logging.getLogger('GraspPredictor')
        
        # 设置阈值
        self.confidence_threshold = config['ml']['inference']['confidence_threshold']
        
    def preprocess_vision(self, 
                         image: np.ndarray) -> torch.Tensor:
        """预处理视觉数据
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的张量
        """
        # 调整大小
        target_size = self.config['ml']['model']['input_size']
        image = cv2.resize(image, target_size)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        
        return image.to(self.device)
        
    def preprocess_tactile(self, 
                          tactile_data: np.ndarray) -> torch.Tensor:
        """预处理触觉数据
        
        Args:
            tactile_data: 触觉数据
            
        Returns:
            处理后的张量
        """
        # 调整大小
        target_size = self.config['ml']['model']['input_size']
        tactile_data = cv2.resize(tactile_data, target_size)
        
        # 归一化
        tactile_data = tactile_data.astype(np.float32) / 255.0
        
        # 转换为张量
        tactile_data = torch.from_numpy(tactile_data).unsqueeze(0).unsqueeze(0)
        
        return tactile_data.to(self.device)
        
    def predict(self, 
                vision_data: np.ndarray,
                tactile_data: np.ndarray) -> Tuple[Optional[Dict], float]:
        """预测抓取参数
        
        Args:
            vision_data: 视觉数据
            tactile_data: 触觉数据
            
        Returns:
            抓取参数和置信度
        """
        try:
            # 预处理数据
            vision_tensor = self.preprocess_vision(vision_data)
            tactile_tensor = self.preprocess_tactile(tactile_data)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                grasp_params, quality_score = self.model(vision_tensor, tactile_tensor)
                
            # 获取预测结果
            grasp_params = grasp_params.cpu().numpy()[0]
            quality_score = quality_score.cpu().numpy()[0][0]
            
            # 检查置信度
            if quality_score < self.confidence_threshold:
                self.logger.warning(f"预测置信度过低: {quality_score:.4f}")
                return None, quality_score
                
            # 解析预测结果
            position = grasp_params[:3]
            direction = grasp_params[3:]
            
            # 归一化方向向量
            direction = direction / np.linalg.norm(direction)
            
            return {
                'position': position,
                'direction': direction,
                'quality_score': quality_score
            }, quality_score
            
        except Exception as e:
            self.logger.error(f"预测错误: {str(e)}")
            return None, 0.0
            
    def predict_batch(self, 
                     vision_batch: np.ndarray,
                     tactile_batch: np.ndarray) -> Tuple[Optional[Dict], float]:
        """批量预测抓取参数
        
        Args:
            vision_batch: 视觉数据批次
            tactile_batch: 触觉数据批次
            
        Returns:
            最佳抓取参数和置信度
        """
        try:
            # 预处理数据
            vision_tensors = []
            tactile_tensors = []
            
            for vision_data, tactile_data in zip(vision_batch, tactile_batch):
                vision_tensor = self.preprocess_vision(vision_data)
                tactile_tensor = self.preprocess_tactile(tactile_data)
                
                vision_tensors.append(vision_tensor)
                tactile_tensors.append(tactile_tensor)
                
            vision_batch = torch.cat(vision_tensors, dim=0)
            tactile_batch = torch.cat(tactile_tensors, dim=0)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                grasp_params, quality_scores = self.model(vision_batch, tactile_batch)
                
            # 获取最佳预测结果
            best_idx = np.argmax(quality_scores.cpu().numpy())
            best_quality = quality_scores[best_idx].cpu().numpy()
            
            if best_quality < self.confidence_threshold:
                self.logger.warning(f"最佳预测置信度过低: {best_quality:.4f}")
                return None, best_quality
                
            # 解析最佳预测结果
            best_params = grasp_params[best_idx].cpu().numpy()
            position = best_params[:3]
            direction = best_params[3:]
            
            # 归一化方向向量
            direction = direction / np.linalg.norm(direction)
            
            return {
                'position': position,
                'direction': direction,
                'quality_score': best_quality
            }, best_quality
            
        except Exception as e:
            self.logger.error(f"批量预测错误: {str(e)}")
            return None, 0.0 