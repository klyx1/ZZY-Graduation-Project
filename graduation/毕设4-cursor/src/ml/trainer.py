import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from typing import Dict, List, Tuple
from datetime import datetime

from .model import GraspNet

class ModelTrainer:
    def __init__(self, 
                 model: GraspNet,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化训练器
        
        Args:
            model: 模型实例
            config: 配置字典
            device: 训练设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 设置日志
        self.logger = logging.getLogger('ModelTrainer')
        
        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['ml']['model']['learning_rate']
        )
        
        # 设置损失函数
        self.position_criterion = nn.MSELoss()
        self.direction_criterion = nn.CosineEmbeddingLoss()
        self.quality_criterion = nn.BCELoss()
        
        # 创建保存目录
        self.save_dir = config['ml']['training']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self, 
                    train_loader: DataLoader,
                    epoch: int) -> Dict[str, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0
        total_position_loss = 0
        total_direction_loss = 0
        total_quality_loss = 0
        
        for batch_idx, (vision_data, tactile_data, 
                       target_position, target_direction,
                       target_quality) in enumerate(train_loader):
            # 数据转移到设备
            vision_data = vision_data.to(self.device)
            tactile_data = tactile_data.to(self.device)
            target_position = target_position.to(self.device)
            target_direction = target_direction.to(self.device)
            target_quality = target_quality.to(self.device)
            
            # 前向传播
            grasp_params, quality_score = self.model(vision_data, tactile_data)
            
            # 计算损失
            position_loss = self.position_criterion(
                grasp_params[:, :3],
                target_position
            )
            
            direction_loss = self.direction_criterion(
                grasp_params[:, 3:],
                target_direction,
                torch.ones(len(target_direction), device=self.device)
            )
            
            quality_loss = self.quality_criterion(
                quality_score,
                target_quality
            )
            
            # 总损失
            loss = (position_loss + direction_loss + quality_loss) / 3
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 累加损失
            total_loss += loss.item()
            total_position_loss += position_loss.item()
            total_direction_loss += direction_loss.item()
            total_quality_loss += quality_loss.item()
            
            # 打印进度
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Epoch: {epoch}, Batch: {batch_idx}, '
                    f'Loss: {loss.item():.4f}'
                )
                
        # 计算平均损失
        num_batches = len(train_loader)
        return {
            'loss': total_loss / num_batches,
            'position_loss': total_position_loss / num_batches,
            'direction_loss': total_direction_loss / num_batches,
            'quality_loss': total_quality_loss / num_batches
        }
        
    def validate(self, 
                val_loader: DataLoader) -> Dict[str, float]:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证指标
        """
        self.model.eval()
        total_loss = 0
        total_position_loss = 0
        total_direction_loss = 0
        total_quality_loss = 0
        
        with torch.no_grad():
            for vision_data, tactile_data, \
                target_position, target_direction, \
                target_quality in val_loader:
                # 数据转移到设备
                vision_data = vision_data.to(self.device)
                tactile_data = tactile_data.to(self.device)
                target_position = target_position.to(self.device)
                target_direction = target_direction.to(self.device)
                target_quality = target_quality.to(self.device)
                
                # 前向传播
                grasp_params, quality_score = self.model(vision_data, tactile_data)
                
                # 计算损失
                position_loss = self.position_criterion(
                    grasp_params[:, :3],
                    target_position
                )
                
                direction_loss = self.direction_criterion(
                    grasp_params[:, 3:],
                    target_direction,
                    torch.ones(len(target_direction), device=self.device)
                )
                
                quality_loss = self.quality_criterion(
                    quality_score,
                    target_quality
                )
                
                # 总损失
                loss = (position_loss + direction_loss + quality_loss) / 3
                
                # 累加损失
                total_loss += loss.item()
                total_position_loss += position_loss.item()
                total_direction_loss += direction_loss.item()
                total_quality_loss += quality_loss.item()
                
        # 计算平均损失
        num_batches = len(val_loader)
        return {
            'loss': total_loss / num_batches,
            'position_loss': total_position_loss / num_batches,
            'direction_loss': total_direction_loss / num_batches,
            'quality_loss': total_quality_loss / num_batches
        }
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = None) -> Dict[str, List[float]]:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            
        Returns:
            训练历史
        """
        if num_epochs is None:
            num_epochs = self.config['ml']['model']['epochs']
            
        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_position_loss': [],
            'val_position_loss': [],
            'train_direction_loss': [],
            'val_direction_loss': [],
            'train_quality_loss': [],
            'val_quality_loss': []
        }
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_position_loss'].append(train_metrics['position_loss'])
            history['val_position_loss'].append(val_metrics['position_loss'])
            history['train_direction_loss'].append(train_metrics['direction_loss'])
            history['val_direction_loss'].append(val_metrics['direction_loss'])
            history['train_quality_loss'].append(train_metrics['quality_loss'])
            history['val_quality_loss'].append(val_metrics['quality_loss'])
            
            # 打印epoch结果
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Val Loss: {val_metrics["loss"]:.4f}'
            )
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model('best_model.pth')
                
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
                
        return history
        
    def save_model(self, filename: str):
        """保存模型
        
        Args:
            filename: 文件名
        """
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, save_path)
        self.logger.info(f'模型已保存到: {save_path}')
        
    def load_model(self, filename: str):
        """加载模型
        
        Args:
            filename: 文件名
        """
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f'模型已从 {load_path} 加载') 