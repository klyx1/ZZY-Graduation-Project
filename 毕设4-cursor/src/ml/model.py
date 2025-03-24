import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraspNet(nn.Module):
    def __init__(self, input_size=(224, 224)):
        """初始化抓取网络
        
        Args:
            input_size: 输入图像大小
        """
        super().__init__()
        
        # 视觉特征提取
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 触觉特征提取
        self.tactile_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(256 * (input_size[0]//8) * (input_size[1]//8) + 64 * (input_size[0]//4) * (input_size[1]//4), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 抓取参数预测
        self.grasp_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 3D位置 + 3D方向
        )
        
        # 抓取质量评估
        self.quality_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vision_input, tactile_input):
        """前向传播
        
        Args:
            vision_input: 视觉输入 [B, 3, H, W]
            tactile_input: 触觉输入 [B, 1, H, W]
            
        Returns:
            grasp_params: 抓取参数 [B, 6]
            quality_score: 抓取质量分数 [B, 1]
        """
        # 特征提取
        vision_features = self.vision_encoder(vision_input)
        tactile_features = self.tactile_encoder(tactile_input)
        
        # 特征展平
        vision_features = vision_features.view(vision_features.size(0), -1)
        tactile_features = tactile_features.view(tactile_features.size(0), -1)
        
        # 特征融合
        fused_features = torch.cat([vision_features, tactile_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # 预测
        grasp_params = self.grasp_predictor(fused_features)
        quality_score = self.quality_predictor(fused_features)
        
        return grasp_params, quality_score
        
    def predict_grasp(self, vision_input, tactile_input, threshold=0.8):
        """预测抓取参数
        
        Args:
            vision_input: 视觉输入
            tactile_input: 触觉输入
            threshold: 质量分数阈值
            
        Returns:
            grasp_params: 抓取参数
            quality_score: 质量分数
            is_valid: 是否有效
        """
        self.eval()
        with torch.no_grad():
            grasp_params, quality_score = self(vision_input, tactile_input)
            is_valid = quality_score > threshold
        return grasp_params, quality_score, is_valid 