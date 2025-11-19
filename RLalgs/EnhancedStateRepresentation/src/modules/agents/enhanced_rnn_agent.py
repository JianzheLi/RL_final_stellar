import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StateEncoder(nn.Module):
    """
    增强的状态编码器：提取更丰富的状态特征
    """
    def __init__(self, input_shape, args):
        super(StateEncoder, self).__init__()
        self.args = args
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_shape, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制：关注重要特征
        self.attention = nn.Sequential(
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(args.enhanced_feature_dim // 2, args.enhanced_feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 注意力加权
        attention_weights = self.attention(features)
        enhanced_features = features * attention_weights
        
        return enhanced_features


class EnhancedRNNAgent(nn.Module):
    """
    增强的RNN智能体：使用改进的状态表示
    """
    def __init__(self, input_shape, args):
        super(EnhancedRNNAgent, self).__init__()
        self.args = args
        
        # 状态编码器
        self.state_encoder = StateEncoder(input_shape, args)
        
        # RNN网络
        self.rnn = nn.GRUCell(
            args.enhanced_feature_dim, 
            args.rnn_hidden_dim
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        
    def init_hidden(self):
        return self.rnn.weight_ih.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        
        # 重塑输入
        inputs_flat = inputs.view(-1, e)
        
        # 状态编码
        encoded_features = self.state_encoder(inputs_flat)
        
        # RNN处理
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(encoded_features, hidden_state)
        
        # 输出Q值
        q = self.fc_out(h)
        q = torch.clamp(q, -5, 2)
        
        return q.view(b, a, -1), h.view(b, a, -1)


class SpatialFeatureExtractor(nn.Module):
    """
    空间特征提取器：提取空间关系特征（可选）
    """
    def __init__(self, input_shape, args):
        super(SpatialFeatureExtractor, self).__init__()
        self.args = args
        
        # 如果观测包含空间信息，可以使用CNN
        if hasattr(args, 'use_spatial_features') and args.use_spatial_features:
            # 假设有空间维度信息
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.spatial_dim = 32
        else:
            self.spatial_conv = None
            self.spatial_dim = 0
    
    def forward(self, x):
        if self.spatial_conv is not None:
            # 假设输入有空间维度（需要根据实际情况调整）
            spatial_features = self.spatial_conv(x)
            return spatial_features.view(x.size(0), -1)
        return None

