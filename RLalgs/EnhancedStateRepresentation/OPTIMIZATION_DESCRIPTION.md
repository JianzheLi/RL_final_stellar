# EnhancedStateRepresentation - 特征工程优化

## 优化思路

通过**改进状态表示**来提升智能体的感知能力，使智能体能够从观测中提取更丰富、更有用的特征信息。

## 核心改进

### 1. 增强的状态编码器

#### 特征提取网络
- **多层特征提取**: 使用多层全连接网络提取深层特征
- **层归一化**: 使用LayerNorm稳定训练
- **Dropout**: 防止过拟合

#### 注意力机制
- **特征注意力**: 使用注意力机制关注重要特征
- **自适应加权**: 自动学习哪些特征更重要
- **提升表达能力**: 增强对关键信息的感知

### 2. 技术细节

#### 网络结构
```
原始观测 → 特征提取网络 → 注意力加权 → RNN → 输出层 → Q值
```

#### 关键组件
1. **StateEncoder**: 状态编码器
   - 多层特征提取
   - 注意力机制
   - 层归一化

2. **EnhancedRNNAgent**: 增强的RNN智能体
   - 使用StateEncoder编码状态
   - GRU处理时序信息
   - 改进的输出层

3. **SpatialFeatureExtractor** (可选): 空间特征提取器
   - 如果观测包含空间信息，可以使用CNN提取

#### 关键参数
- `enhanced_feature_dim`: 128 - 增强特征维度
- `enhanced_dropout`: 0.1 - Dropout率
- `use_spatial_features`: False - 是否使用空间特征

### 3. 优势

1. **更好的特征表示**: 提取更丰富、更有用的特征
2. **注意力机制**: 自动关注重要信息
3. **稳定训练**: 使用LayerNorm和Dropout稳定训练
4. **可扩展性**: 可以添加更多特征提取模块

### 4. 预期效果

- 更好的状态理解能力
- 更快的收敛速度
- 更高的任务完成率
- 更稳定的训练过程

## 使用方法

### 训练单个地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/EnhancedStateRepresentation/train_single_map.sh adcc 3 42
```

### 训练所有HLSMAC地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/EnhancedStateRepresentation/train_hlsmac.sh 3 42
```

## 配置文件

主要配置文件: `src/config/algs/enhanced_qmix.yaml`

### 关键参数说明

- `agent: "enhanced_rnn"` - 使用增强的RNN智能体
- `enhanced_feature_dim: 128` - 增强特征维度（建议64-256）
- `enhanced_dropout: 0.1` - Dropout率（建议0.1-0.3）
- `use_spatial_features: False` - 是否使用空间特征

### 调优建议

1. **特征维度**:
   - 增加`enhanced_feature_dim`提升表达能力
   - 但要注意计算开销

2. **Dropout率**:
   - 如果过拟合，增加`enhanced_dropout`
   - 如果欠拟合，减少`enhanced_dropout`

3. **空间特征**:
   - 如果观测包含空间信息，设置`use_spatial_features: True`
   - 需要根据实际观测格式调整

## 实现细节

### 网络结构
```
原始观测 [batch, agents, obs_dim]
    ↓
状态编码器
    ├─ 特征提取网络 (FC + LayerNorm + ReLU)
    └─ 注意力机制 (FC + Sigmoid)
    ↓
增强特征 [batch, agents, enhanced_feature_dim]
    ↓
RNN (GRU)
    ↓
输出层 (FC + LayerNorm + ReLU + Dropout)
    ↓
Q值 [batch, agents, n_actions]
```

### 关键创新点
1. **多层特征提取**: 提取深层特征表示
2. **注意力机制**: 自动关注重要特征
3. **层归一化**: 稳定训练过程
4. **模块化设计**: 易于扩展和修改

## 与Baseline对比

| 特性 | Baseline | EnhancedStateRepresentation |
|------|----------|----------------------------|
| 状态编码 | 简单FC | 多层+注意力 |
| 特征提取 | 单层 | 多层+归一化 |
| 注意力机制 | 无 | 有 |
| 表达能力 | 中等 | 强 |
| 训练稳定性 | 中等 | 高（LayerNorm）|

## 实验建议

1. **特征维度调优**: 测试不同`enhanced_feature_dim`的影响
2. **注意力可视化**: 可视化注意力权重，理解模型关注什么
3. **消融实验**: 测试各个组件的贡献
4. **不同地图**: 在不同HLSMAC地图上测试效果

## 扩展方向

1. **更多特征类型**: 添加距离、角度、相对位置等特征
2. **多尺度特征**: 提取不同尺度的特征
3. **特征融合**: 融合多种类型的特征
4. **自适应特征选择**: 根据任务自动选择重要特征

## 参考文献

- Attention Is All You Need
- Layer Normalization
- Feature Engineering for Machine Learning

