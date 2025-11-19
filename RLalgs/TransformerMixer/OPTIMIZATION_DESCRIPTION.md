# TransformerMixer - 基于Transformer架构的QMIX Mixer网络优化

## 优化思路

在原始QMIX的基础上，引入**Transformer架构**来改进MIXER网络，使智能体能够更好地捕捉状态中的关键信息和智能体间的协作关系。

## 核心改进

### 1. Transformer架构的Mixer网络
- **多头自注意力机制**: 使用多头注意力捕捉状态中的关键信息
- **Transformer编码器层**: 通过多层Transformer编码器增强状态表示
- **智能体间注意力**: 在混合Q值时考虑智能体间的相互影响
- **位置编码**: 隐式地通过状态嵌入学习位置信息

### 2. 技术细节

#### 架构组件
- **状态编码器**: 将全局状态编码为embedding
- **智能体Q值编码器**: 将每个智能体的Q值编码为embedding
- **Transformer编码器**: 多层Transformer编码器处理智能体-状态联合表示
- **混合网络**: 将Transformer输出聚合为总Q值

#### 关键参数
- `transformer_heads`: 4 - 多头注意力头数
- `transformer_layers`: 2 - Transformer编码器层数
- `transformer_ff_dim`: 256 - Feed-forward网络维度
- `transformer_dropout`: 0.1 - Dropout率
- `mixing_embed_dim`: 64 - 混合网络embedding维度

### 3. 优势

1. **更强的表达能力**: Transformer架构能够捕捉长距离依赖关系
2. **注意力机制**: 自动关注关键状态信息和智能体协作
3. **可扩展性**: 可以通过增加层数和头数进一步提升性能
4. **并行计算**: Transformer架构支持高效的并行计算

### 4. 预期效果

- 更好的状态表示学习
- 更强的智能体协作能力
- 在复杂HLSMAC任务上提升胜率
- 更稳定的训练过程

## 使用方法

### 训练单个地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/TransformerMixer/train_single_map.sh adcc 1 42
```

### 训练所有HLSMAC地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/TransformerMixer/train_hlsmac.sh 1 42
```

## 配置文件

主要配置文件: `src/config/algs/transformer_qmix.yaml`

### 关键参数说明

- `mixer: "transformer_qmix"` - 使用Transformer QMIX
- `transformer_heads: 4` - 多头注意力头数（建议4-8）
- `transformer_layers: 2` - Transformer编码器层数（建议2-4）
- `transformer_ff_dim: 256` - Feed-forward网络维度
- `transformer_dropout: 0.1` - Dropout率（防止过拟合）
- `mixing_embed_dim: 64` - 混合网络embedding维度

### 调优建议

1. **增加表达能力**: 
   - 增加`transformer_layers`到3-4层
   - 增加`transformer_heads`到6-8个
   - 增加`mixing_embed_dim`到128

2. **防止过拟合**:
   - 增加`transformer_dropout`到0.2-0.3
   - 使用更大的batch_size

3. **训练稳定性**:
   - 使用较小的学习率（如0.0005）
   - 增加梯度裁剪阈值

## 实现细节

### 网络结构
```
状态输入 → 状态编码器 → Transformer编码器 → 混合网络 → 总Q值
智能体Q值 → Q值编码器 ↗
```

### 关键创新点
1. **联合编码**: 将智能体Q值和状态信息联合编码
2. **自注意力**: 使用自注意力机制捕捉智能体间关系
3. **残差连接**: 在Transformer层中使用残差连接
4. **层归一化**: 使用LayerNorm稳定训练

## 与Baseline对比

| 特性 | Baseline (QMIX) | TransformerMixer |
|------|-----------------|-------------------|
| Mixer架构 | Hypernetwork | Transformer |
| 注意力机制 | 无 | 多头自注意力 |
| 状态表示 | 简单编码 | Transformer编码 |
| 智能体交互 | 隐式 | 显式（注意力） |
| 计算复杂度 | O(n) | O(n²) |
| 表达能力 | 中等 | 强 |

## 实验建议

1. **对比实验**: 与原始QMIX在相同条件下对比
2. **消融实验**: 测试不同层数和头数的影响
3. **计算资源**: 注意Transformer的计算开销
4. **超参数调优**: 针对不同地图调整参数

## 参考文献

- QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
- Attention Is All You Need (Transformer原始论文)
- Multi-Agent Reinforcement Learning with Transformer-based Value Function Factorization

