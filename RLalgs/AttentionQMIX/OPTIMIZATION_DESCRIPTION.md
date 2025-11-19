# AttentionQMIX - 注意力机制改进的MIXER网络

## 优化思路

在原始QMIX的基础上，引入**多头注意力机制**来改进MIXER网络，使智能体能够更好地关注关键状态信息和智能体间的协作关系。

## 核心改进

### 1. 注意力增强的MIXER网络
- **多头注意力机制**: 在状态编码中加入多头自注意力，捕捉状态中的关键信息
- **智能体间注意力**: 在混合Q值时考虑智能体间的相互影响
- **分层注意力**: 结合局部和全局注意力，提升表达能力

### 2. 技术细节
- 在QMIX的hypernetwork中加入注意力层
- 使用Scaled Dot-Product Attention机制
- 多头注意力头数: 4-8个
- 注意力维度: 64-128

### 3. 预期效果
- 更好的状态表示学习
- 更强的智能体协作能力
- 在复杂HLSMAC任务上提升胜率

## 使用方法

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/AttentionQMIX/train_single_map.sh <map_name> <gpu_id> <seed>
```

## 配置文件

主要配置文件: `src/config/algs/attention_qmix.yaml`

关键参数:
- `mixer: "attention_qmix"` - 使用注意力QMIX
- `attention_heads: 4` - 注意力头数
- `attention_dim: 64` - 注意力维度

