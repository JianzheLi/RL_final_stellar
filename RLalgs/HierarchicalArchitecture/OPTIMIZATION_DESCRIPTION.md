# HierarchicalArchitecture - 分层架构优化

## 优化思路

引入**分层架构**，将智能体决策分为两个层次：
1. **高层策略**：制定宏观策略（目标选择、战术决策）
2. **底层执行**：基于高层策略执行具体动作

这种设计使智能体能够更好地处理HLSMAC任务中的复杂战略决策。

## 核心改进

### 1. 分层架构设计

#### 高层策略网络（High-Level Policy）
- **功能**: 分析全局状态，制定宏观策略
- **输出**:
  - **目标选择**: 选择当前应该关注的目标（如攻击特定敌人、占领位置等）
  - **战术类型**: 选择战术策略（如集中攻击、分散防御、包围等）
- **网络结构**: GRU + 全连接层

#### 底层执行网络（Low-Level Policy）
- **功能**: 基于高层策略执行具体动作
- **输入**: 原始观测 + 高层策略信息
- **输出**: 具体动作的Q值
- **网络结构**: GRU + 全连接层

### 2. 技术细节

#### 架构流程
```
观测输入 → 高层策略网络 → 目标选择 + 战术类型
                ↓
        底层执行网络 ← 原始观测 + 高层策略信息
                ↓
            动作Q值
```

#### 关键参数
- `hierarchical_high_dim`: 128 - 高层策略网络维度
- `hierarchical_n_goals`: 8 - 目标数量（可自定义）
- `hierarchical_n_tactics`: 4 - 战术类型数量（可自定义）

### 3. 优势

1. **战略决策能力**: 高层策略网络能够制定长期战略
2. **可解释性**: 可以观察智能体的目标选择和战术决策
3. **模块化设计**: 高层和底层可以独立优化
4. **适应复杂任务**: 特别适合HLSMAC这种需要战略思考的任务

### 4. 预期效果

- 更好的战略规划能力
- 在需要长期规划的任务上表现更好
- 更高的任务完成率
- 更清晰的决策过程

## 使用方法

### 训练单个地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/HierarchicalArchitecture/train_single_map.sh adcc 2 42
```

### 训练所有HLSMAC地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/HierarchicalArchitecture/train_hlsmac.sh 2 42
```

## 配置文件

主要配置文件: `src/config/algs/hierarchical_qmix.yaml`

### 关键参数说明

- `agent: "hierarchical_rnn"` - 使用分层架构智能体
- `hierarchical_high_dim: 128` - 高层策略网络维度
- `hierarchical_n_goals: 8` - 目标数量（根据任务调整）
- `hierarchical_n_tactics: 4` - 战术类型数量（根据任务调整）

### 调优建议

1. **目标数量设置**:
   - 根据HLSMAC地图特点设置`hierarchical_n_goals`
   - 例如：攻击、防御、占领、撤退、支援等

2. **战术类型设置**:
   - 根据任务需求设置`hierarchical_n_tactics`
   - 例如：集中、分散、包围、诱敌等

3. **网络容量**:
   - 增加`hierarchical_high_dim`提升表达能力
   - 但要注意过拟合风险

## 实现细节

### 网络结构
```
输入观测
    ↓
高层策略网络 (GRU + FC)
    ├─→ 目标选择 (softmax)
    └─→ 战术类型 (softmax)
    ↓
底层执行网络 (GRU + FC)
    ├─ 输入: 原始观测 + 高层策略信息
    └─→ 动作Q值
```

### 关键创新点
1. **分层决策**: 将决策分为战略和战术两个层次
2. **信息传递**: 高层策略信息指导底层执行
3. **端到端训练**: 整个架构可以端到端训练
4. **可扩展性**: 可以添加更多层次或更复杂的策略

## 与Baseline对比

| 特性 | Baseline (单层) | HierarchicalArchitecture |
|------|----------------|-------------------------|
| 决策层次 | 单层 | 两层（高层+底层）|
| 战略规划 | 隐式 | 显式（目标+战术）|
| 可解释性 | 低 | 高 |
| 长期规划 | 弱 | 强 |
| 复杂度 | 低 | 中等 |

## 实验建议

1. **目标设计**: 根据具体HLSMAC地图设计合适的目标类型
2. **战术设计**: 设计符合三十六计策略的战术类型
3. **消融实验**: 测试高层策略对性能的贡献
4. **可视化**: 可视化高层策略的决策过程

## 扩展方向

1. **多层级架构**: 可以扩展到三层或更多层
2. **动态目标**: 根据任务进度动态调整目标
3. **策略迁移**: 在不同地图间迁移高层策略
4. **模仿学习**: 使用专家演示训练高层策略

## 参考文献

- Hierarchical Reinforcement Learning: A Comprehensive Survey
- FeUdal Networks for Hierarchical Reinforcement Learning
- Strategic Attentive Writer for Learning Macro-Actions

