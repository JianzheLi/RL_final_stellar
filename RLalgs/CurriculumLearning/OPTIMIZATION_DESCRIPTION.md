# CurriculumLearning - 课程学习优化

## 优化思路

通过**课程学习（Curriculum Learning）**策略，从简单任务开始，逐步增加任务难度，使智能体能够更好地学习和适应复杂任务。

## 核心改进

### 1. 课程学习策略

#### 线性调度（Linear Schedule）
- **逐步增加难度**: 从简单任务开始，线性增加难度
- **时间调度**: 根据训练步数自动调整难度
- **平滑过渡**: 避免难度突然变化

#### 自适应调度（Adaptive Schedule）
- **性能驱动**: 根据智能体性能自动调整难度
- **动态调整**: 性能好时增加难度，性能差时降低难度
- **稳定学习**: 确保智能体始终在合适的难度下学习

### 2. 技术细节

#### 难度级别
- **范围**: 0.0（最简单）到 1.0（最难）
- **初始难度**: `curriculum_min_difficulty`（默认0.0）
- **最终难度**: `curriculum_max_difficulty`（默认1.0）

#### 调度方式
1. **Linear**: 线性增加难度
   ```
   难度 = min_difficulty + (max_difficulty - min_difficulty) * 进度
   ```

2. **Adaptive**: 基于性能自适应调整
   - 胜率 > 80%: 增加难度
   - 胜率 < 20%: 降低难度

#### 关键参数
- `curriculum_enabled`: True - 是否启用课程学习
- `curriculum_schedule`: "linear" - 调度方式
- `curriculum_start_step`: 0 - 开始步数
- `curriculum_end_step`: 1000000 - 结束步数
- `curriculum_min_difficulty`: 0.0 - 最小难度
- `curriculum_max_difficulty`: 1.0 - 最大难度

### 3. 优势

1. **更好的学习曲线**: 从简单到复杂，学习更稳定
2. **更快的收敛**: 避免一开始就面对困难任务
3. **更高的成功率**: 逐步建立技能，最终完成复杂任务
4. **自适应调整**: 可以根据性能动态调整难度

### 4. 预期效果

- 更快的初始学习速度
- 更稳定的训练过程
- 更高的最终性能
- 更好的任务完成率

## 使用方法

### 训练单个地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/CurriculumLearning/train_single_map.sh adcc 4 42
```

### 训练所有HLSMAC地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/CurriculumLearning/train_hlsmac.sh 4 42
```

## 配置文件

主要配置文件: `src/config/algs/curriculum_qmix.yaml`

### 关键参数说明

- `learner: "curriculum_learner"` - 使用课程学习learner
- `curriculum_enabled: True` - 启用课程学习
- `curriculum_schedule: "linear"` - 调度方式（"linear"或"adaptive"）
- `curriculum_start_step: 0` - 开始课程学习的步数
- `curriculum_end_step: 1000000` - 结束课程学习的步数

### 调优建议

1. **调度方式选择**:
   - **Linear**: 适合已知任务难度分布的情况
   - **Adaptive**: 适合任务难度不确定的情况

2. **难度范围**:
   - 根据具体任务调整`curriculum_min_difficulty`和`curriculum_max_difficulty`
   - 建议从0.0开始，逐步增加到1.0

3. **调度时间**:
   - `curriculum_end_step`应该小于`t_max`
   - 建议设置为总训练步数的50-70%

## 实现细节

### 课程学习流程
```
训练开始
    ↓
初始难度 = min_difficulty
    ↓
每个训练步骤:
    1. 更新难度（根据调度方式）
    2. 应用难度到batch
    3. 训练模型
    4. 记录课程信息
    ↓
最终难度 = max_difficulty
```

### 关键创新点
1. **灵活调度**: 支持线性和自适应两种调度方式
2. **平滑过渡**: 难度变化平滑，避免突然跳跃
3. **性能监控**: 自适应调度基于实际性能
4. **易于扩展**: 可以添加更多调度策略

## 与Baseline对比

| 特性 | Baseline | CurriculumLearning |
|------|----------|-------------------|
| 任务难度 | 固定 | 逐步增加 |
| 学习曲线 | 可能陡峭 | 更平滑 |
| 初始学习 | 可能困难 | 从简单开始 |
| 收敛速度 | 中等 | 更快 |
| 最终性能 | 基准 | 可能更高 |

## 实验建议

1. **调度方式对比**: 测试linear和adaptive的效果
2. **难度范围**: 测试不同的难度范围
3. **调度时间**: 测试不同的课程学习时间
4. **消融实验**: 测试课程学习对性能的贡献

## 扩展方向

1. **多任务课程**: 在不同任务间进行课程学习
2. **技能分解**: 将复杂任务分解为子技能
3. **自动课程生成**: 自动生成课程学习计划
4. **迁移学习**: 在不同地图间迁移课程

## 参考文献

- Curriculum Learning
- Self-Paced Learning for Latent Variable Models
- Automatic Curriculum Learning for Deep RL

