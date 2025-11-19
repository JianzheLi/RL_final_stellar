import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelPolicy(nn.Module):
    """
    高层策略网络：负责制定宏观策略（如目标选择、战术决策）
    """
    def __init__(self, input_shape, args):
        super(HighLevelPolicy, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.fc1 = nn.Linear(input_shape, args.hierarchical_high_dim)
        self.fc2 = nn.Linear(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
        # 输出：目标选择、战术类型等高层决策
        self.goal_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_goals)
        self.tactic_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_tactics)
        
        self.rnn = nn.GRUCell(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        goal_logits = self.goal_head(h)
        tactic_logits = self.tactic_head(h)
        
        return {
            'goal': goal_logits,
            'tactic': tactic_logits,
            'hidden': h
        }


class LowLevelPolicy(nn.Module):
    """
    底层策略网络：负责执行具体动作（基于高层策略的指导）
    """
    def __init__(self, input_shape, args):
        super(LowLevelPolicy, self).__init__()
        self.args = args
        
        # 底层输入：原始观测 + 高层策略信息
        enhanced_input_dim = input_shape + args.hierarchical_high_dim + args.hierarchical_n_goals + args.hierarchical_n_tactics
        
        self.fc1 = nn.Linear(enhanced_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
    def forward(self, inputs, high_level_info, hidden_state=None):
        # 拼接原始观测和高层策略信息
        enhanced_input = torch.cat([
            inputs,
            high_level_info['hidden'],
            F.softmax(high_level_info['goal'], dim=-1),
            F.softmax(high_level_info['tactic'], dim=-1)
        ], dim=-1)
        
        x = F.relu(self.fc1(enhanced_input), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        q = self.fc2(h)
        q = torch.clamp(q, -5, 2)
        
        return q, h


class HierarchicalRNNAgent(nn.Module):
    """
    分层架构智能体：结合高层策略和底层执行
    """
    def __init__(self, input_shape, args):
        super(HierarchicalRNNAgent, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.high_level = HighLevelPolicy(input_shape, args)
        
        # 底层执行网络
        self.low_level = LowLevelPolicy(input_shape, args)
        
        # 高层策略的隐藏状态
        self.high_hidden = None
        self.low_hidden = None
        
    def init_hidden(self):
        high_hidden = self.high_level.fc1.weight.new(1, self.args.hierarchical_high_dim).zero_()
        low_hidden = self.low_level.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return high_hidden, low_hidden
    
    def forward(self, inputs, hidden_state=None):
        """
        Args:
            inputs: [batch, agents, features]
            hidden_state: (high_hidden, low_hidden) tuple
        Returns:
            q: [batch, agents, n_actions]
            hidden: (high_hidden, low_hidden) tuple
        """
        b, a, e = inputs.size()
        
        # 重塑输入
        inputs_flat = inputs.view(-1, e)
        
        # 获取高层策略
        if hidden_state is not None:
            high_hidden, low_hidden = hidden_state
            high_hidden = high_hidden.reshape(-1, self.args.hierarchical_high_dim)
            low_hidden = low_hidden.reshape(-1, self.args.rnn_hidden_dim)
        else:
            high_hidden = None
            low_hidden = None
        
        high_level_info = self.high_level(inputs_flat, high_hidden)
        new_high_hidden = high_level_info['hidden']
        
        # 获取底层Q值
        q, new_low_hidden = self.low_level(inputs_flat, high_level_info, low_hidden)
        
        # 重塑输出
        q = q.view(b, a, -1)
        new_high_hidden = new_high_hidden.view(b, a, -1)
        new_low_hidden = new_low_hidden.view(b, a, -1)
        
        return q, (new_high_hidden, new_low_hidden)

