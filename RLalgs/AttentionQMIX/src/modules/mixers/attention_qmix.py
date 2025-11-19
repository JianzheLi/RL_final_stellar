import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = th.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = th.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        output = self.out_proj(attn_output)
        return output, attn_weights


class AttentionQMixer(nn.Module):
    """带注意力机制的QMIX Mixer"""
    def __init__(self, args):
        super(AttentionQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        
        # 注意力参数
        self.attention_heads = getattr(args, 'attention_heads', 4)
        self.attention_dim = getattr(args, 'attention_dim', 64)
        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, 'abs', True)

        # 状态编码器（带注意力）
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.attention_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.attention_dim * 2, self.attention_dim)
        )
        
        # 多头注意力层
        self.attention = MultiHeadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads,
            dropout=getattr(args, 'attention_dropout', 0.1)
        )
        
        # 注意力后的状态处理
        self.attention_proj = nn.Sequential(
            nn.Linear(self.attention_dim, self.embed_dim),
            nn.ReLU(inplace=True)
        )

        # Hypernetworks (使用注意力增强的状态)
        hypernet_embed = getattr(args, 'hypernet_embed', 64)
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.embed_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.embed_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.embed_dim, hypernet_embed),
                nn.ReLU(inplace=True),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.embed_dim, hypernet_embed),
                nn.ReLU(inplace=True),
                nn.Linear(hypernet_embed, self.embed_dim)
            )
        else:
            raise Exception("Only 1 or 2 hypernet layers supported!")

        # State dependent bias
        self.hyper_b_1 = nn.Linear(self.embed_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states, dropout=False):
        bs = agent_qs.size(0)
        seq_len = agent_qs.size(1)
        
        if dropout:
            states = states.reshape(states.shape[0], states.shape[1], 1, states.shape[2]).repeat(1, 1, self.n_agents, 1)
        
        states = states.reshape(-1, self.state_dim)
        
        # 状态编码
        encoded_states = self.state_encoder(states)
        
        # 自注意力（query=key=value=encoded_states）
        # 将状态扩展为序列形式以应用注意力
        encoded_states = encoded_states.unsqueeze(1)  # [batch*seq, 1, attention_dim]
        attn_states, _ = self.attention(encoded_states, encoded_states, encoded_states)
        attn_states = attn_states.squeeze(1)  # [batch*seq, attention_dim]
        
        # 投影到embed_dim
        enhanced_states = self.attention_proj(attn_states)
        
        # 使用增强的状态生成hypernetwork权重
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        
        # First layer
        w1 = self.hyper_w_1(enhanced_states).abs() if self.abs else self.hyper_w_1(enhanced_states)
        b1 = self.hyper_b_1(enhanced_states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = self.hyper_w_final(enhanced_states).abs() if self.abs else self.hyper_w_final(enhanced_states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(enhanced_states).view(-1, 1, 1)
        
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(bs, seq_len, 1)
        
        return q_tot

    def k(self, states):
        """计算智能体权重（用于分析）"""
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
        
        encoded_states = self.state_encoder(states)
        encoded_states = encoded_states.unsqueeze(1)
        attn_states, _ = self.attention(encoded_states, encoded_states, encoded_states)
        attn_states = attn_states.squeeze(1)
        enhanced_states = self.attention_proj(attn_states)
        
        w1 = th.abs(self.hyper_w_1(enhanced_states))
        w_final = th.abs(self.hyper_w_final(enhanced_states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1, w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        """计算状态相关的偏置"""
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
        
        encoded_states = self.state_encoder(states)
        encoded_states = encoded_states.unsqueeze(1)
        attn_states, _ = self.attention(encoded_states, encoded_states, encoded_states)
        attn_states = attn_states.squeeze(1)
        enhanced_states = self.attention_proj(attn_states)
        
        w_final = th.abs(self.hyper_w_final(enhanced_states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(enhanced_states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(enhanced_states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b

