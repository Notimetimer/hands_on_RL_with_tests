import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboard_visualize import TensorBoardLogger
import matplotlib.pyplot as plt

# 导入 PettingZoo 的并行环境 API
from pettingzoo.mpe import simple_spread_v3

# ============================= Hyperparameters =============================+
# 将所有训练/环境/模型相关的可调超参数集中到文件开头，方便修改与复现。
# 只收集那些当前不是类/函数签名默认值的参数。
ACTOR_HIDDEN_DIMS = [128, 128] # [64, 64]
CRITIC_HIDDEN_DIMS_BACK = [128, 128] # [128]
CRITIC_HIDDEN_DIMS_FRONT = [64]

# 环境相关
ENV_N = 2 # 3 个以上就学不会了?
ENV_MAX_CYCLES = 100

# 训练相关
NUM_EPISODES = 1000 # 2000
UPDATE_INTERVAL = 10
BATCH_SIZE = 128

# 测试/绘图相关
MAX_TEST_STEPS = 1000
# ===========================================================================
# ================================================================= #
#                          工具与网络定义                           #
# ================================================================= #

def compute_gae(rewards, values, next_values, dones, truncs, gamma, lam):
    """
    计算广义优势估计 (GAE)
    区别：terminated (死亡/通关) -> next_value 为 0
          truncated (超时)     -> next_value 正常计算
    """
    advantages = torch.zeros_like(rewards).to(rewards.device)
    last_gae_lam = 0
    step_num = rewards.shape[0]
    
    for t in reversed(range(step_num)):
        # 如果是 terminated，下一个状态价值视为 0
        next_non_terminal = 1.0 - dones[t]
        
        # TD误差计算
        delta = rewards[t] + gamma * next_values[t] * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
        
    returns = advantages + values
    return advantages, returns


def _get_positions_from_env(env):
    """尝试从 env 或其内部 world 提取智能体与地标的全局坐标。
    返回 (agent_pos_dict, landmark_pos_list) 或 (None, None) 如果无法获取。
    """
    try:
        # 常见访问方式
        world = None
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'world'):
            world = env.unwrapped.world
        elif hasattr(env, 'world'):
            world = env.world

        if world is not None:
            agent_pos = {ag.name: np.array(ag.state.p_pos) for ag in world.agents}
            landmark_pos = [np.array(l.state.p_pos) for l in world.landmarks]
            return agent_pos, landmark_pos
    except Exception:
        pass

    # 其他回退方案可加在这里（基于 infos 等），当前返回 None
    return None, None

class OfficialMHALayerWithMask(nn.Module):
    """
    使用 PyTorch 官方 nn.MultiheadAttention 实现的完美平替版本。
    速度更快，底层自动优化，支持多头。
    """
    def __init__(self, in_features, out_features, n_heads=1, dropout=0.0):
        super().__init__()
        
        # 保证输入维度与 MHA 期望的特征维度一致
        self.in_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        # 官方 MHA (设置 batch_first=True 让输入形状保持为 [B, N, E])
        self.mha = nn.MultiheadAttention(
            embed_dim=out_features, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True
        )

    def forward(self, h, adj_mask=None, edge_mask=None, return_attention=False):
        """
        h: (B, N, in_features)
        adj_mask: (B, N) -> 1为存活，0为死亡/Padding
        edge_mask: (B, N, N) -> 1为允许看，0为禁止看
        """
        h_proj = self.in_proj(h)
        
        # 1. 转换 Padding 掩码 (给 key_padding_mask 使用)
        # PyTorch 规则: True 代表被忽略 (所以我们要对 1/0 逻辑取反)
        key_padding_mask = None
        if adj_mask is not None:
            key_padding_mask = ~(adj_mask.bool())  # (B, N)

        # 2. 转换结构/拓扑掩码 (给 attn_mask 使用)
        # PyTorch 规则: True / -inf 代表不允许看。形状必须是 (B * n_heads, N, N)
        attn_mask = None
        if edge_mask is not None:
            bool_edge_mask = ~(edge_mask.bool())  # (B, N, N)，反转逻辑
            n_heads = self.mha.num_heads
            # 按 Batch 维度复制 n_heads 次，以符合官方 API 强制要求
            attn_mask = bool_edge_mask.repeat_interleave(n_heads, dim=0)

        # 3. 前向传播
        # need_weights=return_attention, average_attn_weights=True 保证多头融合后的权重兼容旧代码
        attn_output, attn_weights = self.mha(
            query=h_proj, 
            key=h_proj, 
            value=h_proj,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=True 
        )

        if return_attention:
            return attn_output, attn_weights
        else:
            return attn_output

# 上面那个的手写版本
class QKVLayerWithEdgeMask(nn.Module):
    """
    轻量级 QKV 缩放点积注意力层，作为原 GATLayerWithEdgeMask 的完美平替。
    """
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(QKVLayerWithEdgeMask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        # 分离的 Q, K, V 线性映射 (不再共享同一个 W)
        self.W_q = nn.Linear(in_features, out_features, bias=False)
        self.W_k = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        
        # 缩放因子 (1 / sqrt(d_k))
        self.scale = math.sqrt(out_features)
        
        # 兼容原版的激活逻辑 (多头拼接时使用 ELU，最后一层取特征时不激活)
        self.activation = F.elu if self.concat else nn.Identity()

    def forward(self, h, adj_mask=None, edge_mask=None, return_attention=False):
        """
        参数:
            h: (B, N, in_features) - 节点特征
            adj_mask: (B, N) - 节点活跃度掩码 (1=存活/真实, 0=死亡/Padding)
            edge_mask: (B, N, N) - 拓扑边掩码 (1=允许看, 0=禁止看)
        """
        # 1. 计算 Q, K, V
        Q = self.W_q(h)  # (B, N, out_features)
        K = self.W_k(h)  # (B, N, out_features)
        V = self.W_v(h)  # (B, N, out_features)
        
        # 2. 缩放点积打分 (B, N, N)
        # Q: (B, N, out), K^T: (B, out, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 3. 构建联合掩码 (合并结构掩码与 Padding 掩码)
        B, N, _ = h.shape
        valid_mask = torch.ones(B, N, N, dtype=torch.bool, device=h.device)
        
        # 应用边掩码 (结构化约束)
        if edge_mask is not None:
            valid_mask = valid_mask & edge_mask.bool()
            
        # 应用节点掩码 (Padding约束)
        if adj_mask is not None:
            # 屏蔽掉作为 Target (Query) 的无效节点
            valid_mask = valid_mask & adj_mask.unsqueeze(2).bool()
            # 屏蔽掉作为 Source (Key/Value) 的无效节点
            valid_mask = valid_mask & adj_mask.unsqueeze(1).bool()

        # 4. 掩码阻断：将不允许关注的地方设为 -1e9 (Softmax后趋于0)
        scores = scores.masked_fill(~valid_mask, -1e9)
        
        # 5. 注意力权重及特征聚合
        attention = F.softmax(scores, dim=-1)
        h_prime = torch.matmul(attention, V)
        
        output = self.activation(h_prime)
        
        if return_attention:
            return output, attention
        else:
            return output


class GATBlock(nn.Module):
    """GAT block with optional multi-head support, residual connection and LayerNorm.
    Uses one or multiple internal OfficialMHALayerWithMask instances (heads).
    """
    def __init__(self, in_dim, out_dim, n_heads=1, dropout=0.0, alpha=0.2, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # 随机失活率，用于防止过拟合，强化学习建议关闭
        self.ln = nn.LayerNorm(in_dim) # 层归一化，用于防止过拟合
        # alpha 是leakyrelu的负半轴斜率

        # If multiple heads, split out_dim across heads (evenly)
        if n_heads == 1:
            self.heads = nn.ModuleList([OfficialMHALayerWithMask(in_dim, out_dim, dropout=0.0)])
            self.merge_linear = None
        else:
            head_dim = out_dim // n_heads
            self.heads = nn.ModuleList([OfficialMHALayerWithMask(in_dim, head_dim, dropout=0.0) for _ in range(n_heads)])
            # linear to combine concatenated heads into out_dim
            self.merge_linear = nn.Linear(head_dim * n_heads, out_dim)

        # projection for residual when dims mismatch
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

    def forward(self, h, adj_mask=None, edge_mask=None, return_attention=False):
        # pre-norm
        h_norm = self.ln(h)

        attns = []
        outs = []
        for head in self.heads:
            out = head(h_norm, adj_mask=adj_mask, edge_mask=edge_mask, return_attention=return_attention)
            if return_attention:
                out_feat, attn = out
                outs.append(out_feat)
                attns.append(attn)
            else:
                outs.append(out)

        if self.n_heads == 1:
            out = outs[0]
        else:
            # concat along feature dim
            out = torch.cat(outs, dim=-1)
            if self.merge_linear is not None:
                out = self.merge_linear(out)

        out = self.dropout(out)

        # residual
        res = self.res_proj(h) if self.res_proj is not None else h
        out = out + res

        if return_attention:
            # average attention across heads
            attn = torch.stack(attns, dim=0).mean(dim=0)
            return out, attn
        return out

class StructuredGATActor(nn.Module):
    """
    具备 2 层 GAT 结构的 Actor：
    - 能够理解：队友(Teammate) 与 地标(Landmark) 之间的关系
    - 逻辑：Layer1 (队友看地标) -> Layer2 (我看队友对地标的意图)
    """
    def __init__(self, n_agents, n_landmarks, node_dim, hidden_dim, action_dim):
        super().__init__()
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.total_nodes = 1 + (n_agents - 1) + n_landmarks # 自己 + 队友 + 地标
        self.hidden_dim = hidden_dim
        
        # 异质特征编码器（对应 Critic 的编码逻辑）
        self.self_encoder = nn.Linear(4, hidden_dim)          # 自己: pos(2), vel(2)
        self.other_agent_encoder = nn.Linear(2, hidden_dim)   # 队友: rel_pos(2)
        self.landmark_encoder = nn.Linear(2, hidden_dim)      # 地标: rel_pos(2)
        
        # 单层
        self.gat1 = GATBlock(hidden_dim, hidden_dim, n_heads=1, dropout=0.0, alpha=0.2, concat=False)

        # # 两层 GAT，带边掩码支持
        # # use GATBlock (pre-norm residual + optional multi-head). n_heads=1 to preserve parameters.
        # self.gat1 = GATBlock(hidden_dim, hidden_dim, n_heads=1, dropout=0.0, alpha=0.2, concat=True)
        # self.gat2 = GATBlock(hidden_dim, hidden_dim, n_heads=1, dropout=0.0, alpha=0.2, concat=False)
        
        # 最终决策输出
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def _create_edge_mask(self, batch_size, device):
        """
        按照你的要求精确裁剪拓扑：
        - 节点 0: 自己
        - 节点 1 ~ N-1: 队友
        - 节点 N ~ End: 地标
        """
        # 初始化全 0 (Batch, Target, Source)
        mask = torch.zeros(batch_size, self.total_nodes, self.total_nodes, device=device)
        
        # 1. 我 (0) 和 所有人 (1~End) 必须互看
        mask[:, 0, :] = 1  # 我看所有人
        mask[:, :, 0] = 1  # 所有人看我 (为了让他们躲开我)
        
        # 2. 队友 (1~N-1) 必须能看到 地标 (N~End)
        # 这是二阶推理的核心：第一层 GAT 后，队友特征里会带有地标信息
        mask[:, 1:self.n_agents, self.n_agents:] = 1
        
        # 3. 敲掉：队友(1~N-1) 之间互看 -> 保持 mask 对应位置为 0
        # 4. 敲掉：地标(N~End) 之间互看 -> 保持 mask 对应位置为 0
        # 5. 敲掉：地标(N~End) 看 智能体 -> 保持 mask 对应位置为 0
        
        # 6. 允许自环 (自己看自己，队友看自己，地标看自己 - 保持特征不丢失)
        for i in range(self.total_nodes):
            mask[:, i, i] = 1
        return mask

    def _parse_obs(self, obs):
        """
        根据 MPE simple_spread 布局拆分节点:
        obs: [self_pos(2), self_vel(2), other_agents_rel_pos(2*(N-1)), landmarks_rel_pos(2*M)]
        """
        B = obs.shape[0]
        f_self = obs[:, 0:4]
        others_end = 4 + 2 * (self.n_agents - 1)
        f_others = obs[:, 4:others_end].view(B, self.n_agents - 1, 2)
        f_lms = obs[:, others_end:others_end + 2 * self.n_landmarks].view(B, self.n_landmarks, 2)
        return f_self, f_others, f_lms

    def forward(self, obs, active_mask=None, agent_ids=None):
        B = obs.shape[0]
        device = obs.device
        
        # 1. 特征拆分与编码
        f_self, f_others, f_lms = self._parse_obs(obs)
        h_self = self.self_encoder(f_self).unsqueeze(1)    # (B, 1, H)
        h_others = self.other_agent_encoder(f_others)     # (B, N-1, H)
        h_lms = self.landmark_encoder(f_lms)              # (B, M, H)
        
        # 拼接为图节点矩阵 (B, 1+(N-1)+M, H)
        h = torch.cat([h_self, h_others, h_lms], dim=1)    # (B, Total_Nodes, H)
        
        # 2. 生成边掩码 (Landmark出发边为0)
        edge_mask = self._create_edge_mask(B, device)
        
        # 3. 单层 GAT 传递
        h = self.gat1(h, edge_mask=edge_mask)

        # # 3. 两层 GAT 传递
        # # Layer 1: 所有 Agent 收集周围 Agent 和地标的信息
        # h = self.gat1(h, edge_mask=edge_mask)
        # # Layer 2: 进一步精炼信息（实现二阶协同推理）
        # h = self.gat2(h, edge_mask=edge_mask)
        
        # 4. 只取“自己”节点进行动作输出
        self_feat = h[:, 0, :]
        logits = self.output(self_feat)
        return logits
    


class GATLayerWithEdgeMask(nn.Module):
    """改进的 GAT 层，支持边级掩码（边掩码优先级高于节点掩码）"""
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(GATLayerWithEdgeMask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mask=None, edge_mask=None, return_attention=False):
        """
        参数:
            h: (B, N, in_features) - 节点特征
            adj_mask: (B, N) - 节点活跃度掩码
            edge_mask: (B, N, N) - 边掩码矩阵，1=允许，0=禁止
            return_attention: bool - 是否返回注意力权重
        """
        Wh = self.W(h)
        B, N, _ = Wh.size()

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # 应用边掩码（高优先级）
        if edge_mask is not None:
            e = e.masked_fill(edge_mask == 0, -1e9)
        
        # 应用节点掩码
        if adj_mask is not None:
            mask = adj_mask.unsqueeze(1)
            e = e.masked_fill(mask == 0, -1e9)

        attention = F.softmax(e, dim=-1)
        h_prime = torch.matmul(attention, Wh)
        output = F.elu(h_prime) if self.concat else h_prime
        
        if return_attention:
            return output, attention
        else:
            return output


class StructuredGATCritic(nn.Module):
    """
    改进的 Critic：显式建模 Agent-Landmark 图结构
    - 前 N 个节点：agents
    - 后 M 个节点：landmarks
    
    掩码策略：
    - Agent → Agent：✓ 允许
    - Agent → Landmark：✓ 允许  
    - Landmark → Any：✗ 禁止（landmarks 被动）
    """
    def __init__(self, n_agents, node_dim, n_landmarks, hidden_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.total_nodes = n_agents + n_landmarks
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        self.agent_encoder = nn.Linear(node_dim, hidden_dim)
        self.landmark_encoder = nn.Linear(2, hidden_dim)
        
        # use GATBlock (pre-norm residual + optional multi-head). n_heads=1 keeps structure minimal.
        self.gat1 = GATBlock(hidden_dim, hidden_dim, n_heads=1, dropout=0.0)
        self.gat2 = GATBlock(hidden_dim, hidden_dim, n_heads=1, dropout=0.0, concat=False)
        
        self.v_head = nn.Linear(hidden_dim, 1)

    def _extract_landmark_features(self, obs_matrix):
        """
        从观察向量中提取 landmark 相对位置特征
        
        Simple Spread V3 观察布局：
        [0:2] - 自己位置, [2:4] - 自己速度
        [4:4+2*(N-1)] - 其他 agents 相对位置
        [4+2*(N-1):] - landmarks 相对位置
        """
        batch_size = obs_matrix.shape[0]
        landmark_offset = 4 + 2 * (self.n_agents - 1)
        
        if obs_matrix.shape[2] > landmark_offset:
            landmark_end = landmark_offset + 2 * self.n_landmarks
            landmark_features = obs_matrix[:, 0, landmark_offset:landmark_end]
            return landmark_features.view(batch_size, self.n_landmarks, 2)
        else:
            return torch.zeros(batch_size, self.n_landmarks, 2, device=obs_matrix.device)

    def _create_edge_mask(self, batch_size, device):
        """
        Critic 全局掩码：
        - 节点 0 ~ N-1: 所有的 Agents
        - 节点 N ~ End: 所有的 Landmarks
        """
        # 初始全 0
        mask = torch.zeros(batch_size, self.total_nodes, self.total_nodes, device=device)
        
        # 规则 1: 所有 Agents (0 ~ n_agents-1) 作为 Target 时，可以看所有人
        # (这样 Agent 之间能互看，Agent 也能看地标)
        mask[:, :self.n_agents, :] = 1
        
        # 规则 2: 所有 Landmarks (n_agents ~ End) 作为 Target 时，不准看任何人
        # (这符合地标是被动观测物的逻辑，这一行保持为 0 即可)
        
        # 规则 3: 允许自环
        for i in range(self.total_nodes):
            mask[:, i, i] = 1
            
        return mask

    def forward(self, obs_matrix, active_mask=None):
        """
        参数:
            obs_matrix: (B, N, obs_dim)
            active_mask: (B, N)
        返回:
            values: (B, N) - agents 的价值
        """
        B, N, _ = obs_matrix.shape
        device = obs_matrix.device
        
        # 编码 agents
        agent_features = F.relu(self.agent_encoder(obs_matrix))  # (B, N, H)
        
        # 编码 landmarks
        landmark_features_raw = self._extract_landmark_features(obs_matrix)
        landmark_features = F.relu(self.landmark_encoder(landmark_features_raw))  # (B, M, H)
        
        # 拼接构建完整图
        h = torch.cat([agent_features, landmark_features], dim=1)  # (B, N+M, H)
        
        # 扩展 active mask
        if active_mask is not None:
            landmarks_active = torch.ones(B, self.n_landmarks, device=device)
            full_active_mask = torch.cat([active_mask, landmarks_active], dim=1)
        else:
            full_active_mask = None
        
        # 创建边掩码
        edge_mask = self._create_edge_mask(B, device)
        
        # 应用 GAT
        h = self.gat1(h, adj_mask=full_active_mask, edge_mask=edge_mask)
        h = self.gat2(h, adj_mask=full_active_mask, edge_mask=edge_mask)
        
        # 输出价值
        all_values = self.v_head(h).squeeze(-1)  # (B, N+M)
        agent_values = all_values[:, :N]  # (B, N)
        
        return agent_values

class RolloutBuffer:
    """专为 MARL 设计的顺序轨迹缓冲区，方便进行 GAE 计算"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.data = {
            'obs': [], 'global_states': [], 'actions': [], 'rewards': [],
            'next_global_states': [], 'dones': [], 
            'truncs': [], 'agent_ids': [], 'active_masks': [], 'next_active_masks': []
        }
        
    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.data[key].append(value)
            
    def get_tensors(self, device):
        # Convert lists of arrays/scalars into properly shaped tensors.
        out = {}
        for k, v in self.data.items():
            arr = np.array(v)
            # agent_ids should be integer type
            if k == 'agent_ids':
                out[k] = torch.tensor(arr, dtype=torch.long).to(device)
            else:
                out[k] = torch.tensor(arr, dtype=torch.float32).to(device)
        return out

class MAPPO:
    def __init__(self, obs_dim, action_dim, num_agents, 
                 actor_hidden_dims, critic_hidden_dims_back, critic_hidden_dims_front,
                 actor_lr=3e-4, critic_lr=5e-4, gamma=0.99, lmbda=0.95, epochs=10, eps=0.2, 
                 device='cpu', k_entropy=0.1):
        
        self.num_agents = num_agents
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.k_entropy = k_entropy
        
        # Use the new GAT-based actor and critic
        # actor: expects per-agent observation vector
        actor_hidden = actor_hidden_dims[0] if len(actor_hidden_dims) > 0 else 128
        critic_hidden = critic_hidden_dims_front[0] if len(critic_hidden_dims_front) > 0 else 64
        self.actor = StructuredGATActor(
            n_agents=num_agents,
            n_landmarks=num_agents,
            node_dim=obs_dim, 
            hidden_dim=actor_hidden, 
            action_dim=action_dim
        ).to(device)
        # critic: expects (batch, N, obs_dim) - 使用改进的 StructuredGATCritic 支持边级掩码
        self.critic = StructuredGATCritic(
            n_agents=num_agents,
            node_dim=obs_dim,
            n_landmarks=num_agents,  # Simple Spread V3 通常 landmarks = agents
            hidden_dim=critic_hidden
        ).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, obs, explore=True, agent_id=None):
        state = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
        agent_id_tensor = torch.tensor([agent_id], dtype=torch.long).to(self.device) if agent_id is not None else None
        logits = self.actor(state, agent_ids=agent_id_tensor)
        
        if explore:
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
        else:
            action = torch.argmax(logits, dim=1)
            
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return probs, action.item()

    def update(self, buffer, batch_size=64):
        data = buffer.get_tensors(self.device)
        
        obs = data['obs']
        global_states = data['global_states'] # 特别注意：global_states 现在应该是 (Total_Steps, N, Obs_Dim)
        actions = data['actions'].long().view(-1, 1)
        rewards = data['rewards'].view(-1, 1)
        next_global_states = data['next_global_states']
        dones = data['dones'].view(-1, 1)
        truncs = data['truncs'].view(-1, 1)
        agent_ids = data['agent_ids']  # (Total_Steps,)
        agent_ids = agent_ids.view(-1)
        agent_ids_one_hot = F.one_hot(agent_ids, num_classes=self.num_agents).float()
        active_masks = data['active_masks']  # (Total_Steps, N)
        next_active_masks = data.get('next_active_masks', None)

        # 1. 预计算所有状态的 Value 和 优势函数 (GAE)
        with torch.no_grad():
            # 注意：现在的 global_states 形状必须是 (Batch, Num_Agents, Obs_Dim)
            all_values = self.critic(global_states, active_mask=active_masks) # (Total_Steps, N)
            # pass next_active_masks to critic for next state value computation
            if next_active_masks is not None:
                all_next_values = self.critic(next_global_states, active_mask=next_active_masks)
            else:
                all_next_values = self.critic(next_global_states, active_mask=None)
            
            # 关键：我们需要从全队 Value 中，选出当时产生这条 trajectory 的那个 agent 的 value
            # agent_ids 形状 (Total_Steps,)
            agent_idx = agent_ids.long().view(-1, 1)  # (T,1)
            values = all_values.gather(1, agent_idx)  # (T,1)
            next_values = all_next_values.gather(1, agent_idx)
            
            
            # 使用包含 term 和 trunc 处理的自定义 GAE
            advantages, td_targets = compute_gae(
                rewards, values, next_values, dones, truncs, self.gamma, self.lmbda
            )
            
            # 仅对有效数据进行优势归一化 —— 只考虑当时该样本对应 agent 是否为活跃
            agent_active_mask = active_masks.gather(1, agent_idx)
            active_adv = advantages[agent_active_mask.bool()]
            if active_adv.numel() > 1:
                advantages = (advantages - active_adv.mean()) / (active_adv.std() + 1e-8)
                
            old_logits = self.actor(obs, agent_ids=agent_ids.long().view(-1))
            old_log_probs = Categorical(logits=old_logits).log_prob(actions.squeeze()).view(-1, 1)

        dataset_size = obs.shape[0]

        # prepare lists to collect losses
        actor_loss_list = []
        critic_loss_list = []
        entropy_list = []

        # 2. 多 Epoch PPO 更新
        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                mb_idx = indices[start:start + batch_size]
                
                mb_obs = obs[mb_idx]
                mb_global_states = global_states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_agent_ids_one_hot = agent_ids_one_hot[mb_idx]
                mb_agent_ids_vec = agent_ids[mb_idx].long().view(-1)
                mb_agent_ids = mb_agent_ids_vec.view(-1, 1)
                mb_adv = advantages[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_td_targets = td_targets[mb_idx]
                mb_active_masks = active_masks[mb_idx]
                mb_next_active_masks = next_active_masks[mb_idx] if next_active_masks is not None else None
                
                # --- Actor Loss ---
                # 现在的 mb_obs 也要送入 AttentionActor
                logits = self.actor(mb_obs, agent_ids=mb_agent_ids_vec)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions.squeeze()).view(-1, 1)
                
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_adv
                
                actor_loss_per_sample = -torch.min(surr1, surr2)
                entropy_per_sample = dist.entropy().view(-1, 1)
                
                # 使用 Active Mask 加权 —— 只对当前 agent 的 active 位进行加权
                mb_agent_active = mb_active_masks.gather(1, mb_agent_ids)
                active_sum = mb_agent_active.sum() + 1e-5
                actor_loss = (actor_loss_per_sample * mb_agent_active).sum() / active_sum
                entropy = (entropy_per_sample * mb_agent_active).sum() / active_sum
                tot_actor_loss = actor_loss - self.k_entropy * entropy
                
                # --- Critic Loss ---
                mb_all_values = self.critic(mb_global_states, active_mask=mb_active_masks) # (Batch, N)
                mb_values = mb_all_values.gather(1, mb_agent_ids).view(-1, 1)
                critic_loss = F.mse_loss(mb_values, mb_td_targets)
                
                # --- 优化 ---
                self.actor_optimizer.zero_grad()
                tot_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                actor_loss_list.append(actor_loss.item() if isinstance(actor_loss, torch.Tensor) else float(actor_loss))
                critic_loss_list.append(critic_loss.item() if isinstance(critic_loss, torch.Tensor) else float(critic_loss))
                entropy_list.append(entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy))
            # average over collected minibatches
            self.actor_loss = float(np.mean(actor_loss_list)) if len(actor_loss_list) > 0 else 0.0
            self.critic_loss = float(np.mean(critic_loss_list)) if len(critic_loss_list) > 0 else 0.0
            self.entropy = float(np.mean(entropy_list)) if len(entropy_list) > 0 else 0.0

# ================================================================= #
#                       训练主循环 (PettingZoo)                     #
# ================================================================= #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard logger (在 main 中管理，不在 MAPPO 类内)
    logger = TensorBoardLogger(log_root="./logs", auto_show=False)

    # 1. 初始化 PettingZoo 并行环境
    # simple_spread_v3 是一个协作环境，N个智能体需要分别占领N个地标
    env = simple_spread_v3.parallel_env(N=ENV_N, max_cycles=ENV_MAX_CYCLES, continuous_actions=False)
    obs_dict, infos = env.reset()
    agents = env.agents
    num_agents = len(agents)
    
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).n

    # 2. 初始化 MAPPO
    mappo = MAPPO(
        obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS,
        critic_hidden_dims_back=CRITIC_HIDDEN_DIMS_BACK,
        critic_hidden_dims_front=CRITIC_HIDDEN_DIMS_FRONT,
        device=device
    )
    buffer = RolloutBuffer()
    
    num_episodes = NUM_EPISODES # 20000
    update_interval = UPDATE_INTERVAL  # 每收集若干回合的数据更新一次

    try:
        for ep in range(num_episodes):
            obs_dict, _ = env.reset()
            episode_reward = 0

            # per-episode container to collect attention weights per agent
            attn_by_agent = {ag: {'lm': [], 'other': []} for ag in agents}

            # distance accumulators for this episode
            sum_min_dist = 0.0
            steps_with_dist = 0
            per_agent_sums = np.zeros(num_agents, dtype=np.float32)
            per_agent_counts = np.zeros(num_agents, dtype=np.int32)
            
            while env.agents:
                # 组装上帝视角 global_state (将所有智能体的obs拼接到一起)
                # 形状：(num_agents * obs_dim, )
                global_state = np.stack([obs_dict[agent] for agent in agents]) # [N, Dim]
                
                actions = {}
                for agent_id, agent in enumerate(agents):
                    # 如果这个智能体在当前 step 存活（在 obs_dict 中），就获取动作
                    if agent in obs_dict:
                        _, act = mappo.take_action(obs_dict[agent], explore=True)
                        actions[agent] = act
                        # 读取并存下该 agent 本次前向的注意力权重（若有）
                        # 注意：通过GAT实现，权重结构已改为统一的注意力而非分离的lm/other
                        try:
                            w = getattr(mappo.actor, 'last_attn_weights', None)
                            if w is not None and isinstance(w, torch.Tensor):
                                # w: (B, 1, 1) -> 提取单个样本的权重值
                                w_scalar = float(w[0, 0, 0].item())  # 单节点自注意力值
                                # 为了保持与原有结构的兼容性，同时追加到lm和other
                                attn_by_agent[agent]['lm'].append(w_scalar)
                                attn_by_agent[agent]['other'].append(w_scalar)
                        except Exception:
                            pass

                
                # 环境前进一步
                next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
                
                # 组装下一个上帝视角 next_global_state
                # 注意：如果某个智能体死亡，PettingZoo 会把它从字典中剔除。
                # 这里为了保持 global_state 维度不变，我们使用 0 兜底并使用 stack 保持 (N, obs_dim)
                next_global_state_list = []
                for agent in agents:
                    if agent in next_obs_dict:
                        next_global_state_list.append(next_obs_dict[agent])
                    else:
                        next_global_state_list.append(np.zeros(obs_dim)) # 用 0 兜底
                next_global_state = np.stack(next_global_state_list)

                # --- 计算距离指标（尝试从 env 提取全局坐标） ---
                agent_positions, landmark_positions = _get_positions_from_env(env)
                if landmark_positions is not None:
                    # 对每个智能体，计算到最近地标的距离，并记录整个环境的最小值
                    min_candidates = []
                    for agent_id, agent in enumerate(agents):
                        if agent in next_obs_dict:
                            pos = agent_positions.get(agent, None)
                            if pos is None:
                                # 回退：尝试从 obs 中取前两维作为位置
                                try:
                                    pos = np.array(next_obs_dict[agent][:2])
                                except Exception:
                                    pos = None
                            if pos is not None:
                                dists = [np.linalg.norm(pos - lp) for lp in landmark_positions]
                                if len(dists) > 0:
                                    nearest = float(np.min(dists))
                                    per_agent_sums[agent_id] += nearest
                                    per_agent_counts[agent_id] += 1
                                    min_candidates.append(nearest)
                    if len(min_candidates) > 0:
                        sum_min_dist += float(np.min(min_candidates))
                        steps_with_dist += 1

                # 计算本步与下步的 active mask 向量 (用于 Attention/critic)
                active_mask_vec = np.array([1.0 if agent in obs_dict else 0.0 for agent in agents], dtype=np.float32)
                next_active_mask_vec = np.array([1.0 if agent in next_obs_dict else 0.0 for agent in agents], dtype=np.float32)

                # 将每个智能体的数据存入 Buffer（每个样本包含整队的 global_state 与 active mask）
                for agent_id, agent in enumerate(agents):
                    is_active = float(active_mask_vec[agent_id])

                    agent_obs = obs_dict[agent] if is_active else np.zeros(obs_dim)
                    agent_next_obs = next_obs_dict[agent] if agent in next_obs_dict else np.zeros(obs_dim)
                    agent_action = actions[agent] if is_active else 0
                    agent_reward = rewards[agent] if is_active else 0.0
                    agent_term = terminations[agent] if agent in terminations else True
                    agent_trunc = truncations[agent] if agent in truncations else False

                    buffer.add(
                        obs=agent_obs, global_states=global_state, actions=agent_action,
                        rewards=agent_reward, next_global_states=next_global_state,
                        dones=agent_term, truncs=agent_trunc, agent_ids=agent_id,
                        active_masks=active_mask_vec, next_active_masks=next_active_mask_vec
                    )

                    if is_active:
                        episode_reward += agent_reward
                
                obs_dict = next_obs_dict
                
            # 打印日志
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{num_episodes}, Total Reward: {episode_reward:.2f}")

            # 更新网络
            if (ep + 1) % update_interval == 0:
                mappo.update(buffer, batch_size=BATCH_SIZE)

                # --- 在这里通过读取 MAPPO 的属性，将指标写入 TensorBoard ---
                try:
                    logger.add("episode/total_reward", float(episode_reward), ep)
                except Exception:
                    pass
                try:
                    logger.add("loss/actor", float(mappo.actor_loss), ep)
                    logger.add("loss/critic", float(mappo.critic_loss), ep)
                    logger.add("entropy/entropy", float(mappo.entropy), ep)
                except Exception:
                    pass

                # 距离相关的平均值（如果有收集到）
                if steps_with_dist > 0:
                    avg_min_dist = sum_min_dist / steps_with_dist
                    logger.add("distance/min_dist", float(avg_min_dist), ep)
                    for agent_id in range(num_agents):
                        if per_agent_counts[agent_id] > 0:
                            avg_agent_nearest = per_agent_sums[agent_id] / per_agent_counts[agent_id]
                            logger.add(f"distance/agent_{agent_id}_nearest", float(avg_agent_nearest), ep)

                buffer.reset()
    except:
        pass # 允许中断训练
    
    print("Training finished!")
    # 关闭 TensorBoard logger
    try:
        logger.close()
    except Exception:
        pass

    # ===== 在训练结束后运行一次测试回合并绘制轨迹（静态图，不使用 env.render） =====
    try:
        test_obs, _ = env.reset()
        test_agents = env.agents
        # 初始化轨迹容器
        trajectories = {ag: [] for ag in test_agents}
        landmark_positions = None

        step = 0
        max_test_steps = MAX_TEST_STEPS
        while env.agents and step < max_test_steps:
            # 选择确定性动作（不探索）
            acts = {}
            for agent in test_agents:
                if agent in test_obs:
                    _, a = mappo.take_action(test_obs[agent], explore=False)
                    acts[agent] = a

            next_obs, rewards, terminations, truncations, infos = env.step(acts)

            # 提取位置（优先使用 world/state，如果无法获取则回退到 obs[:2]）
            agent_pos_dict, landmark_pos_list = _get_positions_from_env(env)
            if landmark_pos_list is not None:
                landmark_positions = landmark_pos_list

            for agent in test_agents:
                if agent_pos_dict is not None and agent in agent_pos_dict:
                    p = np.array(agent_pos_dict[agent])
                else:
                    # 回退：尝试从 next_obs 中读取前两维
                    if agent in next_obs:
                        try:
                            p = np.array(next_obs[agent][:2])
                        except Exception:
                            p = np.array([np.nan, np.nan])
                    else:
                        p = np.array([np.nan, np.nan])
                trajectories[agent].append(p)

            test_obs = next_obs
            step += 1

        # 绘制静态轨迹图
        try:
            plt.figure(figsize=(6, 6))
            cmap = plt.cm.get_cmap('tab10')
            for i, agent in enumerate(test_agents):
                traj = np.array(trajectories[agent])
                if traj.size == 0:
                    continue
                mask = ~np.isnan(traj[:, 0])
                if mask.sum() == 0:
                    continue
                plt.plot(traj[mask, 0], traj[mask, 1], '-', color=cmap(i), label=f'agent_{i}')
                # 起点：圆圈；终点：三角形
                plt.scatter(traj[mask, 0][0], traj[mask, 1][0], marker='o', color=cmap(i), s=50)
                plt.scatter(traj[mask, 0][-1], traj[mask, 1][-1], marker='^', color=cmap(i), s=70)

            if landmark_positions is not None:
                lps = np.array(landmark_positions)
                plt.scatter(lps[:, 0], lps[:, 1], marker='X', color='k', s=100, label='landmarks')

            plt.legend()
            plt.title('Test episode: agent trajectories and landmarks (circle=start, triangle=end)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print('Plotting failed:', e)
    except Exception as e:
        print('Test episode failed:', e)

    env.close()

if __name__ == '__main__':
    main()
    