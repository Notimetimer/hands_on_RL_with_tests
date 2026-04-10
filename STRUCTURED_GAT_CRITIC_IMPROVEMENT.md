# 改进方案：显式 Agent-Landmark 图结构的 GAT Critic

这是 `MAPPO_MPE训练3_带GAT.py` 中 GATCritic 的改进版本。

## 关键改进

### 1. 支持边级掩码 (Edge-level Masking)
```python
# 修改 GATLayer 以支持 (N, N) 的边掩码矩阵
```

### 2. 显式构建 Agent-Landmark 节点图
```python
# 新增 StructuredGATCritic，将观察拆解为 agents + landmarks
```

### 3. 掩码策略
- ✓ Agent → Agent：完全允许（协作）
- ✓ Agent → Landmark：完全允许（观察目标）
- ✗ Landmark → Any：完全禁止（landmarks 被动）
- ✗ Landmark → Landmark：完全禁止（无意义）

---

## 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GATLayerWithEdgeMask(nn.Module):
    """
    改进的 GAT 层，支持边级掩码 (N, N) 矩阵
    """
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

    def forward(self, h, adj_mask: Optional[torch.Tensor] = None, 
                edge_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        参数:
            h: (B, N, in_features) - 节点特征
            adj_mask: (B, N) - 节点掩码（来自原实现，掩蔽不活跃节点）
            edge_mask: (B, N, N) - 边掩码矩阵
                edge_mask[b, i, j] = 1 表示允许边 j→i
                edge_mask[b, i, j] = 0 表示禁止边 j→i
            return_attention: bool - 是否返回注意力权重
        
        返回:
            output: (B, N, out_features)
            attention (可选): (B, N, N)
        """
        Wh = self.W(h)  # (B, N, out_features)
        B, N, _ = Wh.size()

        # 广播拼接所有节点对
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, 1, out_f) → (B, N, N, out_f)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, 1, N, out_f) → (B, N, N, out_f)
        
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (B, N, N, 2*out_f)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # (B, N, N)

        # === 掩码应用（按优先级） ===
        
        # 1. 边级掩码（最高优先级）
        if edge_mask is not None:
            # edge_mask: (B, N, N)，其中 edge_mask[b, i, j]=0 表示禁止 j→i
            e = e.masked_fill(edge_mask == 0, -1e9)
        
        # 2. 节点级掩码（传统 active mask）
        if adj_mask is not None:
            # adj_mask: (B, N) → (B, 1, N)
            # 掩蔽不活跃的目标节点（被关注的节点）
            node_mask = adj_mask.unsqueeze(1)
            e = e.masked_fill(node_mask == 0, -1e9)

        # Softmax 注意力
        attention = F.softmax(e, dim=-1)
        h_prime = torch.matmul(attention, Wh)

        output = F.elu(h_prime) if self.concat else h_prime
        
        if return_attention:
            return output, attention
        else:
            return output


class StructuredGATCritic(nn.Module):
    """
    改进的 Critic，显式建模 Agent-Landmark 图结构
    
    图结构：
    - 前 N 个节点：agents
    - 后 M 个节点：landmarks（特征来自观察中的相对位置）
    
    掩码策略：
    - Agent → Agent：✓ 允许（协作学习）
    - Agent → Landmark：✓ 允许（观察环境）
    - Landmark → Any：✗ 禁止（landmarks 被动）
    """
    def __init__(self, n_agents: int, node_dim: int, n_landmarks: int, 
                 hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.total_nodes = n_agents + n_landmarks
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # 特征编码
        self.agent_encoder = nn.Linear(node_dim, hidden_dim)
        # Landmarks 的特征从观察中的相对位置提取
        # Simple Spread 中，landmark 特征通常是相对位置 (2D)
        self.landmark_encoder = nn.Linear(2, hidden_dim)
        
        # GAT 层
        self.gat1 = GATLayerWithEdgeMask(hidden_dim, hidden_dim)
        self.gat2 = GATLayerWithEdgeMask(hidden_dim, hidden_dim, concat=False)
        
        # 价值头
        self.v_head = nn.Linear(hidden_dim, 1)

    def _extract_landmark_features(self, obs_matrix: torch.Tensor) -> torch.Tensor:
        """
        从观察矩阵中提取 landmark 特征。
        
        假设 obs_matrix 中，每个 agent 观察的结构是：
        [self_pos(2), self_vel(2), other_agents_rel_pos(...), landmarks_rel_pos(2 * n_landmarks)]
        
        对于 Simple Spread V3，标准结构通常是：
        - [0:2] - self position
        - [2:4] - self velocity  
        - [4:4+2*(N-1)] - relative positions of other agents
        - [4+2*(N-1):] - relative positions of landmarks (2 * M)
        """
        batch_size = obs_matrix.shape[0]
        
        # 提取 landmark 相对位置（假设在观察向量的末尾）
        # 实际位置取决于环境的观察结构
        landmark_offset = 4 + 2 * (self.n_agents - 1)  # 标准 layout
        
        # 每个 agent 的观察末尾都包含所有 landmarks 相对位置
        # 我们取任意一个 agent（e.g., agent 0）的 landmark 信息
        if obs_matrix.shape[2] > landmark_offset:
            landmark_features = obs_matrix[:, 0, landmark_offset:landmark_offset + 2 * self.n_landmarks]
            # reshape 为 (B, M, 2)
            landmark_features = landmark_features.view(batch_size, self.n_landmarks, 2)
            
            return landmark_features
        else:
            # Fallback：如果观察向量不够长，返回零特征
            return torch.zeros(batch_size, self.n_landmarks, 2, device=obs_matrix.device)

    def _create_edge_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        创建 (B, N+M, N+M) 的边掩码矩阵。
        
        掩码策略：
        - [agent, agent]：✓ 允许（协作）
        - [agent, landmark]：✓ 允许（观察）
        - [landmark, *]：✗ 全禁止（landmarks 被动）
        """
        mask = torch.ones(batch_size, self.total_nodes, self.total_nodes, device=device)
        
        # 禁止从 landmarks 出发的边
        # mask[b, i, j] = 0 表示禁止 j→i
        # landmarks 是节点 [n_agents:] 是"目标"时禁止
        # 但我们要禁止 landmarks 是"源"时，即 j >= n_agents 时
        mask[:, :, self.n_agents:] = 0  # 禁止所有来自 landmarks 的信息
        
        return mask

    def forward(self, obs_matrix: torch.Tensor, 
                active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            obs_matrix: (B, N, obs_dim) - 所有 agents 的观察
            active_mask: (B, N) - agent 活跃度掩码
        
        返回:
            values: (B, N) - 每个 agent 的价值估计
        """
        B, N, _ = obs_matrix.shape
        device = obs_matrix.device
        
        # 1. 编码 agent 节点
        agent_features = F.relu(self.agent_encoder(obs_matrix))  # (B, N, H)
        
        # 2. 编码 landmark 节点
        landmark_features_raw = self._extract_landmark_features(obs_matrix)  # (B, M, 2)
        landmark_features = F.relu(self.landmark_encoder(landmark_features_raw))  # (B, M, H)
        
        # 3. 拼接构建完整的节点特征矩阵
        h = torch.cat([agent_features, landmark_features], dim=1)  # (B, N+M, H)
        
        # 4. 创建 agent active mask（仅对前 N 个节点）
        # Landmarks 始终是"活跃的"
        if active_mask is not None:
            # 补充 landmarks 的活跃标志（全为 1）
            landmarks_active = torch.ones(B, self.n_landmarks, device=device)
            full_active_mask = torch.cat([active_mask, landmarks_active], dim=1)  # (B, N+M)
        else:
            full_active_mask = None
        
        # 5. 创建边级掩码
        edge_mask = self._create_edge_mask(B, device)
        
        # 6. 应用 GAT
        h = self.gat1(h, adj_mask=full_active_mask, edge_mask=edge_mask)
        h = self.gat2(h, adj_mask=full_active_mask, edge_mask=edge_mask)
        
        # 7. 生成价值
        all_values = self.v_head(h).squeeze(-1)  # (B, N+M)
        
        # 8. 仅返回 agent 的价值（前 N 个）
        agent_values = all_values[:, :N]  # (B, N)
        
        return agent_values


class DebugStructuredGATCritic(StructuredGATCritic):
    """
    带调试信息的 StructuredGATCritic，用于检查掩码和注意力权重。
    """
    def forward(self, obs_matrix: torch.Tensor,
                active_mask: Optional[torch.Tensor] = None,
                debug: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        返回 (values, debug_info) 或仅 values（如果 debug=False）
        """
        B, N, _ = obs_matrix.shape
        device = obs_matrix.device
        
        agent_features = F.relu(self.agent_encoder(obs_matrix))
        landmark_features_raw = self._extract_landmark_features(obs_matrix)
        landmark_features = F.relu(self.landmark_encoder(landmark_features_raw))
        
        h = torch.cat([agent_features, landmark_features], dim=1)
        
        if active_mask is not None:
            landmarks_active = torch.ones(B, self.n_landmarks, device=device)
            full_active_mask = torch.cat([active_mask, landmarks_active], dim=1)
        else:
            full_active_mask = None
        
        edge_mask = self._create_edge_mask(B, device)
        
        if debug:
            h1, attn1 = self.gat1(h, adj_mask=full_active_mask, edge_mask=edge_mask, 
                                   return_attention=True)
            h2, attn2 = self.gat2(h1, adj_mask=full_active_mask, edge_mask=edge_mask, 
                                   return_attention=True)
            debug_info = {
                'gat1_attention': attn1.detach().cpu(),  # (B, N+M, N+M)
                'gat2_attention': attn2.detach().cpu(),
                'edge_mask': edge_mask.detach().cpu() if isinstance(edge_mask, torch.Tensor) else edge_mask,
            }
        else:
            h1 = self.gat1(h, adj_mask=full_active_mask, edge_mask=edge_mask)
            h2 = self.gat2(h1, adj_mask=full_active_mask, edge_mask=edge_mask)
            debug_info = {}
        
        all_values = self.v_head(h2).squeeze(-1)
        agent_values = all_values[:, :N]
        
        if debug:
            return agent_values, debug_info
        else:
            return agent_values
```

---

## 集成到训练代码中

### 替换现有的 GATCritic

在 `MAPPO.__init__()` 中：

```python
# 旧代码
self.critic = GATCritic(num_agents, node_dim=obs_dim, 
                         hidden_dim=critic_hidden).to(device)

# 新代码
n_landmarks = num_agents  # Simple Spread V3 默认 N个地标
self.critic = StructuredGATCritic(
    n_agents=num_agents,
    node_dim=obs_dim,
    n_landmarks=n_landmarks,
    hidden_dim=critic_hidden
).to(device)
```

### 验证掩码效果

```python
# 在训练循环中添加调试
critic_debug = DebugStructuredGATCritic(3, obs_dim, 3, 64).to(device)

# 获取附带调试信息的输出
values, debug_info = critic_debug.forward(global_states, 
                                          active_mask=active_masks,
                                          debug=True)

# 分析注意力模式
attn_gat1 = debug_info['gat1_attention'][0]  # (N+M, N+M)
edge_mask = debug_info['edge_mask'][0]

# 检查被屏蔽的边
masked_edges = (edge_mask == 0)
print(f"Masked edges (should be from landmarks): {masked_edges.sum()}")

# 检查 landmark 行的注意力是否为零
landmark_rows = attn_gat1[3:, :]  # 假设前 3 个是 agents
print(f"Landmark attention weights (should be ~0): {landmark_rows.mean()}")
```

---

## 优势对比

| 特性 | 原实现 | 改进后 |
|------|------|------|
| 节点类型区分 | 无（隐式） | 显式（agents + landmarks） |
| 边级掩码 | 无 | ✓ 支持任意 (N,N) 掩码 |
| 掌控力 | 弱 | 强 |
| 可解释性 | 低 | 高 |
| 灵活性 | 有限 | 可支持多种图拓扑 |
| 计算成本 | 稍低 | 稍高（+M个节点） |

---

## 后续可能的改进方向

1. **动态边掩码**：根据物理距离动态调整允许的连接
2. **类型感知注意力**：为不同节点类型的注意力赋予不同权重
3. **多头注意力**：对不同类型的关系使用不同的注意力头
4. **分层图**：agents 形成一个子图，landmarks 形成另一个子图

