# GAT 掩码策略分析

## 当前实现状态

### 1. **Actor 的掩码情况**
```python
# GATActor.forward() - 第 167 行
h_graph, attn_weights = self.gat(h_graph, adj_mask=None, return_attention=True)
```
- **当前**：`adj_mask=None`，无掩码
- **图结构**：单节点图（本身的观察向量）
- **结果**：自注意力权重为 1.0（softmax 作用在单个元素上）
- **掩码需求**：无（单节点无法进行有意义的掩码）

### 2. **Critic 的掩码情况**
```python
# GATCritic.forward() - 第 205-206 行
h = self.gat1(h, adj_mask=active_mask)
h = self.gat2(h, adj_mask=active_mask)
```

#### 当前掩码策略
- **掩码类型**：Agent active/inactive mask（(B, N) 形状）
- **掩码含义**：屏蔽已死亡或不存在的 agents
- **掩码效应**：
  ```
  观察矩阵形状：(B, N, obs_dim)
  - N 个节点代表 N 个 agents
  - 每个节点的观察向量包含：
    • 自己的位置 (2D)
    • 自己的速度 (2D)
    • 其他每个 agent 相对位置 (2D × (N-1))
    • 每个 landmark 相对位置 (2D × N)  ← 关键
  ```

#### GATLayer 掩码应用原理
```python
if adj_mask is not None:
    mask = adj_mask.unsqueeze(1)  # (B, N) → (B, 1, N)
    e = e.masked_fill(mask == 0, -1e9)  # 注意力得分矩阵 (B, N, N)
```

掩码作用在注意力权重矩阵的**列**（被关注的节点）上：
- `mask == 1` 的列：允许关注
- `mask == 0` 的列：被屏蔽为 -1e9，softmax 后变为 ≈ 0

---

## 用户问的关键问题

"是否使用过 padding 强制置零'队友和队友之间'的权重、'地标和地标之间'的权重，只保留'我和地标'、'我和队友'与'队友和地标'之间的权重"

### 理解问题

这个问题假设观察矩阵中既包含 agents 信息也包含 landmarks 信息，并希望通过掩码来**控制节点间的连接方式**。

但这需要澄清：**当前实现没有显式建模 agent-landmark 图结构**。

---

## 两种实现方案对比

### 方案 A：当前实现（隐式信息）
```
观察矩阵：(B, N, obs_dim)
- 每个节点 = 一个 agent 的观察向量
- 该向量已包含所有信息（其他 agents + landmarks）
- GAT 在这 N 个节点间进行注意力计算
- 掩码：只对不活跃的 agents 进行屏蔽
```

**优点**：
- 实现简单
- Critic 可在 N 个 agents 间进行全局推理

**缺点**：
- 无法精细控制哪些信息应被关注
- Landmark 信息隐含在每个 agent 的观察向量中
- 无法区分"agent-agent"、"agent-landmark"、"landmark-landmark"连接

---

### 方案 B：显式图结构（建议改进）
```
扩展的观察矩阵：(B, N+M, obs_dim_expanded)
  - 前 N 个节点：agents
  - 后 M 个节点：landmarks
  - 总共 N+M 个节点

掩码矩阵应该禁止以下连接：
  - agent_i ← landmark_j （列为 landmark，行可来自任何节点）
    原因：landmarks 被动，不应作为注意源
  - landmark_i ← landmark_j （两个 landmarks 间）
    原因：landmarks 间无信息交互

允许的连接：
  - agent_i ← agent_j （agents 可相互协作）
  - agent_i ← landmark_j ← 实际上应禁止，见上
  - landmark_j ← agent_i （agent 可观察 landmark）
```

---

## 当前实现的实际图连接

### Critic 中的实际情况
```python
# 输入：obs_matrix (B, N, obs_dim)
# 其中每个 obs 已包含：
#   - self position/velocity
#   - relative positions of all other agents
#   - relative positions of all landmarks

# GAT 计算 attention (B, N, N)
# attention[b, i, j] = 智能体 i 对智能体 j 的关注权重

# 实际信息流：
# - Agent_0 可以关注 Agent_1 的"知识"
#   （而 Agent_1 的知识中已包含了 landmarks 信息）
```

**这是一种隐式的信息聚合**，不存在明确的"landmark 节点"。

---

## 改进建议

### 如果需要精细掩码控制

**Option 1**：提取并重构观察矩阵（最灵活）
```python
def construct_agent_landmark_graph(obs_dict, n_landmarks):
    """
    obs_dict: {agent_id: obs_vector}
    obs_vector layout: [self_pos(2), self_vel(2), other_agent_rel_pos(...), landmark_rel_pos(...)]
    
    返回：(B, N+M, graph_feature_dim)
    其中前 N 个节点是 agents，后 M 个是 landmarks
    """
    # 从观察向量中提取相应部分
    # 创建新的特征向量，为 landmarks 赋予特殊标记
    pass

def create_structured_mask(n_agents, n_landmarks):
    """
    返回 (N+M, N+M) 的掩码矩阵
    mask[i, j] = 1 表示允许 j → i 的边
    """
    mask = torch.ones(n_agents + n_landmarks, n_agents + n_landmarks)
    
    # 禁止从 landmarks 到任何地方的信息流
    mask[n_agents:, :] = 0  # landmarks 不能作为信息源
    
    # 可选：禁止 agent 之间的直接连接
    # mask[:n_agents, :n_agents] = torch.eye(n_agents)
    
    return mask
```

**Option 2**：修改 GATLayer 以支持二维掩码
```python
def forward(self, h, adj_mask=None, edge_mask=None, return_attention=False):
    # adj_mask: (B, N) - 节点级掩码（当前）
    # edge_mask: (B, N, N) - 边级掩码（建议新增）
    #   edge_mask[b, i, j] = 1 表示允许 j → i 的边
    
    if edge_mask is not None:
        e = e.masked_fill(edge_mask.unsqueeze(1) == 0, -1e9)
```

---

## 现状总结

| 方面 | 当前状态 | 是否实现了用户要求 |
|------|--------|-----------------|
| Actor 掩码 | 无（单节点图） | N/A - 单节点无法应用 |
| Critic 掩码 | 仅 active agent 掩码 | ❌ 否 |
| 显式 agent-landmark 分离 | 未实现 | ❌ 否 |
| Agent-agent 连接 | 完全允许 | ✓ 允许（用户可能想要） |
| Agent-landmark 连接 | 混合在观察向量中 | ⚠️ 隐式存在 |
| Landmark-landmark 连接 | 不存在（无 landmark 节点） | ✓ 自动满足 |

---

## 建议行动

### 如果当前性能满足需求
- **保持现状**：当前的隐式信息流方案已能让 Critic 有效进行全局推理

### 如果需要显式控制可视化/解释性
- **实现 Option 1**：显式构建 agent-landmark 图
- **修改 GATLayer**：支持边级掩码 (N, N)
- **提供可视化工具**：展示不同类型的连接权重分布

### 快速检验方法
```python
# 在 Critic.forward() 后添加：
with torch.no_grad():
    # 提取注意力权重用于分析
    _, attn_weights = self.gat1(h_encoded, adj_mask=active_mask, return_attention=True)
    # attn_weights.shape = (B, N, N)
    # 分析：不同 agent 对彼此的关注权重分布
    print(f"Attention mean: {attn_weights.mean()}")
    print(f"Attention std: {attn_weights.std()}")
```

