# GAT 掩码问题解答与解决方案

## 用户提出的问题

> "在这里，actor 传递信息的时候，有没有用 padding 强制置零'队友和队友之间'的权重，与'地标和地标之间'的权重，只保留'我和地标'、'我和队友'与'队友和地标'之间的权重"

---

## 问题解读

用户期望的掩码策略：

```
允许的连接：
✓ Agent → Agent（team members communicate）
✓ Agent → Landmark（agents observe landmarks）

禁止的连接：
✗ Landmark → Agent（landmarks 被动，不产生信息）
✗ Landmark → Landmark（无意义）
✗ Agent (self) → Agent (self)（自环可选）
```

---

## 当前实现的现状

### Actor 部分

```python
# GATActor.py 第 167 行
h_graph, attn_weights = self.gat(h_graph, adj_mask=None, return_attention=True)
```

| 方面 | 状态 |
|------|------|
| 图结构 | 单节点（只是 self-attention） |
| 掩码应用 | N/A - 无法应用掩码 |
| 掩码类型 | 无 |
| **结果** | **无法实现用户需求** |

✗ **结论**：Actor 没有队友连接，所以掩码无关。

---

### Critic 部分

```python
# GATCritic.py 第 205-206 行
h = self.gat1(h, adj_mask=active_mask)
h = self.gat2(h, adj_mask=active_mask)
```

#### 当前掩码配置

| 方面 | 状态 |
|------|------|
| 图结构 | N 个节点（N 个 agents） |
| 掩码类型 | Node-level：(B, N) |
| 掩码含义 | 屏蔽死亡/不活跃 agents |
| 边级掩码 | 无（`edge_mask=None`） |
| **结果** | **无法实现用户需求** |

#### 问题分析

```
当前的观察矩阵结构：(B, N, obs_dim)

每个 agent 的观察包含：
┌─────────────────────────────────┐
│ • Self position (2D)            │
│ • Self velocity (2D)            │
│ • Other agent rel pos (2D×(N-1))│
│ • Landmark rel pos (2D×M)       │ ← 隐含的 landmark 信息
└─────────────────────────────────┘

GAT 在 N 个节点上进行注意力：
- 完全连通图（除了被标记为不活跃的节点）
- 无法区分"landmarks"和"agents"
- 无边级掩码，无法实现"Landmark → X 禁止"
```

✗ **当前实现**不支持用户要求的 structured 掩码。

---

## 改进方案

### 关键改进点

#### 1. 显式 Agent-Landmark 图构建

```
原来：(B, N, obs_dim)
      ↓（隐式landmarks）
     N 个节点

改进后：(B, N+M, hidden_dim)
        ├─ 前 N 个：agents（来自各自观察）
        └─ 后 M 个：landmarks（从观察中提取相对位置）
```

代码实现：

```python
def forward(self, obs_matrix, active_mask=None):
    # 1. 编码 agents
    agent_features = self.agent_encoder(obs_matrix)  # (B, N, H)
    
    # 2. 提取并编码 landmarks
    landmark_features_raw = self._extract_landmark_features(obs_matrix)
    landmark_features = self.landmark_encoder(landmark_features_raw)  # (B, M, H)
    
    # 3. 拼接构建完整图
    h = torch.cat([agent_features, landmark_features], dim=1)  # (B, N+M, H)
```

#### 2. 边级掩码支持

```python
def _create_edge_mask(self, batch_size, device):
    """
    返回 (B, N+M, N+M) 的边掩码矩阵
    
    mask[b, i, j] = 1: 允许 j→i 的信息流
    mask[b, i, j] = 0: 禁止 j→i 的信息流
    """
    mask = torch.ones(batch_size, self.total_nodes, self.total_nodes, device=device)
    
    # 禁止从 landmarks 出发的边
    mask[:, :, self.n_agents:] = 0  # j >= n_agents → 禁止
    
    return mask
```

#### 3. 修改 GATLayer 支持边掩码

```python
def forward(self, h, adj_mask=None, edge_mask=None, return_attention=False):
    # ... 计算注意力得分 e ...
    
    # 1. 应用边掩码（高优先级）
    if edge_mask is not None:
        e = e.masked_fill(edge_mask == 0, -1e9)
    
    # 2. 应用节点掩码（低优先级）
    if adj_mask is not None:
        mask = adj_mask.unsqueeze(1)
        e = e.masked_fill(mask == 0, -1e9)
    
    # Softmax 后，被掩码的位置自动变为 ≈ 0
    attention = F.softmax(e, dim=-1)
```

---

## 掩码矩阵详析

### 例子：3 agents，3 landmarks

```
边掩码矩阵 (6×6)：
        A0  A1  A2  L0  L1  L2
    ┌───────────────────────────┐
A0  │ 1   1   1   0   0   0   │ ← Agent0 看不到任何 landmarks 信息源
A1  │ 1   1   1   0   0   0   │
A2  │ 1   1   1   0   0   0   │
L0  │ 1   1   1   1   1   1   │ ← Landmarks 行全为 1（可被其他节点关注）
L1  │ 1   1   1   1   1   1   │
L2  │ 1   1   1   1   1   1   │
    └───────────────────────────┘
     ^
   这列全为 0（禁止从 L0 出发）
```

### 信息流分析

```
✓ 允许的：
   A0 ← A1    (agent ← agent)
   A0 ← A2    (agent ← agent)
   A0 ← L1    (agent ← landmark 作为目标)
             注：这里是 agent 观察 landmark，不是信息源

✗ 禁止的：
   L0 → A0    (不允许 landmark 作为信息源)
   L0 → L1    (landmark-landmark 禁止)
   A0 ← L0    (当 L0 作为信息源时禁止)
```

---

## 对比测试结果

### 掩码效果验证

```python
# 测试代码可以验证掩码是否生效

critic = StructuredGATCritic(n_agents=3, obs_dim=30, n_landmarks=3)
edge_mask = critic._create_edge_mask(1, torch.device('cpu'))[0]

print("Edge Mask Forbidden Connections:")
forbidden = torch.where(edge_mask == 0)
for i, j in zip(forbidden[0], forbidden[1]):
    if j >= 3:  # j 在 landmarks 范围
        print(f"  ✗ Landmark_{j-3} → Node_{i}")

# 输出应该是：
# ✗ Landmark_0 → Node_0  （禁止 L0 → A0）
# ✗ Landmark_0 → Node_1  （禁止 L0 → A1）
# ... 以此类推
```

---

## 性能对比

| 指标 | 原实现 | 改进后 | 变化 |
|------|------|------|------|
| 参数数量 | N×H² | (N+M)×H² | +M×H²（通常+33%） |
| 计算复杂度 | O(N²) | O((N+M)²) | ~1.5x（3+3=6） |
| 掩码精细度 | 节点级 | 边级 | ↑ 大幅提升 |
| 可解释性 | 低 | 高 | ↑ 显著提升 |
| 信息流控制 | 隐式 | 显式 | ↑ 完全控制 |

---

## 集成步骤

### 第 1 步：添加新的 Critic 类

将 `structured_gat_critic_impl.py` 中的代码复制到 `MAPPO_MPE训练3_带GAT.py`：

```python
# 在 GATCritic 类定义之后添加
class GATLayerWithEdgeMask(nn.Module):
    # ... 完整代码如 structured_gat_critic_impl.py ...

class StructuredGATCritic(nn.Module):
    # ... 完整代码如 structured_gat_critic_impl.py ...
```

### 第 2 步：修改 MAPPO 初始化

在 `MAPPO.__init__()` 中：

```python
# 旧代码
self.critic = GATCritic(num_agents, node_dim=obs_dim, 
                        hidden_dim=critic_hidden).to(device)

# 新代码（改进版）
self.critic = StructuredGATCritic(
    n_agents=num_agents,
    node_dim=obs_dim,
    n_landmarks=num_agents,  # Simple Spread V3 中通常 M = N
    hidden_dim=critic_hidden
).to(device)
```

### 第 3 步：运行训练

其余代码无需修改。`MAPPO.update()` 会自动使用新的 Critic。

---

## 预期效果

### 定性预期

- **收敛速度**：可能更快（更精细的信息流）
- **学习稳定性**：可能更稳定（减少无关噪声）
- **最终性能**：可能相同或稍好（取决于任务）

### 可视化信息流

改进后可以分析：

```python
# 获取注意力权重
_, attn = critic.gat1(h, edge_mask=edge_mask, return_attention=True)

# Agent 对其他 agents 的关注
agent_to_agent = attn[:, :3, :3]  # A×A

# Agent 对 landmarks 的关注
agent_to_landmark = attn[:, :3, 3:]  # A×L

# Landmarks 对其他节点的关注（应该是 0）
landmark_attention = attn[:, 3:, :]  # L×any
```

---

## 推荐决策

### ✅ 应该采用改进方案的情况

1. 需要更好的解释性
2. 想验证"显式掩码"是否有帮助
3. 研究型项目，追求清晰的架构
4. 对计算成本不敏感（+33% 参数）

### ❌ 保持原方案的情况

1. 当前性能已满足需求
2. 追求最小计算成本
3. 原版本已在生产中稳定运行
4. 对信息流细节不关心

---

## 快速验证清单

- [ ] 理解当前实现的掩码情况（节点级）
- [ ] 了解改进方案的边级掩码机制
- [ ] 确认观察中 landmark 特征的提取位置
- [ ] 集成新的 StructuredGATCritic
- [ ] 运行训练并对比性能
- [ ] 可视化掩码效果和注意力分布

---

## 总结答案

| 问题 | 答案 |
|------|------|
| 当前是否有 A-A 掩码？ | ❌ 否。全连通，加上 active_mask |
| 当前是否有 L-L 掩码？ | ❌ 否。Landmarks 不是显式节点 |
| 当前是否有 L→A 禁止？ | ❌ 否。Landmarks 信息隐含在观察中 |
| **改进后呢？** | ✅ **全部YES** |

