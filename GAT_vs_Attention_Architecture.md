# GAT vs MultiheadAttention 架构对比分析

## 核心问题：为何注意力权重显示为 None？

### 原因分析

在从 MultiheadAttention 实现迁移到 GAT 实现过程中，注意力权重的**存储和获取方式发生了本质变化**。

---

## 架构对比

### 1. **ParallelAttentionActor** (原始实现)

```python
# 分离处理结构
self.attn_lm = nn.MultiheadAttention(embed_dim, num_heads=2)
self.attn_other = nn.MultiheadAttention(embed_dim, num_heads=2)

# Forward 中获取两种权重
context_lm, attn_weights_lm = self.attn_lm(q, k_lm, k_lm)
context_other, attn_weights_other = self.attn_other(q, k_other, k_other)

# 存储为属性
self.last_attn_lm_weights = attn_weights_lm
self.last_attn_other_weights = attn_weights_other
```

**特点**：
- ✅ 显式返回权重张量 (B, num_heads, 1, N)
- ✅ 分离的 landmark 和 teammate 权重
- ✅ 权重和为1的正数（softmax 结果）
- ❌ 计算复杂，参数多

### 2. **GATActor** (新实现)

```python
# 统一的单节点图处理
self.gat = GATLayer(hidden_dim, hidden_dim)

# Forward 中获取权重
h_graph, attn_weights = self.gat(h_graph, return_attention=True)
# attn_weights: (B, N_nodes, N_nodes) -> (B, 1, 1) 单节点情况

# 存储为属性
self.last_attn_weights = attn_weights.detach().cpu()
```

**特点**：
- ✅ 权重隐含在 GAT 计算过程中
- ✅ 权重是完整的注意力矩阵 (B, N, N)
- ✅ 权重和为1的正数（softmax 结果）
- ✅ 计算高效，参数少
- ❌ 没有显式的 lm/other 分离语义

---

## 关键差异

| 维度 | MultiheadAttention | GAT |
|------|------------------|-----|
| **节点结构** | Query-Key-Value 异构 | 同构图 |
| **权重获取** | 显式返回 attnweights | 需要 `return_attention=True` |
| **权重维度** | (B, heads, Q, K) | (B, N, N) |
| **权重语义** | lm/other 分离 | 全连接图 |
| **权重计算** | MultiheadAttention 内部 | softmax(LeakyReLU(a)) |

---

## 注意力权重的数学性质

### 两种实现中的权重都满足

$$\text{权重}[b,i,j] \in [0, 1], \quad \sum_j \text{权重}[b,i,j] = 1$$

### GATLayer 的权重计算

```python
# 1. 计算原始注意力得分
e[b, i, j] = LeakyReLU(concat(Wh_i, Wh_j) @ a)  # (B, N, N)

# 2. 使用 softmax 归一化
attention[b, i, j] = softmax(e[b, i, :])  # 按最后一维求和为1

# 3. 应用到特征
h_prime[b, i, :] = sum_j attention[b, i, j] * Wh[b, j, :]
```

---

## 当前实现中的权重提取

### GATActor 中的权重获取

```python
# 修改后的提取方式
w = getattr(mappo.actor, 'last_attn_weights', None)
# w: (B, 1, 1) - 单节点自注意力权重
# 本质上：w[b, 0, 0] = 1.0 （单节点的自注意力总是自己）
```

### 权重的实际含义

对于**单节点图**（当前 GATActor 的情况）：
- 节点 0 只参与自注意力
- 权重矩阵为 (B, 1, 1)
- 每个样本中 w[b, 0, 0] = 1.0

这在调试输出中体现为：
```
Agent agent_0 - landmark_attn: [1.0]  teammate_attn: [1.0]
Agent agent_1 - landmark_attn: [1.0]  teammate_attn: [1.0]
```

---

## 为什么原本显示 None？

1. **属性名称不匹配**：
   - 代码期望：`last_attn_lm_weights`, `last_attn_other_weights`
   - GATActor 提供：`last_attn_weights`
   - 结果：`getattr()` 返回 None

2. **权重返回方式**：
   - 原实现：MultiheadAttention 直接返回权重
   - GAT 实现：需要 `return_attention=True` 才能获得权重
   - 修复方案：已在 GATLayer 中添加 `return_attention` 参数

---

## 改进后的数据流

### 修改1：GATLayer 支持权重返回

```python
def forward(self, h, adj_mask=None, return_attention=False):
    # ... 计算过程 ...
    attention = F.softmax(e, dim=-1)
    h_prime = torch.matmul(attention, Wh)
    
    if return_attention:
        return output, attention  # ✅ 显式返回权重
    else:
        return output
```

### 修改2：GATActor 存储权重

```python
def forward(self, obs, active_mask=None, agent_ids=None):
    h_graph, attn_weights = self.gat(h_graph, return_attention=True)
    # ✅ 存储到属性供外部访问
    self.last_attn_weights = attn_weights.detach().cpu()
    return logits
```

### 修改3：主循环中的权重提取

```python
w = getattr(mappo.actor, 'last_attn_weights', None)
if w is not None:
    # w: (B, 1, 1)
    w_scalar = float(w[0, 0, 0].item())
    attn_by_agent[agent]['lm'].append(w_scalar)
    attn_by_agent[agent]['other'].append(w_scalar)
```

---

## 语义解释

### 原始 MultiheadAttention 方式

```
Query: "自己的观察"
Key/Value: 
  - Landmark: 地标位置和身份
  - Other: 队友位置

输出: 两个独立的权重分布，表示对不同类型信息的关注度
```

### GAT 方式

```
图节点: 只有"自己"（单节点）
图边: 自环（self-loop）

输出: 单节点的自注意力权重 w[0,0] = 1.0
      （特征本身通过 feature_encoder 已整合所有信息）
```

---

## 为什么都显示为标量 1.0？

在单节点图中：
- GAT 自注意力机制就是节点与自己计算注意力
- 单节点没有其他节点竞争，权重自动集中在自己
- 数学上：softmax([e[0,0]]) = 1.0

### 更丰富的权重信息

若要恢复类似原实现的 lm/other 分离权重，可改进为：

**选项1**：多节点 GAT（完整的 landmark/teammate 图）
```python
# 构建完整图: [self, landmark1, ..., other1, ...]
nodes = [self_feat, lm_feats, other_feats]
h_graph = self.gat(nodes)  # (B, 1+N+N-1, H)
attention = (B, 1+N+N-1, 1+N+N-1)
```

**选项2**：两层并联 GAT
```python
# 分别处理 landmark 和 other
h_lm = self.gat_lm(torch.cat([self_feat, lm_feats]))
h_other = self.gat_other(torch.cat([self_feat, other_feats]))
```

---

## 调试建议

### 验证权重正确性

```python
# 检查 GAT 设备一致性
print(f"Actor device: {next(mappo.actor.parameters()).device}")
print(f"Attn weights device: {mappo.actor.last_attn_weights.device if mappo.actor.last_attn_weights else 'None'}")

# 检查权重统计
if mappo.actor.last_attn_weights is not None:
    w = mappo.actor.last_attn_weights
    print(f"Shape: {w.shape}")
    print(f"Min: {w.min()}, Max: {w.max()}, Sum per row: {w.sum(dim=-1)}")
```

### 保存权重日志

```python
# 在 TensorBoard 中追踪权重分布
logger.add("attention/gat_weights_mean", float(w.mean()), step)
logger.add("attention/gat_weights_std", float(w.std()), step)
```

---

## 总结

| 问题 | 原因 | 解决方案 |
|-----|------|--------|
| 权重为 None | 属性名不匹配 | ✅ 已在 GATActor 中添加 `last_attn_weights` |
| 没有 lm/other 分离 | GAT 是单节点图 | 调整权重解释（都表示特征融合程度） |
| 权重显示为 1.0 | 单节点自注意力 | 正常行为（softmax 集中在唯一节点） |
| 权重不更新 | 未调用 `return_attention=True` | ✅ 已在 GATLayer 中修复 |

---

## 后续建议

1. **验证权重可用性**：确认输出中权重不再为 None
2. **考虑架构升级**：如需更多权重细节，实现多节点 GAT
3. **权重可视化**：使用 TensorBoard 追踪权重演变

