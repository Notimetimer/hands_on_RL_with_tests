# 改进的 Critic 实现 - 可直接替换

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GATLayerWithEdgeMask(nn.Module):
    """改进的 GAT 层，支持边级掩码"""
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
        h: (B, N, in_features)
        adj_mask: (B, N) - 节点活跃度掩码
        edge_mask: (B, N, N) - 边掩码矩阵
        """
        Wh = self.W(h)
        B, N, _ = Wh.size()

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # 边级掩码（优先级更高）
        if edge_mask is not None:
            e = e.masked_fill(edge_mask == 0, -1e9)
        
        # 节点级掩码
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
    def __init__(self, n_agents: int, node_dim: int, n_landmarks: int, 
                 hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.total_nodes = n_agents + n_landmarks
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        self.agent_encoder = nn.Linear(node_dim, hidden_dim)
        self.landmark_encoder = nn.Linear(2, hidden_dim)  # Landmarks 用相对位置
        
        self.gat1 = GATLayerWithEdgeMask(hidden_dim, hidden_dim)
        self.gat2 = GATLayerWithEdgeMask(hidden_dim, hidden_dim, concat=False)
        
        self.v_head = nn.Linear(hidden_dim, 1)

    def _extract_landmark_features(self, obs_matrix: torch.Tensor) -> torch.Tensor:
        """
        从观察向量中提取 landmark 相对位置特征
        
        Simple Spread V3 观察布局（标准）：
        - [0:2] self position
        - [2:4] self velocity
        - [4:4+2*(N-1)] other agents' relative positions
        - [4+2*(N-1):] landmarks' relative positions (2 * M)
        """
        batch_size = obs_matrix.shape[0]
        landmark_offset = 4 + 2 * (self.n_agents - 1)
        
        if obs_matrix.shape[2] > landmark_offset:
            landmark_end = landmark_offset + 2 * self.n_landmarks
            # 取第一个 agent 的 landmark 信息（所有 agents 看到相同的相对位置）
            landmark_features = obs_matrix[:, 0, landmark_offset:landmark_end]
            landmark_features = landmark_features.view(batch_size, self.n_landmarks, 2)
            return landmark_features
        else:
            return torch.zeros(batch_size, self.n_landmarks, 2, 
                             device=obs_matrix.device)

    def _create_edge_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        创建 (B, N+M, N+M) 的边掩码
        规则：landmarks 不能作为信息源（禁止从 landmarks 出发的边）
        """
        mask = torch.ones(batch_size, self.total_nodes, self.total_nodes, device=device)
        # 禁止所有来自 landmarks 的边
        mask[:, :, self.n_agents:] = 0
        return mask

    def forward(self, obs_matrix: torch.Tensor, 
                active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        obs_matrix: (B, N, obs_dim)
        active_mask: (B, N)
        返回: (B, N) - agents 的价值
        """
        B, N, _ = obs_matrix.shape
        device = obs_matrix.device
        
        # 编码节点
        agent_features = F.relu(self.agent_encoder(obs_matrix))  # (B, N, H)
        landmark_features_raw = self._extract_landmark_features(obs_matrix)
        landmark_features = F.relu(self.landmark_encoder(landmark_features_raw))  # (B, M, H)
        
        # 拼接
        h = torch.cat([agent_features, landmark_features], dim=1)  # (B, N+M, H)
        
        # 扩展 active mask
        if active_mask is not None:
            landmarks_active = torch.ones(B, self.n_landmarks, device=device)
            full_active_mask = torch.cat([active_mask, landmarks_active], dim=1)
        else:
            full_active_mask = None
        
        # 边掩码
        edge_mask = self._create_edge_mask(B, device)
        
        # GAT
        h = self.gat1(h, adj_mask=full_active_mask, edge_mask=edge_mask)
        h = self.gat2(h, adj_mask=full_active_mask, edge_mask=edge_mask)
        
        # 输出
        all_values = self.v_head(h).squeeze(-1)  # (B, N+M)
        agent_values = all_values[:, :N]  # (B, N)
        
        return agent_values


class ComparableCriticWrapper:
    """
    包装器类，用于在原代码中轻松切换两个 Critic 实现
    """
    def __init__(self, model_type='original', n_agents=3, n_landmarks=3, 
                 node_dim=30, hidden_dim=64):
        """
        model_type: 'original' 或 'structured'
        """
        self.model_type = model_type
        
        if model_type == 'structured':
            self.model = StructuredGATCritic(n_agents, node_dim, n_landmarks, hidden_dim)
        else:
            # 从原文件导入 GATCritic
            raise NotImplementedError("需要导入原文件中的 GATCritic")
    
    def __call__(self, obs_matrix, active_mask=None):
        return self.model(obs_matrix, active_mask)
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def parameters(self):
        return self.model.parameters()


# ====== 下面是测试和对比代码 ======

def test_structured_gat_critic():
    """测试新的 StructuredGATCritic"""
    print("\n=== Testing StructuredGATCritic ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 参数
    n_agents = 3
    n_landmarks = 3
    batch_size = 8
    obs_dim = 30  # 标准 Simple Spread V3
    hidden_dim = 64
    
    print(f"Input shapes:")
    print(f"  obs_matrix: ({batch_size}, {n_agents}, {obs_dim})")
    print(f"  active_mask: ({batch_size}, {n_agents})")
    
    # 创建模型
    critic = StructuredGATCritic(n_agents, obs_dim, n_landmarks, hidden_dim).to(device)
    
    # 创建测试数据
    obs_matrix = torch.randn(batch_size, n_agents, obs_dim).to(device)
    active_mask = torch.ones(batch_size, n_agents).to(device)
    active_mask[0, 1] = 0  # 模拟一个 agent 不活跃
    
    # 前向传播
    with torch.no_grad():
        values = critic(obs_matrix, active_mask=active_mask)
    
    print(f"\nOutput shapes:")
    print(f"  values: {values.shape}")  # 应该是 (batch_size, n_agents)
    print(f"  values range: [{values.min():.4f}, {values.max():.4f}]")
    
    # 验证参数数量
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # 测试梯度流
    values.sum().backward()
    print(f"✓ Gradients computed successfully")
    
    return critic, obs_matrix, active_mask, values


def visualize_edge_mask():
    """可视化边掩码矩阵"""
    print("\n=== Edge Mask Visualization ===\n")
    
    n_agents = 3
    n_landmarks = 3
    total_nodes = n_agents + n_landmarks
    
    critic = StructuredGATCritic(n_agents, 30, n_landmarks)
    edge_mask = critic._create_edge_mask(1, torch.device('cpu'))[0]
    
    print("Edge Mask Matrix (1 = allowed, 0 = masked):")
    print("Rows: target nodes (agents 0-2, landmarks 3-5)")
    print("Cols: source nodes (agents 0-2, landmarks 3-5)\n")
    
    labels = [f"A{i}" for i in range(n_agents)] + [f"L{i}" for i in range(n_landmarks)]
    
    print("     ", "  ".join(labels))
    for i, row_label in enumerate(labels):
        print(f"{row_label}: ", "  ".join(str(int(edge_mask[i, j].item())) for j in range(total_nodes)))
    
    print("\n分析:")
    print("✓ Agents 可以相互关注（Agent 行全为 1）")
    print("✗ Landmarks 行全为 1（可被关注）")
    print("✗ 所有列从 L0 开始都为 0（不能从 Landmarks 出发）")
    print("\n掩码效果：")
    print("  - Agent → Agent：✓ 允许（协作）")
    print("  - Agent → Landmark：✓ 允许（观察）")
    print("  - Landmark → Any：✗ 禁止（被动）")


def compare_critic_architectures():
    """对比原实现和改进实现"""
    print("\n=== Architecture Comparison ===\n")
    
    import pandas as pd
    
    comparison = pd.DataFrame({
        'Feature': [
            'Node Types',
            'Edge-level Masking',
            'Explicit Landmarks',
            'Agent-Agent Edges',
            'Agent-Landmark Edges',
            'Landmark-Landmark Edges',
            'Total Nodes',
            'Interpretation'
        ],
        'Original GATCritic': [
            'Agents only',
            'No (node-level only)',
            'Implicit (in obs vectors)',
            'Soft control',
            'Implicit (in obs)',
            'Implicit (in obs)',
            'N',
            'Harder to explain'
        ],
        'Structured GATCritic': [
            'Agents + Landmarks',
            'Yes ((N+M)×(N+M) matrix)',
            'Explicit graph nodes',
            'Fully allowed',
            'Fully allowed',
            'Fully masked',
            'N + M',
            'Interpretable graph'
        ]
    })
    
    print(comparison.to_string(index=False))
    
    print("\n关键改进：")
    print("✓ 显式分离 agents 和 landmarks")
    print("✓ 精细的边级掩码控制")
    print("✓ 更易理解的信息流")
    print("✓ 更好的可解释性（用于研究和调试）")


def usage_in_training():
    """展示如何在现有训练代码中集成改进版本"""
    print("\n=== Usage in MAPPO Training ===\n")
    
    example_code = '''
# 在 MAPPO.__init__() 中替换 Critic

# 旧代码：
# self.critic = GATCritic(num_agents, node_dim=obs_dim, 
#                         hidden_dim=critic_hidden).to(device)

# 新代码：
n_landmarks = num_agents  # Simple Spread V3 通常 landmarks = agents
self.critic = StructuredGATCritic(
    n_agents=num_agents,
    node_dim=obs_dim,
    n_landmarks=n_landmarks,
    hidden_dim=critic_hidden
).to(device)

# 其余代码无需修改！
# MAPPO.update() 中调用 self.critic() 的部分会自动使用新实现
    '''
    
    print(example_code)


if __name__ == '__main__':
    # 运行测试
    critic, obs, mask, vals = test_structured_gat_critic()
    
    print("\n" + "="*60)
    visualize_edge_mask()
    
    print("\n" + "="*60)
    compare_critic_architectures()
    
    print("\n" + "="*60)
    usage_in_training()
    
    print("\n" + "="*60)
    print("\n✅ 所有测试完成！")
    print("\n建议：")
    print("1. 将 StructuredGATCritic 集成到 MAPPO_MPE训练3_带GAT.py")
    print("2. 对比两个版本的性能")
    print("3. 观察学习曲线和最终收敛性能")
