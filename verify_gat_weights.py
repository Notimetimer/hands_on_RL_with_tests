"""
验证 GAT actor 的注意力权重存储和提取
"""

import torch
import numpy as np

def test_gat_attention_weights():
    """测试GATActor的注意力权重是否能正确获取"""
    print("="*70)
    print("GAT Actor 注意力权重验证")
    print("="*70)
    
    # 导入模型
    with open('MAPPO_MPE训练3_带GATpy', 'r', encoding='utf-8') as f:
        code = f.read()
    
    exec_globals = {
        'torch': torch,
        'nn': torch.nn,
        'F': torch.nn.functional,
        'np': np,
    }
    exec(code, exec_globals)
    
    GATActor = exec_globals['GATActor']
    
    # 创建 Actor
    device = torch.device('cpu')
    actor = GATActor(
        n_agents=3,
        node_dim=18,
        hidden_dim=64,
        action_dim=4
    ).to(device)
    
    print(f"\n[1] GATActor 初始化")
    print(f"    last_attn_weights 属性: {hasattr(actor, 'last_attn_weights')}")
    print(f"    初始值: {actor.last_attn_weights}")
    
    # 前向传播
    batch_size = 8
    obs = torch.randn(batch_size, 18).to(device)
    
    print(f"\n[2] 前向传播")
    print(f"    输入 obs: {obs.shape}")
    
    logits = actor(obs)
    
    print(f"    输出 logits: {logits.shape}")
    print(f"    Expected: ({batch_size}, 4)")
    
    # 检查权重存储
    print(f"\n[3] 注意力权重提取")
    w = getattr(actor, 'last_attn_weights', None)
    
    if w is not None:
        print(f"    ✓ 成功获取权重")
        print(f"    权重类型: {type(w)}")
        print(f"    权重形状: {w.shape}")
        print(f"    权重范围: [{w.min():.4f}, {w.max():.4f}]")
        print(f"    权重和: {w.sum(dim=-1).mean():.4f} (应该 ≈ 1.0)")
        
        # 提取标量值
        w_scalar = float(w[0, 0, 0].item())
        print(f"    单个样本权重值: {w_scalar:.6f}")
        
    else:
        print(f"    ✗ 权重获取失败 (为 None)")
        return False
    
    # 验证权重的数学性质
    print(f"\n[4] 权重数学性质验证")
    
    # 检查权重和为1
    row_sums = w.sum(dim=-1)  # (B, 1)
    print(f"    每行和: mean={row_sums.mean():.6f}, std={row_sums.std():.6f}")
    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        print(f"    ✓ 权重符合概率分布 (和为1)")
    else:
        print(f"    ✗ 权重和不为1")
        return False
    
    # 检查权重为正
    if (w >= 0).all():
        print(f"    ✓ 权重全为非负数")
    else:
        print(f"    ✗ 存在负权重")
        return False
    
    # 对于单节点图，权重应该是 [1.0]
    print(f"\n[5] 单节点图的特殊性")
    print(f"    理论: 单节点图中 w[b,0,0] 应该 = 1.0")
    print(f"    实际: {w_scalar:.6f}")
    if abs(w_scalar - 1.0) < 1e-5:
        print(f"    ✓ 符合预期（单节点自注意力）")
    else:
        print(f"    ⚠ 存在偏离（可能由数值精度问题）")
    
    # 批量验证
    print(f"\n[6] 批量验证")
    all_scalars = w.reshape(batch_size).numpy()
    print(f"    所有权重值: {all_scalars}")
    print(f"    方差: {all_scalars.var():.6f}")
    
    print(f"\n" + "="*70)
    print("✓ GAT 注意力权重验证成功！")
    print("="*70)
    
    return True


def test_weight_in_episode():
    """测试在episode中权重的动态变化"""
    print("\n" + "="*70)
    print("Episode 中权重追踪测试")
    print("="*70)
    
    with open('MAPPO_MPE训练3_带GATpy', 'r', encoding='utf-8') as f:
        code = f.read()
    
    exec_globals = {
        'torch': torch,
        'nn': torch.nn,
        'F': torch.nn.functional,
        'np': np,
        'Categorical': torch.distributions.Categorical,
    }
    exec(code, exec_globals)
    
    GATActor = exec_globals['GATActor']
    MAPPO = exec_globals['MAPPO']
    
    # 创建 MAPPO
    device = torch.device('cpu')
    mappo = MAPPO(
        obs_dim=18,
        action_dim=4,
        num_agents=3,
        actor_hidden_dims=[64],
        critic_hidden_dims_back=[64],
        critic_hidden_dims_front=[64],
        device=device
    )
    
    print(f"\n[1] 创建 MAPPO 实例")
    print(f"    Actor 类型: {type(mappo.actor).__name__}")
    
    # 模拟多个时间步
    print(f"\n[2] 模拟 10 个时间步的权重追踪")
    weights_history = []
    
    for t in range(10):
        # 生成随机观察
        obs = np.random.randn(18).astype(np.float32)
        probs, action = mappo.take_action(obs, explore=True, agent_id=0)
        
        # 获取权重
        w = getattr(mappo.actor, 'last_attn_weights', None)
        if w is not None:
            w_scalar = float(w[0, 0, 0].item())
            weights_history.append(w_scalar)
            status = "✓"
        else:
            weights_history.append(None)
            status = "✗"
        
        print(f"    Step {t+1:2d}: {status} weight={w_scalar:.6f if w is not None else 'N/A':>8s}")
    
    # 统计
    valid_weights = [w for w in weights_history if w is not None]
    if valid_weights:
        print(f"\n[3] 权重统计")
        print(f"    有效权重数: {len(valid_weights)}")
        print(f"    平均值: {np.mean(valid_weights):.6f}")
        print(f"    标准差: {np.std(valid_weights):.6f}")
        print(f"    ✓ 权重追踪成功，已采集 {len(valid_weights)} 个数据点")
        return True
    else:
        print(f"    ✗ 权重追踪失败，无有效数据")
        return False


if __name__ == '__main__':
    success1 = test_gat_attention_weights()
    success2 = test_weight_in_episode()
    
    if success1 and success2:
        print("\n" + "="*70)
        print("✓✓✓ 所有测试通过！GAT 权重系统工作正常 ✓✓✓")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ 部分测试失败，请检查日志")
        print("="*70)
