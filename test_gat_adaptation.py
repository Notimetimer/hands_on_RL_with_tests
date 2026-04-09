"""
GAT Actor/Critic 适配验证脚本
用于验证基于GAT构建的actor和critic是否正确集成到MAPPO中
"""

import torch
import numpy as np
import sys

def test_gat_models():
    """测试GATActor和GATCritic的生成功能"""
    print("=" * 60)
    print("GAT Actor/Critic 适配验证")
    print("=" * 60)
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    obs_dim = 18
    action_dim = 4
    num_agents = 3
    batch_size = 8
    
    print(f"[CONFIG] obs_dim={obs_dim}, action_dim={action_dim}, num_agents={num_agents}")
    
    # 导入模型类 - 使用exec来执行包含中文注释的文件
    print("\n[STEP 1] 导入模型定义...")
    try:
        with open('MAPPO_MPE训练3_带GATpy', 'r', encoding='utf-8') as f:
            model_code = f.read()
        
        # 提取必要的类定义
        exec_globals = {
            'torch': torch,
            'nn': torch.nn,
            'F': torch.nn.functional,
            'np': np,
            'Categorical': torch.distributions.Categorical,
        }
        exec(model_code, exec_globals)
        
        GATLayer = exec_globals['GATLayer']
        GATActor = exec_globals['GATActor']
        GATCritic = exec_globals['GATCritic']
        MAPPO = exec_globals['MAPPO']
        
        print("  [SUCCESS] 模型类导入成功")
    except Exception as e:
        print(f"  [FAILED] 模型导入失败: {e}")
        return False
    
    # 测试GATActor
    print("\n[STEP 2] 测试 GATActor...")
    try:
        actor = GATActor(
            n_agents=num_agents,
            node_dim=obs_dim,
            hidden_dim=64,
            action_dim=action_dim
        ).to(device)
        
        # 前向传播
        obs = torch.randn(batch_size, obs_dim).to(device)
        logits = actor(obs)
        
        assert logits.shape == (batch_size, action_dim), \
            f"Actor output shape mismatch: {logits.shape} != ({batch_size}, {action_dim})"
        
        print(f"  [SUCCESS] Actor forward: obs {tuple(obs.shape)} -> logits {tuple(logits.shape)}")
    except Exception as e:
        print(f"  [FAILED] Actor 测试失败: {e}")
        return False
    
    # 测试GATCritic
    print("\n[STEP 3] 测试 GATCritic...")
    try:
        critic = GATCritic(
            n_agents=num_agents,
            node_dim=obs_dim,
            hidden_dim=64
        ).to(device)
        
        # 前向传播
        global_states = torch.randn(batch_size, num_agents, obs_dim).to(device)
        values = critic(global_states)
        
        assert values.shape == (batch_size, num_agents), \
            f"Critic output shape mismatch: {values.shape} != ({batch_size}, {num_agents})"
        
        print(f"  [SUCCESS] Critic forward: global_states {tuple(global_states.shape)} -> values {tuple(values.shape)}")
    except Exception as e:
        print(f"  [FAILED] Critic 测试失败: {e}")
        return False
    
    # 测试MAPPO集成
    print("\n[STEP 4] 测试 MAPPO 集成...")
    try:
        mappo = MAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            actor_hidden_dims=[64, 64],
            critic_hidden_dims_back=[64],
            critic_hidden_dims_front=[64],
            device=device
        )
        
        # 验证actor和critic是否为正确的类型
        assert isinstance(mappo.actor, GATActor), "MAPPO的actor不是GATActor"
        assert isinstance(mappo.critic, GATCritic), "MAPPO的critic不是GATCritic"
        
        print("  [SUCCESS] MAPPO 集成正确: actor 和 critic 使用 GAT")
        
        # 测试take_action方法
        obs_sample = np.random.randn(obs_dim).astype(np.float32)
        probs, action = mappo.take_action(obs_sample, explore=True, agent_id=0)
        
        assert probs.shape == (action_dim,), f"Probs shape 错误: {probs.shape}"
        assert isinstance(action, (int, np.integer)), f"Action 类型错误: {type(action)}"
        
        print(f"  [SUCCESS] take_action 工作正常: probs shape {probs.shape}, action {action}")
        
    except Exception as e:
        print(f"  [FAILED] MAPPO 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 最终确认
    print("\n" + "=" * 60)
    print("[VERIFIED] 所有测试通过！")
    print("=" * 60)
    print("\n关键信息:")
    print("  ✓ GATActor 正确处理 per-agent observations")
    print("  ✓ GATCritic 正确处理 global states")
    print("  ✓ MAPPO 成功集成 GAT-based actor 和 critic")
    print("  ✓ 代码准备就绪，可以开始训练")
    print("\n提示：运行 python MAPPO_MPE训练3_带GATpy 来开始训练任务。")
    
    return True


if __name__ == '__main__':
    success = test_gat_models()
    sys.exit(0 if success else 1)
