import random
import gym
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

import os, sys
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))
current_dir = get_current_file_dir()
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
print(project_root)

# 设置后端
matplotlib.use('Qt5Agg') 

from tracking_test2 import testEnv

# ==========================================
# 1. 导入新的混合算法框架组件
# ==========================================
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper, PPOHybrid
from Algorithms.MLP_heads import ValueNet


# ==========================================
# 2. 辅助函数：离散动作 -> 连续动作转换
# ==========================================
class ActionDiscretizer:
    def __init__(self, action_space_low, action_space_high, bins_per_dim):
        """
        action_space_low: np.array, 动作下界, shape (action_dim,)
        action_space_high: np.array, 动作上界, shape (action_dim,)
        bins_per_dim: int, 每个维度切分的份数 (例如 5)
        """
        self.low = action_space_low
        self.high = action_space_high
        self.bins = bins_per_dim
        self.step_size = (self.high - self.low) / self.bins
        self.action_dim = len(action_space_low)

    def discrete_to_continuous(self, discrete_actions):
        """
        discrete_actions: list or np.array of ints (indices), e.g. [2, 0, 4]
        返回: 对应区间的中心值
        """
        # 确保输入是 numpy 数组
        indices = np.array(discrete_actions)
        # 计算中心值: min + index * step + step/2
        continuous_actions = self.low + indices * self.step_size + self.step_size / 2.0
        return continuous_actions


# ==========================================
# 3. 主训练配置
# ==========================================
dt = 0.5
dof = 3
bins = 5  # 将每个动作维度切分为 5 份

# 超参数
actor_lr = 1e-4  
critic_lr = 1e-3
num_episodes = 500
hidden_dim =[128, 128] # 稍微加深一点网络通常对离散动作有帮助
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 初始化环境
env_name = 'testEnv'
env = testEnv(dof=dof, dt=dt)

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

state_dim = env.observation_space.shape[0]
raw_action_dim = env.action_space.shape[0] # 物理环境的动作维度 (3)

# ==========================================
# 定义混合动作空间参数：仅使用多重离散动作(cat)
# ==========================================
# x, y, z 3个维度，每个维度5个选项 -> 'cat': [5, 5, 5]
action_dims_dict = {
    'cont': 0,
    'cat': [bins] * raw_action_dim, 
    'bern': 0
}

# 初始化动作转换器
discretizer = ActionDiscretizer(env.action_space.low, env.action_space.high, bins)

# ==========================================
# 实例化网络和 Agent (使用统一的 PPOHybrid)
# ==========================================
policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
actor = HybridActorWrapper(policy_net, action_dims_dict, device=device).to(device)
critic = ValueNet(state_dim, hidden_dim).to(device)

agent = PPOHybrid(
    actor=actor,
    critic=critic,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    lmbda=lmbda,
    epochs=epochs,
    eps=eps,
    gamma=gamma,
    device=device
)

out_range_count = 0
return_list =[]

import time
start_time = time.time()


# ==========================================
# 4. 训练循环
# ==========================================
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:
    for i_episode in range(int(num_episodes)):
        episode_return = 0
        
        # transition_dict 修改动作存储结构以适配 PPOHybrid23_0
        transition_dict = {
            'states': [], 
            'actions': {'cat':[]},  # 以字典形式存储动作序列
            'next_states': [], 
            'rewards': [], 
            'dones':[]
        }
        
        state, _ = env.reset(train=True)
        done = False
        
        while not done:
            # 1. Agent 决策 
            # take_action 返回: actions_exec, actions_raw, h_state, actions_dist_check
            actions_exec, actions_raw, _, _ = agent.take_action(state, explore=True)
            
            # 获取执行用的离散动作索引
            action_indices = actions_exec['cat']
            
            # 2. 将离散索引转换为连续动作以输入环境
            real_action = discretizer.discrete_to_continuous(action_indices)
            
            # 3. 环境步进
            next_state, reward, terminated, truncated, info = env.step(real_action)
            done = bool(terminated or truncated)
            
            # 4. 存储数据 (注意：存储 actions_raw 中的 raw indices)
            transition_dict['states'].append(state)
            transition_dict['actions']['cat'].append(actions_raw['cat']) # Store array like[2, 0, 4]
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            
            state = next_state
            episode_return += reward

        if hasattr(env, 'out_range') and env.out_range == 1:
            out_range_count += 1
            
        return_list.append(episode_return)
        
        # 5. 更新 Agent
        agent.update(transition_dict)
        
        if (i_episode + 1) >= 10:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                              'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)

end_time = time.time()
training_duration = end_time - start_time


# ==========================================
# 5. 绘图与测试
# ==========================================
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

episodes_list = list(range(len(return_list)))
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO Hybrid (Multi-Discrete) on {}'.format(env_name))

mv_return = moving_average(return_list, 9)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Smoothed Returns')
plt.title('PPO Hybrid (Multi-Discrete) on {}'.format(env_name))

# --- 测试回合 ---
car_trajectory = []
target_trajectory =[]
episode_return = 0

state, _ = env.reset(train=False)
done = False

while not done:
    # 测试时不探索 (explore=False 将透传为 {'cont': False, 'cat': False, 'bern': False})
    actions_exec, _, _, _ = agent.take_action(state, explore=False)
    
    # 提取并转换动作
    action_indices = actions_exec['cat']
    real_action = discretizer.discrete_to_continuous(action_indices)
    
    next_state, reward, terminated, truncated, info = env.step(real_action)
    done = bool(terminated or truncated)
    
    car_trajectory.append(env.state[0:dof].copy())
    target_trajectory.append(env.target_pos_[0:dof].copy())
    
    state = next_state
    episode_return += reward

# 绘制轨迹
plt.figure(4, figsize=(10, 8))
for i in range(dof):
    plt.subplot(dof, 1, i + 1)
    pos_trajectory = [s[i] for s in car_trajectory]
    target_pos_trajectory = [s[i] for s in target_trajectory]
    plt.plot(range(len(pos_trajectory)), pos_trajectory, 'b-', label='Position')
    plt.plot(range(len(target_pos_trajectory)), target_pos_trajectory, 'r--', label='Target')
    plt.xlabel('Step')
    plt.ylabel(f'Dim {i + 1}')
    plt.legend()

plt.tight_layout()
plt.show()

print("出界次数：", out_range_count)
print("训练时长（秒）：", training_duration)