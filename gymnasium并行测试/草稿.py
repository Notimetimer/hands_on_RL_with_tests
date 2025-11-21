import sys
import os
import numpy as np
import torch as th
from math import *
import gymnasium as gymn
import copy
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
# partial
from functools import partial

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取project目录
def get_current_file_dir():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return os.getcwd()
        else:
            return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.dirname(os.path.abspath(__file__))
current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(current_dir))

from Algorithms.SquashedPPOcontinues_dual_a_out import *

# ====== 使用矢量化环境 ======
# 行为：当子环境返回 terminated 或 truncated 时，AutoResetWrapper 会自动 reset
# 返回的 obs 是 reset 后的，但 terminated/truncated 依然为 True 以便算法识别
class AutoResetWrapper(gymn.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done:
            info = info if isinstance(info, dict) else {}
            info['final_observation'] = obs
            new_obs, new_info = self.env.reset()
            if isinstance(new_info, dict):
                info.update({'reset_info': new_info})
            return new_obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, truncated, info

# 可调用类
class EnvFactory:
    def __init__(self, seed):
        self.seed = seed
    def __call__(self):
        env = AutoResetWrapper(gymn.make("Pendulum-v1"))
        env.reset(seed=self.seed)
        return env
    
# 超参数
actor_lr = 1e-4
critic_lr = actor_lr * 50
max_steps = 800*200
hidden_dim = [128]
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
k_entropy = 0
transition_dict_capacity = 200
num_envs = 5 # 子环境数目，建议设置为 1 进行对比测试，之后再增加

# 创建环境
env_fns = [EnvFactory(i) for i in range(num_envs)]
env = gymn.vector.SyncVectorEnv(env_fns)

obs_space = env.single_observation_space
action_space = env.single_action_space

state_dim = obs_space.shape[0]
action_dim = action_space.shape[0]
action_bound = np.array([action_space.low, action_space.high]).T

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device, k_entropy)

# 统计变量
steps_list = []
return_list = []

start_time = time.time()
steps_since_update = 0

# 为每个子环境维护独立缓冲
per_env_buffers = [
    {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
    for _ in range(num_envs)
]

# 【修改2】维护每个子环境当前的 episode return
# 这样任何一个子环境 done 了，我们都记录下来，这与串行代码逻辑一致
current_episode_returns = np.zeros(num_envs)

try:
    rl_steps = 0
    with tqdm(total=int(max_steps), desc='Steps') as pbar:
        states, _ = env.reset()
        
        while rl_steps < max_steps:
            rl_steps += 1
            
            # 1.执行动作
            actions = []
            us = []
            for state in states:
                action, u = agent.take_action(state, action_bounds=action_bound, explore=True)
                actions.append(action)
                us.append(u)
            actions = np.array(actions)

            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(num_envs):
                # 记录 reward
                current_episode_returns[i] += rewards[i]

                # 存入 per-env buffer
                per_env_buffers[i]['states'].append(states[i])
                per_env_buffers[i]['actions'].append(us[i])
                per_env_buffers[i]['next_states'].append(next_states[i])
                per_env_buffers[i]['rewards'].append((rewards[i] + 8.0) / 8.0)
                per_env_buffers[i]['dones'].append(dones[i])
                per_env_buffers[i]['action_bounds'].append(action_bound)

                # 【修改3】事件触发式记录：如果当前子环境结束，记录该回合回报
                if dones[i]:
                    return_list.append(current_episode_returns[i])
                    steps_list.append(rl_steps) # 记录当前的全局步数
                    current_episode_returns[i] = 0.0
                    
            states = next_states
            steps_since_update += 1
            
            # 2.更新策略
            if steps_since_update >= transition_dict_capacity:
                steps_since_update = 0
                
                # 【修改4】分环境计算 GAE，然后合并
                # 这样可以正确处理子环境的截断（Bootstrap），并且不会混淆 Env1 和 Env2 的数据
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 
                                   'dones': [], 'action_bounds': [], 'advantages': [], 'td_targets': []}
                
                for i in range(num_envs):
                    ebuf = per_env_buffers[i]
                    if len(ebuf['states']) == 0: continue
                    
                    # 【重要修复】绝对不要强制置 dones[-1] = True
                    # PPO 算法内部会利用 ebuf['next_states'][-1] 和 ebuf['dones'][-1]
                    # 来正确处理截断情况（如果 done=False，则 V(next) 作为 target）。
                    
                    # 调用新加的 API，在合并前计算该子环境的 GAE
                    ebuf = agent.compute_gae_for_buffer(ebuf)
                    
                    # 合并数据
                    transition_dict['states'].extend(ebuf['states'])
                    transition_dict['actions'].extend(ebuf['actions'])
                    transition_dict['next_states'].extend(ebuf['next_states'])
                    transition_dict['rewards'].extend(ebuf['rewards'])
                    transition_dict['dones'].extend(ebuf['dones'])
                    transition_dict['action_bounds'].extend(ebuf['action_bounds'])
                    transition_dict['advantages'].extend(ebuf['advantages'])
                    transition_dict['td_targets'].extend(ebuf['td_targets'])

                # 【修改5】统一 API，开启 shuffled=1 以打破数据相关性
                agent.update(transition_dict, shuffled=1)
                
                # 清空 buffer
                per_env_buffers = [
                    {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
                    for _ in range(num_envs)
                ]

            # tqdm 更新
            if rl_steps % 200 == 0:
                pbar.update(200)
                if len(return_list) > 0:
                    recent_mean = np.mean(return_list[-10:])
                    pbar.set_postfix({'steps': '%d' % rl_steps, 'return': f'{recent_mean:.3f}'})

except KeyboardInterrupt:
    print("\n检测到 KeyboardInterrupt")
finally:
    def moving_average(a, window_size):
        if len(a) < window_size: return np.array(a)
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size - 1, 2)
        begin = np.cumsum(a[:window_size - 1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    end_time = time.time()
    
    # 绘图（此时 steps_list 和 return_list 的定义与串行代码完全一致）
    plt.figure()
    plt.plot(steps_list, return_list, alpha=0.5, label='Raw')
    plt.xlabel('steps')
    plt.ylabel('Returns')
    plt.title('PPO on pendulum (Parallel Corrected)')
    
    if len(return_list) > 10:
        mv_return = moving_average(return_list, 9)
        plt.plot(steps_list, mv_return, label='Smoothed')
    
    plt.legend()
    plt.show()