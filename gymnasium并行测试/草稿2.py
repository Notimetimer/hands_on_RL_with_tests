import sys
import os
import numpy as np
import torch as torch
from math import *
import gymnasium as gymn
import copy
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取project目录并添加路径
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

# 【修改1】引入新的算法文件
from Algorithms.PPOcontinues_std_no_state_with_truncs2 import PPOContinuous

# 超参数
actor_lr = 1e-4
critic_lr = actor_lr * 50
max_steps = 800 * 200
hidden_dim = [128]
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
k_entropy = 0
transition_dict_capacity = 200 # 现在的含义是：每个环境采集多少步后更新
num_envs = 5 

# 创建环境工厂
def make_env(seed):
    def _thunk():
        env = gymn.make("Pendulum-v1")
        env.reset(seed=seed)
        return env
    return _thunk

# 【修改2】直接使用 Gym 原生矢量化环境，它会自动处理 Reset
env_fns = [make_env(i) for i in range(num_envs)]
env = gymn.vector.SyncVectorEnv(env_fns)

obs_space = env.single_observation_space
action_space = env.single_action_space

state_dim = obs_space.shape[0]
action_dim = action_space.shape[0]
action_bound = np.array([action_space.low, action_space.high]).T

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 初始化 Agent
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device, k_entropy)

# 统计变量
steps_list = []
return_list = []
# 维护每个子环境当前的 episode return
current_episode_returns = np.zeros(num_envs)

start_time = time.time()
steps_since_update = 0

# 【修改3】统一的 Experience Buffer
# 这里的 List 将存储 numpy 数组，例如 states 的每个元素形状为 (num_envs, state_dim)
buffer = {
    'states': [], 
    'actions': [],    # 存 raw_u (pre-tanh)
    'next_states': [], 
    'rewards': [], 
    'dones': [],      # 存 terminated (Done=True, Value=Reward)
    'truncs': [],     # 存 truncated (Done=True, Value=V(s'))
    'action_bounds': []
}

try:
    rl_steps = 0
    with tqdm(total=int(max_steps), desc='Steps') as pbar:
        # VectorEnv reset 返回 (obs, infos)
        states, _ = env.reset()
        
        while rl_steps < max_steps:
            rl_steps += 1
            
            # 1.执行动作
            # 注意：agent.take_action 目前设计是处理单个状态，
            # 为了适配 (N, Dim) 输入，我们在外层循环调用 (或者修改 Agent 支持 batch 输入)
            # 这里保持原逻辑简单循环，虽然牺牲一点效率但兼容性好
            actions_exec = []
            us_raw = []
            
            for i in range(num_envs):
                # take_action 返回 (执行动作, 原始u)
                a, u = agent.take_action(states[i], action_bounds=action_bound, explore=True)
                actions_exec.append(a)
                us_raw.append(u)
            
            actions_exec = np.array(actions_exec) # (num_envs, action_dim)
            us_raw = np.array(us_raw)             # (num_envs, action_dim)

            # 2. 环境交互
            next_states, rewards, terminateds, truncateds, infos = env.step(actions_exec)
            
            # 3. 处理真实 Next State
            # SyncVectorEnv 会自动 reset，此时 next_states 是新回合的初始状态
            # 真实的末端状态藏在 infos 里面
            real_next_states = next_states.copy()
            
            # 检查是否有环境完成
            if "final_observation" in infos:
                # final_observation 是一个列表或数组，对应每个环境
                for i in range(num_envs):
                    # 如果该环境这一步结束了 (terminated 或 truncated)
                    if infos["_final_observation"][i]: 
                        real_next_states[i] = infos["final_observation"][i]

            # 4. 记录数据到 Buffer
            # 这里的 append 操作存入的是 (num_envs, ...) 的切片
            buffer['states'].append(states)
            buffer['actions'].append(us_raw) # 存原始 u
            buffer['next_states'].append(real_next_states)
            buffer['rewards'].append((rewards + 8.0) / 8.0) # 保持原有的 Reward Scaling
            buffer['dones'].append(terminateds) # 只有 terminated 才是真正的结束
            buffer['truncs'].append(truncateds) # truncated 需要特殊处理
            
            # 扩展 action_bound 以匹配 (num_envs, action_dim, 2)
            # 这样算法内部处理时维度一致
            batch_bounds = np.tile(action_bound, (num_envs, 1, 1))
            buffer['action_bounds'].append(batch_bounds)

            # 5. 统计 Return
            current_episode_returns += rewards
            
            # 检查结束的 Episode 并记录
            dones = np.logical_or(terminateds, truncateds)
            for i in range(num_envs):
                if dones[i]:
                    return_list.append(current_episode_returns[i])
                    steps_list.append(rl_steps)
                    current_episode_returns[i] = 0.0
                    
            states = next_states
            steps_since_update += 1
            
            # 6. 更新策略
            if steps_since_update >= transition_dict_capacity:
                steps_since_update = 0
                
                # 【修改4】直接调用 update，无需手动计算 GAE
                # preprocess_parallel_buffer 会自动识别 (Time, N, Dim) 结构
                # 并计算 advantage, td_target，然后 flatten
                agent.update(buffer, shuffled=1)
                
                # 清空 buffer
                buffer = {
                    'states': [], 'actions': [], 'next_states': [], 
                    'rewards': [], 'dones': [], 'truncs': [], 'action_bounds': []
                }

            # tqdm 更新
            if rl_steps % 200 == 0:
                pbar.update(200)
                if len(return_list) > 0:
                    recent_mean = np.mean(return_list[-10:])
                    pbar.set_postfix({
                        'steps': '%d' % rl_steps, 
                        'return': f'{recent_mean:.3f}',
                        'actor_loss': f'{getattr(agent, "actor_loss", 0):.3f}'
                    })

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
    env.close() # 关闭环境
    
    plt.figure()
    plt.plot(steps_list, return_list, alpha=0.5, label='Raw')
    plt.xlabel('steps')
    plt.ylabel('Returns')
    plt.title('PPO on Pendulum (Vectorized & Concise)')
    
    if len(return_list) > 10:
        mv_return = moving_average(return_list, 9)
        plt.plot(steps_list, mv_return, label='Smoothed')
    
    plt.legend()
    plt.show()