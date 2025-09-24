import sys
import os
import numpy as np
import torch as th
from math import *
import gymnasium as gymn
# from gym import spaces
import copy
import matplotlib.pyplot as plt
# import json
# import glob
import time

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取project目录
def get_current_file_dir():
    # 判断是否在 Jupyter Notebook 环境
    try:
        shell = get_ipython().__class__.__name__  # ← 误报，不用管
        if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
            # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
            return os.getcwd()
        else:  # 其他 shell
            return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 普通 Python 脚本
        return os.path.dirname(os.path.abspath(__file__))
current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(current_dir))

from Algorithms.SquashedPPOcontinues_dual_a_out import *

######## 调用环境


env = gymn.make("Pendulum-v1")
from tqdm import tqdm
obs_space = env.observation_space
action_space = env.action_space

# dof = 3
# 超参数
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 50  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
num_episodes = 1000  # 2000 400
hidden_dim = [128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
k_entropy=0
transition_dict_capacity = 200

state_dim = obs_space.shape[0]
action_dim = action_space.shape[0]
action_bound = np.array([action_space.low, action_space.high]).T  # 动作幅度限制


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device, k_entropy)

out_range_count = 0
return_list = []
steps_count = 0

start_time = time.time()
steps_since_update = 0
transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}

try:
    # 强化学习训练
    rl_steps = 0
    with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
        for i_episode in range(int(num_episodes)):
            episode_return = 0         
            state, _ = env.reset()
            done = False
            while not done:  # 每个训练回合
                # 1.执行动作得到环境反馈
                action, u = agent.take_action(state, action_bounds=action_bound, explore=True)
                rl_steps += 1
                total_action = action

                next_state, reward, terminated, truncated, _ = env.step(total_action)
                done = terminated or truncated

                transition_dict['states'].append(state)
                transition_dict['actions'].append(u)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append((reward + 8.0) / 8.0)
                transition_dict['dones'].append(done)
                transition_dict['action_bounds'].append(action_bound)
                state = next_state
                episode_return += reward
                steps_count += 1
                steps_since_update += 1

                if steps_since_update >= transition_dict_capacity:
                    steps_since_update = 0
                    agent.update(transition_dict)
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}

            return_list.append(episode_return)
            
            # tqdm 训练进度显示
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                                'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

except KeyboardInterrupt:
    print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
finally:
    def moving_average(a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size - 1, 2)
        begin = np.cumsum(a[:window_size - 1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))
    # 记录训练结束时间
    end_time = time.time()
    training_duration = end_time - start_time

    episodes_list = list(range(len(return_list)))
    plt.figure()
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on pendulum')

    mv_return = moving_average(return_list, 9)
    plt.figure()
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on pendulum')

    plt.show()