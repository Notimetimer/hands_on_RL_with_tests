import sys
import os
import numpy as np
import torch as th
from math import *
import gymnasium as gymn
# from gymnasium.wrappers import AutoResetEnv
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

# ====== 使用矢量化环境 ======

# 由于 gymnasium 1.0.0 没有 AutoResetEnv，这里实现一个小的兼容包装器。
# 行为：当子环境返回 terminated 或 truncated 时，先保存 final observation，
# 然后立即对该子环境调用 reset() 并把 reset 后的 observation返回给外层，但保留 terminated/truncated=True
# 这样外层可以检测到该子环境完成 episode（通过 terminated/truncated），同时下一步状态就是自动重置后的初始观测。
class AutoResetWrapper(gymn.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done:
            # 保存 final obs 到 info（可选）
            info = info if isinstance(info, dict) else {}
            info['final_observation'] = obs
            # 自动重置该子环境并返回 reset 后的 obs（保持 terminated/truncated=True）
            new_obs, new_info = self.env.reset()
            # 合并 info（保留 reset 返回的 info 字典）
            if isinstance(new_info, dict):
                info.update({'reset_info': new_info})
            return new_obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, truncated, info

# --------------- 在模块顶层定义一个可 picklable 的 env 工厂 ---------------
def make_env_fn(seed_offset=0):
    """返回一个零参数的可调用 thunk，供 AsyncVectorEnv 使用（顶层可 picklable）"""
    def _thunk():
        env = AutoResetWrapper(gymn.make("Pendulum-v1"))
        # 注意：不要在这里调用 env.reset(seed=...)，改为在主流程的 transition_dict_capacity 定义后统一 reset
        return env
    return _thunk

# 将原来把 env 创建放在 main() 的位置改为：把 main guard 提前到超参数之前，
# 在 guard 内用 make_env_fn 构建 env_fns 并创建 AsyncVectorEnv。
# 这样保证 Windows 下 spawn 时不会在子进程重复执行创建子进程的代码。
if __name__ == '__main__':
    # --------- 在这里开始你的超参数设置及训练（原来在 main() 内的内容） ----------
    # dof = 3
    # 超参数
    actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
    critic_lr = actor_lr * 50  # *10 为什么critic学习率大于一都不会梯度爆炸？ ...
    num_episodes = 1000  # 2000 400
    hidden_dim = [128]  # 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10  # 10
    eps = 0.2
    k_entropy=0
    transition_dict_capacity = 200
    # 子环境数量（可以在这里调整）
    num_envs = 5

    from multiprocessing import freeze_support
    freeze_support()  # Windows: 推荐调用
    
    # 为每个子环境生成一个 thunk（顶层函数返回的闭包是可 picklable 的）
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    # 创建异步向量环境（必须在主保护下进行）
    env = gymn.vector.AsyncVectorEnv(env_fns)

    # 把 env.reset 从 make_env_fn 中移出，统一在这里初始化 vector env（放在 transition_dict_capacity 之后）
    # 这会触发每个子环境的第一次 reset（AutoResetWrapper 的 reset 会被调用）
    states, _ = env.reset()

    from tqdm import tqdm
    obs_space = env.single_observation_space
    action_space = env.single_action_space

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
        # 强化学习训练（把原来 main() 中的训练循环原封不动放在这里）
        rl_steps = 0
        with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
            episode_idx = 0
            last_reported = 0  # 上次向 tqdm 报告的已完成 episode 总数（按 10 的倍数报告）
            # 按子环境独立进行 episode 收集，直到收集到 num_episodes 个 episode
            states, _ = env.reset()
            episode_returns = np.zeros(num_envs)
            while episode_idx < num_episodes:
                # 1.执行动作得到环境反馈（对每个子环境）
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
                    transition_dict['states'].append(states[i])
                    transition_dict['actions'].append(us[i])
                    transition_dict['next_states'].append(next_states[i])
                    transition_dict['rewards'].append((rewards[i] + 8.0) / 8.0)
                    transition_dict['dones'].append(dones[i])
                    transition_dict['action_bounds'].append(action_bound)
                    episode_returns[i] += rewards[i]
                    steps_count += 1
                    steps_since_update += 1

                    # 如果子环境完成一个 episode，记录并重置该子环境的累计返回（AutoResetWrapper 已自动重置状态）
                    if dones[i]:
                        return_list.append(episode_returns[i])
                        episode_idx += 1
                        episode_returns[i] = 0.0

                states = next_states

                if steps_since_update >= transition_dict_capacity:
                    steps_since_update = 0
                    agent.update(transition_dict)
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}

                # tqdm 训练进度显示（仅当累计完成的 episode 达到 10 的倍数时更新进度条）
                completed_groups = episode_idx // 10
                last_reported_groups = last_reported // 10
                if completed_groups > last_reported_groups:
                    increments = (completed_groups - last_reported_groups) * 10
                    pbar.update(increments)
                    # 显示最近 10 个 episode 的平均返回（如果不足 10 个则显示已有的平均）
                    if len(return_list) >= 10:
                        recent_mean = np.mean(return_list[-10:])
                    else:
                        recent_mean = np.mean(return_list) if return_list else 0.0
                    # 保证为有限数值并格式化，强制刷新 tqdm 输出
                    if not np.isfinite(recent_mean):
                        recent_mean = 0.0
                    pbar.set_postfix({'episode': int(episode_idx),
                                      'return': f'{recent_mean:.3f}'}, refresh=True)
                    last_reported = completed_groups * 10

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
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
        plt.title('PPO on pendulum (vectorized)')

        mv_return = moving_average(return_list, 9)
        plt.figure()
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('PPO on pendulum (vectorized)')

        plt.show()
    # 训练结束后记得关闭 env
    env.close()