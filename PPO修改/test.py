import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from math import *
def _wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

import random
import gym
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F

from gym import spaces
from numpy.linalg import norm
from torch.distributions import Normal

# 超参数
actor_lr = 1e-3 / 10  # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 10  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
num_episodes = 200  # 2000
hidden_dim = [128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        # self.prelu = torch.nn.PReLU()

        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)  # todo 补充多维输出

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        self.fc_std = torch.nn.Linear(prev_size, action_dim)

    def forward(self, x, action_bound=1.0, min_std=1e-3):
        x = self.net(x)
        mu = action_bound * self.fc_mu(x)
        std = action_bound * F.softplus(self.fc_std(x))  # + 1e-8
        std = torch.clamp(std, min=min_std)  # 设置 std 的最小值
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state, action_bound=2.0, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state, action_bound=action_bound)
        if not explore:
            action = mu
            return action[0].cpu().detach().numpy().flatten()  # 支持一维和多维动作，而不是.item只支持1维或.squeeze只支持多维
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action[0].cpu().detach().numpy().flatten()  # 支持一维和多维动作，而不是.item只支持1维或.squeeze只支持多维
        # return [action.item()]

    def update(self, transition_dict, action_bound=1):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'],
        #                        dtype=torch.float).view(-1, 1).to(self.device)
        # fixme actions不适合flatten
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        mu, std = self.actor(states, action_bound=action_bound)  # 均值、方差
        # 添加Actor NaN检查
        if torch.isnan(mu).any() or torch.isnan(std).any():
            raise ValueError("NaN in Actor outputs before loop")
        # 添加Critic NaN检查
        critic_values = self.critic(states)
        if torch.isnan(critic_values).any():
            raise ValueError("NaN in Critic outputs before loop")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # 时序差分回报值
        td_delta = td_target - self.critic(states)  # 优势函数用时序差分回报与Critic网络输出作差表示
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        if torch.isnan(old_log_probs).any():
            raise ValueError("替代动作无法被转换")

        for _ in range(self.epochs):
            mu, std = self.actor(states, action_bound=action_bound)
            # 添加Actor NaN检查
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values = self.critic(states)
            # 添加Critic NaN检查
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)

            # print(ratio)
            # test 给actor更新添加熵正则项
            dist_entropy = action_dists.entropy()  # 获取熵

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            # actor_loss = torch.mean(-torch.min(surr1, surr2)) # original
            actor_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)  # test
            # print(actor_loss)
            # 计算 dist_entropy 的均值

            entropy_term = 0.1 * dist_entropy.mean()  # 添加熵正则项
            actor_loss = -actor_loss.mean() - entropy_term  # 合并熵项 # test Actor更新加一个熵项

            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)

            self.actor_optimizer.step()
            self.critic_optimizer.step()


###################################################################

class CarHeadingEnv(gym.Env):
    """
    Simple planar car with constant forward speed.
    - State: [heading_error, heading_error_rate]
      heading_error = wrap(desired_heading - current_heading) in radians
      heading_error_rate approximated by finite difference
    - Action: angular velocity (rad/s), continuous scalar
    - Each episode runs fixed n_steps (default 100)
    - Reward: dense, negative quadratic of heading_error: r = - (heading_error**2)
    - render(): plots trajectory using matplotlib (collects positions during episode)
    Gymnasium API: reset() -> (obs, info), step(a) -> (obs, reward, terminated, truncated, info)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 max_steps: int = 100,
                 dt: float = 0.1,
                 speed: float = 1.0,
                 max_omega: float = 2.0,
                 seed: int = None):
        super().__init__()
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.speed = float(speed)
        self.max_omega = float(max_omega)

        # action: angular velocity scalar
        self.action_space = spaces.Box(low=np.array([-self.max_omega], dtype=np.float32),
                                       high=np.array([self.max_omega], dtype=np.float32),
                                       shape=(1,), dtype=np.float32)
        # observation: heading_error in radians (-pi,pi), derivative (rad/s) bounded
        obs_high = np.array([math.pi, self.max_omega * 2], dtype=np.float32)
        obs_low = -obs_high
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.seed(seed)

        # internal state
        self.current_heading = 0.0
        self.desired_heading = 0.0
        self.prev_error = 0.0
        self.step_count = 0
        self.error_dot = 0.0

        # rendering buffers
        self.x = 0.0
        self.y = 0.0
        self.trajectory = []

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def get_state(self,):
        error = _wrap_angle(self.desired_heading - self.current_heading)
        error_dot = self.error_dot
        return np.array([sin(error), cos(error), error_dot])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        # randomize headings (uniform -pi..pi)
        self.current_heading = self._rng.uniform(-math.pi, math.pi)
        self.desired_heading = self._rng.uniform(-math.pi, math.pi)
        self.prev_error = _wrap_angle(self.desired_heading - self.current_heading)
        self.step_count = 0

        # reset pose for render
        self.x, self.y = 0.0, 0.0
        self.trajectory = [(self.x, self.y)]

        obs = np.array([sin(self.prev_error), cos(self.prev_error), 0.0], dtype=np.float32)  # error, error_dot (0 at reset)
        return obs, {}

    def step(self, action):
        # clip action
        omega = float(np.clip(action, -self.max_omega, self.max_omega).item() if hasattr(action, "item") else np.clip(action, -self.max_omega, self.max_omega))
        # apply continuous-time approximate dynamics
        # heading integrates angular velocity
        self.current_heading = _wrap_angle(self.current_heading + omega * self.dt)

        # update position (for render) with constant forward speed along current heading
        self.x += self.speed * math.cos(self.current_heading) * self.dt
        self.y += self.speed * math.sin(self.current_heading) * self.dt
        self.trajectory.append((self.x, self.y))

        # compute new error and derivative
        error = _wrap_angle(self.desired_heading - self.current_heading)
        self.error_dot = _wrap_angle(error-self.prev_error) / self.dt

        # reward: dense, negative quadratic of heading error (you can scale)
        reward = + (error ** 2)

        self.prev_error = error
        self.step_count += 1
        terminated = bool(self.step_count >= self.max_steps)
        truncated = False

        obs = np.array([sin(error), cos(error), self.error_dot], dtype=np.float32)
        info = {"current_heading": float(self.current_heading),
                "desired_heading": float(self.desired_heading),
                "step": int(self.step_count)}
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human", show=True):
        # visualize trajectory and headings
        xs, ys = zip(*self.trajectory) if self.trajectory else ([0.0], [0.0])
        plt.figure(figsize=(6,6))
        plt.plot(xs, ys, '-o', markersize=3, label='trajectory')
        plt.scatter([xs[0]], [ys[0]], c='green', label='start')
        plt.scatter([xs[-1]], [ys[-1]], c='red', label='end')

        # draw desired heading vector from origin
        origin = (0.0, 0.0)
        dhx = math.cos(self.desired_heading)
        dhy = math.sin(self.desired_heading)
        plt.arrow(origin[0], origin[1], dhx, dhy, head_width=0.05, color='orange', length_includes_head=True, label='desired')

        # draw final heading vector from last position
        fhx = math.cos(self.current_heading)
        fhy = math.sin(self.current_heading)
        plt.arrow(xs[-1], ys[-1], fhx * 0.5, fhy * 0.5, head_width=0.03, color='blue', length_includes_head=True, label='final heading')

        plt.axis('equal')
        plt.grid(True)
        plt.title(f"Car trajectory (steps={len(xs)})")
        plt.legend()
        if show:
            plt.show()

    def close(self):
        plt.close('all')


import numpy as np
import matplotlib.pyplot as plt

# 假设 CarHeadingEnv 已在当前命名空间定义为 CarHeadingEnv（如 notebook 前面单元格）
env = CarHeadingEnv(max_steps=100, dt=0.1, speed=1.0, max_omega=2.0, seed=0)
env.seed(0)
np.random.seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
state_dim = 3
action_dim = 1
action_bound = 1  # 动作最大值
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

returns = []

for ep in range(num_episodes):
    if ep %10 ==0:
        print(ep, "in", num_episodes, "epidosdes")
    obs, info = env.reset()
    ep_ret = 0.0
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    done = False
    for t in range(env.max_steps):
        # 随机策略：从动作空间采样
        action = agent.take_action(obs, action_bound=action_bound, explore=True)
        step_res = env.step(action)
        # 兼容不同 gym / gymnasium 返回签名
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, info = step_res
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step_res
        transition_dict['states'].append(obs)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_obs)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        ep_ret += float(reward)
        obs = next_obs
        if done:
            break
    returns.append(ep_ret)
    agent.update(transition_dict)

print(f"episodes: {num_episodes}, mean return: {np.mean(returns):.3f}, std: {np.std(returns):.3f}")

# 绘制每回合总回报
plt.figure()
plt.plot(returns, '-o')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Random policy returns')
plt.grid(True)
plt.show()

# 可视化一个示例回合轨迹（重新运行一次并调用 render）
obs, info = env.reset()
for t in range(env.max_steps):
    print("状态：", env.get_state())
    # action = env.action_space.sample()
    action = agent.take_action(obs, action_bound=action_bound, explore=False)
    step_res = env.step(action)
    if len(step_res) == 5:
        next_obs, reward, terminated, truncated, info = step_res
        done = bool(terminated or truncated)
    else:
        next_obs, reward, done, info = step_res
    obs= next_obs
    if done:
        break

env.render()   # matplotlib 显示轨迹
