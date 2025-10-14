import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

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
from math import *
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


def sub_of_radian(input1, input2):
    # 弧度减法
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

def calc_intern_dist2cylinder(R, pos_, psi, theta):
    """
    计算飞机到圆柱形边界的斜距离
    
    参数:
    R: float, 圆柱形边界半径
    rho: float, 飞机到圆心的距离
    eta: float, 飞机相对于圆心的方位角（弧度）
    psi: float, 飞机航向角（弧度）
    theta: float, 飞机俯仰角（弧度）
    
    返回:
    d: float, 飞机到边界的斜距离
    dh: float, 飞机到边界的水平距离
    pos_: ndarray, 飞机位置坐标 [北、天、东]
    """
    # 计算飞机位置
    pos_on_floor_ = np.array([pos_[0], 0, pos_[2]])
    rho = norm(pos_on_floor_)
    eta = atan2(pos_[2], pos_[0])
    
    # 计算水平距离
    dh_list = rho*cos(pi+eta-psi) + sqrt(R**2-rho**2*sin(pi+eta-psi)**2)
    dh = dh_list
    
    # 计算斜距离
    d = dh/(cos(theta)+1e-5)

    # 边界在飞机的左边还是右边
    left_or_right = np.sign(sub_of_radian(eta, psi)) # -1 左边，0 中间，1 右边
    return d, dh, left_or_right


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


class CarHeadingEnv(gym.Env):
    """
    Car driving in a circle boundary:
    - Constant speed: 10 m/s
    - Max angular velocity: 5 rad/s
    - Circle boundary: radius 30m
    - State: [dh/R, left_or_right] where dh is distance to boundary
    - Action: angular velocity (rad/s), continuous scalar
    - Episode ends when car exits circle
    - Reward: dh/R (normalized distance to boundary) if inside, large negative if outside
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 max_steps: int = 100,
                 dt: float = 0.1,
                 speed: float = 10.0,  # 固定速度10
                 max_omega: float = 5.0,  # 最大角速度5
                 circle_radius: float = 30.0,  # 圆形边界半径
                 seed: int = None):
        super().__init__()
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.speed = float(speed)
        self.max_omega = float(max_omega)
        self.R = float(circle_radius)

        # action: angular velocity scalar
        self.action_space = spaces.Box(
            low=np.array([-self.max_omega], dtype=np.float32),
            high=np.array([self.max_omega], dtype=np.float32),
            shape=(1,), dtype=np.float32)
        
        # observation: [dh/R, left_or_right]
        # dh/R normalized to [0,1], left_or_right in [-1,0,1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32)

        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.seed(seed)

        # internal state
        self.current_heading = 0.0  # 当前航向角
        self.step_count = 0
        
        # position state
        self.x = 0.0  # 北向位置
        self.y = 0.0  # 东向位置
        self.trajectory = []  # 轨迹存储

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def get_state(self):
        # 计算到边界距离
        pos_ = np.array([self.x, 0.0, self.y])  # [北,天,东]
        d, dh, left_or_right = calc_intern_dist2cylinder(
            self.R, pos_, self.current_heading, 0.0)
        
        # 加入遮罩：当 dh > 20 时，将 dh 限制为 20，left_or_right 置为 0
        mask_threshold = 20.0
        if dh > mask_threshold:
            dh = mask_threshold
            left_or_right = 0.0
            
        # 归一化 dh/R 到 [0,1]（使用遮罩后的 dh）
        norm_dh = min(max(dh/self.R, 0.0), 1.0)
        return np.array([norm_dh, left_or_right], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        # 重置位置到圆心
        self.x = 0.0
        self.y = 0.0
        # 随机初始航向
        self.current_heading = self._rng.uniform(-np.pi, np.pi)
        self.step_count = 0
        
        # 重置轨迹
        self.trajectory = [(self.x, self.y)]
        
        obs = self.get_state()
        return obs, {}

    def step(self, action):
        # clip action
        omega = float(np.clip(action, -self.max_omega, self.max_omega).item() 
                     if hasattr(action, "item") else np.clip(action, -self.max_omega, self.max_omega))
        
        # update heading
        self.current_heading = _wrap_angle(self.current_heading + omega * self.dt)
        
        # update position
        self.x += self.speed * math.cos(self.current_heading) * self.dt
        self.y += self.speed * math.sin(self.current_heading) * self.dt
        self.trajectory.append((self.x, self.y))

        # get new state (with masking)
        obs = self.get_state()
        
        # check if outside circle
        r = math.sqrt(self.x * self.x + self.y * self.y)
        outside = r > self.R
        
        # compute reward using masked dh
        if outside:
            reward = -10.0  # 出界惩罚
            terminated = True
        else:
            # obs[0] already contains the normalized masked dh/R
            reward = obs[0]  # 奖励使用遮罩后的归一化边界距离 dh/R
            terminated = False

        self.step_count += 1
        truncated = bool(self.step_count >= self.max_steps)
        
        info = {
            "position": (self.x, self.y),
            "heading": float(self.current_heading),
            "step": int(self.step_count)
        }
        
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human", show=True):
        # 创建图形
        plt.figure(figsize=(8,8))
        
        # 画圆形边界
        circle = plt.Circle((0, 0), self.R, fill=False, color='red', linestyle='--', label='boundary')
        plt.gca().add_patch(circle)
        
        # 画轨迹
        xs, ys = zip(*self.trajectory) if self.trajectory else ([0.0], [0.0])
        plt.plot(xs, ys, '-o', markersize=3, label='trajectory')
        plt.scatter([xs[0]], [ys[0]], c='green', label='start')
        plt.scatter([xs[-1]], [ys[-1]], c='red', label='end')

        # 画最后位置的航向
        fhx = math.cos(self.current_heading)
        fhy = math.sin(self.current_heading)
        plt.arrow(xs[-1], ys[-1], fhx * 2, fhy * 2, 
                 head_width=0.5, color='blue', 
                 length_includes_head=True, label='heading')

        plt.axis('equal')
        plt.grid(True)
        plt.title(f"Car trajectory (steps={len(xs)})")
        plt.legend()
        
        # 设置显示范围略大于圆形边界
        margin = self.R * 0.2
        plt.xlim(-self.R-margin, self.R+margin)
        plt.ylim(-self.R-margin, self.R+margin)
        
        if show:
            plt.show()

    def close(self):
        plt.close('all')


import numpy as np
import matplotlib.pyplot as plt

dt = 0.2
max_steps = int(2*pi*30/10/dt)

# 假设 CarHeadingEnv 已在当前命名空间定义为 CarHeadingEnv（如 notebook 前面单元格）
env = CarHeadingEnv(max_steps=max_steps, dt=dt, speed=10.0, max_omega=5.0, seed=0)
env.seed(0)
np.random.seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
state_dim = 2
action_dim = 1
action_bound = 1  # 动作最大值
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

returns = []

for ep in range(num_episodes):
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