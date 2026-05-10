import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from gym import spaces

# 修改后端为 Tkinter
matplotlib.use('TkAgg') 

class OrbitEnv(gym.Env):
    def __init__(self, dt=0.1, agent_speed=2.0, min_dist=1.0):
        super(OrbitEnv, self).__init__()
        self.dt = dt
        self.v = agent_speed
        self.min_dist = min_dist
        
        # --- 难度提升修改点 ---
        # 1. 期望距离变化率提升至智能体速度的 80% (4/5)
        self.max_radius_change = 0.8 * self.v * self.dt 
        
        # 2. 引入物理限制：最大角速度 (单位: rad/s)
        # 假设智能体每秒最多转向 90 度
        self.max_omega = np.pi / 2  
        # ---------------------

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reset()

    def reset(self, train=True):
        self.t = 0
        # 初始位置在 (5, 0)
        self.pos = np.array([5.0, 0.0])
        # 初始航向：垂直向上
        self.heading = np.pi / 2
        # 初始期望距离
        self.target_r = 5.0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        dist = np.linalg.norm(self.pos)
        dist_error = dist - self.target_r
        
        # 计算 目标方向角 (指向中心点 [0,0])
        target_angle = np.arctan2(-self.pos[1], -self.pos[0])
        
        # 相对角 phi = 目标方位角 - 当前航向角
        phi = target_angle - self.heading
        return np.array([dist_error, np.cos(phi), np.sin(phi)], dtype=np.float32)

    def step(self, action):
        self.t += self.dt
        
        # 1. 目标距离剧烈随机游走 (速度提升至 0.8v)
        noise = np.sin(0.1 * self.t) * self.max_radius_change
        # noise = (np.random.rand()) * self.max_radius_change
        # noise = (np.random.rand() * 2 - 1) * self.max_radius_change
        self.target_r = max(self.min_dist + 0.5, self.target_r + noise)

        # 2. 解析动作并实施角速度限制
        act_norm = np.linalg.norm(action) + 1e-8
        cos_a, sin_a = action / act_norm
        
        # 计算期望的目标方位角 (相对于中心)
        beta = np.arctan2(-self.pos[1], -self.pos[0])
        # 计算动作要求的“期望航向”
        alpha = np.arctan2(sin_a, cos_a)
        desired_heading = beta + alpha 
        
        # --- 核心修改：角速度约束 ---
        # 计算当前航向与期望航向的偏差，并映射到 [-pi, pi]
        delta_h = (desired_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
        
        # 限制单步最大转角
        max_delta_h = self.max_omega * self.dt
        actual_delta_h = delta_h # np.clip(delta_h, -max_delta_h, max_delta_h)
        
        # 更新实际航向
        self.heading += actual_delta_h
        # --------------------------
        
        # 3. 更新位置 (恒定速率)
        self.pos[0] += self.v * np.cos(self.heading) * self.dt
        self.pos[1] += self.v * np.sin(self.heading) * self.dt
        
        # 4. 计算奖励 (增加对极端情况的惩罚)
        dist = np.linalg.norm(self.pos)
        dist_error = abs(dist - self.target_r)
        
        # 基础奖励：误差越小奖励越高
        reward = 1.0 - (dist_error / 5.0) 
        
        # 失败判定：进入最小值范围或飞得太远
        if dist < self.min_dist or dist > 15.0:
            reward -= 20.0 # 提高惩罚权值
            self.done = True
        
        if self.t > 20: 
            self.done = True

        return self._get_obs(), reward, self.done, 0

# ==========================================
# 2. 算法工具与网络定义
# ==========================================
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
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

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

    def forward(self, x, action_bound=1.0):
        x = self.net(x)
        mu = action_bound * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 1e-5
        return mu, std

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lmbda, self.epochs, self.eps, self.device = gamma, lmbda, epochs, eps, device

    def take_action(self, state, action_bound=1.0, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state, action_bound=action_bound)
        if not explore:
            return mu[0].cpu().detach().numpy().flatten()
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return torch.clamp(action, -action_bound, action_bound)[0].cpu().detach().numpy().flatten()

    def update(self, transition_dict, action_bound=1.0):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        mu, std = self.actor(states, action_bound=action_bound)
        old_log_probs = torch.distributions.Normal(mu.detach(), std.detach()).log_prob(actions).sum(dim=-1, keepdim=True)

        for _ in range(self.epochs):
            mu, std = self.actor(states, action_bound=action_bound)
            log_probs = torch.distributions.Normal(mu, std).log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

# ==========================================
# 3. 主训练流程与可视化
# ==========================================
if __name__ == "__main__":
    actor_lr, critic_lr = 1e-4, 1e-3
    num_episodes = 1000
    hidden_dim = [128]
    gamma, lmbda, epochs, eps = 0.9, 0.9, 10, 0.2
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = OrbitEnv(dt=0.1)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = 1.0
    
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                          lmbda, epochs, eps, gamma, device)

    return_list = []
    with tqdm(total=int(num_episodes), desc='Training') as pbar:
        for i_episode in range(int(num_episodes)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state, action_bound=action_bound, explore=True)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict, action_bound=action_bound)
            pbar.update(1)

    # 可视化测试
    agent_traj, target_traj = [], []
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state, action_bound=action_bound, explore=False)
        next_state, reward, done, _ = env.step(action)
        agent_traj.append(env.pos.copy())
        unit_vec = env.pos / (np.linalg.norm(env.pos) + 1e-8)
        target_traj.append(unit_vec * env.target_r)
        state = next_state

    agent_traj, target_traj = np.array(agent_traj), np.array(target_traj)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(return_list)
    plt.title("Learning Curve")
    
    plt.subplot(1, 2, 2)
    plt.plot(agent_traj[:, 0], agent_traj[:, 1], 'b-', label='Agent')
    plt.plot(target_traj[:, 0], target_traj[:, 1], 'r--', label='Target Radius')
    plt.scatter([0], [0], c='k', marker='x', label='Center')
    plt.axis('equal')
    plt.legend()
    plt.title("Trajectory")
    plt.show()