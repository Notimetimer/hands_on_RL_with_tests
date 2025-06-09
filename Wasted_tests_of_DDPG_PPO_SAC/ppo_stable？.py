import random
import gym
import numpy as np
from tqdm import tqdm
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')  # 使用Qt5作为后端
from gym import spaces
from numpy.linalg import norm

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.prelu = nn.PReLU()
        layers = []
        prev_size = state_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev_size, h))
            layers.append(self.prelu)
            prev_size = h
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_size, 1)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)


# ## 连续环境
class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.prelu = nn.PReLU()
        layers = []
        prev_size = state_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev_size, h))
            layers.append(self.prelu)
            prev_size = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_size, action_dim)
        self.fc_log_std = nn.Linear(prev_size, action_dim)
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_log_std.weight, gain=0.01)

    def forward(self, x):
        x = self.net(x)
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(-5, 2)  # 限制 log_std 范围防止过小/大
        std = torch.exp(log_std)
        return mu, std

class PPOContinuous:
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

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action[0].cpu().detach().numpy().flatten()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 标准化奖励（可选）
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)

        # 标准化优势
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # 固定旧策略 log_prob
        with torch.no_grad():
            old_mu, old_std = self.actor(states)
            old_dist = torch.distributions.Normal(old_mu, old_std)
            old_log_probs = old_dist.log_prob(actions)

        for epoch in range(self.epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO 目标
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            # critic loss
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # 优化器清零
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            # 裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # ✅ 新增：KL 监控，防止爆炸
            approx_kl = (old_log_probs - log_probs).mean().item()
            if approx_kl > 0.02:
                print(f"[Early Stop] KL too high at epoch {epoch+1}: {approx_kl:.4f}")
                break
