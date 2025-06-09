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

# import rl_utils
dt = 0.2
dof = 3
action_bound = 3

# 超参数
# actor_lr = 1e-3  # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
# critic_lr = actor_lr * 8  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
# 原始学习率：
# actor_lr = 1e-3
# critic_lr = actor_lr * 8 = 8e-3

# 新推荐学习率（更稳定）：
actor_lr = 1e-4
critic_lr = 5e-4

num_episodes = 2000  # 2000
hidden_dim = [64]  # 128
gamma = 0.9
lmbda = 0.9  # 0.95~0.99
epochs = 4  # 10
eps = 0.2  # 0.1~0.3


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
        # self.fc_std = torch.nn.Linear(prev_size, action_dim)

        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_log_std.weight, gain=0.01)
        # nn.init.xavier_normal_(self.fc_std.weight, gain=0.01)

    def forward(self, x):
        x = self.net(x)
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(-5, 2)  # 限制 log_std 范围防止过小/大
        std = torch.exp(log_std)
        # std = F.softplus(self.fc_std(x))  # + 1e-8
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
        action = torch.clamp(action, -action_bound, action_bound)  # 限幅动作
        return action[0].cpu().detach().numpy().flatten()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        rewards = torch.clamp(rewards, -10, 10)
        # 标准化奖励（可选）
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # 时序差分回报值
        td_delta = td_target - self.critic(states)  # 优势函数用时序差分回报与Critic网络输出作差表示
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 标准化优势
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = torch.clamp(advantage, -10, 10)

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
            critic_loss = 0.5 * F.mse_loss(self.critic(states), td_target.detach())

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

            # ✅ 新增：KL 监控，防止爆炸，正常的KL散度应该在0.1以下，超过0.3需立即干预
            approx_kl = (old_log_probs - log_probs).mean().item()
            if approx_kl > 0.02:
                print(f"[Early Stop] KL too high at epoch {epoch+1}: {approx_kl:.4f}")
                break


class testEnv(gym.Env):
    def __init__(self):
        super(testEnv, self).__init__()
        # 观测空间：相对于点的位置和速度
        low1 = np.ones(dof * 2) * -np.inf
        high1 = np.ones(dof * 2) * np.inf
        self.observation_space = spaces.Box(low=low1, high=high1, dtype=np.float32)
        # 动作空间：三轴加速度
        self.action_space = spaces.Box(low=-1, high=1, shape=(dof,), dtype=np.float32)
        self.state = None
        self.done = False

    def reset(self, train=True):
        # 初始化状态
        self.t = 0
        if train:
            pos_ = np.random.rand(dof) * 3
            vel_ = np.random.rand(dof) * 0.3
        else:
            pos_ = np.ones(dof)
            vel_ = np.ones(dof) * 0.1
        self.state = np.hstack((pos_, vel_))  # 初始位置
        self.done = False
        return self.state

    def step(self, action):
        pos_ = self.state[0:dof]  # 从数组中提取向量
        vel_ = self.state[dof:]

        # # # 更新状态
        action = np.clip(action, -3, 3)  # 限制动作范围
        vel_ += action * dt
        pos_ += vel_ * dt

        # vel_ = action
        # pos_ += vel_ * dt

        self.state = np.hstack((pos_, vel_))
        self.t += dt

        # 定义奖励函数
        reward = -np.linalg.norm(pos_)  # 奖励与位置偏差的绝对值成反比
        # reward_plus = min(0, -np.linalg.norm(np.dot(vel_, pos_)/np.linalg.norm(pos_))) if np.linalg.norm(pos_) > 0 else -np.linalg.norm(vel_)

        reward_plus = 10  # 存活奖励
        if self.t > 8:
            self.done = True
        if norm(pos_) > 10:
            reward += -100
            self.done = True  # 惩罚过大的位置但不结束运行
        return self.state, reward, self.done, reward_plus

    def render(self, mode='human'):
        # 可视化小车的位置和方向
        pass


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'testEnv'
env = testEnv()
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
# action_bound = env.action_space.high[0]  # 动作最大值
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

# state_check=[] # 查看输入
## todo 打开这里的函数
# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
return_list = []
# global state_check
# for i in range(1):  # 10
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
    for i_episode in range(int(num_episodes)):  # 每个1/10的训练轮次
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset(train=True)
        done = False
        while not done:  # 每个训练回合
            # state_check=state
            # 1.执行动作得到环境反馈
            action = agent.take_action(state)
            next_state, reward, done, reward_plus = env.step(action)  # pendulum中的action一定要是ndarray才能输入吗？
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward + reward_plus)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        episode_return = np.clip(episode_return, -1000, 1000)  # 不这样都没法看
        return_list.append(episode_return)
        agent.update(transition_dict)
        if (i_episode + 1) >= 10:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)
    # return return_list

# return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))

mv_return = moving_average(return_list, 9)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))

# trajectory = []
#
# state = env.reset()
# done = False
# while not done:
#     action = agent.take_action(state)
#     next_state, _, done, _ = env.step(action)
#     trajectory.append(state)
#     state = next_state


# # 绘制第三个图形
# plt.figure()
# plt.plot(range(len(trajectory)), trajectory, label='Car Trajectory')  # 轨迹
# plt.axhline(y=0, color='r', linestyle='--', label='Target Line')  # 目标直线
# plt.xlabel('Frap')
# plt.ylabel('Location')
# plt.title('Car Trajectory vs Target Line')
# plt.legend()
#
# # 显示所有图形
plt.show()
