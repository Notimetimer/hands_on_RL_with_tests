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
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import copy

matplotlib.use('Qt5Agg')  # 使用Qt5作为后端
from gym import spaces
from numpy.linalg import norm
from torch.distributions import Normal

# import rl_utils
dt = 0.5
dof = 3

# 超参数
actor_lr = 1e-3 / 10  # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 10  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
num_episodes = 400  # 2000
hidden_dim = [128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
action_std_init = 0.6


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
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)  # todo 补充多维输出

        # # 添加参数初始化
        # for layer in self.net:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_normal_(layer.weight, gain=0.01)
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        # self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        self.fc_std = torch.nn.Linear(prev_size, action_dim)
        # # 固定神经网络初始化参数
        # torch.nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        # torch.nn.init.xavier_normal_(self.fc_std.weight, gain=0.01)

    def forward(self, x, action_bound=2.0):
        x = self.net(x)
        mu = action_bound * torch.tanh(self.fc_mu(x))
        std = action_bound * F.softplus(self.fc_std(x))  # + 1e-8
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.action_mu = None
        self.action_std = None
        self.action_dist = None
        # 添加一个参数版本计数器
        self.network_version = 0
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        
    def take_action(self, state, action_bound=2.0, explore=True):
        # 记录当前网络版本
        # print(f"Network version in take_action: {self.network_version}")

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        with torch.no_grad():
            mu, std = self.actor(state, action_bound=action_bound)
        self.action_mu = mu.detach()
        self.action_std = std.detach()
        if not explore:
            action = mu
            return action[0].cpu().detach().numpy().flatten()  # 支持一维和多维动作，而不是.item只支持1维或.squeeze只支持多维
        action_dist = torch.distributions.Normal(mu, std)
        self.action_dist = action_dist
        action = action_dist.sample()
        return action[0].cpu().detach().numpy().flatten(), [mu, std]  # 支持一维和多维动作，而不是.item只支持1维或.squeeze只支持多维
        # return [action.item()]

    def update(self, replay_buffer):

        states = torch.tensor(replay_buffer['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(replay_buffer['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(replay_buffer['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(replay_buffer['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        old_log_probs1 = torch.cat(replay_buffer['old_log_probs'], dim=0).to(self.device)
        values = torch.tensor(replay_buffer['values'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        next_values = torch.tensor(replay_buffer['next_values'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # mu_recorded = torch.cat(replay_buffer['mu'], dim=0).to(self.device)
        # std_recorded = torch.cat(replay_buffer['std'], dim=0).to(self.device)

        td_target = rewards + self.gamma * next_values * (1 - dones)  # 时序差分回报值
        td_delta = td_target - values  # 优势函数用时序差分回报与Critic网络输出作差表示
        advantage1 = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        advantage = advantage1
        old_log_probs = old_log_probs1

        for _ in range(self.epochs):
            mu, std = self.actor(states, action_bound=action_bound)
            critic_values = self.critic(states)

            # 添加Actor NaN检查
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in actor network outputs")
            # 添加Critic NaN检查
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in critic network outputs")

            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)

            # print(ratio)
            dist_entropy = action_dists.entropy()  # 获取熵

            # # 添加KL检查
            # approx_kl = (old_log_probs - log_probs).mean()
            # test = 0.02
            # if abs(approx_kl) > test:
            #     # print('approx_kl',approx_kl) # 这个好像绝对值大于1就会有问题
            #     ratio = torch.exp((log_probs - old_log_probs) / abs(approx_kl) * test)
            #     # print('ratio', ratio)  # 这个好像绝对值大于1就会有问题
            # # test advantage归一化，可能不是好做法
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            # actor_loss = torch.mean(-torch.min(surr1, surr2))  # original
            actor_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)  # test
            actor_loss = -actor_loss.mean()
            # critic_loss = torch.mean(
            #     F.mse_loss(self.critic(states), td_target.detach())) # original
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())  # 简化
            loss = actor_loss + 1 * critic_loss - 0.01 * dist_entropy  # 添加一个熵正则项

            self.optimizer.zero_grad()
            loss.mean().backward()
            # test 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.optimizer.step()

            # 标记网络更新
            # print(f"Network version before update: {self.network_version}")
            self.network_version += 1  # 更新版本号
            # print(f"Network version after update: {self.network_version}")


from tracking_test import testEnv

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'testEnv'
env = testEnv(dof=dof, dt=dt)
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
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
        replay_buffer = \
            {
                'states': [],
                'actions': [],
                'old_log_probs': [],
                'values': [],
                'rewards': [],
                'dones': [],
                'next_values': [],
                # 'next_states': [],
                # 'mu': [],
                # 'std': []
             }
        state = env.reset(train=True)
        done = False
        while not done:  # 每个训练回合
            # state_check=state
            # 1.执行动作得到环境反馈
            action, mu_std = agent.take_action(state, action_bound=action_bound, explore=True)
            next_state, reward, done, reward_plus = env.step(action)

            replay_buffer['states'].append(state)
            replay_buffer['actions'].append(action)

            action_dist = torch.distributions.Normal(copy.deepcopy(agent.action_mu), copy.deepcopy(agent.action_std))
            old_log_prob = action_dist.log_prob(torch.tensor(action, dtype=torch.float).to(device))
            replay_buffer['old_log_probs'].append(copy.deepcopy(old_log_prob.detach()))

            value = agent.critic(torch.tensor(state, dtype=torch.float).to(device))
            replay_buffer['values'].append(value)

            next_value = agent.critic(torch.tensor(next_state, dtype=torch.float).to(device))
            replay_buffer['next_values'].append(next_value)

            replay_buffer['rewards'].append(reward + reward_plus)
            replay_buffer['dones'].append(done)

            # replay_buffer['mu'].append(mu_std[0])
            # replay_buffer['std'].append(mu_std[1])

            state = next_state
            episode_return += reward
        # episode_return = np.clip(episode_return, -1000, 1000)  # 不这样都没法看
        return_list.append(episode_return)

        agent.update(replay_buffer)
        if (i_episode + 1) >= 10:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                              'return': '%.3f' % np.mean(return_list[-10:])})
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

car_trajectory = []
target_trajectory = []

episode_return = 0
state = env.reset(train=False)
done = False
while not done:  # 测试回合
    action = agent.take_action(state, action_bound=action_bound, explore=False)
    next_state, reward, done, reward_plus = env.step(action)
    car_trajectory.append(env.state[0:dof].copy())
    target_trajectory.append(env.target_pos_[0:dof].copy())
    state = next_state
    episode_return += reward

# 新增代码：绘制每个坐标分量的轨迹和目标值
plt.figure(4)
for i in range(dof):
    plt.subplot(dof, 1, i + 1)
    # 提取每个坐标分量的轨迹
    pos_trajectory = [state[i] for state in car_trajectory]
    # 假设 target_pos_ 是一个数组，每个元素对应一个时间步的目标位置
    target_pos_trajectory = [state[i] for state in target_trajectory]
    plt.plot(range(len(pos_trajectory)), pos_trajectory, 'b-', label='Position')
    plt.plot(range(len(target_pos_trajectory)), target_pos_trajectory, 'r--', label='Target Position')
    plt.xlabel('Step')
    plt.ylabel(f'Coordinate {i + 1}')
    plt.legend()

# # 显示所有图形
plt.show()
