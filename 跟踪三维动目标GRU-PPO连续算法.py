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
num_layers = 1
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
    def __init__(self, state_dim, hidden_dim, gru_hidden_size=64):
        super(ValueNet, self).__init__()
        self.gru = nn.GRU(state_dim, gru_hidden_size, num_layers=num_layers, batch_first=True)  # 添加GRU层
        self.gru_hidden_size = gru_hidden_size
        # self.hidden = None  # 在类中保存隐藏状态

        layers = []
        prev_size = gru_hidden_size
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_size, 1)

    def forward(self, x, hidden=None):
        # 假设输入x的形状为(batch_size, seq_len, state_dim)
        x, hidden = self.gru(x, hidden)  # output, h_n = gru(input, h_0)
        # self.hidden = hidden.detach()  # 保存并截断梯度
        # 取最后一个时间步的输出作为状态价值
        x = x[:, -1, :]  # (batch_size, seq_len, input_size)
        y = self.net(x)
        return self.fc_out(y), hidden


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gru_hidden_size=64):
        super(PolicyNetContinuous, self).__init__()
        self.hidden = None
        self.gru = nn.GRU(state_dim, gru_hidden_size, num_layers=num_layers, batch_first=True)  # 添加GRU层
        self.gru_hidden_size = gru_hidden_size

        layers = []
        prev_size = gru_hidden_size
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_size, action_dim)
        self.fc_std = nn.Linear(prev_size, action_dim)

    def forward(self, x, action_bound=2.0, hidden=None):
        # 假设输入x的形状为(batch_size, seq_len, state_dim)
        x, hidden = self.gru(x, hidden)  # output, h_n = gru(input, h_0)
        # self.hidden = hidden.detach()  # 保存并截断梯度
        x = x[:, -1, :]  # 取最后一个时间步的输出 (batch_size, seq_len, input_size)
        x = self.net(x)
        mu = action_bound * torch.tanh(self.fc_mu(x))
        std = action_bound * F.softplus(self.fc_std(x))  # + 1e-8
        return mu, std, hidden


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, gru_hidden_size):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, gru_hidden_size).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, gru_hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.actor_h = None
        self.critic_h = None

    def take_action(self, state, action_bound=2.0, explore=True):
        # 将当前状态转换为 GRU 所需的时间序列格式
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)  # (1, 1, state_dim)

        mu, sigma, self.actor_h = self.actor(state, action_bound=action_bound, hidden=self.actor_h)
        # self.hidden = self.actor.hidden  # 更新隐藏状态

        if not explore:
            action = mu
            return action[0].cpu().detach().numpy().flatten()

        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action[0].cpu().detach().numpy().flatten()

    #     # 添加Actor NaN检查
    #     if torch.isnan(mu).any() or torch.isnan(std).any():
    #         print("WARNING: NaN detected in mu or std!")
    #         # print(f"mu: {mu}\nstd: {std}")
    #         raise ValueError("NaN in actor network outputs")
    #     # 添加Critic NaN检查
    #     if torch.isnan(critic_values).any():
    #         print("WARNING: NaN detected in critic values!")
    #         # print(f"critic_values: {critic_values}")
    #         raise ValueError("NaN in critic network outputs")

    #         # # 添加KL检查
    #         # approx_kl = (old_log_probs - log_probs).mean()
    #         # test = 0.02
    #         # if abs(approx_kl) > test:
    #         #     # print('approx_kl',approx_kl) # 这个好像绝对值大于1就会有问题
    #         #     ratio = torch.exp((log_probs - old_log_probs) / abs(approx_kl) * test)
    #         #     # print('ratio', ratio)  # 这个好像绝对值大于1就会有问题


    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        states = states.unsqueeze(1)  # 添加序列维度: (batch_size, 1, state_dim)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        next_states = next_states.unsqueeze(1)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        actor_hs=None
        critic_hs=None
        # print(states.shape)
        # print(type(transition_dict['actor_hs']))

        actor_hs = torch.cat(transition_dict['actor_hs'], dim=1).to(device)
        # # 或者下面这样
        # actor_hs = torch.cat(transition_dict['actor_hs'], dim=0)  # shape: [8, 1, 64]
        # # 然后交换维度
        # actor_hs = actor_hs_tensor.transpose(0, 1)  # shape: [1, 8, 64]
        # # 或者这样调整维度顺序
        # actor_hs = actor_hs.permute(1, 0, 2)  # shape: [1, 8, 64]

        # critic_hs = torch.cat(transition_dict['critic_hs'], dim=0).to(device)
        critic_hs = None

        next_critic, _ = self.critic(next_states, critic_hs)
        td_target = rewards + self.gamma * next_critic * (1 - dones)
        current_critic, _ = self.critic(states, critic_hs)
        td_delta = td_target - current_critic
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # print('state shape', states.shape)
        # print('action shape', actions.shape)
        # print('rewards shape', rewards.shape)
        # print('next_states', next_states.shape)
        # print('actor hidden shape', actor_hs.shape)
        # print('critic hidden shape', critic_hs.shape)

        mu, std, _ = self.actor(states, action_bound=action_bound, hidden=actor_hs)

        # NaN检查略去...

        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std, _ = self.actor(states, action_bound=action_bound, hidden=actor_hs)
            critic_values, _ = self.critic(states, hidden=critic_hs)

            # NaN检查略去...

            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_values, _ = self.critic(states, hidden=critic_hs)

            critic_loss = torch.mean(
                F.mse_loss(critic_values, td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            with torch.autograd.detect_anomaly():
                actor_loss.backward()
                critic_loss.backward()

            # 梯度裁剪（可选）
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)

            self.actor_optimizer.step()
            self.critic_optimizer.step()


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
gru_hidden_size = 64
hidden_dim = [128]  # 全连接层维度

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device,
                      gru_hidden_size=gru_hidden_size)

# state_check=[] # 查看输入
## todo 打开这里的函数
# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
return_list = []
# global state_check
# for i in range(1):  # 10
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
    for i_episode in range(int(num_episodes)):  # 每个1/10的训练轮次
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [],
                           'dones': [], 'actor_hs': [], 'critic_hs': []}
        state = env.reset(train=True)
        agent.hidden = None  # 每个 episode 开始前重置隐藏状态
        done = False
        while not done:  # 每个训练回合
            # state_check=state
            # 1.执行动作得到环境反馈
            action = agent.take_action(state, action_bound=action_bound, explore=True)
            next_state, reward, done, reward_plus = env.step(action)  # pendulum中的action一定要是ndarray才能输入吗？
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward + reward_plus)
            transition_dict['dones'].append(done)
            # print(type(agent.actor_h))
            # print(agent.actor_h.shape)
            transition_dict['actor_hs'].append(agent.actor_h.detach().clone())
            # transition_dict['critic_hs'].append(agent.critic_h.detach().clone())
            state = next_state
            episode_return += reward
        # episode_return = np.clip(episode_return, -1000, 1000)  # 不这样都没法看
        return_list.append(episode_return)
        agent.update(transition_dict)
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
agent.hidden = None  # 每个 episode 开始前重置隐藏状态
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
