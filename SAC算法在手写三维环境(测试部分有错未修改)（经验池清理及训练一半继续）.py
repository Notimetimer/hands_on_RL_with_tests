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
import os
import pickle

matplotlib.use('Qt5Agg')  # 使用Qt5作为后端
from gym import spaces
from torch.distributions import Normal

# import rl_utils
dt = 0.5
dof = 3
actor_lr = 3e-4  # 3e-4
critic_lr = actor_lr * 10  # 3e-3
num_episodes = 400  # 500
hidden_dim_1 = [32, 32]

gamma = 0.7  # 0.99
alpha_lr = 3e-4
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 1000  # 1000
batch_size = 100  # 64
sigma = 0.01  # 0.01  # 高斯噪声标准差


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

    def clear_half_old(self):
        # 将 deque 转换为 list 后再进行切片
        if len(self.buffer) == self.capacity:
            buffer_list = list(self.buffer)
            new_size = len(buffer_list) // 2
            self.buffer = collections.deque(buffer_list[-new_size:], maxlen=self.capacity)

    def clear_all(self):
        self.buffer = collections.deque(maxlen=self.capacity)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, file_path):
        print('load replay buffer')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = collections.deque(data, maxlen=self.capacity)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.action_bound = action_bound
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)
            # layers.append(nn.ReLU())
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        self.fc_std = torch.nn.Linear(prev_size, action_dim)
        # torch.nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        # torch.nn.init.xavier_normal_(self.fc_std.weight, gain=0.01)
        nn.init.kaiming_uniform_(self.fc_mu.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc_std.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6

        # x = F.relu(self.fc1(x))
        # mu = self.fc_mu(x)
        # std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.action_bound = action_bound
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim + action_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)
            # layers.append(nn.ReLU())
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

        # self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        y = self.net(cat)
        return self.fc_out(y)

        # x = F.relu(self.fc1(cat))
        # x = F.relu(self.fc2(x))
        # return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0].detach().numpy().flatten()  # self.actor(state)[0]
        return action  # [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        # entropy = -log_prob # 这里不对，应该把三维变成一维的
        entropy = -log_prob.sum(dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        if td_target.shape[1] > 1:
            print('error')
            raise ValueError
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        # rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)

        # # test 确保td_target的尺寸和Q网络输出的尺寸一致
        # td_target = td_target.view(-1, 1)  # test 添加这一行

        # print('states:',states.shape)
        # print('actions:',actions.shape)
        # print('td_target',td_target.shape)
        # print('self.critic_1(states, actions)',self.critic_1(states, actions).shape)
        # print('self.critic_2(states, actions)', self.critic_2(states, actions).shape)

        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


# 更改目标：跟踪动目标
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
        self.target_vel_ = None
        self.target_pos_ = None
        self.t = None

    def reset(self):
        # 初始化状态
        self.t = 0
        self.target_pos_ = np.zeros(dof)
        self.target_vel_ = np.ones(dof)

        pos_ = np.random.rand(dof) * 3
        vel_ = np.random.rand(dof) * 1
        self.vel0_=vel_
        self.A = 1

        self.state = np.hstack((pos_, vel_))  # 初始位置
        self.done = False
        observe = self.state
        return observe

    def step(self, action):
        self.t += dt
        # 目标运动
        self.target_vel_ = self.vel0_  # + self.A * np.sin(0.5*1/np.pi*self.t)
        self.target_pos_ += self.target_vel_ * dt
        # 小车运动
        pos_ = self.state[0:dof]  # 从数组中提取向量
        # 更新状态
        vel_ = action * 3
        pos_ += vel_ * dt

        self.state = np.hstack((pos_, vel_))
        observe = np.hstack((self.target_pos_ - pos_, self.target_vel_ - vel_))

        # 定义奖励函数
        reward = -np.linalg.norm(observe[0:dof])  # 奖励与位置偏差的绝对值成负正比
        # reward_plus = -np.linalg.norm(observe[0:dof]) ** 2 * 10
        # if np.linalg.norm(observe[dof:])<1e-2:
        #     reward_plus-=np.linalg.norm(observe[0:dof]) ** 2 * 100

        # 如果和目标的相对速度足够小并且接下来的距离比当前更短，那么增加奖励
        L_ = self.target_pos_ - pos_
        current_distance = np.linalg.norm(L_)
        vr_ = self.target_vel_ - vel_
        q_dot = np.dot(vr_, L_) / current_distance if current_distance > 0 else 0
        reward_plus = -10 * q_dot

        if self.t > 20:
            self.done = True
        if np.linalg.norm(observe[0:dof]) > 10:
            reward -= 100
            self.done = True
        return observe, reward, self.done, reward_plus

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
replay_buffer = ReplayBuffer(buffer_size)  # 经验回放
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
target_entropy = -env.action_space.shape[0]

agent = SACContinuous(state_dim, hidden_dim_1, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)

# # （可选）加载上一次训练的模型参数
# # 定义保存文件的名称
# actor_path = 'sac_actor01.pth'
# critic1_path = 'sac_critic101.pth'
# critic2_path = 'sac_critic201.pth'
# target_critic1_path = 'sac_target_critic101.pth'
# target_critic2_path = 'sac_target_critic201.pth'
# replay_buffer = ReplayBuffer(buffer_size)  # 经验回放
# replay_buffer_path = 'replay_buffer01.pkl'
# # 尝试加载经验回放池数据
# if os.path.exists(replay_buffer_path):
#     replay_buffer.load(replay_buffer_path)
#
# # 检查文件是否存在并加载参数
# if os.path.exists(actor_path):
#     agent.actor.load_state_dict(torch.load(actor_path))
#     print('load up actor')
# if os.path.exists(critic1_path):
#     agent.critic_1.load_state_dict(torch.load(critic1_path))
#     print('load up critic1')
# if os.path.exists(critic2_path):
#     agent.critic_2.load_state_dict(torch.load(critic2_path))
#     print('load up critic2')
# if os.path.exists(target_critic1_path):
#     agent.target_critic_1.load_state_dict(torch.load(target_critic1_path))
#     print('load up target_critic1')
# if os.path.exists(target_critic2_path):
#     agent.target_critic_2.load_state_dict(torch.load(target_critic2_path))
#     print('load up target_critic2')

# state_check=[] # 查看输入
return_list = []
test_trigger = -100

# global state_check
for i in range(10):  # 10
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  # 进度条
        for i_episode in range(int(num_episodes / 10)):  # 每个1/10的训练轮次
            episode_return = 0
            state = env.reset()
            done = False
            while not done:  # 每个训练回合
                # state_check=state
                # 1.执行动作得到环境反馈
                action = agent.take_action(state)
                next_state, reward, done, reward_plus = env.step(action)  # pendulum中的action一定要是ndarray才能输入吗？
                # 2.运行记录添加回放池
                replay_buffer.add(state, action, reward + reward_plus, next_state, done)
                state = next_state
                episode_return += reward
                # 3.从回放池采样更新智能体
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)
            episode_return = np.clip(episode_return, -1000, 1000)  # 不这样都没法看
            return_list.append(episode_return)

            # # 抛弃一半旧经验
            # if episode_return > test_trigger:
            #     replay_buffer.clear_half_old()
            #     test_trigger /= 2

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))

mv_return = moving_average(return_list, 9)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))

car_trajectory = []
target_trajectory = []

state = env.reset()
done = False
while not done:
    action = agent.take_action(state)
    _, _, done, _ = env.step(action)
    car_trajectory.append(env.state[0:dof].copy())
    target_trajectory.append(env.target_pos_[0:dof].copy())
    state = next_state

# # 绘制第三个图形
# distances = []
# for state in trajectory:
#     distance = np.linalg.norm(state[0:dof])
#     distances.append(distance)
# plt.figure()
# plt.plot(range(len(trajectory)), distances, label='Car Trajectory')  # 轨迹
# plt.axhline(y=0, color='r', linestyle='--', label='Target Line')  # 目标直线
# plt.xlabel('Frap')
# plt.ylabel('Location')
# plt.title('Car Trajectory vs Target Line')
# plt.legend()

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

# # 保存网络参数
# torch.save(agent.actor.state_dict(), actor_path)
# torch.save(agent.critic_1.state_dict(), critic1_path)
# torch.save(agent.critic_2.state_dict(), critic2_path)
# torch.save(agent.target_critic_1.state_dict(), target_critic1_path)
# torch.save(agent.target_critic_2.state_dict(), target_critic2_path)
# # 保存经验回放池数据
# replay_buffer.save(replay_buffer_path)
