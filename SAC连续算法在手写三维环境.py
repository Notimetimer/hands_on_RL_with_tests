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
from torch.distributions import Normal

# import rl_utils
dt = 0.5
dof = 3
actor_lr = 3e-4
critic_lr = actor_lr * 10  # 3e-3
num_episodes = 500
hidden_dim_1 = [128]

gamma = 0.99
alpha_lr = 3e-4
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 1000  # 1000
batch_size = 64
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

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


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

    def forward(self, x, explore=True):
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6
        if explore:
            dist = Normal(mu, std)
            normal_sample = dist.rsample()  # rsample()是重参数化采样
            log_prob = dist.log_prob(normal_sample)
            action = torch.tanh(normal_sample)
            # 计算tanh_normal分布的对数概率密度
            log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
            action = action * self.action_bound
            return action, log_prob
        else:
            action = torch.tanh(mu) * self.action_bound
            return action, None

        # # x = F.relu(self.fc1(x))
        # # mu = self.fc_mu(x)
        # # std = F.softplus(self.fc_std(x))
        # dist = Normal(mu, std)
        # normal_sample = dist.rsample()  # rsample()是重参数化采样
        # log_prob = dist.log_prob(normal_sample)
        # action = torch.tanh(normal_sample)
        # # 计算tanh_normal分布的对数概率密度
        # log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # action = action * self.action_bound
        # return action, log_prob


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

    # test 更改学习率
    def update_actor_lr(self, new_lr):
        """
        更新Actor网络优化器的学习率
        :param new_lr: 新的学习率
        """
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr

    def update_critic_lr(self, new_lr):
        """
        更新Critic网络优化器的学习率
        :param new_lr: 新的学习率
        """
        for param_group in self.critic_1_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_2_optimizer.param_groups:
            param_group['lr'] = new_lr

    def take_action(self, state, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, _ = self.actor(state, explore)
        return action.detach().numpy().flatten()

        # action, _ = self.actor(state)[0].detach().numpy().flatten()  # self.actor(state)[0]
        # return action  # [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states, explore=True)
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
        # actions = torch.tensor(transition_dict['actions'],
        #                        dtype=torch.float).view(-1, 1).to(self.device)
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
        new_actions, log_prob = self.actor(states, explore=True)
        entropy = -log_prob.sum(dim=1, keepdim=True)
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
        self.action_space = spaces.Box(low=-5, high=5, shape=(dof,), dtype=np.float32)
        self.state = None
        self.done = False
        self.target_vel_ = None
        self.target_pos_ = None
        self.t = None

    def reset(self, train=True):
        # 初始化状态
        self.t = 0
        self.target_pos_ = np.zeros(dof)
        self.target_vel_ = np.ones(dof)

        if train:
            pos_ = np.random.rand(dof) * 3
            vel_ = np.random.rand(dof) * 0.3
        else:
            pos_ = np.ones(dof)
            vel_ = np.ones(dof) * 0.1
        self.state = np.hstack((pos_, vel_))  # 初始位置
        self.done = False
        observe = self.state
        return observe

    def step(self, action):
        self.t += dt
        pos_ = self.state[0:dof]  # 从数组中提取向量
        vel_ = self.state[dof:]

        # # # 更新状态
        vel_ += action * dt
        pos_ += vel_ * dt
        self.target_pos_ += self.target_vel_ * dt

        self.state = np.hstack((pos_, vel_))
        observe = np.hstack((self.target_pos_ - pos_, self.target_vel_ - vel_))

        # 定义奖励函数
        reward = 2 * (5 - np.linalg.norm(observe[0:dof]))  # 奖励与位置偏差的绝对值成负正比

        # reward_plus = -reward**2 * 5
        # reward_plus -= np.linalg.norm(observe[dof:])**2

        # reward_plus = min(0, -np.linalg.norm(np.dot(vel_, pos_)/np.linalg.norm(pos_))) if np.linalg.norm(pos_) > 0 else -np.linalg.norm(vel_)

        if self.t > 20:
            self.done = True
        if np.linalg.norm(observe[0:dof]) > 10:
            reward -= 100
            self.done = True

        reward_plus = 0
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
# state_check=[] # 查看输入
## todo 打开这里的函数
# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
return_list = []
# global state_check
for i in range(10):  # 10
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  # 进度条
        for i_episode in range(int(num_episodes / 10)):  # 每个1/10的训练轮次
            episode_return = 0
            # state = env.reset(train=True)
            state = env.reset(train=False)
            done = False
            while not done:  # 每个训练回合
                # state_check=state
                # 1.执行动作得到环境反馈
                action = agent.take_action(state, explore=True)
                next_state, reward, done, reward_plus = env.step(action)  # pendulum中的action一定要是ndarray才能输入吗？
                # 2.运行记录添加回放池
                replay_buffer.add(state, action, reward + reward_plus, next_state, done)
                state = next_state
                episode_return += reward

                if episode_return > 200:
                    # print("episode_return",episode_return)
                    # print("distance",np.linalg.norm(next_state[0:dof]))
                    # print("time", env.t)
                    pass

                # 3.从回放池采样更新智能体
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)
            episode_return = np.clip(episode_return, -1000, 1000)  # 不这样都没法看
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
    # return return_list

# return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

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

episode_return = 0
state = env.reset(train=False)
done = False
while not done:  # 测试回合
    action = agent.take_action(state, explore=False)
    next_state, reward, done, reward_plus = env.step(action)
    car_trajectory.append(env.state[0:dof].copy())
    target_trajectory.append(env.target_pos_[0:dof].copy())
    state = next_state
    episode_return += reward

    if episode_return > 200:
        # print("episode_return",episode_return)
        # print("distance",np.linalg.norm(next_state[0:dof]))
        # print("time", env.t)
        pass

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
