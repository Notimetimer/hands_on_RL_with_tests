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
'''更改内容：增加了多层神经网络'''
matplotlib.use('Qt5Agg')  # 使用Qt5作为后端
from gym import spaces
# import rl_utils
dt = 0.5
dof = 3
actor_lr = 3e-4
critic_lr = actor_lr*10
num_episodes = 500
hidden_dim = [64]
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 1000
minimal_size = 100 #1000
batch_size = 64
sigma = 0.01 #  0.01  # 高斯噪声标准差

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
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

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.action_bound=action_bound # action_bound是环境可以接受的动作最大值
        self.prelu1 = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)
            # layers.append(nn.ReLU())
            prev_size = layer_size
        layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)

        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim[0])
        # self.fc2 = torch.nn.Linear(hidden_dim[0], hidden_dim[1])
        # self.fc3 = torch.nn.Linear(hidden_dim[1], action_dim)
        # self.prelu1 = torch.nn.PReLU()

    def forward(self, x):
        # print(self.net(x))
        return torch.tanh(self.net(x)) * self.action_bound
        # x1 = self.prelu1(self.fc1(x))
        # x2 = self.prelu1(self.fc2(x1))
        # x3 = torch.tanh(self.fc3(x2))
        # return x3 * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.prelu1 = torch.nn.PReLU()
        layers = []
        prev_size = state_dim+action_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)
            # layers.append(nn.ReLU())
            prev_size = layer_size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

        # self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim[0])
        # self.fc2 = torch.nn.Linear(hidden_dim[0], hidden_dim[1])
        # self.fc_out = torch.nn.Linear(hidden_dim[1], 1)
        # self.prelu1 = torch.nn.PReLU()

    def forward(self, x, a):
        cat=torch.cat([x,a],dim=1)
        return self.net(cat)

        # # print('a',a.shape)
        # # print('x',x.shape)
        # cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        # # print('cat',cat.shape)
        # x = self.prelu1(self.fc1(cat))
        # x = self.prelu1(self.fc2(x))
        # return self.fc_out(x)

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        # DDPG四个网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        # print(state)
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).detach().numpy().flatten()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        # 放入设备
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones) # 对每个元组，用目标网络计算y
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets)) # 自动计算目标的方均误差
        self.critic_optimizer.zero_grad()# 清除累积的梯度
        critic_loss.backward()
        self.critic_optimizer.step()# 通过指定的优化器更新当前Critic网络
 #         因为优化器默认采用梯度下降而非上升，而奖励函数的值是负的，且越负效果越差，所以需要乘以一个负数转换优化方向。
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
    # 为啥是负数？pendulum的奖励函数确实是负的，这个负数如果删除掉会训练出一个"正立摆"
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

class testEnv(gym.Env):
    def __init__(self):
        super(testEnv, self).__init__()
        # 观测空间：相对于点的位置和速度
        low1 = np.ones(dof * 2)*-np.inf
        high1 = np.ones(dof * 2)*np.inf
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
            vel_ = np.ones(dof)*0.1
        self.state = np.hstack((pos_, vel_))  # 初始位置
        self.done = False
        return self.state

    def step(self, action):
        pos_ = self.state[0:dof]  # 从数组中提取向量
        vel_ = self.state[dof:]

        # # # 更新状态
        vel_ += action * dt
        pos_ += vel_ * dt

        # vel_ = action
        # pos_ += vel_ * dt

        self.state = np.hstack((pos_, vel_))
        self.t += dt

        # 定义奖励函数
        reward = -np.linalg.norm(pos_)  # 奖励与位置偏差的绝对值成负正比
        reward_plus = 0
        # reward_plus = min(0, -np.linalg.norm(np.dot(vel_, pos_)/np.linalg.norm(pos_))) if np.linalg.norm(pos_) > 0 else -np.linalg.norm(vel_)

        if self.t > 8:
            self.done = True
        if max(pos_) > 10:
            reward -= -10
            self.done = True
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
replay_buffer = ReplayBuffer(buffer_size)  # 经验回放
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
# state_check=[] # 查看输入
## todo 打开这里的函数
# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
return_list = []
# global state_check
for i in range(10):  # 10
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar: # 进度条
        last_action = np.zeros(3)
        for i_episode in range(int(num_episodes/10)): # 每个1/10的训练轮次
            episode_return = 0
            state = env.reset(train=True)
            done = False
            while not done: # 每个训练回合
                # state_check=state
                # 1.执行动作得到环境反馈
                action = agent.take_action(state)

                next_state, reward, done, reward_plus = env.step(action) # pendulum中的action一定要是ndarray才能输入吗？
                # 2.运行记录添加回放池
                replay_buffer.add(state, action, reward+reward_plus, next_state, done)
                state = next_state
                episode_return += reward
                # 3.从回放池采样更新智能体
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)
            episode_return=np.clip(episode_return,-1000,1000) # 不这样都没法看
            return_list.append(episode_return)
            if (i_episode+1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
    # return return_list

# return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))

mv_return = moving_average(return_list, 9)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))

#
trajectory = []

state = env.reset()
done = False
last_action = np.zeros(3)
while not done:
    action = agent.take_action(state)

    next_state, _, done, _ = env.step(action)
    trajectory.append(state)
    state = next_state

distances = []
for state in trajectory:
    distance = np.linalg.norm(state[0:dof])
    distances.append(distance)

# 绘制第三个图形
plt.figure()
plt.plot(range(len(trajectory)), distances, label='Car Trajectory')  # 轨迹
plt.axhline(y=0, color='r', linestyle='--', label='Target Line')  # 目标直线
plt.xlabel('Frap')
plt.ylabel('Location')
plt.title('Car Trajectory vs Target Line')
plt.legend()
#
# # 显示所有图形
plt.show()