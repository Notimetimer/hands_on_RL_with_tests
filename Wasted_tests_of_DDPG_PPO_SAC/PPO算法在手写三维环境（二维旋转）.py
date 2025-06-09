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
# import rl_utils
dt = 0.5
dof = 2

# 超参数
actor_lr = 1e-4  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 15  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
num_episodes = 1000 # 2000
hidden_dim = 64  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10 # 10
eps = 0.2

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
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
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ## 连续环境pendulum

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_bound * torch.tanh(self.fc_mu(x)) # 激活函数不一样了
        std = F.softplus(self.fc_std(x))
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, action_bound):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim, action_bound).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.detach().cpu().numpy().flatten()  # 支持一维和多维动作，而不是.item只支持1维或.squeeze只支持多维

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
        
        # 添加奖励缩放
        # rewards/=100
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        # # 添加优势函数缩放
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        mu, std = self.actor(states)

        # 添加NaN检查
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print("WARNING: NaN detected in mu or std!")
            print(f"mu: {mu}\nstd: {std}")
            raise ValueError("NaN in actor network outputs")

        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)

            # 添加循环内的NaN检查
            if torch.isnan(mu).any() or torch.isnan(std).any():
                print("WARNING: NaN detected in mu or std during training!")
                print(f"mu: {mu}\nstd: {std}")
                raise ValueError("NaN in actor network outputs during training")

            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)

            # 添加KL计算
            approx_kl = (old_log_probs - log_probs).mean()
            test=0.02
            if abs(approx_kl)>test:
                # print('approx_kl',approx_kl) # 这个好像绝对值大于1就会有问题
                ratio = torch.exp((log_probs - old_log_probs)/abs(approx_kl)*test)
                # print('ratio', ratio)  # 这个好像绝对值大于1就会有问题

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # print('minofsurr',torch.min(surr1, surr2))
            # print('surr2',surr2)
            # print('advantage',advantage)

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # # print('actor_loss',actor_loss)
            # test=0.1 # 不成功
            # actor_loss=test*torch.sign(actor_loss) if abs(actor_loss)>test else actor_loss

            # # 添加损失缩放
            # actor_loss = actor_loss * 0.5
            # critic_loss = critic_loss * 0.5
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

class testEnv(gym.Env):
    def __init__(self):
        super(testEnv, self).__init__()
        # 观测空间：相对于点的位置和速度
        low1 = np.ones(dof * 2)*-np.inf
        high1 = np.ones(dof * 2)*np.inf
        self.observation_space = spaces.Box(low=low1, high=high1, dtype=np.float32)
        # 动作空间：加速度和角速度
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
        speed = np.linalg.norm(vel_)
        theta = np.arctan2(vel_[1], vel_[0])
        speed += action[0]*dt
        theta += action[1]*np.pi/4*dt

        # # # 更新状态
        vel_ = speed*np.array([np.cos(theta), np.sin(theta)])
        pos_ += vel_ * dt

        self.state = np.hstack((pos_, vel_))
        self.t += dt

        # 定义奖励函数
        reward = -np.linalg.norm(pos_)  # 奖励与位置偏差的绝对值成反比
        reward_plus = 0
        reward_plus = -5*np.dot(vel_, pos_)/(np.linalg.norm(pos_)*np.linalg.norm(vel_)) if \
            np.linalg.norm(pos_)*np.linalg.norm(vel_)>0 else 0
        reward_plus -= -0.1 * np.linalg.norm(vel_)/np.linalg.norm(pos_) if np.linalg.norm(pos_)>0 else 0

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
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device, action_bound=action_bound)

# state_check=[] # 查看输入
## todo 打开这里的函数
# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
return_list = []
# global state_check
for i in range(10):  # 10
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar: # 进度条
        for i_episode in range(int(num_episodes/10)): # 每个1/10的训练轮次
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset(train=True)
            done = False
            while not done: # 每个训练回合
                # state_check=state
                # 1.执行动作得到环境反馈
                action = agent.take_action(state)
                next_state, reward, done, reward_plus = env.step(action) # pendulum中的action一定要是ndarray才能输入吗？
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward+reward_plus)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            episode_return=np.clip(episode_return,-1000,1000) # 不这样都没法看
            return_list.append(episode_return)
            agent.update(transition_dict)
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
plt.title('PPO on {}'.format(env_name))

mv_return = moving_average(return_list, 9)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))


trajectory = []

state = env.reset()
done = False
while not done:
    action = agent.take_action(state)
    next_state, _, done, _ = env.step(action)
    trajectory.append(state)
    state = next_state

dist = []
for i in range(len(trajectory)):
    dist.append(np.linalg.norm(trajectory[i][0:2]))
dist = np.array(dist)

# 绘制第三个图形
plt.figure()
plt.plot(range(len(trajectory)), dist, label='Car Trajectory')  # 轨迹
# plt.axhline(y=0, color='r', linestyle='--', label='Target Line')  # 目标直线
plt.xlabel('x')
plt.ylabel('y')
plt.title('Car Trajectory vs Target Line')
plt.legend()
#
# # 显示所有图形
plt.show()