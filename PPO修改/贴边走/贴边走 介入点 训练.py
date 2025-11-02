# 游标训练环境
import random
import matplotlib
import matplotlib.pyplot as plt
import gym
from gym import spaces
from numpy.linalg import norm
# --- 修改开始 ---
from torch.distributions import Categorical # 使用离散分布
# --- 修改结束 ---
import random
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F

# matplotlib.use('Qt5Agg')

# --- 修改开始 ---
# 将Env类完全重构以适应新问题
class SupervisoryEnv:
    def __init__(self, default_move_std=1.0):
        self.min_pos = -10
        self.max_pos = 10
        self.position = None
        self.out_range = None
        self.steps = 0
        self.default_move_std = default_move_std
        # 智能体需要观察的，除了位置，还有小车“打算”怎么走
        self.intended_displacement = np.array([0.0], dtype='float64')

    def reset(self):
        self.position = np.array([random.uniform(self.min_pos/2, self.max_pos/2)], dtype='float64')
        self.steps = 0
        self.out_range = 0
        # 在reset时，生成第一次的意图位移
        self.intended_displacement = np.random.normal(0, self.default_move_std, 1)
        return self.get_obs()

    def get_obs(self):
        # 状态是2维：[当前位置, 意图位移]
        return np.concatenate([self.position, self.intended_displacement]).copy()

    def step(self, intervention_action): # 动作是离散的 {0:不介入, 1:介入}
        # 1. 根据智能体的动作，决定实际的位移
        if intervention_action == 1: # 介入
            actual_move = np.array([0.0])
        else: # 不介入
            actual_move = self.intended_displacement
        
        # 2. 更新小车位置
        self.position += actual_move
        self.steps += 1

        # 3. 检查是否完成
        done = self.get_done()

        # 4. 计算奖励
        reward = self.get_reward(intervention_action)

        # 5. 生成下一步的意图位移，供下一次观察
        self.intended_displacement = np.random.normal(0, self.default_move_std, 1)

        # 成本在这里不再需要
        cost = 0 
        
        return self.get_obs(), reward, done, cost

    def get_done(self):
        self.out_range = 0
        done = 0
        if self.position[0] < self.min_pos or self.position[0] > self.max_pos:
            done = 1
            self.out_range = 1
        if self.steps >= 50: # 可以适当延长episode长度
            done = 1
        return done

    def get_reward(self, intervention_action):
        reward = 0
        
        # 规则1: 出界永远是最大的惩罚
        if self.out_range:
            return -50

        trigger_zone_width = 4.0 # 离边界4个单位内算危险区
        
        if intervention_action == 1: # 如果决定介入
            dist_to_boundary = min(self.max_pos - self.position[0], self.position[0] - self.min_pos)
            
            # 规则2: 在危险区内介入是好的，离边界越近越好
            if dist_to_boundary <= trigger_zone_width:
                # 奖励与靠近边界的程度成正比，范围 [0, 5]
                reward = 5.0 * (1 - dist_to_boundary / trigger_zone_width)
            # 规则3: 在安全区内介入是坏的（胆小鬼）
            else:
                reward = -1.0 # 惩罚不必要的介入
        
        # 规则4 (可选): 每一步都有一个微小的生存惩罚，鼓励智能体不要无限期地介入
        reward -= 0.1 

        return reward
# --- 修改结束 ---

# (辅助函数 model_grad_norm, check_weights_bias_nan, moving_average, compute_advantage 保持不变)
# ...

# --- 修改开始 ---
# 删除了 ValueNet, CostNet, SquashedNormal, PolicyNetContinuous, PPOLagCont
# 引入新的、更简单的 Actor-Critic 网络和 PPO 智能体

class PolicyNetDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 输出的是每个离散动作的 logits，不需要 softplus 或 clamp
        return self.fc2(x)

class ValueNetSimple(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetSimple, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPODiscrete:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNetSimple(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 优势归一化
        adv_mean = advantage.mean()
        adv_std = advantage.std()
        advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            logits = self.actor(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions.squeeze()).view(-1, 1) # 计算新策略的 log_prob
            
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
# --- 修改结束 ---

# 超参数
actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 2000 # 可能需要更多 episodes 来学习
hidden_dims = 64
gamma = 0.98 # 对于更长的episodes，可以稍稍提高gamma
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --- 修改开始 ---
env = SupervisoryEnv() # 使用新的环境

state_dim = 2  # [position, intended_displacement]
action_dim = 2 # {0: Don't Intervene, 1: Intervene}

agent = PPODiscrete(state_dim, hidden_dims, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
# --- 修改结束 ---

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

out_range_count = 0
return_list = []
replay_buffer = collections.deque()

with tqdm(total=int(num_episodes), desc='Iteration') as pbar:
    for i_episode in range(int(num_episodes)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            
            state = next_state
            episode_return += reward
        
        if env.out_range:
            out_range_count += 1
            
        return_list.append(episode_return)
        agent.update(transition_dict) # 每回合更新一次

        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                              'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)

# (绘图和测试部分代码基本类似，但需要修改以适应新的环境和智能体)
# ... 这里为了简洁，省略了绘图和测试代码的修改，但主要就是调用 agent.take_action(state) 和 env.step(action)

