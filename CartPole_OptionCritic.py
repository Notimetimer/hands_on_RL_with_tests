"""
算法和训练
"""
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 超参数调整
GAMMA = 0.98  # 稍微减少折扣因子，减少远期奖励影响
LEARNING_RATE = 0.0005  # 降低学习率，提高训练的稳定性
BATCH_SIZE = 128  # 增加批次大小以稳定训练
MEMORY_SIZE = 20000  # 增大经验回放缓冲区
EPSILON_DECAY = 0.99  # 减慢 epsilon 的衰减速度
MIN_EPSILON = 0.05  # 增大最小 epsilon，保持一定探索
NUM_OPTIONS = 2
NUM_EPISODES = 1000  # 增加训练回合数


# 修改后的网络结构
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, num_options):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # 增加隐藏层神经元数量
        self.fc2 = nn.Linear(256, num_options)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


class IntraOptionPolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_options, action_dim):
        super(IntraOptionPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, num_options * action_dim)
        self.num_options = num_options
        self.action_dim = action_dim

    def forward(self, state, option):
        x = torch.relu(self.fc1(state))
        policy_logits = self.fc2(x)
        option_policy = policy_logits.view(-1, self.num_options, self.action_dim)
        return option_policy[:, option, :]


class TerminationNetwork(nn.Module):
    def __init__(self, state_dim, num_options):
        super(TerminationNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, num_options)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        termination_probs = torch.sigmoid(self.fc2(x))
        return termination_probs


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, option, action, reward, next_state, done):
        self.buffer.append((state, option, action, reward, next_state, done))

    def sample(self, batch_size):
        states, options, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), options, actions, rewards, np.stack(next_states), dones

    def size(self):
        return len(self.buffer)


# 选择动作
def select_action(policy_net, state, option, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1])  # CartPole 动作空间为 2（0 或 1）
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = torch.softmax(policy_net(state, option), dim=-1)
        return torch.argmax(action_probs).item()


# Option-Critic 智能体
class OptionCriticAgent:
    def __init__(self, state_dim, action_dim, num_options):
        self.policy_net = IntraOptionPolicyNetwork(state_dim, num_options, action_dim)
        self.q_net = CriticNetwork(state_dim, num_options)
        self.termination_net = TerminationNetwork(state_dim, num_options)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.optimizer_q = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.optimizer_term = optim.Adam(self.termination_net.parameters(), lr=LEARNING_RATE)
        self.epsilon = 1.0
        self.num_options = num_options
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def train(self, batch_size):
        if self.memory.size() < batch_size:
            return

        states, options, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        options = torch.LongTensor(options)
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)

        # 更新 Q 函数
        q_values = self.q_net(states)
        next_q_values = self.q_net(next_states).detach()
        target_q_values = rewards + GAMMA * next_q_values.max(1)[0] * (1 - dones)
        loss_q = nn.functional.mse_loss(q_values.gather(1, options.unsqueeze(1)).squeeze(), target_q_values)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        # 更新选项内策略
        for option in range(self.num_options):
            policy_logits = self.policy_net(states, option)
            action_probs = torch.softmax(policy_logits, dim=-1)
            log_action_probs = torch.log(action_probs)
            policy_loss = -log_action_probs.gather(1, actions.unsqueeze(1)).mean()
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

        # 更新终止概率
        terminations = self.termination_net(states)
        termination_loss = nn.functional.binary_cross_entropy(terminations.gather(1, options.unsqueeze(1)).squeeze(),
                                                              dones)
        self.optimizer_term.zero_grad()
        termination_loss.backward()
        self.optimizer_term.step()

    def remember(self, state, option, action, reward, next_state, done):
        self.memory.push(state, option, action, reward, next_state, done)


# 训练智能体
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = OptionCriticAgent(state_dim, action_dim, NUM_OPTIONS)

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    option = random.choice(range(NUM_OPTIONS))  # 随机选择一个选项
    done = False
    episode_reward = 0

    while not done:
        action = select_action(agent.policy_net, state, option, agent.epsilon)
        next_state, reward, done, _, __ = env.step(action)
        agent.remember(state, option, action, reward, next_state, done)
        agent.train(BATCH_SIZE)

        # 选项终止时选择新选项
        if random.random() < agent.termination_net(torch.FloatTensor(state))[option].item():
            option = random.choice(range(NUM_OPTIONS))

        state = next_state
        episode_reward += reward

    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

env.close()


# Option-Critic 智能体
class OptionCriticAgent:
    def __init__(self, state_dim, action_dim, num_options):
        self.policy_net = IntraOptionPolicyNetwork(state_dim, num_options, action_dim)
        self.q_net = CriticNetwork(state_dim, num_options)
        self.termination_net = TerminationNetwork(state_dim, num_options)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.optimizer_q = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.optimizer_term = optim.Adam(self.termination_net.parameters(), lr=LEARNING_RATE)
        self.epsilon = 1.0
        self.num_options = num_options
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def train(self, batch_size):
        if self.memory.size() < batch_size:
            return

        states, options, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        options = torch.LongTensor(options)
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)

        # 更新 Q 函数
        q_values = self.q_net(states)
        next_q_values = self.q_net(next_states).detach()
        target_q_values = rewards + GAMMA * next_q_values.max(1)[0] * (1 - dones)
        loss_q = nn.functional.mse_loss(q_values.gather(1, options.unsqueeze(1)).squeeze(), target_q_values)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        # 更新选项内策略
        for option in range(self.num_options):
            policy_logits = self.policy_net(states, option)
            action_probs = torch.softmax(policy_logits, dim=-1)
            log_action_probs = torch.log(action_probs)
            policy_loss = -log_action_probs.gather(1, actions.unsqueeze(1)).mean()
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

        # 更新终止概率
        terminations = self.termination_net(states)
        termination_loss = nn.functional.binary_cross_entropy(terminations.gather(1, options.unsqueeze(1)).squeeze(),
                                                              dones)
        self.optimizer_term.zero_grad()
        termination_loss.backward()
        self.optimizer_term.step()

    def remember(self, state, option, action, reward, next_state, done):
        self.memory.push(state, option, action, reward, next_state, done)


# 在 CartPole 环境中训练智能体
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = OptionCriticAgent(state_dim, action_dim, NUM_OPTIONS)

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    option = random.choice(range(NUM_OPTIONS))  # 初始化为随机选项
    done = False
    episode_reward = 0

    while not done:
        action = select_action(agent.policy_net, state, option, agent.epsilon)
        next_state, reward, done, _, __ = env.step(action)
        agent.remember(state, option, action, reward, next_state, done)
        agent.train(BATCH_SIZE)

        if random.random() < agent.termination_net(torch.FloatTensor(state))[option].item():
            option = random.choice(range(NUM_OPTIONS))  # 终止当前选项并选择新选项

        state = next_state
        episode_reward += reward

    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

env.close()

'''测试和可视化'''
import gym
import torch


# 测试 Option-Critic 模型并显示动画
def test_option_critic(agent, env, num_episodes=5):
    for episode in range(num_episodes):
        state, _ = env.reset()
        option = random.choice(range(agent.num_options))  # 随机选择一个选项
        done = False
        episode_reward = 0
        env.render()  # 初始化渲染环境

        while not done:
            env.render()  # 渲染环境，显示动画
            action = select_action(agent.policy_net, state, option, epsilon=0.0)  # 使用已学策略选择动作
            next_state, reward, done, _, __ = env.step(action)

            # 检查选项是否应终止，并在终止时重新选择新选项
            if random.random() < agent.termination_net(torch.FloatTensor(state))[option].item():
                option = random.choice(range(agent.num_options))

            state = next_state
            episode_reward += reward

        print(f"测试 Episode {episode + 1}: Total Reward: {episode_reward}")
    env.close()


# 创建 CartPole 环境并调用测试函数
env = gym.make('CartPole-v1', render_mode='human')
test_option_critic(agent, env)


'''
自动调整在GPU上运行，否则会在CPU上运行，且并不会报错

# 在文件开头添加设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 修改 OptionCriticAgent 类的初始化
class OptionCriticAgent:
    def __init__(self, state_dim, action_dim, num_options):
        self.policy_net = IntraOptionPolicyNetwork(state_dim, num_options, action_dim).to(device)
        self.q_net = CriticNetwork(state_dim, num_options).to(device)
        self.termination_net = TerminationNetwork(state_dim, num_options).to(device)
        # ... 其余代码保持不变 ...

# 修改 select_action 函数
def select_action(policy_net, state, option, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1])
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # ... 其余代码保持不变 ...

# 修改 train 方法中的张量创建
def train(self, batch_size):
    if self.memory.size() < batch_size:
        return

    states, options, actions, rewards, next_states, dones = self.memory.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    options = torch.LongTensor(options).to(device)
    actions = torch.LongTensor(actions).to(device)
    dones = torch.FloatTensor(dones).to(device)
    # ... 其余代码保持不变 ...
'''