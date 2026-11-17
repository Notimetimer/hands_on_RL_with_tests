# 离散动作空间的 DDPG 算法示例（CartPole）
# 原始 DDPG 只适用于连续动作：策略网络直接输出一个动作值，Critic 接收 (状态, 动作)。
# 离散动作的关键问题：策略网络输出的是动作概率分布，argmax / 采样都不可导，
# 无法把 Critic 的梯度传回策略网络。
# 解决思路：用 Gumbel-Softmax (重参数化) 采样得到“近似 one-hot”的动作向量，
# 前向是离散选择、反向可导，于是可以像连续 DDPG 一样对动作求梯度。
import random
import collections

import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------- 工具函数
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def reset_env(env, seed=None):
    """兼容 gym(旧) 与 gymnasium(新) 的 reset 返回值。"""
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    """兼容 gym(4 元组) 与 gymnasium(5 元组) 的 step 返回值。"""
    out = env.step(action)
    if len(out) == 5:  # gymnasium: obs, reward, terminated, truncated, info
        next_state, reward, terminated, truncated, _ = out
        return next_state, reward, terminated or truncated
    next_state, reward, done, _ = out  # gym: obs, reward, done, info
    return next_state, reward, done


def gumbel_softmax(logits, tau=1.0, hard=True):
    """Gumbel-Softmax 重参数化: 前向返回离散的 one-hot, 反向梯度用 softmax 近似。"""
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~ Gumbel(0, 1)
    y = F.softmax((logits + gumbels) / tau, dim=-1)
    if not hard:
        return y
    index = y.max(dim=-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    return (y_hard - y).detach() + y  # 前向 one-hot, 反向等价于对 y 求导


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


# ---------------------------------------------------------------- 网络
class PolicyNet(torch.nn.Module):
    """策略网络: 输出动作 logits, 经 Gumbel-Softmax 得到可导的离散动作。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # 返回 logits, 不接 softmax


class QValueNet(torch.nn.Module):
    """价值网络: Q(s, a), 其中 a 是 one-hot 离散动作向量。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a_onehot):
        cat = torch.cat([x, a_onehot], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DiscreteDDPG:
    ''' 处理离散动作的 DDPG 算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, tau, gamma, device):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络初始参数与在线网络一致
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, epsilon=0.0):
        """epsilon-贪心: 以 eps 的概率随机探索, 否则取策略网络的 argmax 动作。"""
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            return int(torch.argmax(logits, dim=-1).item())

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.as_tensor(np.asarray(transition_dict['states'], dtype=np.float32)).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long).view(-1).to(self.device)
        rewards = torch.as_tensor(np.asarray(transition_dict['rewards'], dtype=np.float32)
                                  ).view(-1, 1).to(self.device)
        next_states = torch.as_tensor(np.asarray(transition_dict['next_states'], dtype=np.float32)
                                      ).to(self.device)
        dones = torch.as_tensor(np.asarray(transition_dict['dones'], dtype=np.float32)
                                ).view(-1, 1).to(self.device)

        # 离散动作 -> one-hot 向量, 才能喂给 Critic
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()

        # 1. 更新价值网络: y = r + gamma * Q_target(s', a*(s'))
        with torch.no_grad():
            next_logits = self.target_actor(next_states)
            next_actions = gumbel_softmax(next_logits, hard=True)  # 可导的离散动作
            next_q_values = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        critic_loss = torch.mean(
            F.mse_loss(self.critic(states, actions_onehot), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. 更新策略网络: 最大化 Critic 对当前动作的评估
        logits = self.actor(states)
        actor_actions = gumbel_softmax(logits, hard=True)
        actor_loss = -torch.mean(self.critic(states, actor_actions))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. 软更新两个目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


def train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                           minimal_size, batch_size, epsilon_start=1.0,
                           epsilon_end=0.05):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # 探索率线性衰减
                done_ratio = (i * int(num_episodes / 10) + i_episode) / num_episodes
                epsilon = epsilon_start + (epsilon_end - epsilon_start) * done_ratio

                episode_return = 0
                state = reset_env(env)
                done = False
                while not done:
                    action = agent.take_action(state, epsilon)
                    next_state, reward, done = step_env(env, action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 样本足够多时才开始更新
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a,
                                           'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005          # 软更新参数
    buffer_size = 100000
    minimal_size = 1000  # 经验回放中样本数超过该值才开始训练
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # CartPole-v2: 若本机 gym 未注册该 ID, 则自动回退到 CartPole-v1
    env_name = 'CartPole-v2'
    try:
        env = gym.make(env_name)
    except Exception:
        env_name = 'CartPole-v1'
        env = gym.make(env_name)
        print('本机 gym 未注册 CartPole-v2, 已自动回退到 CartPole-v1')

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if hasattr(env, 'seed'):
        env.seed(0)
    env.action_space.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DiscreteDDPG(state_dim, hidden_dim, action_dim,
                         actor_lr, critic_lr, tau, gamma, device)
    replay_buffer = ReplayBuffer(buffer_size)

    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                                         minimal_size, batch_size)

    # ------------------------------------------------------------ 结果可视化
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Discrete DDPG on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Discrete DDPG on {} (moving average)'.format(env_name))
    plt.show()
