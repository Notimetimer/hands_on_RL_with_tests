# 不带 Gumbel-Softmax 的离散 DDPG 示例（CartPole）
# 思路：把 Critic 改成“枚举式多头输出”（输入 s，输出所有动作的 Q 值向量），
# Actor 直接输出动作概率分布 probs，更新时直接最大化期望 sum(probs * Q(s))，
# 从而在保持全可微的同时，彻底省去 Gumbel-Softmax 重参数化采样。
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
    if len(out) == 5:  # gymnasium
        next_state, reward, terminated, truncated, _ = out
        return next_state, reward, terminated or truncated
    next_state, reward, done, _ = out  # gym
    return next_state, reward, done


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
    """【改动1】策略网络: 直接输出动作概率分布 probs, 不需要后续再做 Gumbel-Softmax。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc_out(x), dim=-1)


class QValueNet(torch.nn.Module):
    """【改动2】价值网络: 改成枚举式多头输出, 输入 state, 输出每个离散动作的 Q 值。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # shape: (batch_size, action_dim)


# ---------------------------------------------------------------- 算法主体
class DiscreteDDPGNoGumbel:
    """不带 Gumbel-Softmax 的离散 DDPG (也被称为 Mean Actor-Critic)。"""

    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, tau, gamma, device):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, epsilon=0.0):
        """
        epsilon-贪心探索:
        注意这里保留 epsilon-贪心很重要, 因为没有 SAC 的熵正则项时,
        actor 的输出概率会迅速向 0/1 逼近, 仅靠 probs 采样会很快停止探索。
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
            return int(torch.argmax(probs, dim=-1).item())

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.as_tensor(np.asarray(transition_dict['states'], dtype=np.float32)).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.as_tensor(np.asarray(transition_dict['rewards'], dtype=np.float32)
                                  ).view(-1, 1).to(self.device)
        next_states = torch.as_tensor(np.asarray(transition_dict['next_states'], dtype=np.float32)
                                      ).to(self.device)
        dones = torch.as_tensor(np.asarray(transition_dict['dones'], dtype=np.float32)
                                ).view(-1, 1).to(self.device)

        # 【改动3】1. 更新 Critic: 用 gather 取出历史动作对应的 Q 值, 目标值用 target_actor 概率求期望
        with torch.no_grad():
            next_probs = self.target_actor(next_states)
            next_q_all = self.target_critic(next_states)
            next_q_values = torch.sum(next_probs * next_q_all, dim=1, keepdim=True)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        q_values = self.critic(states).gather(1, actions)
        critic_loss = F.mse_loss(q_values, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 【改动4】2. 更新 Actor: 直接对全动作分布求期望 E_{a~pi}[Q(s, a)], 完全可导且无偏
        probs = self.actor(states)
        q_all = self.critic(states).detach()  # Q 不依赖 probs, detach 掉更省计算且不污染 Critic 梯度
        actor_loss = -torch.mean(torch.sum(probs * q_all, dim=1))
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
    tau = 0.005
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if hasattr(env, 'seed'):
        env.seed(0)
    env.action_space.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DiscreteDDPGNoGumbel(state_dim, hidden_dim, action_dim,
                                 actor_lr, critic_lr, tau, gamma, device)
    replay_buffer = ReplayBuffer(buffer_size)

    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                                         minimal_size, batch_size)

    # ------------------------------------------------------------ 结果可视化
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Discrete DDPG (No Gumbel) on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Discrete DDPG (No Gumbel) on {} (moving average)'.format(env_name))
    plt.show()