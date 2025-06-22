import torch
import numpy as np
import argparse
import os
import time
from collections import defaultdict, deque

from pettingzoo.mpe import simple_adversary_v3
from torch.utils.tensorboard import SummaryWriter

# PettingZoo库
from pettingzoo.mpe import *


# 设置随机种子以确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 策略网络
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach().numpy()


# 价值网络
class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# 经验回放缓冲区
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        del self.log_probs[:]


# MAPPO智能体
class MAPPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer()

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer_actor = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.value.parameters(), lr=lr_critic)

    def select_action(self, state):
        action, log_prob = self.policy.get_action(state)
        return action, log_prob

    def update(self):
        # 获取缓冲区数据
        states = torch.FloatTensor(np.array(self.buffer.states))
        actions = torch.LongTensor(np.array(self.buffer.actions))
        rewards = torch.FloatTensor(np.array(self.buffer.rewards))
        next_states = torch.FloatTensor(np.array(self.buffer.next_states))
        dones = torch.FloatTensor(np.array(self.buffer.dones))
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs))

        # 计算优势函数
        with torch.no_grad():
            values = self.value(states).squeeze()
            next_values = self.value(next_states).squeeze()

        returns = []
        advantages = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)

            advantage = R - values[i]
            advantages.insert(0, advantage)

        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算策略损失
        log_probs = torch.log(self.policy(states).gather(1, actions.unsqueeze(1)).squeeze())
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # 计算价值损失
        critic_loss = ((self.value(states).squeeze() - returns) ** 2).mean()

        # 优化网络
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # 清空缓冲区
        self.buffer.clear()

        return actor_loss.item(), critic_loss.item()


# 训练函数
def train(args):
    set_seed(args.seed)

    # 创建环境
    env = args.env_fn()
    env.reset()

    # 获取状态和动作维度
    state_dims = {}
    action_dims = {}
    agents = env.possible_agents

    for agent in agents:
        state_dims[agent] = env.observation_space(agent).shape[0]
        action_dims[agent] = env.action_space(agent).n

    # 创建智能体
    agents_dict = {}
    for agent in agents:
        agents_dict[agent] = MAPPOAgent(
            state_dim=state_dims[agent],
            action_dim=action_dims[agent],
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            eps_clip=args.eps_clip
        )

    # 创建日志目录
    log_dir = f"logs/mappo_{args.scenario_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 创建模型保存目录
    model_dir = f"models/mappo_{args.scenario_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)

    # 训练循环
    episode_rewards = defaultdict(list)
    episode_lengths = []
    total_steps = 0

    for episode in range(args.max_episodes):
        env.reset()
        states = {agent: env.observe(agent) for agent in env.agents}
        episode_reward = defaultdict(float)
        episode_step = 0

        while env.agents:  # 当还有活跃智能体时
            actions = {}
            log_probs = {}

            # 选择动作
            for agent in env.agents:
                action, log_prob = agents_dict[agent].select_action(states[agent])
                actions[agent] = action
                log_probs[agent] = log_prob

            # 执行动作
            env.step(actions)

            next_states = {agent: env.observe(agent) for agent in env.agents}
            rewards = {agent: env.rewards[agent] for agent in env.agents}
            dones = {agent: env.terminations[agent] for agent in env.agents}

            # 存储转换
            for agent in env.agents:
                agents_dict[agent].buffer.states.append(states[agent])
                agents_dict[agent].buffer.actions.append(actions[agent])
                agents_dict[agent].buffer.rewards.append(rewards[agent])
                agents_dict[agent].buffer.next_states.append(next_states[agent])
                agents_dict[agent].buffer.dones.append(dones[agent])
                agents_dict[agent].buffer.log_probs.append(log_probs[agent])

                episode_reward[agent] += rewards[agent]

            states = next_states
            episode_step += 1
            total_steps += 1

            # 检查是否达到最大步数
            if episode_step >= args.max_episode_length:
                break

        # 更新策略
        for agent in agents:
            actor_loss, critic_loss = agents_dict[agent].update()

            # 记录日志
            writer.add_scalar(f'loss/actor_{agent}', actor_loss, episode)
            writer.add_scalar(f'loss/critic_{agent}', critic_loss, episode)

        # 记录本轮奖励
        for agent in agents:
            writer.add_scalar(f'reward/{agent}', episode_reward[agent], episode)
            episode_rewards[agent].append(episode_reward[agent])

        episode_lengths.append(episode_step)
        writer.add_scalar('episode_length', episode_step, episode)

        # 打印训练进度
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode + 1}/{args.max_episodes}")
            for agent in agents:
                avg_reward = np.mean(episode_rewards[agent][-args.log_interval:])
                print(f"  Agent {agent}: Average Reward = {avg_reward:.3f}")
            print(f"  Episode Length: {np.mean(episode_lengths[-args.log_interval:]):.1f}")
            print("----------------------------------------")

        # 保存模型
        if (episode + 1) % args.save_interval == 0:
            for agent in agents:
                torch.save(agents_dict[agent].policy.state_dict(),
                           f"{model_dir}/{agent}_episode_{episode + 1}.pth")

    env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAPPO on PettingZoo MPE')
    parser.add_argument('--scenario_name', type=str, default='simple_spread',
                        help='Name of the MPE scenario (e.g., simple_spread, simple_adversary)')
    parser.add_argument('--max_episodes', type=int, default=10000,
                        help='Maximum number of training episodes')
    parser.add_argument('--max_episode_length', type=int, default=25,
                        help='Maximum length of each episode')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Learning rate for actor network')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                        help='Learning rate for critic network')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                        help='Clipping parameter for PPO')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Interval between logging')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Interval between saving models')
    args = parser.parse_args()

    # 根据场景名称选择环境创建函数
    env_fns = {
        'simple': simple_v3.parallel_env,
        'simple_adversary': simple_adversary_v3.parallel_env,
        'simple_crypto': simple_crypto_v2.parallel_env,
        'simple_push': simple_push_v2.parallel_env,
        'simple_reference': simple_reference_v2.parallel_env,
        'simple_speaker_listener': simple_speaker_listener_v4.parallel_env,
        'simple_spread': simple_spread_v2.parallel_env,
        'simple_tag': simple_tag_v2.parallel_env,
        'simple_world_comm': simple_world_comm_v2.parallel_env
    }

    if args.scenario_name not in env_fns:
        print(f"不支持的场景: {args.scenario_name}")
        print(f"支持的场景: {list(env_fns.keys())}")
        sys.exit(1)

    args.env_fn = env_fns[args.scenario_name]

    train(args)