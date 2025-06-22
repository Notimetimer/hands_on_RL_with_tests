import torch
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
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
        return action.item()


# 评估函数
def evaluate(args):
    set_seed(args.seed)

    # 创建环境
    env = args.env_fn(render_mode="human" if args.render else None)
    env.reset()

    # 获取状态和动作维度
    state_dims = {}
    action_dims = {}
    agents = env.possible_agents

    for agent in agents:
        state_dims[agent] = env.observation_space(agent).shape[0]
        action_dims[agent] = env.action_space(agent).n

    # 加载模型
    policies = {}
    for agent in agents:
        policy = PolicyNetwork(state_dims[agent], action_dims[agent])
        model_path = f"{args.model_dir}/{agent}_episode_{args.episode}.pth"

        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return

        policy.load_state_dict(torch.load(model_path))
        policy.eval()
        policies[agent] = policy

    # 评估参数
    total_episodes = args.eval_episodes
    episode_rewards = {agent: [] for agent in agents}
    episode_lengths = []

    # 评估循环
    for episode in range(total_episodes):
        env.reset()
        states = {agent: env.observe(agent) for agent in env.agents}
        episode_reward = {agent: 0 for agent in agents}
        episode_step = 0

        while env.agents:  # 当还有活跃智能体时
            # 选择动作
            actions = {}
            for agent in env.agents:
                with torch.no_grad():
                    action = policies[agent].get_action(states[agent])
                actions[agent] = action

            # 执行动作
            env.step(actions)

            # 记录奖励
            for agent in env.agents:
                episode_reward[agent] += env.rewards[agent]

            # 更新状态
            states = {agent: env.observe(agent) for agent in env.agents}
            episode_step += 1

            # 渲染环境
            if args.render:
                env.render()
                time.sleep(args.render_delay)

            # 检查是否达到最大步数
            if episode_step >= args.max_episode_length:
                break

        # 记录本轮结果
        for agent in agents:
            episode_rewards[agent].append(episode_reward[agent])
        episode_lengths.append(episode_step)

        print(f"Episode {episode + 1}/{total_episodes}")
        for agent in agents:
            print(f"  Agent {agent} Reward: {episode_reward[agent]:.3f}")
        print(f"  Episode Length: {episode_step}")
        print("----------------------------------------")

    # 计算平均结果
    print("\n===== 评估结果 =====")
    for agent in agents:
        avg_reward = np.mean(episode_rewards[agent])
        std_reward = np.std(episode_rewards[agent])
        print(f"Agent {agent}: Average Reward = {avg_reward:.3f} ± {std_reward:.3f}")

    avg_length = np.mean(episode_lengths)
    print(f"平均回合长度: {avg_length:.1f}")

    # 绘制奖励曲线
    if args.plot:
        plt.figure(figsize=(10, 6))
        for agent in agents:
            plt.plot(episode_rewards[agent], label=f'Agent {agent}')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'MAPPO Performance on {args.scenario_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'mappo_{args.scenario_name}_performance.png')
        print(f"奖励曲线已保存至 mappo_{args.scenario_name}_performance.png")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MAPPO on PettingZoo MPE')
    parser.add_argument('--scenario_name', type=str, default='simple_spread',
                        help='Name of the MPE scenario (e.g., simple_spread, simple_adversary)')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained models')
    parser.add_argument('--episode', type=int, required=True,
                        help='Episode number of the saved model')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--max_episode_length', type=int, default=25,
                        help='Maximum length of each episode')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--render_delay', type=float, default=0.1,
                        help='Delay between render frames (seconds)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot and save the reward curve')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # 根据场景名称选择环境创建函数
    env_fns = {
        'simple': simple_v2.parallel_env,
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

    evaluate(args)