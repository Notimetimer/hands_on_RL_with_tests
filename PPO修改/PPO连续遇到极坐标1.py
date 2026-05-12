"""
这个尝试是失败的，
用独立的多元高斯分布去拟合非欧几里得空间耦合的角度关系不可取
"""
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from gym import spaces
from tqdm import tqdm

matplotlib.use('TkAgg')

# ==========================================
# 1. 视线坐标系环境 (LOSCommandEnv - V2)
# ==========================================
class LOSCommandEnv(gym.Env):
    def __init__(self, dt=0.1, agent_speed=2.0, change_steps=60, max_steps=600):
        super(LOSCommandEnv, self).__init__()
        self.dt = dt
        self.v = agent_speed
        self.change_steps = change_steps
        self.max_steps = max_steps
        
        # 观测 (5,7维): [距离r, 径向速度vr, 周向速度vt, cmd0, cmd1, cmd2, cmd3]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32) # (7,)
        # 动作 (2维): [cos(alpha), sin(alpha)] alpha是相对于视线矢量的角度
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.t_step = 0
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(5, 12)
        self.pos = np.array([r * np.cos(angle), r * np.sin(angle)])
        self.heading = np.random.uniform(0, 2 * np.pi)
        self.cmd = np.random.randint(0, 4)
        return self._get_obs()

    def _get_obs(self):
        r = np.linalg.norm(self.pos)
        psi = np.arctan2(self.pos[1], self.pos[0]) # 视线角
        
        # 速度分量计算
        # vr = v * cos(heading - psi)
        # vt = v * sin(heading - psi)
        relative_angle = self.heading - psi
        # v_noise = np.random.uniform(5,10)
        vr = self.v * np.cos(relative_angle)
        vt = self.v * np.sin(relative_angle)
        
        one_hot = np.zeros(4)
        one_hot[self.cmd] = 1.0
        r_spotted = min(r, 25)/25
        return np.concatenate([[r_spotted], one_hot]).astype(np.float32) # (7,) [r_spotted, vr, vt]

    def step(self, action):
        self.t_step += 1
        
        # 计算当前时刻视线角
        psi = np.arctan2(self.pos[1], self.pos[0])
        
        # 解析相对动作 alpha
        act_norm = np.linalg.norm(action) + 1e-8
        alpha = np.arctan2(action[1]/act_norm, action[0]/act_norm)
        
        # 更新绝对航向
        self.heading = psi + alpha
        
        # 更新物理位置
        self.pos[0] += self.v * np.cos(self.heading) * self.dt
        self.pos[1] += self.v * np.sin(self.heading) * self.dt
        
        # 奖励函数 (直接根据相对角 alpha 定义)
        if self.cmd == 0:   # 靠近 (期望 alpha = pi)
            reward = -np.cos(alpha)
        elif self.cmd == 1: # 远离 (期望 alpha = 0)
            reward = np.cos(alpha)
        elif self.cmd == 2: # 逆时针 (期望 alpha = pi/2)
            reward = np.sin(alpha)
        else:               # 顺时针 (期望 alpha = -pi/2)
            reward = -np.sin(alpha)
        
        r = np.linalg.norm(self.pos)

        # 周期性切换指令
        if self.t_step % self.change_steps == 0:
            self.cmd = np.random.randint(0, 4)
        if r < 5:
            self.cmd = max(1, self.cmd)

        
        done = (r < 0.5 or self.t_step >= self.max_steps)
        if r < 0.5: reward -= 5.0
            
        return self._get_obs(), reward, done, {}


# ==========================================
# 2. 算法工具与网络定义
# ==========================================
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
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
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        self.mu_layer_norm = torch.nn.LayerNorm(action_dim)
        self.fc_std = torch.nn.Linear(prev_size, action_dim)

    def forward(self, x, action_bound=1.0):
        x = self.net(x)
        mu = action_bound * torch.tanh(self.fc_mu(x))
        mu = self.mu_layer_norm(mu)
        std = F.softplus(self.fc_std(x)) + 1e-5
        return mu, std

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lmbda, self.epochs, self.eps, self.device = gamma, lmbda, epochs, eps, device

    def take_action(self, state, action_bound=1.0, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state, action_bound=action_bound)
        if not explore:
            return mu[0].cpu().detach().numpy().flatten()
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return torch.clamp(action, -action_bound, action_bound)[0].cpu().detach().numpy().flatten()

    def update(self, transition_dict, action_bound=1.0):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # GAE 计算 (保持不变)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        mu, std = self.actor(states, action_bound=action_bound)
        old_log_probs = torch.distributions.Normal(mu.detach(), std.detach()).log_prob(actions).sum(dim=-1, keepdim=True)

        for _ in range(self.epochs):
            mu, std = self.actor(states, action_bound=action_bound)
            log_probs = torch.distributions.Normal(mu, std).log_prob(actions).sum(dim=-1, keepdim=True)
            
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

# ==========================================
# 3. 主训练流程
# ==========================================
if __name__ == "__main__":
    # --- 环境与算法超参数 ---
    dt = 0.1
    change_steps = 60
    max_steps = 600
    env = LOSCommandEnv(dt=dt, change_steps=change_steps, max_steps=max_steps)
    
    actor_lr = 2e-4
    critic_lr = 1e-3
    num_episodes = 1200
    hidden_dim = [128, 128]
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 初始化智能体
    agent = PPOContinuous(
        state_dim=env.observation_space.shape[0], 
        hidden_dim=hidden_dim, 
        action_dim=2, # 策略网络输出维度为1 (角度 mu)
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
        lmbda=lmbda, 
        epochs=epochs, 
        eps=eps, 
        gamma=gamma, 
        device=device
    )

    return_list = []
    
    # --- 训练循环 ---
    print(f"开始训练，设备: {device}...")
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0
        transition_dict = {
            'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []
        }
        
        while not done:
            action_vec = agent.take_action(state, explore=True)
            next_state, reward, done, _ = env.step(action_vec)
            
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action_vec) # 记录向量
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            
            state = next_state
            episode_return += reward
        
        return_list.append(episode_return)
        
        # PPO 更新
        agent.update(transition_dict)

        # 打印进度 (每10回合)
        if (i_episode + 1) % 10 == 0:
            avg_return = np.mean(return_list[-10:])
            print(f"Episode: {i_episode+1:4d} | 10-Episode Avg Return: {avg_return:.3f}")

    # --- 结果可视化 ---
    
    # 1. 绘制训练奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(return_list, alpha=0.3, color='blue', label='Raw Return')
    plt.plot(moving_average(return_list, 19), color='red', label='Moving Average (n=19)')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO Training Reward Curve (Von Mises Distribution)')
    plt.legend()
    plt.grid(True, linestyle='--')

    # 2. 测试推理与轨迹记录
    state = env.reset()
    traj, cmds = [], []
    done = False
    test_return = 0
    
    # 记录初始位置作为起点
    start_pos = env.pos.copy()
    
    while not done:
        action_vec = agent.take_action(state, explore=False)
        next_state, reward, done, _ = env.step(action_vec)
        traj.append(env.pos.copy())
        cmds.append(env.cmd)
        state = next_state
        test_return += reward
    
    traj = np.array(traj)
    print(f"\n测试回合结束 | 总奖励: {test_return:.3f}")

    # 3. 绘制测试轨迹与指令流
    fig, (ax_traj, ax_cmd) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 轨迹图：颜色随指令 ID 变化
    ax_traj.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.15, zorder=1) # 连线轨迹
    sc = ax_traj.scatter(traj[:, 0], traj[:, 1], c=cmds, cmap='viridis', s=10, zorder=2)
    
    # 特殊标注：起点与中心点
    ax_traj.plot(start_pos[0], start_pos[1], 'g*', markersize=15, label='START', zorder=5)
    ax_traj.plot(0, 0, 'ro', markersize=8, label='CENTER', zorder=5)
    
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, linestyle=':')
    ax_traj.set_title("Inference Trajectory (LOS Frame)")
    ax_traj.legend(loc='upper right')
    plt.colorbar(sc, ax=ax_traj, ticks=[0, 1, 2, 3], label='0:Toward, 1:Away, 2:CCW, 3:CW')

    # 指令时序图
    ax_cmd.step(range(len(cmds)), cmds, where='post', color='navy', linewidth=1.5)
    ax_cmd.set_yticks([0, 1, 2, 3])
    ax_cmd.set_yticklabels(['Toward', 'Away', 'CCW', 'CW'])
    ax_cmd.set_title("Command Transitions Over Time")
    ax_cmd.set_xlabel("Steps")
    ax_cmd.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()