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

# 圆周空间下随机概率分布
class CircularContDist4OnPolciy:
    """
    Continuous circular distribution wrapper (von Mises) parameterized by
    mean angle `mu` (radians) and an "equivalent std" `std` such that
        kappa = 1.0 / (std**2 + eps)

    Interface:
      - __init__(mu, std, eps=1e-8)
      - log_prob(theta): returns (batch, 1) log-probability for given angle(s)
      - sample(n=None): CPU-side sampling via numpy.random.vonmises, returns tensor
      - entropy(): approximate entropy if i1 available, else returns None
      - mean(): returns mu tensor
    """
    def __init__(self, mu, std, eps=1e-8):
        # mu: tensor (B,1) or (B,) in radians
        # std: tensor or scalar, interpreted as equivalent standard deviation
        self.eps = float(eps)
        if not torch.is_tensor(mu):
            mu = torch.as_tensor(mu)
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, dtype=mu.dtype)

        # keep device for later
        self.device = mu.device if hasattr(mu, 'device') else torch.device('cpu')

        # normalize shapes to (B, 1)
        self.mu = mu.view(-1, 1).to(self.device)
        self.std = std.to(self.device).view(-1, 1)

        # kappa from equivalent std
        self.kappa = 1.0 / (self.std.pow(2) + self.eps)

    def log_prob(self, theta):
        """Compute von-Mises log pdf for `theta` (radians).
        theta: tensor broadcastable to mu/kappa shapes
        returns tensor shaped (batch, 1)
        """
        if not torch.is_tensor(theta):
            theta = torch.as_tensor(theta, device=self.device)
        # ensure same shape
        theta = theta.view(-1, 1).to(self.device)

        # Use scaled i0 if available for better stability
        if hasattr(torch.special, 'i0e'):
            # log I0 = log(i0e(kappa)) + kappa
            log_i0 = torch.log(torch.special.i0e(self.kappa).clamp(min=self.eps)) + self.kappa
        else:
            log_i0 = torch.log(torch.special.i0(self.kappa).clamp(min=self.eps))

        log_norm = torch.log(torch.tensor(2.0 * np.pi, device=self.device)) + log_i0

        # von Mises log pdf: kappa * cos(theta - mu) - log(2π I0(kappa))
        return (self.kappa * torch.cos(theta - self.mu)) - log_norm

    def sample(self, n=None):
        """CPU sampling via numpy.random.vonmises.
        n: number of samples to draw; if None, draws one per mu in batch.
        Returns tensor on self.device with shape (n, 1) or (batch, 1).
        Note: sampling is non-differentiable (intended for env collection).
        """
        mu_np = self.mu.detach().cpu().numpy().reshape(-1)
        kappa_np = self.kappa.detach().cpu().numpy().reshape(-1)

        if n is None:
            n_draw = mu_np.shape[0]
        else:
            n_draw = int(n)

        samples = np.zeros((n_draw,), dtype=float)
        # if batch provided and n_draw equals batch, sample per-element; else cycle over params
        for i in range(n_draw):
            idx = i % mu_np.shape[0]
            samples[i] = np.random.vonmises(mu_np[idx], kappa_np[idx])

        out = torch.as_tensor(samples, dtype=self.mu.dtype, device=self.device).view(n_draw, 1)
        return out

    def entropy(self):
        """Return approximate entropy H = -kappa * I1(kappa)/I0(kappa) + log(2π I0(kappa)).
        If torch.special.i1 is unavailable, return None.
        """
        if hasattr(torch.special, 'i1'):
            i0 = torch.special.i0(self.kappa).clamp(min=self.eps)
            i1 = torch.special.i1(self.kappa)
            # compute term safely
            ratio = (i1 / i0)
            log_i0 = torch.log(i0)
            ent = - self.kappa * ratio + torch.log(torch.tensor(2.0 * np.pi, device=self.device)) + log_i0
            return ent
        else:
            return None

    def mean(self):
        return self.mu


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

class PolicyNetCircular(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim=1): # 动作维度变为1(角度)
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]), nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[0]), nn.ReLU()
        )
        # 输出 mu 的范围通过 tanh 映射到 [-pi, pi]
        self.fc_mu = nn.Linear(hidden_dim[0], 1)
        self.fc_std = nn.Linear(hidden_dim[0], 1)

    def forward(self, x):
        x = self.fc(x)
        # 将 mu 映射到 [-pi, pi] 弧度空间
        mu = torch.tanh(self.fc_mu(x)) * np.pi 
        # std 映射为正数，作为集中度参数的倒数根
        std = F.softplus(self.fc_std(x)) + 1e-5
        return mu, std

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetCircular(state_dim, hidden_dim).to(device) # 使用圆周网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lmbda, self.epochs, self.eps, self.device = gamma, lmbda, epochs, eps, device

    def take_action(self, state, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        
        # 构造 Von Mises 分布包装器
        dist = CircularContDist4OnPolciy(mu, std)
        
        if not explore:
            angle = mu.detach().cpu().numpy().flatten()
        else:
            # 采样得到标量角度 theta
            angle = dist.sample().detach().cpu().numpy().flatten()
            
        # [关键点] 动作存储角度，但返回给环境时转换为 [cos, sin]
        action_vector = np.array([np.cos(angle[0]), np.sin(angle[0])])
        return angle, action_vector # 返回角度供记录，返回向量供环境执行

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # 注意：这里的 actions 应该是采样得到的标量角度 theta
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # GAE 计算 (保持不变)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 获取旧策略的对数概率
        mu, std = self.actor(states)
        old_dist = CircularContDist4OnPolciy(mu, std)
        old_log_probs = old_dist.log_prob(actions).detach()

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            new_dist = CircularContDist4OnPolciy(mu, std)
            # 使用 Von Mises 的 log_prob
            log_probs = new_dist.log_prob(actions)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

# ==========================================
# 3. 训练与结果展示
# ==========================================
# ==========================================
# 3. 运行主程序 (if __name__ == "__main__")
# ==========================================
# ==========================================
# 3. 运行主程序 (if __name__ == "__main__")
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
    # 注意：使用针对圆周分布设计的 PolicyNetCircular
    agent = PPOContinuous(
        state_dim=env.observation_space.shape[0], 
        hidden_dim=hidden_dim, 
        action_dim=1, # 策略网络输出维度为1 (角度 mu)
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
            # take_action 返回采样角度(用于计算log_prob)和单位向量(用于环境step)
            angle, action_vec = agent.take_action(state, explore=True)
            
            next_state, reward, done, _ = env.step(action_vec)
            
            # 存储时使用标量角度 angle
            transition_dict['states'].append(state)
            transition_dict['actions'].append(angle) 
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
        # 测试模式关闭探索
        angle, action_vec = agent.take_action(state, explore=False)
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