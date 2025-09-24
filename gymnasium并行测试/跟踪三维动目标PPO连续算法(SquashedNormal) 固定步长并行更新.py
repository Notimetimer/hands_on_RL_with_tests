import random
import gymnasium as gymn
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use('Qt5Agg')
from numpy.linalg import norm
from torch.distributions import Normal

# import rl_utils
dt = 0.5
dof = 3

# 超参数
actor_lr = 1e-3 /10 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 10  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
episodes_limit = 500  # 2000
hidden_dim = [128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
update_steps = 42 # num_envs * 42  # 每次更新的步数 512

steps_limit = 350*update_steps # 未完待续

# 改进算法

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
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


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


class SquashedNormal:
    """带 tanh 压缩的高斯分布。

    采样：u ~ N(mu, std)（使用 rsample 支持 reparam），a = tanh(u)
    log_prob：基于 u 的 normal.log_prob(u) 并加上 tanh 的 Jacobian 修正项：-sum log(1 - tanh(u)^2)
    注意：外部需要把动作缩放到环境动作空间（仿射变换）。
    """

    def __init__(self, mu, std, eps=1e-6):
        self.mu = mu
        self.std = std
        self.normal = Normal(mu, std)
        self.eps = eps

    def sample(self):
        # rsample 以支持 reparameterization 重参数化采样, 结果是可导的
        u = self.normal.rsample()
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        # a: tanh(u)
        # log_prob(u) - sum log(1 - tanh(u)^2)
        # normal.log_prob 返回每个维度的 log_prob，需要 sum
        # 为数值稳定性添加小量
        log_prob_u = self.normal.log_prob(u)
        # jacobian term
        jacobian = torch.log(1 - a.pow(2) + self.eps) # fixme 应该+还是-？
        # sum over action dim, keep dims consistent: return (N, 1)
        # 取消提前求和 # return (log_prob_u - jacobian).sum(-1, keepdim=True)
        return log_prob_u - jacobian  # 返回形状为 (batch_size, action_dim)

    def entropy(self):
        # 近似：使用 base normal 的熵之和（不考虑 tanh 的修正）
        # 这在实践中通常足够，若需精确熵可用采样估计
        ent = self.normal.entropy().sum(-1)
        return ent


class PolicyNetContinuous(torch.nn.Module):
    """输出未压缩（pre-squash）的 mu 和 std。

    网络输出的 mu 是未经过 tanh 的原始均值，std 用 softplus 保证正值。
    不在网络内部做 action scaling，统一在采样/执行阶段处理。
    """

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
        self.fc_std = torch.nn.Linear(prev_size, action_dim)

    def forward(self, x, min_std=1e-3):
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        std = torch.clamp(std, min=min_std)
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法，支持时变动作区间（每步 amin/amax 不同）。

    设计说明（必须注意）：
    - 如果环境的动作约束随状态变化（amin/amax 为时变），则经验回放需保存当时的
      amin/amax（请把它放到 transition_dict['action_bounds']，形状为 (N, 2) 或每步的 (amin, amax)）。
    - 如果 action_bounds 在训练时始终恒定（标量或单个区间），也可以直接把 action_bound
      作为常数传入 update()。
    - 在本实现中，策略内部输出的是标准化前的 mu 和 std（即对 u 的分布参数）。
      对应的执行动作为：a = tanh(u)  -> normalized in (-1,1)
      最后缩放到真实区间： a_exec = amin + (a+1)/2 * (amax-amin)
    - update() 中会把存储的 a_exec "反归一化" 回 normalized a（[-1,1]），以便计算 log_prob。
    '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def _scale_action_to_exec(self, a, action_bounds):
        """把 normalized action a (in [-1,1]) 缩放到环境区间。

        action_bounds: 形状为 (action_dim, 2) 的二维 NumPy 数组，
                       每行对应 amin 和 amax。
        """
        action_bounds = torch.as_tensor(action_bounds, dtype=a.dtype, device=a.device)
        if action_bounds.dim() == 2:
            # 处理二维张量 (action_dim, 2)
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            # 处理三维张量 (batch, action_dim, 2)
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        
        # a in (-1,1) -> scale to [amin, amax]
        return amin + (a + 1.0) * 0.5 * (amax - amin)

    def _unscale_exec_to_normalized(self, a_exec, action_bounds):
        """把执行动作 a_exec 反向归一化到 [-1,1]。

        action_bounds: 形状为 (action_dim, 2) 的二维 NumPy 数组，
                       每行对应 amin 和 amax。
        """
        action_bounds = torch.as_tensor(action_bounds, dtype=a_exec.dtype, device=a_exec.device)
        if action_bounds.dim() == 2:
            # 处理二维张量 (action_dim, 2)
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            # 处理三维张量 (batch, action_dim, 2)
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        
        # 防止除以零
        span = (amax - amin)
        span = torch.where(span == 0, torch.tensor(1e-6, device=span.device, dtype=span.dtype), span)
        a = 2.0 * (a_exec - amin) / span - 1.0
        # numerical stability
        return a.clamp(-0.999999, 0.999999)

    def take_action(self, state, action_bounds, explore=True):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        dist = SquashedNormal(mu, std)
        if explore:
            a_norm, u = dist.sample()
        else:
            # use mean action: tanh(mu)
            u = mu
            a_norm = torch.tanh(u)

        a_exec = self._scale_action_to_exec(a_norm, action_bounds)
        return a_exec[0].cpu().detach().numpy().flatten()
    

    def update(self, transition_dict, action_bounds=None):
        """更新函数兼容以下几种调用方式：
        - 如果 action_bounds 是 None: 期望 transition_dict 中包含 'action_bounds'，其形状为 (N,2) 或每步 (amin,amax)
        - 如果 action_bounds 是标量/二元元组/数组：作为全局固定区间使用

        transition_dict 必须包含 keys: 'states','actions','rewards','next_states','dones'
        当动作区间随步变化时，必须包含 'action_bounds' 与之对应。
        存储的 'actions' 应当是环境执行动作 (a_exec 未归一化）。
        """
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions_exec = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        action_bounds = torch.tensor(np.array(transition_dict['action_bounds']), dtype=torch.float).to(self.device)

        # 计算 td_target, advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 策略输出（未压缩的 mu,std）
        mu, std = self.actor(states)
        # 构造 SquashedNormal 并计算 old_log_probs
        dist = SquashedNormal(mu.detach(), std.detach())

        # 将执行动作反向归一化到 [-1,1]，以便计算 log_prob
        actions_normalized = self._unscale_exec_to_normalized(actions_exec, action_bounds)
        
        # 反算 u = atanh(a)
        u_old = torch.atanh(actions_normalized)
        old_log_probs = dist.log_prob(actions_normalized, u_old)

        if torch.isnan(old_log_probs).any():
            raise ValueError("old_log_probs 包含 NaN，检查 action_bounds 或 actions 的合法性")

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            dist = SquashedNormal(mu, std)
            # 计算当前策略对历史执行动作的 log_prob（使用同一个 u_old）
            log_probs = dist.log_prob(actions_normalized, u_old)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # 取消提前求和 # actor_loss = -torch.min(surr1, surr2).mean() - 0.1 * dist.entropy().mean()
            actor_loss = -torch.min(surr1, surr2).sum(-1).mean() - 0.1 * dist.entropy().mean()
            # ↑如果求和之和还要保留原先的张量维度，用torch.sum(torch.min(surr1,surr2),dim=-1,keepdim=True)

            critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)

            self.actor_optimizer.step()
            self.critic_optimizer.step()


from tracking_test2 import testEnv

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'testEnv'
# 并行环境数量
num_envs =  8 # 4
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)

state_dim = testEnv(dof=3, dt=0.5).observation_space.shape[0]
action_dim = testEnv(dof=3, dt=0.5).action_space.shape[0]

# 全局动作区间（每个维度相同）
single_env = testEnv(dof=3, dt=0.5)
action_bound = np.array([[single_env.action_space.low[0], single_env.action_space.high[0]]] * action_dim, dtype=np.float32)

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

out_range_count = 0
return_list = []

def make_env_fn(seed_offset):
    def _thunk():
        env = testEnv(dof=dof, dt=dt)
        return env
    return _thunk

if __name__ == "__main__":
    # 创建 AsyncVectorEnv（在 Windows 上必须放到 main guard 内）
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    vec_env = gymn.vector.AsyncVectorEnv(env_fns)

    # reset vector env
    obs, infos = vec_env.reset()
    # 每个子环境单独收集轨迹，结束后调用 agent.update()
    # remove env_alive array — always produce actions for all subenvs

    per_env_buffers = [{
        'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []
    } for _ in range(num_envs)]

    # 新增变量：经验缓冲区
    global_buffer = {
        'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []
    }
    steps_collected = 0
    total_steps = 0
    
    # 使用 steps_limit 作为进度条总量，每次仿真步推进 1
    start_time = time.time()  # 记录训练开始时间
    episode_counter = 0
    with tqdm(total=int(steps_limit), desc='Steps') as step_bar:
        while total_steps < steps_limit:  # episode_counter < episodes_limit and 
            # 对所有子环境根据当前 obs 一次性生成动作（不区分 alive 状态）
            states_batch = obs  # shape (num_envs, state_dim)
            actions_batch = []
            for i in range(num_envs):
                s = states_batch[i]
                a = agent.take_action(s, action_bounds=action_bound, explore=True)
                actions_batch.append(a)
            actions_np = np.array(actions_batch, dtype=np.float32)

            # vector step
            next_obs, rewards, terminateds, truncs, infos = vec_env.step(actions_np)
            dones = np.logical_or(terminateds, truncs)

            # 保存到全局缓冲区 & per-env 缓冲区，并在对应子环境 done 时处理该回合
            for i in range(num_envs):
                per_env_buffers[i]['states'].append(states_batch[i].copy())
                per_env_buffers[i]['actions'].append(actions_np[i].copy())
                per_env_buffers[i]['next_states'].append(next_obs[i].copy())
                per_env_buffers[i]['rewards'].append(float(rewards[i]))
                per_env_buffers[i]['dones'].append(bool(dones[i]))
                per_env_buffers[i]['action_bounds'].append(action_bound.copy())

                # 同时也填充到全局缓冲区（用于固定步数更新）
                global_buffer['states'].append(states_batch[i].copy())
                global_buffer['actions'].append(actions_np[i].copy())
                global_buffer['next_states'].append(next_obs[i].copy())
                global_buffer['rewards'].append(float(rewards[i]))
                global_buffer['dones'].append(bool(dones[i]))
                global_buffer['action_bounds'].append(action_bound.copy())

                steps_collected += 1
                total_steps += 1

                if dones[i]:
                    # 处理该子环境的整回合经验并清空其 per_env buffer
                    transition = per_env_buffers[i]
                    return_list.append(sum(transition['rewards']))
                    # agent.update(transition)
                    
                    episode_counter += 1

                    # 会自动重置停下的子环境？


                    # 立即只重置第 i 个子环境（不影响其它子环境）
                    # 使用 VectorEnv.reset(indices=...)，兼容不同 gymnasium 版本的返回格式
                    # try:
                    #     # 尝试以 list 形式传入 indices（多数版本接受 list）
                    #     reset_obs_batch, reset_infos = vec_env.reset(indices=[i])
                    #     # reset_obs_batch 通常是形状 (1, obs_dim)
                    #     reset_obs_i = reset_obs_batch[0] if isinstance(reset_obs_batch, np.ndarray) else reset_obs_batch
                    #     print('list传入indices通过')
                    # except TypeError:
                    #     # 有些版本接受单个 int 作为 indices
                    #     try:
                    #         reset_obs_batch, reset_infos = vec_env.reset(indices=i)
                    #         reset_obs_i = reset_obs_batch[0] if isinstance(reset_obs_batch, np.ndarray) else reset_obs_batch
                    #         print('可以接受单个int作为indices')
                    #     except Exception:
                    #         # 若 reset(indices=...) 不可用或失败，跳过，让 vec_env 在后续 step 自动重置
                    #         reset_obs_i = None
                    #         print('上述办法失败')
                    # if reset_obs_i is not None:
                    #     next_obs[i] = np.asarray(reset_obs_i, dtype=next_obs.dtype).copy()

                    # agent.update(transition)
                    # 清空该子环境的 per-env buffer
                    per_env_buffers[i] = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
                    
                    # if episode_counter >= episodes_limit:
                    #     break
                else:
                    obs = next_obs

            # 如果收集的步数达到更新步数，则用 global_buffer 做一次批量更新
            if steps_collected >= update_steps:
                # agent.update(transition)
                # per_env_buffers[i] = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
                agent.update(global_buffer)
                global_buffer = {key: [] for key in global_buffer}  # 清空缓冲区
                steps_collected = 0
                 # 更新进度条
                if total_steps<=steps_limit:
                    step_bar.set_postfix({'episode': '%d' % episode_counter,
                                    'return': '%.3f' % (np.mean(return_list[-10:]))}) # if len(return_list) >= 10 else float(np.nan))})
                step_bar.update(update_steps)

            # # 更新 obs（注意：AsyncVectorEnv 会为已结束的子环境在 next_obs 中返回 reset 后的 init obs）
            # obs = next_obs

    vec_env.close()

    end_time = time.time()
    training_duration = end_time - start_time
    # 简短打印训练时长（秒）与格式化时长
    print(f"训练时长（秒）：{training_duration:.2f}")
    # hrs = int(training_duration // 3600)
    # mins = int((training_duration % 3600) // 60)
    # secs = int(training_duration % 60)
    # print(f"训练时长（H:M:S）：{hrs:d}:{mins:02d}:{secs:02d}")

    # 绘图及测试部分保持原逻辑（可用单环境测试）
    # ... existing plotting and single-env test code ...
    episodes_list = list(range(len(return_list)))
    plt.figure()
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))

    mv_return = moving_average(return_list, 9) if len(return_list) >= 9 else return_list
    plt.figure()
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))

    # 单环境测试回合
    single_env = testEnv(dof=dof, dt=dt)
    state, _ = single_env.reset(train=False)
    done = False
    car_trajectory = []
    target_trajectory = []
    episode_return = 0
    while not done:
        action = agent.take_action(state, action_bounds=action_bound, explore=False)
        next_state, reward, terminated, truncated, info = single_env.step(action)
        done = terminated or truncated
        car_trajectory.append(single_env.state[0:dof].copy())
        target_trajectory.append(single_env.target_pos_[0:dof].copy())
        state = next_state
        episode_return += reward

    plt.figure(4)
    for i in range(dof):
        plt.subplot(dof, 1, i + 1)
        pos_trajectory = [s[i] for s in car_trajectory]
        target_pos_trajectory = [t[i] for t in target_trajectory]
        plt.plot(range(len(pos_trajectory)), pos_trajectory, 'b-', label='Position')
        plt.plot(range(len(target_pos_trajectory)), target_pos_trajectory, 'r--', label='Target Position')
        plt.xlabel('Step')
        plt.ylabel(f'Coordinate {i + 1}')
        plt.legend()

    plt.show()
    print("出界次数：", out_range_count)
    

    '''
    在 Windows 上必须把 AsyncVectorEnv 的创建放到 if name == "main" 下（上面已处理）。
    agent.take_action 保留单 env 接口，训练时对每个子环境分别调用（性能仍有提升因为环境在子进程并行）。
    testEnv 已改为符合 gymnasium 的 reset/step 签名，并把额外信息放到 info 中。
    若要进一步把 agent.take_action 向量化以减少 Python 循环，可在 PPOContinuous 中添加 batch 版的 take_action（可以后续改进）。
    '''

