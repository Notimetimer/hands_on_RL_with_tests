# 游标训练环境
import random
import matplotlib
import matplotlib.pyplot as plt
import gym
from gym import spaces
from numpy.linalg import norm
from torch.distributions import Normal
import random
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F

# matplotlib.use('Qt5Agg')  # 使用Qt5作为后端
def softplus(x):
    """
    计算softplus函数值
    
    参数:
    x -- 输入值（可以是标量、列表或NumPy数组）
    
    返回:
    计算后的softplus值，与输入形状相同
    """
    x_np = np.asarray(x)  # 确保输入是NumPy数组
    return np.log(1 + np.exp(x_np))

class Env:
    def __init__(self):
        self.bounce_back = None
        self.min_pos = -10
        self.max_pos = 10
        self.position = None
        self.out_range = None

    def reset(self):
        self.position = np.array([random.randint(self.min_pos, self.max_pos)],dtype='float64')
        self.steps=0
        self.out_range = 0
        return self.get_obs()

    def get_obs(self):
        # 返回 position 的副本，避免外部保存到 transition_dict 时成为引用
        return self.position.copy()

    def step(self, move):
        self.bounce_back = 0
        self.position += move # + np.random.normal()
        if self.position < self.min_pos or self.position > self.max_pos:
            self.bounce_back=1
        # self.position = np.clip(self.position, self.min_pos, self.max_pos) # 栏杆

        self.steps+=1
        done = self.get_done()
        reward = self.get_reward()
        cost = self.get_cost(move) # 计算成本
        return self.get_obs(), reward, done, cost

    def get_done(self):
        done=0
        if self.position < self.min_pos or self.position > self.max_pos:
            done=1
            self.out_range = 1
        if self.steps>=20:
            done=1
        return done

    def get_reward(self):
        pos_opt = 9
        reward1 = self.position[0]/10
        # if self.min_pos <= self.position <= self.max_pos:
        #     reward1 = 1 - np.linalg.norm(self.position - pos_opt) / 10
        # else:
        #     reward1 = 0 # -3
        # if self.bounce_back:
        #     reward1-=2

        if self.out_range:
            reward1 -= 100
        return reward1 - 1
    
    def get_cost(self, action):
        cost = -100000
        # trigger = 3
        # temp = 0

        # # # 惩罚对边界的靠近程度
        # # temp = min(np.linalg.norm(self.position - self.max_pos)/trigger, np.linalg.norm(self.position - self.min_pos)/trigger, 1)
        # # cost = 1-temp

        # # 惩罚靠近上边界还输出上的动作
        # if self.position[0] > self.max_pos - trigger and action[0] > 0: # 假设边界区域为2个单位
        #     temp = action[0] * (self.position[0] - (self.max_pos - trigger))/trigger # 离边界越近，惩罚越大
        # # 惩罚靠近下边界还输出下的动作
        # if self.position[0] < self.min_pos + trigger and action[0] < 0:
        #     temp = abs(action[0]) * ((self.min_pos + trigger) - self.position[0])/trigger # 离边界越近，惩罚越大
        # cost=temp
        
        return softplus(cost)


# 改进算法


# 计算并记录 actor / critic 的梯度范数（L2）
def model_grad_norm(model):
    total_sq = 0.0
    found = False
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().cpu()
            total_sq += float(g.norm(2).item()) ** 2
            found = True
    return float(total_sq ** 0.5) if found else float('nan')

def check_weights_bias_nan(model, model_name="model", place=None):
    """检查模型中名为 weight/bias 的参数是否包含 NaN，发现则抛出异常。
    参数:
      model: torch.nn.Module
      model_name: 用于错误消息中标识模型（如 "actor"/"critic"）
      place: 字符串，调用位置/上下文（如 "update_loop","pretrain_step"），用于更明确的错误报告
    """
    for name, param in model.named_parameters():
        if ("weight" in name) or ("bias" in name):
            if param is None:
                continue
            if torch.isnan(param).any():
                loc = f" at {place}" if place else ""
                raise ValueError(f"NaN detected in {model_name} parameter '{name}'{loc}")


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

class CostNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CostNet, self).__init__()
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
        self.mean = mu

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
        jacobian = 0 # 保存u的话就不需要该修正项
        # jacobian = 2*(np.log(2.0)-u-F.softplus(-2*u))
        # jacobian = torch.log(1 - a.pow(2) + self.eps)
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

    def forward(self, x, min_std=1e-6, max_std=0.3): # max_std=0.6
        # 最小方差 1e-3, 最大方差不要超过0.707否则tanh后会出现双峰函数
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        std = torch.clamp(std, min=min_std, max=max_std)
        return mu, std


class PPOLagCont:
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

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, cost_critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, 
                 critic_max_grad=2, actor_max_grad=2, max_std=0.3,
                 nu_lr=0.01, lambda_cost_init=0.0, target_cost=0.0,
                 lambda_min=1.0, lambda_max=10.0): # 新增 lambda_min/lambda_max 参数
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.cost_critic = CostNet(state_dim, hidden_dim).to(device) # 新增 CostNet
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=cost_critic_lr) # 新增优化器

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.critic_max_grad=critic_max_grad
        self.actor_max_grad=actor_max_grad
        self.max_std = max_std

        # PPO-Lagrangian 参数
        # 保存最小/最大约束
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        # 初始化并约束 lambda_cost 到 [lambda_min, lambda_max]
        init_val = float(lambda_cost_init)
        init_val = min(max(init_val, self.lambda_min), self.lambda_max)
        self.lambda_cost = torch.tensor(init_val, dtype=torch.float, device=device, requires_grad=True)
        self.nu_optimizer = torch.optim.Adam([self.lambda_cost], lr=nu_lr)
        self.target_cost = target_cost # 目标成本，通常设为0或者一个很小的值
    
    def set_learning_rate(self, actor_lr=None, critic_lr=None, cost_critic_lr=None, nu_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    
        if cost_critic_lr is not None:
            for param_group in self.cost_critic_optimizer.param_groups:
                param_group['lr'] = cost_critic_lr
        if nu_lr is not None:
            for param_group in self.nu_optimizer.param_groups:
                param_group['lr'] = nu_lr

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
    
    # 对外接口（保证numpy输入和numpy输出）
    def unscale_exec_to_normalized(self, a_exec, action_bounds):
        """Public wrapper: accepts numpy or torch, returns numpy on CPU."""
        a_exec_t = torch.as_tensor(a_exec, dtype=torch.float, device=self.device)
        action_bounds_t = torch.as_tensor(action_bounds, dtype=torch.float, device=self.device)
        a_norm_t = self._unscale_exec_to_normalized(a_exec_t, action_bounds_t)
        return a_norm_t.cpu().numpy()
    
    def take_action(self, state, action_bounds, explore=True):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 检查state中是否存在nan
        if torch.isnan(state).any() or torch.isinf(state).any():
            print('state', state)
        # 检查actor参数中是否存在nan
        check_weights_bias_nan(self.actor, "actor", "take action中")
        mu, std = self.actor(state, min_std=1e-6, max_std=self.max_std)
        # 检查mu, std是否含有nan
        if torch.isnan(mu).any() or torch.isnan(std).any() or torch.isinf(mu).any() or torch.isinf(std).any():
            print('mu', mu)
            print('std', std)
            raise ValueError(
                f"NaN/Inf detected in actor outputs: mu_nan={torch.isnan(mu).any().item()}, "
                f"std_nan={torch.isnan(std).any().item()}, mu_inf={torch.isinf(mu).any().item()}, "
                f"std_inf={torch.isinf(std).any().item()}"
            ) 

        dist = SquashedNormal(mu, std)
        if explore:
            a_norm, u = dist.sample()
        else:
            # use mean action: tanh(mu)
            u = mu
            a_norm = torch.tanh(u)

        a_exec = self._scale_action_to_exec(a_norm, action_bounds)
        return a_exec[0].cpu().detach().numpy().flatten(), u[0].cpu().detach().numpy().flatten()
    

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        """更新函数兼容以下几种调用方式：
        - 如果 action_bounds 是 None: 期望 transition_dict 中包含 'action_bounds'，其形状为 (N,2) 或每步 (amin,amax)
        - 如果 action_bounds 是标量/二元元组/数组：作为全局固定区间使用

        transition_dict 必须包含 keys: 'states','actions','rewards','next_states','dones', 'costs'
        当动作区间随步变化时，必须包含 'action_bounds' 与之对应。
        存储的 'actions' 应当是环境执行动作 (a_exec 未归一化）。
        """
        # print(np.array(transition_dict['states']))
        # print(np.array(transition_dict['costs']))

        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        u_s = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        costs = torch.tensor(np.array(transition_dict['costs']), dtype=torch.float).view(-1, 1).to(self.device) # 新增 costs
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        action_bounds = torch.tensor(np.array(transition_dict['action_bounds']), dtype=torch.float).to(self.device)

        # 计算 td_target, advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        # 计算 cost_td_target, cost_advantage
        cost_td_target = costs + self.gamma * self.cost_critic(next_states) * (1 - dones)
        cost_td_delta = cost_td_target - self.cost_critic(states)
        cost_advantage = compute_advantage(self.gamma, self.lmbda, cost_td_delta.cpu()).to(self.device)

        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            # advantage = torch.clamp((advantage - adv_mean) / (adv_std + 1e-8) -10.0, 10.0)
            cost_adv_mean, cost_adv_std = cost_advantage.detach().mean(), cost_advantage.detach().std(unbiased=False)
            cost_advantage = (cost_advantage - cost_adv_mean) / (cost_adv_std + 1e-8)
            # cost_advantage = torch.clamp((cost_advantage - cost_adv_mean) / (cost_adv_std + 1e-8), -10.0, 10.0)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        # 策略输出（未压缩的 mu,std）
        mu, std = self.actor(states, min_std=1e-6, max_std=self.max_std)
        # 构造 SquashedNormal 并计算 old_log_probs
        dist = SquashedNormal(mu.detach(), std.detach())

        u_old = u_s
        old_log_probs = dist.log_prob(0, u_old) # (N,1)
        # 提前在action_dim维度求和
        old_log_probs = dist.log_prob(0, u_old).sum(-1, keepdim=True)    # -> (N,1)

        if torch.isnan(old_log_probs).any():
            raise ValueError("old_log_probs 包含 NaN，检查 action_bounds 或 actions 的合法性")

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        post_clip_actor_grad = []
        post_clip_critic_grad = []
        critic_loss_list = []
        cost_critic_loss_list = [] # 新增
        entropy_list = []
        ratio_list = []
        cost_advantage_mean_list = []

        for _ in range(self.epochs):
            mu, std = self.actor(states, min_std=1e-6, max_std=self.max_std)
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values = self.critic(states)
            cost_critic_values = self.cost_critic(states) # 新增
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")
            if torch.isnan(cost_critic_values).any():
                raise ValueError("NaN in Cost Critic outputs in loop")

            # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")
            check_weights_bias_nan(self.cost_critic, "cost_critic", "update循环中") # 新增

            dist = SquashedNormal(mu, std)
            # 计算当前策略对历史执行动作的 log_prob（使用同一个 u_old）
            log_probs = dist.log_prob(0, u_old) # (N,1)

            # 提前在action_dim维度求和
            log_probs = dist.log_prob(0, u_old).sum(-1, keepdim=True)   # -> (N,1)

            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            # surr1 = ratio * advantage
            # calmp surr1
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # PPO-Lagrangian Actor 损失
            # objective = -torch.min(surr1, surr2).sum(-1).mean()
            # cost_objective = (ratio * cost_advantage).sum(-1).mean() # 注意，这里 cost_advantage 应该是非负的，或者直接使用成本

            # 另一种 PPO-Lagrangian 的 Actor 损失形式，使用 lambda_cost
            actor_loss_reward_term = -torch.min(surr1, surr2).sum(-1).mean()
            actor_loss_cost_term = self.lambda_cost * (ratio * cost_advantage).sum(-1).mean()

            ### test
            actor_loss_cost_term = 0.0  # << 临时置 0 做对照

            actor_loss = actor_loss_reward_term + actor_loss_cost_term - self.k_entropy * dist.entropy().mean()
            # google AI 建议不对 ratio*cost_advatage 进行裁剪，因此没有事先将两个advantage合并

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            cost_critic_loss = F.mse_loss(self.cost_critic(states), cost_td_target.detach()) # 新增

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.cost_critic_optimizer.zero_grad() # 新增

            actor_loss.backward()
            critic_loss.backward()
            cost_critic_loss.backward() # 新增
            
            # 裁剪前梯度
            post_clip_actor_grad.append(model_grad_norm(self.actor))
            post_clip_critic_grad.append(model_grad_norm(self.critic))  

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=self.critic_max_grad) # 新增

            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.cost_critic_optimizer.step() # 新增

            # # 更新 lambda_cost (Lagrangian 乘子)
            # # lambda_cost 的梯度是 (cost_mean - target_cost)
            # # 我们希望 lambda_cost 增加以惩罚更高的成本，所以是 (cost_mean - target_cost)
            # with torch.no_grad():
            #     current_cost_mean = (cost_advantage).mean().item()
            # # print(f"current_cost_mean: {current_cost_mean}, target_cost: {self.target_cost}")
            # lambda_grad = current_cost_mean - self.target_cost

            ###

            # 推荐：用实测 cost（或 episodic cost mean）来更新 lambda，而不是直接用 advantage 的均值
            # 这里使用 batch 的 costs.mean() 作为 λ 的信号（更稳妥）
            empirical_cost_mean = float(costs.mean().item())
            lambda_grad = empirical_cost_mean - self.target_cost
            
            # 由于 self.lambda_cost 是 requires_grad=True 的张量，可以直接对其进行梯度下降
            # (Lagrangian 乘子的更新是梯度上升，因为目标是最大化 Lagrangian 函数)
            # torch.nn.utils.clip_grad_norm_([self.lambda_cost], max_norm=0.1) # 可选：裁剪lambda_cost的梯度
            # self.lambda_cost.grad = torch.tensor(lambda_grad, device=self.device)
            # self.nu_optimizer.step()
            # self.nu_optimizer.zero_grad()
            
            # 更直接的更新方式，因为 lambda_cost 只有一个值
            self.lambda_cost.data.add_(self.nu_optimizer.param_groups[0]['lr'] * lambda_grad)
            # 限制 lambda_cost 在构造函数指定的范围内
            self.lambda_cost.data.clamp_(min=self.lambda_min, max=self.lambda_max)
                       

            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            cost_critic_loss_list.append(cost_critic_loss.detach().cpu().item()) # 新增
            entropy_list.append(dist.entropy().mean().detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())
            cost_advantage_mean_list.append(cost_advantage.mean().detach().cpu().item())
        
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.cost_critic_loss = np.mean(cost_critic_loss_list) # 新增
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.post_clip_critic_grad = np.mean(post_clip_critic_grad)
        self.post_clip_actor_grad = np.mean(post_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        self.cost_advantage_mean = np.mean(cost_advantage_mean_list) # 新增
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")
        check_weights_bias_nan(self.cost_critic, "cost_critic", "update后")

# 超参数
actor_lr = 1e-3 /10 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 10  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
cost_critic_lr = critic_lr
num_episodes = 800 # fixme 如果不限制最小方差，500 的时候会梯度爆炸, 限制后1000 也会爆炸
hidden_dims = [128]  # 128 fixme 层数大时actor梯度也会爆炸
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10 # fixme 4的时候也会梯度爆炸
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# env_name = 'testEnv'
env = Env()
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
state_dim = 1
action_dim = 1


agent = PPOLagCont(state_dim, hidden_dims, action_dim, actor_lr, critic_lr, cost_critic_lr,
                      lmbda, epochs, eps, gamma, device, lambda_min=0, lambda_max=10.0)

out_range_count = 0
return_list = []
clear_batch_flag=1
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
    for i_episode in range(int(num_episodes)):  # 每个1/10的训练轮次
        episode_return = 0
        if clear_batch_flag:
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'costs': [], 'dones': [], 'action_bounds': []}
            clear_batch_flag=0
        state = env.reset()
        done = False
        while not done:  # 每个训练回合
            # state_check=state
            # 1.执行动作得到环境反馈

            # print('state', state, flush=True)
            
            max_action_bound = 3

            # 栏杆
            max_action = max_action_bound # min(10-state[0], max_action_bound)
            min_action = -max_action_bound # max(-10-state[0], -max_action_bound)

            action_bound = [[min_action, max_action]]

            action, u = agent.take_action(state, action_bounds=action_bound, explore=True)

            next_state, reward, done, cost = env.step(action)  # pendulum中的action一定要是ndarray才能输入吗？
            # print(reward)
            transition_dict['states'].append(np.array(state, copy=True))
            transition_dict['actions'].append(u)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['costs'].append(cost)
            transition_dict['dones'].append(done)
            transition_dict['action_bounds'].append(action_bound)
            state = next_state
            episode_return += reward
        
        if env.out_range==1:
            out_range_count+=1
        return_list.append(episode_return)
        if 1: # len(transition_dict['dones'])>20: # 逐batch更新
            agent.update(transition_dict, adv_normed=0)
            clear_batch_flag=1
        if (i_episode + 1) >= 10:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                              'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)
    # return return_list

# return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

# %matplotlib inline

episodes_list = list(range(len(return_list)))
plt.figure()
plt.title("Lagrangian")
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))


mv_return = moving_average(return_list, 9)
plt.figure()
plt.title("Lagrangian")
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))

print("出界次数：", out_range_count)


# 测试回合
ups = []
track = []
downs = []
step = 0
episode_return = 0
state = env.reset()
done = False
while not done:  # 每个训练回合
    step += 1
    max_action_bound = 3

    # 栏杆
    max_action = max_action_bound # min(10-state[0], max_action_bound)
    min_action = -max_action_bound # max(-10-state[0], -max_action_bound)
    action_bound = [[min_action, max_action]]

    action, _ = agent.take_action(state, action_bounds=action_bound, explore=False)
    next_state, reward, done, cost = env.step(action)
    state = next_state
    episode_return += reward

    track.append((step, env.position[0]))
    ups.append((step, env.max_pos))
    downs.append((step, env.min_pos))

times, pos_list = zip(*track)
_, up_list = zip(*ups)
_, down_list = zip(*downs)

plt.figure()
plt.title("Lagrangian")
plt.plot(times, pos_list)
plt.plot(times, up_list)
plt.plot(times, down_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()