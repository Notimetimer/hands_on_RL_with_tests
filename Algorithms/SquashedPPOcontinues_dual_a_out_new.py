'''
注意：take_action 有两个输出，即actor原始输出u和tanh和scale后的action_exec
有监督预训练时经验池存 action_exec, 强化学习训练时经验池存 u
'''

from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# --- 保持 compute_advantage 函数，但根据传入参数数量切换逻辑 ---
def compute_advantage(gamma, lmbda, td_delta, dones, truncateds=None): # truncateds 默认为 None
    # 确保输入转为 numpy
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy() # 假设这里的 dones 是 terminateds (term)

    if truncateds is None:
        # --- 旧式/兼容模式：dones = term OR trunc ---
        # 此时，dones 就是 $\text{done}_t$
        advantage_list = []
        advantage = 0.0
        
        for delta, done in zip(td_delta[::-1], dones[::-1]):
            mask = 1.0 - done # $\text{Mask}_t = 1 - \text{done}_t$
            advantage = delta + gamma * lmbda * advantage * mask
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)
    
    else:
        # --- 新式模式：需要 term (dones) 和 trunc (truncateds) ---
        truncateds = truncateds.detach().cpu().numpy()
        terminateds = dones # $\text{term}_t$
        
        advantage_list = []
        advantage = 0.0
        
        for delta, term, trunc in zip(td_delta[::-1], terminateds[::-1], truncateds[::-1]):
            # 1. GAE 传递项的修正因子: $\gamma \lambda (1 - \text{term}_t) A_{t+1}$
            next_advantage_term = gamma * lmbda * advantage * (1.0 - term)
            
            # 2. 预估 A_t: $A'_t = \delta_t + \text{next\_advantage\_term}$
            advantage = delta + next_advantage_term
            
            # 3. 最终 A_t 屏蔽: $A_t = (1 - \text{trunc}_t) \cdot A'_t$
            advantage = advantage * (1.0 - trunc)
            
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
    """带 tanh 压缩的高斯分布。"""
    def __init__(self, mu, std, eps=1e-6):
        self.mu = mu
        self.std = std
        self.normal = Normal(mu, std)
        self.eps = eps
        self.mean = mu

    def sample(self):
        u = self.normal.rsample()
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        log_prob_u = self.normal.log_prob(u)
        jacobian = 0 
        return log_prob_u - jacobian

    def entropy(self):
        ent = self.normal.entropy().sum(-1)
        return ent


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
        self.fc_std = torch.nn.Linear(prev_size, action_dim)

    def forward(self, x, min_std=1e-7, max_std=5):
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        std = torch.clamp(std, min=min_std, max=max_std)
        return mu, std


class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
    
    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    def _scale_action_to_exec(self, a, action_bounds):
        action_bounds = torch.as_tensor(action_bounds, dtype=a.dtype, device=a.device)
        if action_bounds.dim() == 2:
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        return amin + (a + 1.0) * 0.5 * (amax - amin)

    def _unscale_exec_to_normalized(self, a_exec, action_bounds):
        action_bounds = torch.as_tensor(action_bounds, dtype=a_exec.dtype, device=a_exec.device)
        if action_bounds.dim() == 2:
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        span = (amax - amin)
        span = torch.where(span == 0, torch.tensor(1e-6, device=span.device, dtype=span.dtype), span)
        a = 2.0 * (a_exec - amin) / span - 1.0
        return a.clamp(-0.999999, 0.999999)

    def take_action(self, state, action_bounds, explore=True):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        dist = SquashedNormal(mu, std)
        if explore:
            a_norm, u = dist.sample()
        else:
            u = mu
            a_norm = torch.tanh(u)

        a_exec = self._scale_action_to_exec(a_norm, action_bounds)
        return a_exec[0].cpu().detach().numpy().flatten(), u[0].cpu().detach().numpy().flatten()
    
    def compute_gae_for_buffer(self, transition_dict):
        """
        [新增方法] 专门为单个子环境的 buffer 计算 GAE。
        该方法会计算 advantage 和 td_target，并将结果（list形式）存回 transition_dict。
        
        为什么需要这个？
        并行训练时，不同子环境的经验是独立的。如果合并后再算 GAE，
        backward 循环会将 Env2 的开头混入 Env1 的结尾。
        因此需要在合并前，对每个子环境单独调用此函数。
        """
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        use_new_gae = 'truncs' in transition_dict and len(transition_dict['truncs']) > 0

        with torch.no_grad():
            next_vals = self.critic(next_states)
            curr_vals = self.critic(states)

        if use_new_gae:
            # 新逻辑：区分 terminated (dones) 和 truncated (truncs)
            terminateds = dones
            truncateds = torch.tensor(np.array(transition_dict['truncs']), dtype=torch.float).view(-1, 1).to(self.device)
            
            # td_target 计算时，只有 terminated 才阻止 V(s_t+1) 的引导
            td_target = rewards + self.gamma * next_vals * (1 - terminateds)
            td_delta = td_target - curr_vals
            
            # GAE 计算需要同时传入 terminateds 和 truncateds
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta, terminateds, truncateds).to(self.device)
        else:
            # 旧逻辑：dones = terminated or truncated
            # td_target 计算时，任何 done 都会阻止 V(s_t+1) 的引导
            td_target = rewards + self.gamma * next_vals * (1 - dones)
            td_delta = td_target - curr_vals
            
            # GAE 计算只传入 dones
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta, dones).to(self.device)
        
        # 将结果转回 list 并存入 dict，方便后续合并
        transition_dict['advantages'] = advantage.cpu().numpy().flatten().tolist()
        # 注意 td_target 保持 (N, 1) 或 Flatten 都可以，Update 中会再次转 tensor
        transition_dict['td_targets'] = td_target.cpu().numpy().flatten().tolist()
        
        return transition_dict

    def update(self, transition_dict, advantage_norm=0, shuffled=0):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        u_s = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)

        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        action_bounds = torch.tensor(np.array(transition_dict['action_bounds']), dtype=torch.float).to(self.device)

        # --- 检查是否需要新式 GAE 计算 ---
        use_new_gae = 'truncs' in transition_dict and len(transition_dict['truncs']) > 0
        
        if use_new_gae: # 区分截断和终止
            if 'advantages' in transition_dict and 'td_targets' in transition_dict:
                # 使用预计算的值（适用于并行环境已做好拼接）
                advantage = torch.tensor(np.array(transition_dict['advantages']), dtype=torch.float).view(-1, 1).to(self.device)
                td_target = torch.tensor(np.array(transition_dict['td_targets']), dtype=torch.float).view(-1, 1).to(self.device)
            else:
                # 新式 GAE 流程：需要 $\text{term}$ ($\text{dones}$) 和 $\text{trunc}$ ($\text{truncs}$)
                # 我们假设 transition_dict['dones'] 此时只包含 $\text{terminated}$ 信息
                # 如果您的环境数据无法区分 $\text{term}$ 和 $\text{trunc}$，您需要手动解析数据。
                # 为简化，我们暂时**假设** 'dones' 字段在新模式下就是 $\text{terminateds}$
                terminateds = dones 
                truncateds = torch.tensor(np.array(transition_dict['truncs']), dtype=torch.float).view(-1, 1).to(self.device)
                
                # 1. 计算 td_target (使用 terminateds 屏蔽 V(s_{t+1}))
                td_target = rewards + self.gamma * self.critic(next_states) * (1 - terminateds)
                td_delta = td_target - self.critic(states)
                
                # 2. 计算 advantage (传入 terminateds 和 truncateds)
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta, terminateds, truncateds).to(self.device)
            
        else:  # 合并截断和终止
            if 'advantages' in transition_dict and 'td_targets' in transition_dict:
                # 使用预计算的值（适用于并行环境已做好拼接）
                advantage = torch.tensor(np.array(transition_dict['advantages']), dtype=torch.float).view(-1, 1).to(self.device)
                td_target = torch.tensor(np.array(transition_dict['td_targets']), dtype=torch.float).view(-1, 1).to(self.device)
            else:
                # 旧式/兼容流程：使用 $\text{done} = \text{term} \lor \text{trunc}$ (即原有的 $\text{dones}$)
                td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
                td_delta = td_target - self.critic(states)

                # 使用原有的 compute_advantage 签名（只传入 dones 作为旧的 done）
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta, dones).to(self.device)

        # 可选1：对 advantage 做归一化（默认关闭）
        if advantage_norm:
            adv_mean = advantage.mean()
            adv_std = advantage.std(unbiased=False)
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 可选2：按 episode 打乱顺序（shuffle episodes, 保持每个 episode 内部顺序）
        # shuffled=0/1 控制（默认 0）
        if shuffled:
            N = len(transition_dict['dones'])
            if N > 1:
                idx = torch.randperm(N, device=states.device)
                states = states[idx]
                u_s = u_s[idx]
                rewards = rewards[idx]
                next_states = next_states[idx]
                dones = dones[idx]
                action_bounds = action_bounds[idx]
                td_target = td_target[idx]
                # td_delta = td_delta[idx] # td_delta 没用到backward，可以不shuffle
                advantage = advantage[idx]

        # 策略输出（未压缩的 mu,std）
        mu, std = self.actor(states)
        dist = SquashedNormal(mu.detach(), std.detach())
        
        # 反算 u = atanh(a)
        u_old = u_s
        old_log_probs = dist.log_prob(0, u_old)

        if torch.isnan(old_log_probs).any():
            raise ValueError("old_log_probs 包含 NaN，检查 action_bounds 或 actions 的合法性")

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            dist = SquashedNormal(mu, std)
            # 计算当前策略对历史执行动作的 log_prob（使用同一个 u_old）
            log_probs = dist.log_prob(0, u_old)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            entropy_factor = torch.clamp(dist.entropy().mean(), -20, 7)
            actor_loss = -torch.min(surr1, surr2).sum(-1).mean() - self.k_entropy * entropy_factor

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

            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            entropy_list.append(dist.entropy().mean().detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())
        
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)

    # update_actor_supervised 和 update_critic_only 保持原样即可，略