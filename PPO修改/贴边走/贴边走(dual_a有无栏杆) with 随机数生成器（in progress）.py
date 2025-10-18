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

class Env:
    def __init__(self):
        self.bounce_back = None
        self.min_pos = -10
        self.max_pos = 10
        self.position = None
        self.out_range = None

    def reset(self):
        # MODIFICATION: Use python's global random for env reset
        # This part is separate from torch's randomness
        self.position = np.array([random.randint(self.min_pos, self.max_pos)],dtype='float64')
        self.steps=0
        self.out_range = 0
        return self.get_obs()

    def get_obs(self):
        return self.position.copy()

    def step(self, move):
        self.bounce_back = 0
        self.position += move
        if self.position < self.min_pos or self.position > self.max_pos:
            self.bounce_back=1

        self.steps+=1
        done = self.get_done()
        reward = self.get_reward()
        return self.get_obs(), reward, done

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
        if self.out_range:
            reward1 -= 50
        return reward1 - 1

# Helper functions (unchanged)
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

# Neural Network definitions (unchanged)
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
        self.fc_std = torch.nn.Linear(prev_size, action_dim)
    def forward(self, x, min_std=1e-6, max_std=0.3):
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        std = torch.clamp(std, min=min_std, max=max_std)
        return mu, std

# MODIFICATION: SquashedNormal class updated to use a generator for sampling
class SquashedNormal:
    """
    带 tanh 压缩的高斯分布。
    MODIFICATION: This class now accepts a torch.Generator to isolate randomness.
    """
    def __init__(self, mu, std, generator=None, eps=1e-6):
        self.mu = mu
        self.std = std
        self.generator = generator  # Store the generator
        self.eps = eps
        self.mean = mu

    def sample(self):
        # MODIFICATION: Manually perform the reparameterization trick using the generator
        # 1. Sample noise from a standard normal distribution using our specific generator
        epsilon = torch.randn(self.mu.shape,
                              generator=self.generator,
                              device=self.mu.device,
                              dtype=self.mu.dtype)
        # 2. Apply the reparameterization trick to get the pre-squashed action 'u'
        u = self.mu + self.std * epsilon
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        # MODIFICATION: Re-create a Normal distribution on-the-fly for calculations
        # This is necessary because we no longer store a persistent `self.normal` object
        normal = Normal(self.mu, self.std)
        log_prob_u = normal.log_prob(u)
        jacobian = 0
        return log_prob_u - jacobian

    def entropy(self):
        # MODIFICATION: Re-create a Normal distribution on-the-fly for calculations
        normal = Normal(self.mu, self.std)
        ent = normal.entropy().sum(-1)
        return ent

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01,
                 critic_max_grad=2, actor_max_grad=2, max_std=0.3, seed=0): # MODIFICATION: Added seed parameter

        self.device = device
        
        # MODIFICATION: Create and manage isolated random number generators
        # 1. A generator dedicated to initializing network weights
        self.init_generator = torch.Generator(device=self.device).manual_seed(seed)
        # 2. A generator dedicated to sampling actions during training
        self.sampling_generator = torch.Generator(device=self.device).manual_seed(seed)

        # Helper function to initialize network weights using our dedicated generator
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # Use a standard initialization method, but pass our specific generator
                # This ensures that initial weights are always the same for a given seed
                nn.init.xavier_uniform_(m.weight, generator=self.init_generator)
                m.bias.data.fill_(0.01)

        # MODIFICATION: Initialize networks and then apply our custom initialization
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        self.actor.apply(_init_weights)
        self.critic.apply(_init_weights)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.k_entropy = k_entropy
        self.critic_max_grad=critic_max_grad
        self.actor_max_grad=actor_max_grad
        self.max_std = max_std

    # ... (Helper methods like _scale_action_to_exec are unchanged) ...
    def _scale_action_to_exec(self, a, action_bounds):
        action_bounds = torch.as_tensor(action_bounds, dtype=a.dtype, device=a.device)
        if action_bounds.dim() == 2:
            amin, amax = action_bounds[:, 0], action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            amin, amax = action_bounds[:, :, 0], action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        return amin + (a + 1.0) * 0.5 * (amax - amin)

    def _unscale_exec_to_normalized(self, a_exec, action_bounds):
        action_bounds = torch.as_tensor(action_bounds, dtype=a_exec.dtype, device=a_exec.device)
        if action_bounds.dim() == 2:
            amin, amax = action_bounds[:, 0], action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            amin, amax = action_bounds[:, :, 0], action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        span = (amax - amin)
        span = torch.where(span == 0, torch.tensor(1e-6, device=span.device, dtype=span.dtype), span)
        a = 2.0 * (a_exec - amin) / span - 1.0
        return a.clamp(-0.999999, 0.999999)

    def take_action(self, state, action_bounds, explore=True):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        check_weights_bias_nan(self.actor, "actor", "take action中")
        mu, std = self.actor(state, min_std=1e-6, max_std=self.max_std)

        # MODIFICATION: Pass the dedicated sampling generator to the distribution
        dist = SquashedNormal(mu, std, generator=self.sampling_generator)

        if explore:
            a_norm, u = dist.sample()
        else:
            u = mu
            a_norm = torch.tanh(u)

        a_exec = self._scale_action_to_exec(a_norm, action_bounds)
        return a_exec[0].cpu().detach().numpy().flatten(), u[0].cpu().detach().numpy().flatten()
    
    # ... (The update method remains unchanged internally) ...
    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        u_s = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        v_pred_old = self.critic(states).detach()

        mu, std = self.actor(states, min_std=1e-6, max_std=self.max_std)
        dist_old = SquashedNormal(mu.detach(), std.detach()) # No generator needed here, just for calculations
        
        u_old = u_s
        old_log_probs = dist_old.log_prob(0, u_old).sum(-1, keepdim=True)

        if torch.isnan(old_log_probs).any():
            raise ValueError("old_log_probs 包含 NaN")

        for _ in range(self.epochs):
            mu, std = self.actor(states, min_std=1e-6, max_std=self.max_std)
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")

            dist = SquashedNormal(mu, std) # No generator needed here
            log_probs = dist.log_prob(0, u_old).sum(-1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            entropy_factor = dist.entropy().mean()
            actor_loss = -torch.min(surr1, surr2).mean() - self.k_entropy * entropy_factor

            if clip_vf:
                v_pred = self.critic(states)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

# --- Main Training Loop ---
# 超参数
actor_lr = 1e-3 / 10
critic_lr = actor_lr * 10
num_episodes = 800
hidden_dims = [128]
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 0 # MODIFICATION: Define a seed constant

# MODIFICATION: Set all relevant seeds for full reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

env = Env()
state_dim = 1
action_dim = 1

# MODIFICATION: Pass the seed to the agent
agent = PPOContinuous(state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device, seed=SEED)

out_range_count = 0
return_list = []
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:
    for i_episode in range(int(num_episodes)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
        state = env.reset()
        done = False
        while not done:
            max_action_bound = 3
            action_bound = [[-max_action_bound, max_action_bound]]
            action, u = agent.take_action(state, action_bounds=action_bound, explore=True)
            next_state, reward, done = env.step(action)
            
            transition_dict['states'].append(np.array(state, copy=True))
            transition_dict['actions'].append(u)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['action_bounds'].append(action_bound)
            state = next_state
            episode_return += reward
        
        if env.out_range==1:
            out_range_count+=1
        return_list.append(episode_return)
        
        agent.update(transition_dict, adv_normed=0)
        
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                              'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)

# Plotting results
episodes_list = list(range(len(return_list)))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episodes_list, return_list)
plt.title("Per-Episode Return")
plt.xlabel('Episodes')
plt.ylabel('Return')

mv_return = moving_average(return_list, 9)
plt.subplot(1, 2, 2)
plt.plot(episodes_list, mv_return)
plt.title("Moving Average Return")
plt.xlabel('Episodes')
plt.ylabel('Smoothed Return')

print(f"出界次数： {out_range_count} / {num_episodes} episodes")

# Test trajectory plotting
ups, track, downs = [], [], []
step = 0
state = env.reset()
done = False
while not done:
    step += 1
    max_action_bound = 3
    action_bound = [[-max_action_bound, max_action_bound]]
    action, _ = agent.take_action(state, action_bounds=action_bound, explore=False)
    next_state, reward, done = env.step(action)
    state = next_state
    track.append((step, env.position[0]))
    ups.append((step, env.max_pos))
    downs.append((step, env.min_pos))

plt.figure()
if track: # Ensure track is not empty
    times, pos_list = zip(*track)
    _, up_list = zip(*ups)
    _, down_list = zip(*downs)
    plt.plot(times, pos_list, label='Agent Position')
    plt.plot(times, up_list, label='Upper Bound', linestyle='--', color='r')
    plt.plot(times, down_list, label='Lower Bound', linestyle='--', color='r')
    plt.title("Test Trajectory (No Exploration)")
    plt.xlabel('Steps')
    plt.ylabel('Position')
    plt.legend()

plt.tight_layout()
plt.show()