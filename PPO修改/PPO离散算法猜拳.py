# 相比书籍原版，新增了列表定义多层神经网络形状的方法

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import rl_utils
from tqdm import tqdm
from torch import nn

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
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(ValueNet, self).__init__()
        # self.prelu = torch.nn.PReLU()

        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

        # # 添加参数初始化
        # for layer in self.net:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_normal_(layer.weight, gain=0.01)
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)


class PolicyNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, action_dim)

        # # 固定神经网络初始化参数
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        x = self.net(x)
        return F.softmax(self.fc_out(x), dim=1)


class PPO_discrete:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, critic_max_grad=2, actor_max_grad=2):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dims, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.critic_max_grad=critic_max_grad
        self.actor_max_grad=actor_max_grad

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    # take action
    def take_action(self, state, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs) # 离散的输出为类别分布
        if explore:
            action = action_dist.sample()
        else:
            action = torch.argmax(probs)
        # 返回动作索引与对应的概率分布（numpy array）
        probs_np = probs.detach().cpu().numpy()[0].copy() # [0]是batch维度
        return action.item(), probs_np

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions 必须为 long 用于 gather 索引
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # 添加Actor NaN检查
        if torch.isnan(log_probs).any():
            raise ValueError("NaN in Actor outputs")
        # 添加Critic NaN检查
        critic_values = self.critic(states)
        if torch.isnan(critic_values).any():
            raise ValueError("NaN in Critic outputs")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            # advantage = torch.clamp((advantage - adv_mean) / (adv_std + 1e-8) -10.0, 10.0)
            
            # adv_mean, adv_std = advantage.mean(), advantage.std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        post_clip_actor_grad = []
        post_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 添加Actor NaN检查
            if torch.isnan(log_probs).any():
                raise ValueError("NaN in Actor outputs in loop")
            # 添加Critic NaN检查
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")

            log_probs = torch.log(self.actor(states).gather(1, actions)) # (N,1)
            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            probs = self.actor(states)
            action_dist = torch.distributions.Categorical(probs)

            entropy_factor = action_dist.entropy().mean() # torch.clamp(dist.entropy().mean(), -20, 70) # -20, 7 e^2

            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.k_entropy * entropy_factor # 标量

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                # critic_loss = F.mse_loss(self.critic(states), td_target.detach()) # 原有
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            # 裁剪前梯度
            post_clip_actor_grad.append(model_grad_norm(self.actor))
            post_clip_critic_grad.append(model_grad_norm(self.critic))  

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            entropy_list.append(action_dist.entropy().mean().detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())
        
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.post_clip_critic_grad = np.mean(post_clip_critic_grad)
        self.post_clip_actor_grad = np.mean(post_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")


# 加入新的 仿真环境：石头剪刀布，每回合10场。对手策略：初始随机，
# 若对手赢则保持不变；若对手输則按 rock->paper, paper->scissors, scissors->rock 旋转（opp = (opp+1)%3）
class RPS_Env(gym.Env):
    """
    Observation: 4-dim vector:
      - prev opponent move (one-hot, 3)
      - prev result scalar: win=1.0, tie=0.0, loss=-1.0 (1 dim)
    Action: 0=rock,1=paper,2=scissors
    Reward: +1 win, -1 loss, 0 tie
    Episode length: rounds_per_episode (内部计数，不作为观测)
    Opponent rule: initial random; if opponent loses (agent wins) -> rotate opp move (0->1->2->0),
                   if opponent wins or tie -> keep same.
    """
    metadata = {'render.modes': []}
    def __init__(self, rounds_per_episode=10, seed=None):
        super().__init__()
        self.rounds_per_episode = rounds_per_episode
        self.action_space = gym.spaces.Discrete(3)
        # prev opp one-hot (3) + prev result scalar (1)
        self.observation_space = gym.spaces.Box(low=np.array([0.0,0.0,0.0,-1.0], dtype=np.float32),
                                                high=np.array([1.0,1.0,1.0, 1.0], dtype=np.float32),
                                                shape=(4,), dtype=np.float32)
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.opp_move = 0          # opponent's current move (used this step)
        self.prev_opp_move = 0     # opponent move from previous step (for observation)
        self.prev_result = 0.0     # previous result scalar: 1.0 win, 0.0 tie, -1.0 loss
        self.round_idx = 0

    def seed(self, s=None):
        self._rng = np.random.RandomState(s)

    def reset(self):
        # initialize opponent current move
        self.opp_move = int(self._rng.randint(0, 3))
        # previous info for the first step should be random per要求
        self.prev_opp_move = int(self._rng.randint(0, 3))
        self.prev_result = float(self._rng.choice([1.0, 0.0, -1.0]))
        self.round_idx = 0
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        agent = int(action)
        opp = int(self.opp_move)            # opponent's action used this round (current)
        # outcome from agent perspective: (agent - opp) mod 3 -> 1 win, 2 lose, 0 tie
        diff = (agent - opp) % 3
        if diff == 1:
            reward = 1.0
            # opponent lost -> update opponent move for next round by rotating
            next_opp = (self.opp_move + 1) % 3
            result_scalar = 1.0   # agent win
        elif diff == 2:
            reward = -1.0
            next_opp = self.opp_move  # opponent won -> keep same
            result_scalar = -1.0  # agent loss
        else:
            reward = 0.0
            next_opp = self.opp_move  # tie -> keep same
            result_scalar = 0.0   # tie

        # set previous info for next state's observation:
        self.prev_opp_move = opp
        self.prev_result = result_scalar

        # update opponent for next round
        self.opp_move = next_opp

        self.round_idx += 1
        done = (self.round_idx >= self.rounds_per_episode)
        obs = self._get_obs()
        info = {'opp_move': int(self.opp_move), 'prev_result': float(self.prev_result)}
        return obs, float(reward), bool(done), info

    def _get_obs(self):
        # prev opponent one-hot
        onehot_prev_opp = np.zeros(3, dtype=np.float32)
        onehot_prev_opp[self.prev_opp_move] = 1.0
        # prev result scalar as one float
        res_scalar = np.array([self.prev_result], dtype=np.float32)
        return np.concatenate([onehot_prev_opp, res_scalar])

# 超参数
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 200 # 500
hidden_dim = [128]
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = RPS_Env(rounds_per_episode=10, seed=0)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]  # 应为4
action_dim = env.action_space.n  # 应为3
agent = PPO_discrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                     epochs, eps, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            done = False
            while not done:
                action, _ = agent.take_action(state, explore=1)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict, adv_normed=0)
            if (i_episode+1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

# 画图展示收敛过程
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
plt.show()


state = env.reset()
done = False
steps = 0
while not done:
    steps += 1
    action, _ = agent.take_action(state, explore=0)
    next_state, reward, done, _ = env.step(action)
    # 拆分 next_state：前3维为 one-hot（对手动作），第4维为上一步胜负标量
    onehot = np.asarray(next_state[:3], dtype=np.float32)
    # 如果 one-hot 全 0，返回 None，否则 argmax +1 -> 1,2,3
    if onehot.sum() == 0:
        opp_move = None
    else:
        opp_move = int(np.argmax(onehot))
    prev_result = float(next_state[3])
    state = next_state
    print("第", steps, "步 动作", action, "对手动作", opp_move, "当前步胜负", prev_result)

