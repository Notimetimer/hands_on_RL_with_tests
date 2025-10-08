# 相比书籍原版，新增了列表定义多层神经网络形状的方法

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import rl_utils
from tqdm import tqdm
from torch import nn

# 示例代码为PPO-截断的代码
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
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
        # self.prelu = torch.nn.PReLU()

        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
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
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
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
        # 返回 logits
        return self.fc_out(x)


class PPO_discrete:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def choose_action(self, state): # todo 加上shield输入
        logits = self.actor(state)                     
        probs = F.softmax(logits, dim=1)
        return probs
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # logits = self.actor(state)                     
        # probs = F.softmax(logits, dim=1)
        probs = self.choose_action(state)
        action_dist = torch.distributions.Categorical(probs) # 离散的输出为类别分布
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        
        # test 时间平滑奖励， 过于鸡肋只是绊脚石
        rewards_temp = np.array(transition_dict['rewards'])
        rewards_temp = moving_average(rewards_temp, window_size=3)
        rewards = torch.tensor(rewards_temp,  # transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        
        # 先从 actor 得到 logits，再做soft max以获得概率
        # logits_old = self.actor(states)
        # probs_old = F.softmax(logits_old, dim=1)
        probs_old = self.choose_action(states)
        old_log_probs = torch.log(probs_old.gather(1, actions)).detach()

        for _ in range(self.epochs):
            # logits = self.actor(states)
            # probs = F.softmax(logits, dim=1)
            probs = self.choose_action(states)
            log_probs = torch.log(probs.gather(1, actions)) # 从actor输出序列中的每一行抽取编号为actions同行的动作
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,      # torch.clamp(x,min,max)裁剪
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数，Actor的损失函数
            critic_loss = torch.mean( # PPO Critic损失函数
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

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
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
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