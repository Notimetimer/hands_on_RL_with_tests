import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import collections, random
from tqdm import tqdm

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def compute_monte_carlo_reward(reward, dones, gamma):
    """
    reward: [T, 1] or [T]
    dones:    [T, 1] or [T]  (1 = terminal)
    """
    advantage = torch.zeros_like(reward)
    G = 0.0

    for t in reversed(range(len(reward))):
        mask = 1.0 - dones[t]
        gae = reward[t] + gamma * G * mask
        advantage[t] = G

    return advantage


# 定义策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
# 定义价值网络
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim): # 价值网络不看动作，但是有输出给策略网络
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state) # 策略？
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()# 从中随机抽取一个操作，可如果操作是连续的呢？
        return action.item()

    def update(self, transition_dict):
        # 将数值放入计算设备
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 蒙特卡洛累积回报
        Returns = compute_monte_carlo_reward(rewards, dones, gamma)

        with torch.no_grad():
            values = self.critic(states)
            advantage = Returns - values

        F_mask = torch.where(
            advantage > 0,
            torch.ones_like(advantage),
            0.1 * torch.ones_like(advantage)
        )

        # value filter：论文里的 F(s)
        F_mask = (advantage > 0).float().detach()
        
        # log π(a|s)
        probs_taken = self.actor(states).gather(1, actions)
        log_probs = torch.log(probs_taken + 1e-8)

        actor_loss = -torch.mean(
            log_probs * advantage.detach() * F_mask.detach()
        )
        
        critic_loss = F.mse_loss(
            self.critic(states),
            Returns.detach()
        )

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
buffer_size = 10000
replay_buffer = ReplayBuffer(buffer_size)
minimal_size = 500
batch_size = 64

env_name = 'CartPole-v1'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

if __name__=='__main__':
    return_list = []
    
    for i in range(10):  # 10
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action) # pendulum中的action一定要是ndarray才能输入吗？
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)


    # 画图展示训练过程
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()