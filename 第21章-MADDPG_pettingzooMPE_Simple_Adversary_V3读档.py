# Import libraries
import random
from collections import namedtuple, deque
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from pettingzoo.mpe import simple_adversary_v3
from psutil import virtual_memory

# 回放缓存 Replay Buffer，大小固定，并在队列满时自动丢弃最旧的数据
"""Transition is a namedtuple used to store a transition.

The structure of Transition looks like this:
    (state, action, reward, next_state, done)
"""
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """ReplayBuffer is a class used to achieve experience replay.

    It's a buffer composed of a deque with a certain capacity. When the deque is full, it will automatically remove the oldest transition in the buffer.

    Attributes:
        _storage: The buffer to store the transitions.

    Examples:
        A replay buffer structure looks like below:
        [
            (state_1, action_1, reward_1, next_state_1, done_1),
            (state_2, action_2, reward_2, next_state_2, done_2),
            ...
            (state_n, action_n, reward_n, next_state_n, done_n),
        ]
        Each tuple is a transition.
    """

    def __init__(self, capacity=100_000):
        """Initial a replay buffer with capacity.

        Args:
            capacity: Max length of the buffer.
        """
        self._storage = deque([], maxlen=capacity) # 从list创建deque

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        Args:
            state:          The state of the agents.
            action:         The action of the agents.
            reward:         The reward of the agents.
            next_state:     The next state of the agents.
            done:           The termination of the agents.

        Returns: None

        """
        transition = Transition(state, action, reward, next_state, done)
        self._storage.append(transition) # 加入状态转移元组

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: The number of transitions that we want to sample from the buffer.

        Returns: A batch of transitions.

        Example:
            Assuming that batch_size=3, we'll randomly sample 3 transitions from the buffer:
                state_batch = (state_1, state_2, state_3)
                action_batch = (action_1, action_2, action_3)
                reward_batch = (reward_1, reward_2, reward_3)
                next_state_batch = (next_state_1, next_state_2, next_state_3)
                done_batch = (done_1, done_2, done_3)
        """
        transitions = random.sample(self._storage, batch_size)
        # 这部分仅在训练时候用到
        # print('hello')
        # print(transitions)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions) # 通过zip(*)转置提取各项的元组
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """Buffer的长度"""
        return len(self._storage)


# 2. Simple Neural Network
class SimpleNet(nn.Module):
    """SimpleNet is a 3-layer simple neural network.

    It's used to approximate the policy functions and the value functions.
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        """Initial a simple network for an actor or a critic.

        Args:
            input_dim: The input dimension of the network.
            output_dim: The output dimension of the network.
            hidden_dim: The hidden layer dimension of the network.
        """
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 这里因为输出的动作空间是离散的，所以用独热编码进行表示
# 3. Gumbel Softmax Trick
def trans2onehot(logits, eps=0.01):
    """Transform the output of the actor network to a one-hot vector.
    Args:
        logits:     The output value of the actor network.
        eps:        The epsilon parameter in the epsilon-greedy algorithm used to choose an action.
    Returns: An action in one-hot vector form.
    """
    # Generates a one-hot vector form of the action selected by the actor network.
    best_action = (logits == logits.max(1, keepdim=True)[0]).float() # 贪婪行为，为啥要转为float()？
    # Generate a one-hot vector form of a random action.
    size, num_actions = logits.shape # 后者是行为数，前者是啥？智能体数吗？
    random_index = np.random.choice(range(num_actions), size=size) # 随机从num_actions中抽取size个动作放入一维数组random_index
    random_actions = torch.eye(num_actions)[[random_index]].to(logits.device) # 从对角矩阵中按random_index序列抽取行并自上而下堆叠
    # Select an action using the epsilon-greedy algorithm.
    random_mask = torch.rand(size, device=logits.device) <= eps # 有的随机行动，有的智能体不随机行动
    selected_action = torch.where(random_mask.view(-1, 1), random_actions, best_action) # torch.where(条件，是的行动，否的行动),
    # .view(-1, 1)给tensor变形，行数随动，列数为1
    return selected_action

def sample_gumbel(shape, eps=1e-20, tensor_type=torch.float32):
    """采样自Gumbel(0,1)的噪声，Sample a Gumbel noise from the Gumbel(0,1) distribution."""
    U = torch.rand(shape, dtype=tensor_type, requires_grad=False)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    return gumbel_noise

def gumbel_softmax_sample(logits, temperature):
    """Sample from the Gumbel-Softmax distribution."""
    gumbel_noise = sample_gumbel(logits.shape, tensor_type=logits.dtype).to(logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=1) # 这里的dim有些反直觉，0表示按列归一化（列和为1），1表示按行归一化（行和为1）

def gumbel_softmax(logits, temperature=1.0):
    """Sample from Gumbel-Softmax distribution and discretize it.
    By returning a one-hot vector of y_hard, but with a gradient of y, we can get both a discrete action interacting with the environment and a correct inverse gradient.
    """
    y = gumbel_softmax_sample(logits, temperature) # Gumbel softmax采样公式
    y_hard = trans2onehot(y) # 已选动作的独热编码
    y = (y_hard.to(logits.device) - y).detach() + y # 奇怪的写法，后面会对它求导吗？ todo
    return y

# 4. DDPG
class DDPG:
    """The DDPG Algorithm.
    Attributes:
        actor:              The Actor (Policy Network).
        target_actor:       The Target Actor (Target Policy Network).
        critic:             The Critic (value Network).
        target_critic:      The Target Critic (Target Value Network).
        actor_optimizer:    The optimizer of Actor.
        critic_optimizer:   The optimizer of Critic.
    """

    def __init__(self, state_dim, action_dim, critic_dim, hidden_dim, actor_lr, critic_lr, device):
        """Initialize a DDPG instance for an agent.

        Args:
            state_dim:      The dimension of the state, which is also the input dimension of Actor and a part of Critics' input.
            action_dim:     The dimension of the action, which is also a part of Critics' input.
            critic_dim:     The dimension of the Critics' input (Critic Dimension = State Dimensions + Action Dimension).
            hidden_dim:     The dimension of the hidden layer of the networks.
            actor_lr:       The learning rate for the Actor.
            critic_lr:      The learning rate for the Critic.
            device:         The device to compute.
        """
        # Set Actor with Target Network
        self.actor = SimpleNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = SimpleNet(state_dim, action_dim, hidden_dim).to(device)
        # Set Critic with Target Network
        self.critic = SimpleNet(critic_dim, 1, hidden_dim).to(device) # 只输出一个值的评价
        self.target_critic = SimpleNet(critic_dim, 1, hidden_dim).to(device)
        # Load parameters from Actor and Critic to their target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, explore=False):
        """Take action from the Actor (Policy Network).

        1. State -> the Actor -> Action Value
        2. Choose action according to the value of explore.
            - explore == True: Choose an action based on the gumbel trick.
            - explore == False: Choose the action from the Actor, and transform the value to a one-hot vector.

        Args:
            state:      The partial observation of the agent.
            explore:    The strategy to choose action.

        Returns: The action that the Actor has chosen.

        """
        # Choose an action from actor network (deterministic policy network).
        action = self.actor(state)
        # Exploration and Exploitation
        if explore:
            action = gumbel_softmax(action) # 探索过程需要考虑可导性
        else:
            action = trans2onehot(action) # 利用过程不反向传播，不考虑可导性
        # TODO: Find out why the action need to be transferred to the CPU
        # return action.detach().cpu().numpy()[0]
        # todo：为啥返回动作值也需要detach？
        return action.detach().cpu().numpy()[0]

    @staticmethod # 类静态方法
    def soft_update(net, target_net, tau):
        """Soft update function, which is used to update the parameters in the target network.

        Args:
            net:            The original network.
            target_net:     The target network.
            tau:            Soft update parameter.

        Returns: None

        """
        # 用主网络参数更新目标网络 Update target network's parameters using soft update strategy
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# 5. MADDPG
# The multi-agent DDPG algorithm literally means an algorithm that implements a DDPG for each agent.
# All agents share a centralised critic network that simultaneously guides each agent's actor network during the training process,
# and each agent's actor network acts completely independently, i.e. decentralised execution.
# The main detail of the MADDPG algorithm is that each agent is trained with the Actor-Critic method, but unlike the traditional single agent
class MADDPG:
    """The Multi-Agent DDPG Algorithm.

    1. The instance of MADDPG is the Center Controller in the algorithm.
    2. The instance of MADDPG contains a list of DDPG instances, which are corresponded with the agents in the environment one by one.

    Attributes:
        agents:     A list of DDPG instances, which are corresponded with the agents in the environment one by one.
        device:     The device to compute.
        gamma:      The gamma parameter in TD target.
        tau:        The tau parameter for soft update.
        critic_criterion: The loss function for the Critic networks.
    """

    def __init__(self, state_dims, action_dims, critic_dim, hidden_dim, actor_lr, critic_lr, device, gamma, tau):
        """Initialize a MADDPG instance as the Center Controller.

        Args:
            state_dims: A list of dimensions of each agent's observation.
            action_dims: A list of dimensions of each agent's action.
            critic_dim: The dimension of the Critic networks' input.
            hidden_dim: The dimension of the networks' hidden layers.
            actor_lr: The learning rate for the Actor.
            critic_lr: The learning rate for the Critic.
            device: The device to compute.
            gamma: The gamma parameter in TD target.
            tau: The tau parameter for soft update.
        """
        # TODO: Should we use dict to combine the DDPG instance with agents?
        self.agents = [
            # 对象组合，这里不是继承关系
            DDPG(state_dim, action_dim, critic_dim, hidden_dim, actor_lr, critic_lr, device)
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = nn.MSELoss() # 看起来真的像类组合

    @property # 将函数转为类属性
    def policies(self):
        """A list of Actors for the agents."""
        return [agent.actor for agent in self.agents]

    @property # 不一定要紧跟着setter和deleter
    def target_policies(self):
        """A list of Target Actors for the agents."""
        return [agent.target_actor for agent in self.agents]

    def take_action(self, states, explore):
        """Take actions from the Actors (Policy Networks).

        Args:
            states: A list of observations from all the agents.
            explore: The strategy to choose action (Exploration or Exploitation).

        Returns: A list of actions in one-hot vector form.

        """
        states = [torch.tensor(np.array([state]), dtype=torch.float, device=self.device) for state in states]
        return [agent.take_action(state, explore) for agent, state in zip(self.agents, states)]

    def update(self, sample, agent_idx):
        """Update parameters for the agent whose index is agent_idx.
        Args:
            sample: A batch of transitions used to update all the parameters.
            agent_idx: The index of the agent whose Actor and Critic would be updated in this function.
        Returns: None
        Process:
            Parse Data from Sample to Observation, Action, Reward, Next Observation and Done Tag.
            Set up Current Agent
            Update Current Critic (Value Network) with TD Algorithm:
                1. Initialize the Gradient to Zero.
                2. Build a Tensor Contains All Target Actions Using Target Actor Networks and Next Observations.
                3. Calculate Target Critic Value:
                    - Combine Next Observation and Target Action in One To One Correspondence.
                    - Calculate Target Critic Value (TD Target).
                4. Calculate Critic Value:
                    - Combine Observation and Action in One To One Correspondence.
                    - Calculate Critic Value (TD Target).
                5. Calculate Critic's Loss Using MSELoss Function
                6. Backward Propagation.
                7. Update Parameters with Gradient Descent.
            Update Current Actor (Policy Network) with Deterministic Policy Gradient:
                1. Initialize the Gradient to Zero.
                2. Get Current Actors' Action in the Shape of One-Hot Vector.
                    - Get Current Actor Network Output with Current Observation.
                    - Transform the Output into a One-Hot Action Vector.
                3. Build the Input of Current Actor's Value Function.
                    - Build a tensor that contains all the actions.
                    - Combine Observations and Actions in One To One Correspondence.
                4. Calculate Actor's Loss Using Critic Network (Value Function) Output.
                5. Backward Propagation.
                6. Update Parameters with Gradient Descent.

        TODO:
        1. Why there is a term, "(1 - dones[agent_idx].view(-1, 1))", during calculation of TD target?
            To take termination in to consideration (To be verified).
        2. Why there is a term, "(current_actor_action_value ** 2).mean() * 1e-3", during calculation of Actor Loss?
            To make the output of the trained Actor more stable and smooth with this regular term (To be verified).
        """
        states, actions, rewards, next_states, dones = sample
        current_agent = self.agents[agent_idx]
        # Update Current Critic (Value Network) with TD Algorithm
        current_agent.critic_optimizer.zero_grad()
        # Here is the step to choose the actions from the actor network and we have two strategies.
        # Option 1: Use the target network strategy.
        target_action = [
            trans2onehot(_target_policy(_next_obs))
            for _target_policy, _next_obs in zip(self.target_policies, next_states)
        ]
        # Option 2: Use the double DQN strategy.
        # target_action = [
        #     # Choose actions from the original network!!!
        #     trans2onehot(policy(_next_obs))
        #     for policy, _next_obs in zip(self.policies, next_states)
        # ]
        target_critic_input = torch.cat((*next_states, *target_action), dim=1)
        target_critic_value = (rewards[agent_idx].view(-1, 1) +
                               self.gamma * current_agent.target_critic(target_critic_input) *
                               (1 - dones[agent_idx].view(-1, 1)))
        critic_input = torch.cat((*states, *actions), dim=1)
        critic_value = current_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        current_agent.critic_optimizer.step()
        # Update Current Actor (Policy Network) with Deep Deterministic Policy Gradient
        current_agent.actor_optimizer.zero_grad()
        current_actor_action_value = current_agent.actor(states[agent_idx])
        current_actor_action_onehot = gumbel_softmax(current_actor_action_value)
        all_actor_actions = [
            current_actor_action_onehot if i == agent_idx else trans2onehot(_policy(_state))
            for i, (_policy, _state) in enumerate(zip(self.policies, states))
        ]
        current_critic_input = torch.cat((*states, *all_actor_actions), dim=1)
        actor_loss = (-current_agent.critic(current_critic_input).mean() +
                      (current_actor_action_value ** 2).mean() * 1e-3)
        actor_loss.backward()
        current_agent.actor_optimizer.step()

    def update_all_targets_params(self):
        """Update all Target network's parameters using soft update method."""
        for agent in self.agents:
            agent.soft_update(agent.actor, agent.target_actor, self.tau)
            agent.soft_update(agent.critic, agent.target_critic, self.tau)


# 6. Evaluate Function
# The function takes a MADDPG agent as input and evaluates its performance over a specified number of episodes and episode length.

def evaluate(num_agents, maddpg, num_episode=10, len_episode=25):
    """Evaluate the strategies for learning, and no exploration is undertaken at this time.

    Args:
        num_agents: The number of the agents.
        maddpg: The Center Controller.
        num_episode: The number of episodes.
        len_episode: The length of each episode.

    Returns: Returns list.

    """
    env = simple_adversary_v3.parallel_env(N=num_agents, max_cycles=25, continuous_actions=False)
    env.reset()
    returns = np.zeros(env.max_num_agents)
    # Create an array returns of zeros with a length equal to the number of agents in the environment.
    # This array will store the cumulative returns for each agent.
    for episode in range(num_episode):
        states_dict, rewards_dict = env.reset()
        states = [state for state in states_dict.values()]
        for episode_step in range(len_episode):
            actions = maddpg.take_action(states, explore=False)
            # Take actions using the MADDPG agent (maddpg.take_action) based on the current states. 
            actions_SN = [np.argmax(onehot) for onehot in actions]
            actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
            next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(actions_dict)
            # Take a step in the environment by passing the actions dictionary to env.step.  
            rewards = [reward for reward in rewards_dict.values()]
            next_states = [next_state for next_state in next_states_dict.values()]
            states = next_states
            rewards = np.array(rewards)
            returns += rewards / num_episode
    env.close()
    return returns.tolist()


# 7. Auxiliary Function
# The function rearranges the elements in the input list x and converts them into PyTorch tensors on the specified device.
def sample_rearrange(x, device):
    """Rearrange the transition in the sample."""
    rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
    return [torch.FloatTensor(np.vstack(attribute)).to(device) for attribute in rearranged]
    # np.vstack is used to vertically stack the elements in each sublist before converting them to a tensor.
    # The resulting tensors are then converted to the appropriate device (GPU).

# 8. Set up Parameters and Initialize Variables
# It is time to initialize the necessary components and parameters for training an MADDPG agent in a multi-agent environment using the specified settings.

# Set up Parameters
NUM_EPISODES = 5000 # 5000
LEN_EPISODES = 20 # 25  # The maximum length of each episode is 25
BUFFER_SIZE = 100_000
HIDDEN_DIM = 64  # denotes the dimension of the hidden layers in the actor and critic networks.
ACTOR_LR = 1e-2
CRITIC_LR = 1e-2
GAMMA = 0.95  # determines the discount factor for future rewards.
TAU = 1e-2  # sets the soft update coefficient for target network updates.
BATCH_SIZE = 1024
UPDATE_INTERVAL = 100  # determines the number of steps before performing an update on the networks.
MINIMAL_SIZE = 4000  # is a minimum threshold for the replay buffer size before starting the training.
# Initialize Environment
env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=False)
env.reset()
# Initialize Replay Buffer
buffer = ReplayBuffer(BUFFER_SIZE)  # stores experiences from the environment.
# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Return List
return_list = []
# Initialize Total Step
total_step = 0
# Initialize MADDPG
# 1. Get State and Action Dimensions from the Environment
state_dims = [env.observation_space(env.agents[i]).shape[0] for i in range(env.num_agents)]
action_dims = [env.action_space(env.agents[i]).n for i in range(env.num_agents)]
critic_dim = sum(state_dims) + sum(action_dims)  # calculates the total dimension of the critic network's input
# 2. Create Center Controller
maddpg = MADDPG(state_dims, action_dims, critic_dim, HIDDEN_DIM, ACTOR_LR, CRITIC_LR, device, GAMMA, TAU)
# maddpg is created as an instance of the MADDPG class using the specified dimensions, learning rates and other parameters.

#
# # 9. Train
# for episode in range(NUM_EPISODES):
#     # Reset the Environment
#     states_dict, _ = env.reset()
#     states = [state for state in states_dict.values()]
#     # Start a New Game
#     for episode_step in range(LEN_EPISODES):
#         # Initial States and Actions
#         actions = maddpg.take_action(states, explore=True)
#         actions_SN = [np.argmax(onehot) for onehot in actions]
#         actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
#         # Step
#         next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(actions_dict)
#         # Add to buffer
#         rewards = [reward for reward in rewards_dict.values()]
#         next_states = [next_state for next_state in next_states_dict.values()]
#         terminations = [termination for termination in terminations_dict.values()]
#         buffer.add(states, actions, rewards, next_states, terminations)
#         # Update States
#         states = next_states
#         # Count += 1
#         total_step += 1
#         # When the replay buffer reaches a certain size and the number of steps reaches the specified update interval
#         # 1. Sample from the replay buffer.
#         # 2. Update Actors and Critics.
#         # 3. Update Target networks' parameters.
#         if len(buffer) >= MINIMAL_SIZE and total_step % UPDATE_INTERVAL == 0:
#             sample = buffer.sample(BATCH_SIZE)
#             sample = [sample_rearrange(x, device) for x in sample]
#             # Update Actors and Critics
#             for agent_idx in range(env.max_num_agents):
#                 maddpg.update(sample, agent_idx)
#             # Update Target Parameters
#             maddpg.update_all_targets_params()
#     # After every 100 rounds, use the evaluate function to evaluate the trained agents, get the reward list ep_returns, and add it to return_list.
#     if (episode + 1) % 100 == 0:
#         episodes_returns = evaluate(env.max_num_agents - 1, maddpg, num_episode=100)
#         return_list.append(episodes_returns)
#         print(f"Episode: {episode + 1}, {episodes_returns}")
# # Close the Environment
# env.close()
# return_array = np.array(return_list)

# 读取智能体存档
maddpg.agents[0].actor.load_state_dict(torch.load('maddpgzoo0.pth'))
maddpg.agents[1].actor.load_state_dict(torch.load('maddpgzoo1.pth'))
maddpg.agents[2].actor.load_state_dict(torch.load('maddpgzoo2.pth'))

# 可视化运行
import time
env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=False, render_mode="human" )
# Reset the Environment
states_dict, _ = env.reset()
states = [state for state in states_dict.values()]
# agent读档：


# Start a New Game
for episode_step in range(LEN_EPISODES):
    # Initial States and Actions
    actions = maddpg.take_action(states, explore=False)
    actions_SN = [np.argmax(onehot) for onehot in actions]
    actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
    # Step
    next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(actions_dict)
    # Add to buffer
    rewards = [reward for reward in rewards_dict.values()]
    next_states = [next_state for next_state in next_states_dict.values()]
    terminations = [termination for termination in terminations_dict.values()]
    buffer.add(states, actions, rewards, next_states, terminations)
    # Update States
    states = next_states
    time.sleep(0.08)
# Close the Environment
env.close()


