# pettingzoo==1.23.1
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils
import sys
import gym
import pettingzoo
from pettingzoo.mpe import simple_adversary_v3
import time

# num_episodes = 5000
episode_length = 25  # 每条序列的最大长度
buffer_size = 100000
hidden_dim = 64
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100
minimal_size = 4000
env_id = "simple_adversary"

# ############定义神经网络#######################

# def make_env(scenario_name):
#     # 从环境文件脚本中创建环境
#     scenario = scenarios.load(scenario_name + ".py").Scenario()
#     world = scenario.make_world()
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
#                         scenario.observation)
#     return env

def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dim_int, critic_input_dim, gamma, tau):
    # def __init__(self, env, device, actor_lr, critic_lr, hidden_dim_1,
    #              state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                # DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     # hidden_dim_1, actor_lr, critic_lr, device))
                DDPG(state_dims[i], action_dim_int[i], critic_input_dim,
                 hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    # def take_action(self, states, explore):
    #     states = [
    #         torch.tensor([states[i]], dtype=torch.float, device=self.device)
    #         for i in range(len(env.agents))
    #     ]
    #     return [
    #         agent.take_action(state, explore)
    #         for agent, state in zip(self.agents, states)
    #     ]
    # 这里修改为单个agent执行动作
    def take_action(self, agent, observation, explore):
        state_temp=torch.tensor(observation, dtype=torch.float, device=self.device)
        if agent=='adversary_0':
            return self.agents[0].take_action(state_temp, explore)
        if agent=='agent_0':
            return self.agents[1].take_action(state_temp, explore)
        if agent=='agent_1':
            return self.agents[2].take_action(state_temp, explore)
        # states = [
        #     torch.tensor([states[i]], dtype=torch.float, device=self.device)
        #     for i in range(len(env.agents))
        # ]
        # return [
        #     agent.take_action(state, explore)
        #     for agent, state in zip(self.agents, states)
        # ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(
                target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

# ###########结束神经网络定义####################

# 初始化训练环境和智能体
env = simple_adversary_v3.env(render_mode="human")  # human, rgb_array
env.reset(seed=42)
total_step = 45
count = 0

state_dims = []
action_dims = []
for action_space in env.action_spaces.values():
    action_dims.append(action_space)
for state_space in env.observation_spaces.values():
    state_dims.append(state_space.shape[0])
# critic_input_dim = sum(state_dims) + sum(action_dims)   # debug
temp=0
action_dim_int=[]
for m in range( len(action_dims) ):
    temp+=action_dims[m].n
    action_dim_int.append(action_dims[m].n)
critic_input_dim = int(sum(state_dims) + temp)

# maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim_1, state_dims,
#                 action_dims, critic_input_dim, gamma, tau)
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dim_int, critic_input_dim, gamma, tau)

return_list = []  # 记录每一轮的回报（return）
total_step = 0.95

# 读取智能体存档
maddpg.agents[0].actor.load_state_dict(torch.load('maddpgzoo0.pth'))
maddpg.agents[1].actor.load_state_dict(torch.load('maddpgzoo1.pth'))
maddpg.agents[2].actor.load_state_dict(torch.load('maddpgzoo2.pth'))

# # 运行智能体，在修改

state = env.reset()

# AEC API 写法（Debugging）
for agent in env.agent_iter():
    print(env.agent_selection)
    observation, reward, termination, truncation, info = env.last()
    if env.agent_selection == 'adversary_0':
        count += 1
        # time.sleep(0.05)
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        # 这里变化挺大，首先状态空间从包含了三个agent的state到各个智能体的observation分开，原版take_action所调用的agent.take_action也不一定有效
        actions = maddpg.take_action(agent, observation, explore=False)
        action = env.action_space(agent).sample()

    env.step(action)
    # time.sleep(0.05)
env.close()
# '''