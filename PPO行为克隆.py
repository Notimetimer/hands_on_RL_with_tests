# 游标训练环境
import random
import matplotlib.pyplot as plt
import matplotlib

class Env:
    def __init__(self):
        self.min_pos = -10
        self.max_pos = 10
        self.position = None

    def reset(self):
        self.position = np.array([random.randint(self.min_pos, self.max_pos)],dtype='float64')
        self.steps=0
        return self.get_obs()

    def get_obs(self):
        return self.position

    def step(self, move):
        self.position += move
        self.steps+=1
        done = self.get_done()
        reward = self.get_reward()
        # print(reward)
        return self.get_obs(), reward, done

    def get_done(self):
        done=0
        if self.position < self.min_pos or self.position > self.max_pos:
            done=1
        if self.steps>=20:
            done=1
        return done

    def get_reward(self):
        if self.min_pos <= self.position <= self.max_pos:
            return 1-np.linalg.norm(self.position-(self.max_pos+self.min_pos)/2)/10
        else:
            return -3
    def teach(self):
        teach_action_mu=np.clip((self.max_pos+self.min_pos)/2-self.position, -action_bound, action_bound)
        teach_action_std=action_bound*0.3
        return teach_action_mu, teach_action_std
    

import sys
import os
sys.path.append(os.path.abspath(".."))

from Algorithms.PPOcontinues import *

# 超参数
actor_lr = 8e-5 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 10  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
num_episodes = 200  # 2000
hidden_dims = [128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
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
action_bound = 3  # 动作最大值
actor = PolicyNetContinuous(state_dim, hidden_dims, action_dim)

loss_list=[]
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
    for i_episode in range(int(num_episodes)):  # 每个1/10的训练轮次
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        env.reset()
        state = env.get_obs()
        # print(state)
        
        done = False
        while not done:  # 每个训练回合
            mu_teach, std_teach = env.teach()
            
            # 接管
            action_exec = mu_teach

            next_state, reward, done = env.step(action_exec) 
            # print(reward)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action_exec) # action_exec action_raw
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
        
        # 有监督学习
        import torch.optim as optim
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(actor.parameters(), lr=0.01)

        states=transition_dict['states']
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float)
        teach_actions = transition_dict['actions']
        teach_actions=np.array(teach_actions)
        teach_actions = torch.tensor(teach_actions, dtype=torch.float)

        losses = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            mu, sigma = actor(states, action_bound=action_bound)
            loss = loss_func(mu, teach_actions)
            loss.backward()
            optimizer.step()
            losses+=loss.item()/epochs

        loss_list.append(loss.item())

        if (i_episode + 1) >= 10:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1),})
        pbar.update(1)
    # return return_list


episodes_list = list(range(len(loss_list)))
plt.figure()
plt.plot(episodes_list, loss_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))


mv_return = moving_average(loss_list, 9)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))



# 测试
env.reset()
state = env.get_obs()
print(state)

done = False
state_list=[state.copy()]

while not done:  # 每个训练回合
    
    # Actor动作
    state = np.array(state)
    state = torch.tensor(state, dtype=torch.float)
    mu, sigma = actor(state, action_bound=action_bound)
    action_exec = mu.detach().cpu().numpy()
    
    # # 教练动作
    # mu_teach, std_teach = env.teach()
    # action_exec = mu_teach
    
    next_state, reward, done = env.step(action_exec)
    state_list.append(next_state.copy())
    state = next_state

# print('state_list',state_list)

steps_list = list(range(len(state_list)))
plt.figure()
plt.plot(steps_list, state_list)
plt.xlabel('steps_list')
plt.ylabel('state')