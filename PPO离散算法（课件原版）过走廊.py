import random
import gym
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import norm
from math import *
import gym
import numpy as np
from gym import spaces
import math
import keyboard

matplotlib.use('Qt5Agg')  # 使用Qt5作为后端
# from gym import spaces
from torch.distributions import Normal

stop_flag = False  # 控制退出的全局变量

# import rl_utils
dt = 0.1
# dof = 3
actor_lr = 1e-3 /10
critic_lr = 1e-2 /10
num_episodes = 420*5  # 800 400
hidden_dim = 128

gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(0)

def stop_loop():
    global stop_flag
    stop_flag = True
    print("检测到 'esc' 键，准备退出所有循环！")  # 也可以用 pause、scroll lock等几乎无用的按键


keyboard.add_hotkey('esc', stop_loop, suppress=False, trigger_on_release=False)  # 监听 'esc' 键，按下时执行 `stop_loop()`


def sub_of_radian(input1, input2):
    # 弧度减法
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


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
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class PPO_discrete:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
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

    def take_action(self, state, explore=True, epsilon=0.0):
        if not explore:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action = torch.argmax(probs).item()
            return action

        # 随机选择动作, 对于多个维度的是离散控制似乎是必须的
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(action_list))

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)  # 离散的输出为类别分布
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
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
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
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

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) # np.insert(a, 0, 0) 在数组 a 的位置0（第二项）插入一个 0(第三项），cumsum是计算累积和
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))



# dt = 0.1
corridor_length = 900  # 900
corridor_y_half_width = 30
corridor_z_half_width = 30
agent_initial_speed = 30
max_angle_diff = 10 * (math.pi / 180)  # 最大偏移角度，弧度制


class testEnv(gym.Env):
    def __init__(self):
        super(testEnv, self).__init__()
        # 观测空间：俯仰角、航向角、归一化威胁方向和强度（假设威胁信息有 4 个维度）
        # np.array([rel_heading_angle, rel_pitch_angle, rel_psi_threat_, rel_theta_threat_, threat_strength], dtype=np.float32)
        self.target_pos_ = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.state = None
        self.done = False
        self.corridor_direction = None
        self.corridor_end = None
        self.agent_pos = None
        self.agent_vel = None
        self.t = None

    def reset(self, train=True):
        self.t = 0
        # 随机生成走廊方向
        self.corridor_direction = self._generate_random_direction()  # , direction=np.array([0, 0, -1]) 1, 0, 0 ...  0, 0, 1
        # 走廊方向角
        self.psi_corridor = np.arctan2(self.corridor_direction[2],
                                       self.corridor_direction[0])
        # 计算走廊终点
        self.corridor_end = np.array([0.0, 0.0, 0.0]) + self.corridor_direction * corridor_length
        self.target_pos_ = self.corridor_end
        # 智能体出生点
        self.agent_pos = np.array([0.0, 0.0, 0.0])
        # 智能体初始速度方向与走廊方向呈 ±10° 以内的角度
        self.agent_vel = self._generate_initial_velocity()
        self.agent_speed = norm(self.agent_vel)
        self.done = False
        return self._get_observation()

    def _generate_random_direction(self, randomed=True, direction=np.array([-5, 0, 1])):  # 1,0,0
        if not randomed:
            # direction = np.array([0, 0, 1])
            direction[1] = 0.0  # 确保走廊方向与水平平面平行
            direction = direction / np.linalg.norm(direction)
            return direction
        else:
            # 随机生成一个三维单位向量作为走廊方向
            direction = np.random.randn(3)
            direction[1] = 0.0  # 确保走廊方向与水平平面平行
            direction = direction / np.linalg.norm(direction)
            return direction

    def _generate_initial_velocity(self):
        # 生成与走廊方向呈 ±10° 以内角度的初始速度
        angle = np.random.uniform(-max_angle_diff, max_angle_diff)  # 30
        rotation_axis = np.array([0, -1, 0])
        # rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix = self._rotation_matrix(rotation_axis, angle)
        initial_vel = np.dot(rotation_matrix, self.corridor_direction) * agent_initial_speed + np.array([0, 1, 0]) \
                      * np.random.uniform(-10, 10)
        return initial_vel

    def _rotation_matrix(self, axis, theta):
        # 生成绕给定轴旋转 theta 角度的旋转矩阵
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]).T

    def _get_observation(self):
        # 计算走廊终点中心处相对当前智能体速度的俯仰角和航向角
        relative_pos = self.corridor_end - self.agent_pos
        horizontal_tar_get_line = relative_pos.copy()
        horizontal_tar_get_line[1] = 0
        rel_heading_angle = sub_of_radian(math.atan2(horizontal_tar_get_line[2], horizontal_tar_get_line[0]),
                                          math.atan2(self.agent_vel[2], self.agent_vel[0]))
        rel_pitch_angle = sub_of_radian(math.atan2(relative_pos[1], np.linalg.norm(horizontal_tar_get_line)),
                                        math.atan2(self.agent_vel[1],
                                                   np.sqrt(self.agent_vel[0] ** 2 + self.agent_vel[2] ** 2)))

        # 计算走廊边界相对智能体的归一化威胁方向和强度
        threat_info = self._calculate_threat()

        return np.array([rel_heading_angle, rel_pitch_angle, *threat_info], dtype=np.float32)

    def _calculate_threat(self):
        '''重写这部分'''
        # 计算智能体到走廊边界的距离
        psi_corridor = atan2(self.corridor_direction[2], self.corridor_direction[0])
        # psi_agent = atan2(self.agent_vel[2], self.agent_vel[0])
        # psi_diff = sub_of_radian(psi_corridor, psi_agent)
        # 转化为旋转矩阵，将智能体位置和速度表示到相对于通道正方向的坐标中,使得x_corridor表示进度，z_corridor表示距离左右两边的宽度
        rotation_corridor = self._rotation_matrix(np.array([0, 1, 0]), -psi_corridor)  # 变坐标系属于被动旋转
        # 将智能体坐标投影到相对走廊的坐标系中
        agent_pos_corridor = np.dot(rotation_corridor, self.agent_pos)
        # 将智能体速度投影到相对走廊的坐标系中
        agent_vel_corridor = np.dot(rotation_corridor, self.agent_vel)
        # 计算智能体到走廊边界的距离
        x_corridor = agent_pos_corridor[0]
        y_corridor = agent_pos_corridor[1]
        z_corridor = agent_pos_corridor[2]
        # 速度（暂时不考虑）
        vx_corridor = agent_vel_corridor[0]
        vy_corridor = agent_vel_corridor[1]
        vz_corridor = agent_vel_corridor[2]
        # 计算智能体到走廊各个边界的距离
        distance_to_bounds = np.array([
            x_corridor,
            corridor_y_half_width - y_corridor,
            y_corridor - -corridor_y_half_width,
            corridor_z_half_width - z_corridor,
            z_corridor - -corridor_z_half_width,
        ])
        # print("distance_to_bounds", distance_to_bounds)

        threat_to_bounds = 1 - distance_to_bounds / min(corridor_length, corridor_y_half_width, corridor_z_half_width)
        threat_to_bounds = np.clip(threat_to_bounds, 0, 1)

        # print("threat_to_bounds", threat_to_bounds)

        # 边界威胁加权向量
        threat_sum_ = np.dot(
            np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ]).T, threat_to_bounds
        )

        temp = norm(threat_sum_)
        if temp > 1:
            threat_sum_ /= temp
        if norm(threat_sum_) > 1+1e-4:
            print("超出设计范围")
            print(norm(threat_sum_))

        # print("threat_sum_走廊坐标系", threat_sum_)
        # threat_sum 从走廊坐标系转回惯性系
        threat_sum_ = np.dot(
            np.array([
                [cos(psi_corridor), 0, -sin(psi_corridor)],
                [0, 1, 0],
                [sin(psi_corridor), 0, cos(psi_corridor)],
            ]),
            threat_sum_
        )

        # print("threat_sum_惯性系", threat_sum_)

        # 威胁强度
        threat_strength = np.linalg.norm(threat_sum_)

        # 威胁方向单位向量
        threat_direction_ = threat_sum_ / (threat_strength + 1e-6)
        psi_threat_ = atan2(threat_direction_[2], threat_direction_[0])
        theta_threat_ = atan2(threat_direction_[1], np.sqrt(threat_direction_[0] ** 2 + threat_direction_[2] ** 2))
        psi_agent = atan2(self.agent_vel[2], self.agent_vel[0])
        theta_agent = atan2(self.agent_vel[1], np.sqrt(self.agent_vel[0] ** 2 + self.agent_vel[2] ** 2))

        rel_psi_threat_ = sub_of_radian(psi_threat_, psi_agent)
        rel_theta_threat_ = sub_of_radian(theta_threat_, theta_agent)

        return [rel_psi_threat_, rel_theta_threat_, threat_strength]

    def step(self, action, train=True):
        az, ay = action
        # 将加速度旋转到惯性系下
        a_rel_ = np.array([0, ay, az])
        a = 0.0

        agent_vv = self.agent_vel[1]
        agent_vh = self.agent_vel
        agent_vh[1] = 0
        theta = atan2(agent_vv, norm(agent_vh))
        psi = atan2(self.agent_vel[2], self.agent_vel[0])
        # 弹道系下的加速度先转过psi然后转过theta到达惯性系
        active_rotation = np.dot(
            np.array([
                [cos(theta), -sin(theta), 0],
                [0, cos(theta), sin(theta)],
                [0, 0, 1],
            ]),
            np.array([
                [cos(-psi), 0, sin(-psi)],
                [0, 1, 0],
                [-sin(-psi), 0, cos(-psi)]
            ])
        )
        a_inertial_ = np.dot(active_rotation, a_rel_)

        # print(a_inertial_)

        # 更新智能体速度和位置
        self.t += dt

        self.agent_speed += a * dt
        self.agent_vel += a_inertial_ * dt
        self.agent_vel = self.agent_vel * self.agent_speed / norm(self.agent_vel)
        self.agent_pos += self.agent_vel * dt

        # 判断是否结束
        self.done = self._check_done()

        # 计算奖励
        reward, reward_plus = self.get_reward()


        return self._get_observation(), reward, self.done, reward_plus

    def get_reward(self):
        # 密集奖励：根据和目标的距离和接近边界的惩罚制定
        distance_to_target = np.linalg.norm(self.corridor_end - self.agent_pos)
        target_reward1 = (corridor_length - distance_to_target) / corridor_length
        target_reward2 = np.abs(sub_of_radian(atan2(self.agent_vel[2], self.agent_vel[0]),
                                              atan2(self.corridor_direction[2], self.corridor_direction[0]))) / (pi / 2)

        threat_info = self._calculate_threat()
        threat_strength = threat_info[2]
        therat_direction_ = threat_info[0:2]
        boundary_penalty = 0.5  # threat_strength , 1.01-threat_strength , 2-threat_strength
        # fixme 用 小于1的数-threat_strength会自杀，用2-threat_strength又不会很精确

        dense_reward = target_reward1 * 10 + target_reward2 * 1 + boundary_penalty * 10

        reward_plus = [target_reward1, target_reward2, boundary_penalty]

        # 稀疏奖励
        sparse_reward = 0
        if self.done:
            if np.linalg.norm(self.corridor_end - self.agent_pos) <= 10:
                sparse_reward = 100
            elif self._is_outside_boundary():
                sparse_reward = -100

        return dense_reward + sparse_reward, reward_plus

    def _is_outside_boundary(self):
        '''补充向起点方向出界的部分'''
        '''改为用坐标转换实现'''
        # 计算智能体到走廊边界的距离
        psi_corridor = atan2(self.corridor_direction[2], self.corridor_direction[0])
        # 转化为旋转矩阵，将智能体位置和速度表示到相对于通道正方向的坐标中,使得x_corridor表示进度，z_corridor表示距离左右两边的宽度
        rotation_corridor = self._rotation_matrix(np.array([0, -1, 0]), - psi_corridor)  # 变坐标系属于被动旋转, 旋转角度带个负号
        # 将智能体坐标投影到相对走廊的坐标系中
        agent_pos_corridor = np.dot(rotation_corridor, self.agent_pos)
        # 计算智能体到走廊边界的距离
        x_corridor = agent_pos_corridor[0]
        y_corridor = agent_pos_corridor[1]
        z_corridor = agent_pos_corridor[2]

        if x_corridor < -10 or np.abs(y_corridor) > corridor_y_half_width or np.abs(z_corridor) > corridor_z_half_width:
            return True
        else:
            return False

    def _check_done(self):
        '''重写'''
        # print("out", self._is_outside_boundary())
        # print("到达", np.linalg.norm(self.corridor_end - self.agent_pos) <= 10)
        if self._is_outside_boundary() or np.linalg.norm(self.corridor_end - self.agent_pos) <= 10:
            gameover = True
        else:
            gameover = False
        return gameover

    def render(self, mode='human'):
        # 可视化小车的位置和方向
        pass


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'testEnv'
env = testEnv()
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值，
###### 当把连续空间适配离散动作空间时action_dim不能跟着env走 ######
action_list = np.array([
    # [0, 0],
    # [0, 1],  # 上
    # [0, -1],  # 下
    [-1, 0],  # 左
    [1, 0]  # 右
]) * action_bound

action_dim = len(action_list)
target_entropy = -action_dim  # 修改目标熵的计算方式

agent = PPO_discrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                     epochs, eps, gamma, device)

# state_check=[] # 查看输入
## todo 打开这里的函数
# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
return_list = []
steps_list = []
steps_count = 0
# global state_check
epsilon = 0.1

with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
    for i_episode in range(int(num_episodes)):  # 每个1/10的训练轮次

        episode_return = 0
        episode_return_plus = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        actions = []
        while not done:  # 每个训练回合
            steps_count += 1
            # state_check=state
            # 1.执行动作得到环境反馈
            action = agent.take_action(state, epsilon=epsilon)  # 动作编号
            actions.append(action)

            true_action = action_list[action]
            next_state, reward, done, reward_plus = env.step(true_action)  # pendulum中的action一定要是ndarray才能输入吗？
            # 2.运行记录添加回放池
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
            episode_return_plus += np.array(reward_plus)

            if stop_flag:
                break
        episode_return_plus /= (len(actions) + 1e-6)  # 平均的密集奖励
        return_list.append(episode_return)
        # 3.从回放池采样更新智能体
        agent.update(transition_dict)

        pbar.set_postfix({'episode': '%d' % (i_episode),
                          'return': '%.3f' % episode_return,
                          'return_vector': episode_return_plus,
                          'time_last': '%.3f' % env.t})
        pbar.update(1)
        print("psi_corridor:", env.psi_corridor * 180 / pi)
        if i_episode % 10==0:
            print("actions:", actions)
            # print(env.t)
        steps_list.append(steps_count)  # 记录每回合结束时的累积步数
        if stop_flag:
            break
    # return return_list

# return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)


episodes_list = list(range(len(return_list)))
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))

mv_return = moving_average(return_list, 21)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))

plt.figure()
plt.plot(steps_list, return_list)
plt.xlabel('steps')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))

steps_return = moving_average(return_list, 21)
plt.figure()
plt.plot(steps_list, steps_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))

car_trajectory = []
target_trajectory = []

episode_return = 0
state = env.reset(train=False)
done = False
actions = []

while not done:  # 测试回合
    action = agent.take_action(state, explore=False)  # 1 # random.randint(0, 1)
    actions.append(action)
    true_action = action_list[action]
    next_state, reward, done, reward_plus = env.step(true_action)
    car_trajectory.append(env.agent_pos.copy())
    target_trajectory.append(env.target_pos_.copy())
    state = next_state
    episode_return += reward

    psi_agent = atan2(env.agent_vel[2], env.agent_vel[0])
    print("psi_agent", psi_agent * 180 / pi)

print("psi_corridor:", env.psi_corridor * 180 / pi)
print("actions:", actions)
print("test_return:", episode_return)

plt.figure(5)
# 绘制走廊的俯视图
plt.subplot(2, 1, 1)
psi_corridor = atan2(env.corridor_direction[2], env.corridor_direction[0])
i = np.linspace(0, 1, 100)
# 绘制走廊中心的参数方程
x_center_ = i * corridor_length * np.cos(psi_corridor)
z_center_ = i * corridor_length * np.sin(psi_corridor)
x_left_border_ = x_center_ + corridor_z_half_width * np.sin(psi_corridor)
x_right_border_ = x_center_ - corridor_z_half_width * np.sin(psi_corridor)
z_left_border_ = z_center_ - corridor_z_half_width * np.cos(psi_corridor)
z_right_border_ = z_center_ + corridor_z_half_width * np.cos(psi_corridor)
plt.plot(z_center_, x_center_, 'k-', label='Corridor Center')
plt.plot(z_left_border_, x_left_border_, 'k--', label='Corridor Left Border')
plt.plot(z_right_border_, x_right_border_, 'k--', label='Corridor Right Border')
plt.xlabel('z')
plt.ylabel('x')
plt.legend()
plt.title('Corridor View')
# 绘制智能体轨迹
move_trajectory = np.array(car_trajectory)
plt.plot(move_trajectory[:, 2], move_trajectory[:, 0], label='Corridor Center')

# 绘制走廊侧视图
plt.subplot(2, 1, 2)
d_center_ = np.sqrt(x_center_ ** 2 + z_center_ ** 2)
y_center_ = i * 0
y_down_border_ = y_center_ - corridor_y_half_width
y_up_border_ = y_center_ + corridor_y_half_width
plt.plot(d_center_, y_center_, 'k-', label='Corridor Center')
plt.plot(d_center_, y_down_border_, 'k--', label='Corridor Left Border')
plt.plot(d_center_, y_up_border_, 'k--', label='Corridor Right Border')
plt.xlabel('d')
plt.ylabel('y')
plt.legend()
plt.title('Corridor View')
# 绘制智能体轨迹
move_trajectory = np.array(car_trajectory)
d_ = np.sqrt(move_trajectory[:, 0] ** 2 + move_trajectory[:, 2] ** 2)
plt.plot(d_, move_trajectory[:, 1], label='Corridor Center')

# # 显示所有图形
plt.show()
