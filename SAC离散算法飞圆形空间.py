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
actor_lr = 3e-4
critic_lr = actor_lr * 1.5  # 3e-3
num_episodes = 1 * 420  # 800 400
max_train_steps = 2e5
hidden_dim_1 = [256]  # [256, 256]

gamma = 0.99
alpha_lr = 3e-4
tau = 0.005  # 软更新参数
buffer_size = 1000
minimal_size = 200  # 1000
batch_size = minimal_size
sigma = 0.01  # 0.01  # 高斯噪声标准差


def stop_loop():
    global stop_flag
    stop_flag = True
    print("检测到 'esc' 键，准备退出所有循环！")  # 也可以用 pause、scroll lock等几乎无用的按键


keyboard.add_hotkey('esc', stop_loop, suppress=False, trigger_on_release=False)  # 监听 'esc' 键，按下时执行 `stop_loop()`


def calc_angles_of_NUE(inpur_array):
    inpur_array_h = inpur_array.copy()
    inpur_array_h[1] = 0
    psi = np.arctan2(inpur_array_h[2], inpur_array_h[0])
    theta = np.arctan2(inpur_array_h[1], norm(inpur_array_h))
    return psi, theta


def calc_intern_dist2cylinder(R, rho, eta, psi, theta):
    """
    计算飞机到圆柱形边界的斜距离

    参数:
    R: float, 圆柱形边界半径
    rho: float, 飞机到圆心的距离
    eta: float, 飞机相对于圆心的方位角（弧度）
    psi: float, 飞机航向角（弧度）
    theta: float, 飞机俯仰角（弧度）

    返回:
    d: float, 飞机到边界的斜距离
    dh: float, 飞机到边界的水平距离
    pos_: ndarray, 飞机位置坐标 [x, y, z]
    """
    # 计算飞机位置
    pos_ = np.array([rho * cos(eta), 0, rho * sin(eta)])

    # 计算水平距离
    dh_list = rho * cos(pi + eta - psi) + sqrt(R ** 2 - rho ** 2 * sin(pi + eta - psi) ** 2)
    dh = dh_list

    # 计算斜距离
    d = dh / cos(theta)

    return d, dh, pos_


def sub_of_radian(input1, input2):
    # 弧度减法
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


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


class PolicyNet_discrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNet_discrete, self).__init__()
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)

            # layers.append(nn.ReLU())
            # layers.append(nn.LayerNorm(layer_size))  # test 添加LayerNorm归一化层， 效果不太好
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_size, action_dim)

        # 修改初始化方式
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # 输出层使用更小的初始化范围
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        x = self.net(x)
        x = self.fc_out(x)
        return F.softmax(x, dim=1)


class QValueNet_discrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(QValueNet_discrete, self).__init__()
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)
            # layers.append(nn.ReLU())
            # layers.append(nn.LayerNorm(layer_size))  # test 添加LayerNorm归一化层， 效果不太好
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, action_dim)

    def forward(self, x):
        x = self.net(x)
        return self.fc_out(x)


class SAC_discrete:
    ''' 处理离散动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet_discrete(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet_discrete(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet_discrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet_discrete(state_dim, hidden_dim,
                                                  action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet_discrete(state_dim, hidden_dim,
                                                  action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, explore=True, epsilon=0.0):
        if not explore:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            actions = self.actor(state)
            action = torch.argmax(actions, dim=1)
            return action.item()

        # 随机选择动作, 对于多个维度的是离散控制似乎是必须的
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(action_list))  # 从1开始，随机动作不许没有动作

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        # print(actions)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


# dt = 0.1
cage_length = 900  # 900
cage_y_half_width = 30
cage_z_half_width = 30
agent_initial_speed = 30
max_angle_diff = 10 * (math.pi / 180)  # 最大偏移角度，弧度制
agent_axcell = 10


class testEnv(gym.Env):
    def __init__(self):
        super(testEnv, self).__init__()
        # 观测空间：俯仰角、航向角、归一化威胁方向和强度（假设威胁信息有 4 个维度）
        # np.array([rel_heading_angle, rel_pitch_angle, rel_psi_threat_, rel_theta_threat_, threat_strength], dtype=np.float32)
        self.a_rel_ = None
        self.agent_speed = None
        self.target_pos_ = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-agent_axcell, high=agent_axcell, shape=(2,), dtype=np.float32)
        self.state = None
        self.done = False
        self.cage_direction = None
        # self.cage_end = None
        self.agent_pos = None
        self.agent_vel = None
        self.t = None
        self.R = None

    def reset(self, train=True):
        self.R = 2.1 * agent_initial_speed ** 2 / agent_axcell  # 场地半径,略高于转弯半径的两倍
        # self.R = 10 * agent_initial_speed ** 2 / agent_axcell
        self.t = 0
        # 智能体出生点
        self.agent_pos = np.array([0.0, 0.0, 0.0])
        # self.agent_pos[2] += -0.2 * self.R
        self.agent_pos[2] += np.random.uniform(0.7 * self.R, 0.9 * self.R)*(random.randint(0, 1) * 2 - 1)
        # 智能体初始速度方向与走廊方向呈 ±10° 以内的角度
        self.agent_vel = self._generate_initial_velocity()
        self.agent_speed = norm(self.agent_vel)
        self.done = False
        return self._get_observation()

    def _generate_initial_velocity(self):
        # 生成与走廊方向呈 ±10° 以内角度的初始速度
        angle = np.random.uniform(-max_angle_diff, max_angle_diff)  # 30
        rotation_axis = np.array([0, -1, 0])
        rotation_matrix = self._rotation_matrix(rotation_axis, angle)
        # temp = np.array([1, 0, 0]) * (random.randint(0, 1) * 2 - 1) \
        #                 * agent_initial_speed + np.array([0, 1, 0]) \
        #                 * np.random.uniform(-8, 8)
        temp = np.array([1, 0, 0]) \
               * agent_initial_speed + np.array([0, 1, 0]) \
               * np.random.uniform(-8, 8)
        initial_vel = np.dot(rotation_matrix, temp)

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
        # relative_pos = self.cage_end - self.agent_pos
        # horizontal_tar_get_line = relative_pos.copy()
        # horizontal_tar_get_line[1] = 0
        rel_heading_angle = 0  # sub_of_radian(math.atan2(horizontal_tar_get_line[2], horizontal_tar_get_line[0]),
        # math.atan2(self.agent_vel[2], self.agent_vel[0]))
        rel_pitch_angle = 0  # sub_of_radian(math.atan2(relative_pos[1], np.linalg.norm(horizontal_tar_get_line)),
        # math.atan2(self.agent_vel[1],
        #            np.sqrt(self.agent_vel[0] ** 2 + self.agent_vel[2] ** 2)))

        # 计算走廊边界相对智能体的归一化威胁方向和强度
        threat_info = self._calculate_threat()

        return np.array([rel_heading_angle, rel_pitch_angle, *threat_info], dtype=np.float32)

    def _is_outside_boundary(self):
        if self.agent_pos[0] ** 2 + self.agent_pos[2] ** 2 >= self.R ** 2 or abs(
                self.agent_pos[1]) >= cage_y_half_width:
            return True
        else:
            return False

    def _calculate_threat(self):
        '''重写这部分'''
        rho = sqrt(self.agent_pos[2] ** 2 + self.agent_pos[0] ** 2)
        eta = atan2(self.agent_pos[2], self.agent_pos[0])

        # print("eta", eta)

        psi_agent, theta_agent = calc_angles_of_NUE(self.agent_vel)

        # 水平边界威胁
        # 计算到预计落点的方向
        d, _, _ = calc_intern_dist2cylinder(self.R, rho, eta, psi_agent, theta_agent)
        # print("d", d)

        # 计算剩余时间
        t_remain1 = d / self.agent_speed
        t_min = pi / 2 / (agent_axcell/self.agent_speed)
        t_save = 2 * t_min
        k = log(0.05) / (t_save - t_min)
        # 计算圆形边界威胁的大小
        # print("t_remain", t_remain1)

        threat_strength_h = min(exp(k * (t_remain1 - t_min)), 1)


        # 圆形边界威胁的方向取在飞机指向圆外部的方向
        # threat_direction_h = np.array([-self.agent_pos[0], 0, -self.agent_pos[2]])  # 圆心（之前写错）
        threat_direction_h = np.array([cos(eta), 0, sin(eta)]) * threat_strength_h if norm(threat_strength_h) > 0 \
            else np.array([1, 0, 0])


        # 垂直边界威胁
        t_remain_up = (cage_y_half_width - self.agent_pos[1]) / self.agent_vel[1] \
            if abs(self.agent_vel[1]) > 1e-3 else np.inf
        if t_remain_up < 0:
            t_remain_up = np.inf
        t_remain_down = (self.agent_pos[1] - -cage_y_half_width) / -self.agent_vel[1] \
            if abs(self.agent_vel[1]) > 1e-3 else np.inf
        if t_remain_down < 0:
            t_remain_down = np.inf
        t_remain2 = min(t_remain_up, t_remain_down)
        # 威胁方向单位向量
        threat_direction_v = np.array([
            [0, 1, 0],
            [0, -1, 0]
        ])
        threat_direction_v = threat_direction_v[np.argmin([t_remain_up, t_remain_down])]
        if t_remain2 <= t_min:
            threat_strength_v = 1
        else:
            threat_strength_v = min(exp(k * (t_remain2 - t_min)), 1)
        # 综合计算威胁方向和大小
        threat_ = threat_strength_v * threat_direction_v + threat_strength_h * threat_direction_h

        threat_strength = norm(threat_)  # 补充归一化
        threat_strength = min(threat_strength, 1)

        psi_threat, theta_threat = calc_angles_of_NUE(threat_)
        # print("psi_threat", psi_threat)
        # print("thetta_threat", theta_threat)

        rel_psi_threat_ = sub_of_radian(psi_threat, psi_agent)
        rel_theta_threat_ = sub_of_radian(theta_threat, theta_agent)

        return [rel_psi_threat_, rel_theta_threat_, threat_strength]

    def step(self, action, train=True):
        az, ay = action
        # 将加速度旋转到惯性系下
        a_rel_ = np.array([0, ay, az])
        self.a_rel_ = a_rel_
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
        # distance_to_target = np.linalg.norm(self.cage_end - self.agent_pos)
        # target_reward1 = (cage_length - distance_to_target) / cage_length
        # target_reward2 = np.abs(sub_of_radian(atan2(self.agent_vel[2], self.agent_vel[0]),
        #                                       atan2(self.cage_direction[2], self.cage_direction[0]))) / (pi / 2)

        rel_psi_threat_, rel_theta_threat_, threat_strength = self._calculate_threat()

        # threat_info = self._calculate_threat()
        # threat_strength = threat_info[2]
        # therat_direction_ = threat_info[0:2]

        boundary_penalty = 1.1-threat_strength  # threat_strength , 1.01-threat_strength , 2-threat_strength
        # fixme 用 小于1的数-threat_strength会自杀，用2-threat_strength又不会很精确

        # control_overhead = 1.1 - norm(self.a_rel_)/agent_axcell

        dense_reward = boundary_penalty * 10  # + control_overhead * 1

        reward_plus = [boundary_penalty]

        # 稀疏奖励
        sparse_reward = 0
        if self.done:
            if self._is_outside_boundary():
                sparse_reward = -100

        return dense_reward + sparse_reward, reward_plus

    def _check_done(self):
        '''重写'''
        if self._is_outside_boundary() or self.t >= 2.0 * 2 * pi / (agent_axcell / agent_initial_speed):
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
replay_buffer = ReplayBuffer(buffer_size)  # 经验回放
state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值，
###### 当把连续空间适配离散动作空间时action_dim不能跟着env走 ######
action_list = np.array([
    [0, 0],  # 直航
    # [0, 1],  # 上
    # [0, -1],  # 下
    [-1, 0],  # 左
    [1, 0]  # 右
]) * action_bound

action_dim = len(action_list)
target_entropy = -action_dim  # 修改目标熵的计算方式

agent = SAC_discrete(state_dim, hidden_dim_1, action_dim,
                     actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                     gamma, device)

# state_check=[] # 查看输入
## todo 打开这里的函数
# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):

return_list = []
steps_list = []
steps_count = 0
# global state_check
epsilon = 0.1
i_episode = -1


'''tensorboard记录'''
from torch.utils.tensorboard import SummaryWriter
import os
import webbrowser
import threading

# 创建 SummaryWriter 实例，指定日志保存目录
writer = SummaryWriter('runs/sac_training')
# 启动 TensorBoard 服务的函数
def open_tensorboard():
    os.system('tensorboard --logdir=runs/sac_training')

# 启动一个线程来运行 TensorBoard
tensorboard_thread = threading.Thread(target=open_tensorboard, daemon=True)
tensorboard_thread.start()

# 打印手动打开的链接
print("请在浏览器中打开以下链接查看 TensorBoard: http://localhost:6006")

# 可选：自动打开浏览器（若要启用，取消下面这行注释）
# webbrowser.open('http://localhost:6006')



with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # episode进度条
# with tqdm(total=int(max_train_steps), desc='Steps') as pbar:  # steps进度条
    # for i_episode in range(int(num_episodes)):  # 每个训练轮次
    while steps_count <= max_train_steps:
        i_episode += 1
        episode_return = 0
        episode_return_plus = 0
        # state = env.reset(train=True)
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
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            episode_return_plus += np.array(reward_plus)
            # 3.从回放池采样更新智能体
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)

                if stop_flag:
                    break

        episode_return_plus /= (len(actions) + 1e-6)  # 平均的密集奖励
        return_list.append(episode_return)
        pbar.set_postfix({'steps': '%d' % (steps_count),
                          'return': '%.3f' % episode_return,
                          'return_vector': episode_return_plus,
                          'time_last': '%.3f' % env.t})
        pbar.update(1)

        print("actions:", actions)
        # print(env.t)
        steps_list.append(steps_count)  # 记录每回合结束时的累积步数

        '''tensorboard记录'''
        # 将数据写入 TensorBoard
        writer.add_scalar('Return', episode_return, i_episode)
        writer.add_scalar('Time Last', env.t, i_episode)
        # 假设 episode_return_plus 是一维数组，记录其每个元素
        for idx, value in enumerate(episode_return_plus):
            writer.add_scalar(f'Return Vector/{idx}', value, i_episode)

        if stop_flag:
            break
    # return return_list

# return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

if i_episode>=10:
    episodes_list = list(range(len(return_list)))
    plt.figure()
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))

    mv_return = moving_average(return_list, 9)
    plt.figure()
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))

    plt.figure()
    plt.plot(steps_list, return_list)
    plt.xlabel('steps')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))

'''停止 tensorboard 记录'''
# writer.close()



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
    state = next_state
    episode_return += reward

    psi_agent = atan2(env.agent_vel[2], env.agent_vel[0])
    print("obs_agent", next_state)
    print("agent_pos", env.agent_pos)
    # print("agent_vel", env.agent_vel)

print("actions:", actions)
print("test_return:", episode_return)


fig = plt.figure(5)
fig.clear()

# 绘制场地俯视图
ax1 = fig.add_subplot(2, 1, 1)
i = np.linspace(0, 1, 100)
# 绘制天花板圆形边界的参数方程
x_border_ceiling = env.R * np.cos(i * 2 * pi)
y_boder_ceiling = np.ones_like(i) * cage_y_half_width
z_border_ceiling = env.R * np.sin(i * 2 * pi)

x_border_floor = env.R * np.cos(i * 2 * pi)
y_boder_floor = np.ones_like(i) * -cage_y_half_width
z_border_floor = env.R * np.sin(i * 2 * pi)

ax1.plot(z_border_floor, x_border_floor, 'k-', label='Corridor border')
ax1.set_xlabel('z')
ax1.set_ylabel('x')
ax1.legend()
ax1.set_title('Corridor View')
# 绘制智能体轨迹
move_trajectory = np.array(car_trajectory)
ax1.plot(move_trajectory[:, 2], move_trajectory[:, 0], label='Corridor Center')

# 绘制高度随时间变化曲线
ax2 = fig.add_subplot(2, 1, 2)
# move_trajectory[:, 1]
length = len(move_trajectory[:, 1])
y_down_border_ = np.ones_like(move_trajectory[:, 1]) * -cage_y_half_width
y_up_border_ = np.ones_like(move_trajectory[:, 1]) * cage_y_half_width
steps_in_one_episode = np.linspace(1, length, length)
ax2.plot(steps_in_one_episode, y_down_border_, 'k-', label='floor')
ax2.plot(steps_in_one_episode, y_up_border_, 'k-', label='ceiling')
ax2.plot(steps_in_one_episode, move_trajectory[:, 1], label='trajectory')

ax2.set_xlabel('steps')
ax2.set_ylabel('y')
ax2.legend()
# ax2.title('height View')

# 强制刷新图形
fig.canvas.draw()
fig.canvas.flush_events()

# # 显示所有图形
plt.show()
