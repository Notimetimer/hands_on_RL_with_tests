import random
import gym
import numpy as np
import collections
from gym import spaces
from numpy.linalg import norm

# 更改目标：跟踪动目标
class testEnv(gym.Env):
    def __init__(self, dof=3, dt=0.5):
        super(testEnv, self).__init__()
        # 观测空间：相对于点的位置和速度
        self.dof = dof
        self.dt = dt
        low1 = np.ones(self.dof * 2) * -np.inf
        high1 = np.ones(self.dof * 2) * np.inf
        self.observation_space = spaces.Box(low=low1, high=high1, dtype=np.float32)
        # 动作空间：三轴加速度
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.dof,), dtype=np.float32)
        self.state = None
        self.done = False
        self.target_vel_ = None
        self.target_pos_ = None
        self.t = None

    def reset(self, train=True):
        # 初始化状态
        self.t = 0
        self.target_pos_ = np.zeros(self.dof)
        self.target_vel_ = np.ones(self.dof)

        if train:
            pos_ = np.random.rand(self.dof) * 3
            vel_ = np.random.rand(self.dof) * 0.3
        else:
            pos_ = np.ones(self.dof)
            vel_ = np.ones(self.dof) * 0.1
        self.state = np.hstack((pos_, vel_))  # 初始位置
        self.done = False
        observe = self.state
        return observe

    def step(self, action):
        self.t += self.dt
        pos_ = self.state[0:self.dof]  # 从数组中提取向量
        vel_ = self.state[self.dof:]

        # # # 更新状态
        vel_ += action * self.dt
        pos_ += vel_ * self.dt
        self.target_pos_ += self.target_vel_ * self.dt

        self.state = np.hstack((pos_, vel_))
        observe = np.hstack((self.target_pos_ - pos_, self.target_vel_ - vel_))

        # 定义奖励函数
        reward = 2 * (5 - np.linalg.norm(observe[0:self.dof]))  # 奖励与位置偏差的绝对值成负正比

        # reward_plus = -reward**2 * 5
        # reward_plus -= np.linalg.norm(observe[self.dof:])**2

        # reward_plus = min(0, -np.linalg.norm(np.dot(vel_, pos_)/np.linalg.norm(pos_))) if np.linalg.norm(pos_) > 0 else -np.linalg.norm(vel_)

        if self.t > 20:
            self.done = True
        if np.linalg.norm(observe[0:self.dof]) > 10:
            reward -= 100
            self.done = True

        reward_plus = 0
        return observe, reward, self.done, reward_plus

    def render(self, mode='human'):
        # 可视化小车的位置和方向
        pass
