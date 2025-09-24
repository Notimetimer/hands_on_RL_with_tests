import random
import gymnasium as gymn
import numpy as np
from numpy.linalg import norm

class testEnv(gymn.Env):
    """gymnasium 兼容的测试环境：跟踪三维动目标"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, dof=3, dt=0.5):
        super(testEnv, self).__init__()
        self.dof = dof
        self.dt = dt
        low1 = np.ones(self.dof * 2) * -np.inf
        high1 = np.ones(self.dof * 2) * np.inf
        self.observation_space = gymn.spaces.Box(low=low1.astype(np.float32), high=high1.astype(np.float32), dtype=np.float32)
        self.action_space = gymn.spaces.Box(low=-5.0, high=5.0, shape=(self.dof,), dtype=np.float32)
        self.state = None
        self.done = False
        self.target_vel_ = None
        self.target_pos_ = None
        self.t = None
        self.out_range = 0

    def reset(self, *, seed=None, options=None, train=True):
        # gymnasium reset signature: return obs, info
        if seed is not None:
            np.random.seed(seed)
        self.t = 0.0
        self.target_pos_ = np.zeros(self.dof, dtype=np.float32)
        self.target_vel_ = np.ones(self.dof, dtype=np.float32)
        self.out_range = 0

        if train:
            pos_ = np.random.rand(self.dof).astype(np.float32) * 3.0
            vel_ = (np.random.rand(self.dof).astype(np.float32)) * 0.3
        else:
            pos_ = np.ones(self.dof, dtype=np.float32)
            vel_ = np.ones(self.dof, dtype=np.float32) * 0.1
        self.state = np.hstack((pos_, vel_)).astype(np.float32)
        self.done = False
        observe = self.state.copy().astype(np.float32)
        info = {}
        return observe, info

    def step(self, action):
        # gymnasium step signature: return obs, reward, terminated, truncated, info
        self.t += self.dt
        pos_ = self.state[0:self.dof].astype(np.float32)
        vel_ = self.state[self.dof:].astype(np.float32)

        action = np.asarray(action, dtype=np.float32)
        vel_ = vel_ + action * self.dt
        pos_ = pos_ + vel_ * self.dt
        self.target_pos_ = (self.target_pos_ + self.target_vel_ * self.dt).astype(np.float32)

        self.state = np.hstack((pos_, vel_)).astype(np.float32)
        observe = np.hstack((self.target_pos_ - pos_, self.target_vel_ - vel_)).astype(np.float32)

        reward = float(2.0 * (5.0 - np.linalg.norm(observe[0:self.dof])))

        terminated = False
        truncated = False
        if self.t > 20.0:
            truncated = True
        if np.linalg.norm(observe[0:self.dof]) > 10.0:
            self.out_range = 1
            reward -= 30.0
            terminated = True

        info = {"reward_plus": 0.0, "out_range": float(self.out_range)}
        return observe, reward, terminated, truncated, info

    def render(self):
        pass