import gymnasium as gymn
from gymnasium import spaces
import numpy as np
import pygame
import math
import time

class ArcherEnv(gymn.Env):
    """
    弓箭手射击移动目标的环境。

    - **观测空间 (Observation Space)**: 3维Box
        1. [0] 与目标的距离 (m)
        2. [1] 距离上次射击的时间 (s)
        3. [2] 剩余箭的数量

    - **动作空间 (Action Space)**: 离散(2)
        - 0: 不发射
        - 1: 发射

    - **奖励 (Reward)**:
        - 获胜: +20
        - 失败: -20
        - 每射一箭: -0.5
        - 快速射击奖励: (5 - 射击间隔) / 5，仅当间隔 < 5s

    - **终止条件 (Termination)**:
        - 射中5箭 (获胜)
        - 箭用完 (失败)
        - 目标距离小于2m (失败)
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- 环境参数 ---
        self.initial_distance = 100.0  # 初始距离 (m)
        self.target_speed = 4.0        # 目标速度 (m/s)
        self.min_safe_distance = 2.0   # 弓箭手失败的最小距离 (m)
        self.total_arrows = 10         # 初始箭数
        self.hits_to_win = 5           # 获胜所需命中数

        # --- 时间步长 ---
        self.decision_dt = 1.0         # 决策步长 (s)
        self.physics_dt = 0.1          # 物理模拟步长 (s)
        self.physics_steps_per_decision = int(self.decision_dt / self.physics_dt)

        # --- 命中率参数 ---
        self.guaranteed_hit_range = 20.0 # 必中范围 (m)
        # 根据 P(d) = exp(-k * (d - 20)) 和 P(50)=0.5 计算衰减系数k
        self.hit_decay_k = -math.log(0.5) / (50.0 - self.guaranteed_hit_range)

        # --- 定义观测空间和动作空间 ---
        # [距离, 上次射击间隔, 剩余箭数]
        obs_low = np.array([0, 0, 0], dtype=np.float32)
        obs_high = np.array([self.initial_distance + 10, 100, self.total_arrows], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # [不发射, 发射]
        # 动作为单维 0/1，使用 float32 的 Box 更兼容外部传入的标量或数组
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Pygame 可视化 ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    def get_obs(self):
        return np.array([
            self.target_pos,
            self.time_since_last_shot,
            self.arrows_left
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "distance": self.target_pos,
            "arrows_left": self.arrows_left,
            "hits": self.hits
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化状态
        self.target_pos = self.initial_distance
        self.arrows_left = self.total_arrows
        self.hits = 0
        self.time_since_last_shot = 0.0 # 可设为较大值表示从未射击

        # 用于可视化的箭矢列表
        self.projectiles = []

        if self.render_mode == "human":
            self._render_frame()

        return self.get_obs(), self._get_info()

    def step(self, action):
        # 兼容外部可能传入的标量 0/1 或形如 (1,) 的数组
        a = np.asarray(action)
        if a.ndim == 0:
            a = np.array([a])
        # 转为 float32 以匹配 action_space.dtype，再裁剪到 [0,1]
        a = np.clip(a.astype(np.float32), 0.0, 1.0)
        if not self.action_space.contains(a):
            # 保险起见打印并抛出更明确的错误
            raise ValueError(f"Action {action!r} is not within the action space {self.action_space}")
        # 最终用整数 0/1 表示是否开火
        action = int(a.flat[0])

        reward = 0.0
        terminated = False

        # --- 1. 处理射击动作 ---
        shoot_action = (action == 1)
        if shoot_action and self.arrows_left > 0:
            # a. 计算射击奖励
            reward -= 0.5  # 每射一箭的固定惩罚
            if self.time_since_last_shot < 5.0:
                reward += (5.0 - self.time_since_last_shot) / 5.0

            # b. 判定是否命中
            distance_at_shot = self.target_pos
            hit_prob = 0.0
            if distance_at_shot <= self.guaranteed_hit_range:
                hit_prob = 1.0
            else:
                hit_prob = math.exp(-self.hit_decay_k * (distance_at_shot - self.guaranteed_hit_range))
            
            if self.np_random.random() < hit_prob:
                self.hits += 1
                # 可视化：添加一个表示命中的标记
                if self.render_mode == "human":
                    self.projectiles.append({"pos": distance_at_shot, "hit": True})
            else:
                if self.render_mode == "human":
                    self.projectiles.append({"pos": distance_at_shot, "hit": False})

            # c. 更新状态
            self.arrows_left -= 1
            self.time_since_last_shot = 0.0
        
        # --- 2. 模拟世界演进 ---
        # 一个决策步包含多个物理步
        for _ in range(self.physics_steps_per_decision):
            if not terminated: # 如果已结束，则不再移动目标
                self.target_pos -= self.target_speed * self.physics_dt
                self.target_pos = max(0, self.target_pos) # 防止穿过弓箭手

        self.time_since_last_shot += self.decision_dt

        # --- 3. 检查终止条件 ---
        if self.hits >= self.hits_to_win:
            reward += 20  # 获胜奖励
            terminated = True
        elif self.arrows_left <= 0 and self.hits < self.hits_to_win:
            reward -= 20  # 失败奖励
            terminated = True
        elif self.target_pos <= self.min_safe_distance:
            reward -= 20  # 失败奖励
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self.get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Archer vs Target")
            self.screen = pygame.display.set_mode((800, 200))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        # 绘制背景
        self.screen.fill((210, 210, 210)) # 浅灰色背景

        # 坐标转换：将米映射到像素
        def to_px(pos_m):
            return int(50 + (pos_m / self.initial_distance) * 700)

        # 绘制弓箭手 (绿色方块)
        archer_px = to_px(0)
        pygame.draw.rect(self.screen, (0, 150, 0), (archer_px - 10, 100 - 20, 20, 40))

        # 绘制目标 (红色方块)
        target_px = to_px(self.target_pos)
        pygame.draw.rect(self.screen, (150, 0, 0), (target_px - 10, 100 - 15, 20, 30))

        # 绘制箭矢和命中效果
        for proj in self.projectiles:
            proj_px = to_px(proj["pos"])
            if proj["hit"]:
                # 命中：在目标位置画一个黄色星星
                pygame.draw.circle(self.screen, (255, 255, 0), (proj_px, 100), 8)
            else:
                # 未命中：在目标位置画一个灰色叉
                pygame.draw.line(self.screen, (50, 50, 50), (proj_px - 5, 95), (proj_px + 5, 105), 2)
                pygame.draw.line(self.screen, (50, 50, 50), (proj_px - 5, 105), (proj_px + 5, 95), 2)
        
        # 清理旧的箭矢效果（只显示一小段时间）
        if len(self.projectiles) > 0:
            self.projectiles.pop(0)

        # 绘制状态信息
        info_text = (
            f"Distance: {self.target_pos:.1f}m | "
            f"Arrows: {self.arrows_left}/{self.total_arrows} | "
            f"Hits: {self.hits}/{self.hits_to_win}"
        )
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

if __name__ == '__main__':
    # --- 环境测试代码 ---
    # 使用随机动作策略来运行环境并观察效果
    # env = ArcherEnv(render_mode="human")
    
    env = ArcherEnv(render_mode="human")
    
    num_episodes = 5
    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {ep + 1} ---")
        
        while not terminated:
            obs = env.get_obs()
            # 随机策略：有15%的几率射击
            action = 1 if np.random.rand() < 0.15 else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 打印每一步的信息
            print(
                f"Step {step_count}: Action={action}, "
                f"Dist={info['distance']:.1f}, "
                f"Reward={reward:.2f}, "
                f"Total Reward={total_reward:.2f}"
            )
            time.sleep(0.5)

        print(f"Episode finished after {step_count} steps. Final reward: {total_reward:.2f}")
        print(f"Final state: Hits={info['hits']}, Arrows Left={info['arrows_left']}")

    env.close()