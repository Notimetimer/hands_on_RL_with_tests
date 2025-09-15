import random
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from tracking_test import testEnv

# 固定随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 环境 / 任务 配置
dt = 0.5
dof = 3
env_kwargs = dict(dof=dof, dt=dt)

# 超参数（可按需调整）
actor_lr = 1e-4
num_episodes = 100
hidden_dim = [128, 128]
gamma = 0.9
epochs = 10
clip_eps = 0.2
ent_coef = 0.1
batch_size = 64
total_timesteps = 200_000
device = "cuda" if torch.cuda.is_available() else "cpu"

# 辅助函数：平滑曲线
def moving_average(a, window_size):
    if len(a) < window_size:
        return np.array(a)
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 创建单环境工厂以便 DummyVecEnv 使用
def make_env():
    def _init():
        env = testEnv(**env_kwargs)
        # 兼容性包装：确保 step 返回的第四项是 dict（不修改 tracking_test）
        class StepInfoFixWrapper(gym.Wrapper):
            def step(self, action):
                ret = self.env.step(action)
                # 兼容不同返回格式
                if isinstance(ret, tuple) and len(ret) == 4:
                    obs, reward, done, info = ret
                    if not isinstance(info, dict):
                        info = {'extra_info': info}
                    return obs, reward, done, info
                # 如果已经是 Gymnasium (obs, reward, terminated, truncated, info)，直接返回
                return ret
            def reset(self, **kwargs):
                ret = self.env.reset(**kwargs)
                # 某些实现返回 (obs, info)，兼容为 obs
                if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
                    return ret
                return ret
        env = StepInfoFixWrapper(env)
        return env
    return _init

vec_env = DummyVecEnv([make_env()])

# 自定义 policy 网络结构（pi 和 vf 分别为两个 128 单元的隐藏层）
policy_kwargs = dict(
    activation_fn=nn.ReLU,
    net_arch=dict(pi=hidden_dim, vf=hidden_dim)  # 使用 dict 以兼容 SB3 新推荐格式
)

model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=actor_lr,
    n_epochs=epochs,
    batch_size=batch_size,
    clip_range=clip_eps,
    ent_coef=ent_coef,
    verbose=1,
    policy_kwargs=policy_kwargs,
    device=device,
    tensorboard_log="./tb_logs_sb3"
)

# 简单回调用于在训练过程中收集 episode returns（可选）
class ReturnLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.returns = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', None)
        if infos:
            for info in infos:
                if 'episode' in info:
                    self.returns.append(info['episode']['r'])
        return True

callback = ReturnLogger()

# 训练
model.learn(total_timesteps=total_timesteps, callback=callback)

# 保存模型
model.save("ppo_testenv_sb3")

# 评估 / 测试：运行若干个 episode 并记录轨迹 / returns
eval_env = testEnv(**env_kwargs)
num_eval_episodes = 5
return_list = []
trajectories = []

for ep in range(num_eval_episodes):
    obs = eval_env.reset(train=False)
    done = False
    ep_ret = 0.0
    traj = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        traj.append(eval_env.state[0:dof].copy())
        ep_ret += reward
    return_list.append(ep_ret)
    trajectories.append(np.array(traj))

print("Eval returns:", return_list)

# 绘图：训练过程若使用 callback 收集到 returns 则绘出；否则只画 eval returns
if callback.returns:
    plt.figure()
    plt.plot(callback.returns)
    plt.xlabel('Logged episodes')
    plt.ylabel('Return')
    plt.title('Training returns (SB3 PPO)')
else:
    plt.figure()
    plt.plot(return_list, marker='o')
    plt.xlabel('Eval episode')
    plt.ylabel('Return')
    plt.title('Eval returns (SB3 PPO)')

# 若有轨迹数据，绘制每个坐标分量的最后一次评估轨迹与目标（若环境提供 target）
if len(trajectories) > 0:
    last_traj = trajectories[-1]
    plt.figure()
    for i in range(dof):
        plt.subplot(dof, 1, i + 1)
        plt.plot(last_traj[:, i], 'b-', label='Position')

        # 如果 env 暴露 target_pos_ 属性，可绘制目标轨迹（兼容多种形状）
        if hasattr(eval_env, "target_pos_"):
            tp = eval_env.target_pos_
            target_traj = None
            if isinstance(tp, (list, np.ndarray)):
                tp_arr = np.array(tp)
                # tp 形状为 (T, dof)
                if tp_arr.ndim == 2 and tp_arr.shape[0] == last_traj.shape[0]:
                    target_traj = tp_arr[:, i]
                # tp 为每维固定目标，形状 (dof,)
                elif tp_arr.ndim == 1 and tp_arr.shape[0] == dof:
                    target_traj = np.full(last_traj.shape[0], tp_arr[i])
                # tp 为每步一维目标，长度等于轨迹步数
                elif tp_arr.ndim == 1 and tp_arr.shape[0] == last_traj.shape[0]:
                    target_traj = tp_arr
                # 其他不匹配的形状则忽略
            elif isinstance(tp, (int, float)):
                target_traj = np.full(last_traj.shape[0], float(tp))

            if target_traj is not None and target_traj.shape[0] == last_traj.shape[0]:
                plt.plot(target_traj, 'r--', label='Target')

        plt.xlabel('Step')
        plt.ylabel(f'Coordinate {i+1}')
        plt.legend()

plt.show()