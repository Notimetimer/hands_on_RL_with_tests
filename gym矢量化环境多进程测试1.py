import gymnasium as gymn
import numpy as np

# 子环境构造函数（返回可调用对象）
def make_env():
    return lambda: gymn.make("CartPole-v1")

if __name__ == "__main__":
    # 1. 同步版本 (safe to run anywhere)
    sync_env = gymn.vector.SyncVectorEnv([make_env() for _ in range(4)])
    obs, info = sync_env.reset()
    print("Sync obs shape:", obs.shape)
    # 为 Discrete 空间构造整型数组，长度等于子环境数量
    actions = np.array([int(sync_env.single_action_space.sample()) for _ in range(sync_env.num_envs)], dtype=np.int64)
    obs, rewards, terminations, truncations, infos = sync_env.step(actions)
    dones = np.logical_or(terminations, truncations)
    print("Sync step returns:", obs.shape, rewards.shape, dones)
    sync_env.close()

    # 2. 异步版本 (must be created under the main guard on Windows)
    async_env = gymn.vector.AsyncVectorEnv([make_env() for _ in range(4)])
    obs, info = async_env.reset()
    print("Async obs shape:", obs.shape)
    actions = np.array([int(async_env.single_action_space.sample()) for _ in range(async_env.num_envs)], dtype=np.int64)
    obs, rewards, terminations, truncations, infos = async_env.step(actions)
    dones = np.logical_or(terminations, truncations)
    print("Async step returns:", obs.shape, rewards.shape, dones)
    async_env.close()
