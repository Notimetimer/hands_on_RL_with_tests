# # OpenAI的MPE环境开启可视化之后会在运行停止时让python崩掉，但是不可视化就不会有事。

# # 其中一个版本的实例，似乎并不会动
# import make_env
#
# env = make_env.make_env('simple')
# for _ in range(50):
#     env.render()
# env.close()
# # 另一个版本的实例（捕食者、猎物、食物、障碍物、树林）
import time
import numpy as np
import make_env as make_env

env = make_env.make_env('simple_world_comm')
obs = env.reset()
print(env.observation_space)
print(env.action_space)

steps = 0
print(steps)
print(obs)

for _ in range(25):
    steps += 1
    print(steps)
    action_n = [np.array([0, 1, 0, 1, 0, 1, 1, 1, 1],dtype=np.float32),
                np.array([0, 10, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32)]
    next_obs_n, reward_n, done_n, _ = env.step(action_n)
    print(next_obs_n)
    print(reward_n)
    print(done_n)
    # env.render() # 可视化
    time.sleep(0.1)
    if all(done_n):
        break
env.close()