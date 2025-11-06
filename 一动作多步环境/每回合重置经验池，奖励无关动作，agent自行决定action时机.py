import random
import pprint # 使用 pprint 模块让字典输出更美观

# 1. 模拟一个极简环境
class SimpleEnv:
    """
    一个极简的环境：
    - 状态就是当前的步数。
    - 每个回合的长度是随机的。
    """
    def __init__(self):
        self.current_step = 0
        self.max_steps = 0

    def reset(self):
        """重置环境，开始新回合。"""
        self.current_step = 0
        # 随机设置回合长度在 19 到 30 之间
        self.max_steps = random.randint(19, 30) 
        print(f"\n---[ 环境重置: 新回合总长度为 {self.max_steps} 步 ]---")
        return self.obs()

    def obs(self):
        """获取当前观测/状态。"""
        return self.current_step

    def step(self, action):
        """
        环境前进一步。
        - action: 在这个例子中我们忽略它，因为状态只和步数有关。
        - 返回: next_state, reward, done
        """
        if self.current_step >= self.max_steps:
            # 如果已经结束，直接返回
            return self.current_step, 0, True

        self.current_step += 1
        reward = 1.0  # 每一步都给一个固定的奖励，方便验证
        done = self.current_step >= self.max_steps
        next_state = self.current_step
        
        return next_state, reward, done

# 2. 新增一个 Agent 类用于决策
class Agent:
    """
    一个简单的智能体，用于决定何时以及如何行动。
    - 决策间隔是随机的。
    """
    def __init__(self, min_interval=5, max_interval=12):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._interval = 0
        print(f"智能体已创建，决策间隔将在 [{min_interval}, {max_interval}] 之间随机。")

    def is_time_to_act(self, current_interval):
        """
        根据当前步数判断是否到了决策时刻。
        """
        if current_interval >= self._interval:
            # 到达或超过了计划的决策点，是时候行动了
            # 计算下一个决策点
            interval = random.randint(self.min_interval, self.max_interval)
            self._interval = interval
            print(f"    (决策点判断: 是。下次决策将在 {interval} 步之后)")
            return True
        return False
    
    def take_action(self, state):
        """
        根据当前状态产生一个动作。
        在这个简化例子中，动作就等于状态。
        """
        action = state
        return action

    def update(self, transition_dict):
        """
        使用收集到的经验来更新策略。
        在这个例子中，此方法为空，仅作演示。
        """
        # 在实际的强化学习中，这里会执行类似策略更新、价值函数拟合等操作
        # print("\n[Agent Update]... (此处省略具体实现)")
        pass

    def reset(self):
        """在新回合开始时重置智能体的内部状态。"""
        self._interval = 0


# 3. 模拟训练主循环
if __name__ == "__main__":
    
    env = SimpleEnv()
    agent = Agent(min_interval=4, max_interval=10) # 创建智能体，设置随机间隔范围
    num_episodes = 3   # 运行 3 个回合进行演示
    gamma = 1 # 0.9
    

    for i_episode in range(num_episodes):
        
        # 每个回合开始时，重置经验池、环境和智能体
        transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        agent.reset()
        state = env.reset()
        done = False
        
        # 用于存储一个动作周期内的信息
        last_decision_state = None
        current_action = None
        cycle_accumulated_reward = 0.0
        current_interval = -1

        # --- 回合主循环 ---
        # 只要回合没结束，就一直运行
        while not done:
            current_interval += 1 # 初始是0
            current_step_in_episode = env.obs()

            # --- 智能体决策 ---
            # 使用 agent 的方法来判断是否是决策点
            if agent.is_time_to_act(current_interval):
                current_interval = 0
                
                # **关键点 1: 完成并存储【上一个】动作周期的经验**
                # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                if current_step_in_episode > 0:
                    transition_dict['states'].append(last_decision_state)
                    transition_dict['actions'].append(current_action)
                    transition_dict['rewards'].append(cycle_accumulated_reward)
                    transition_dict['next_states'].append(current_step_in_episode) # 当前状态是上个周期的 next_state
                    transition_dict['dones'].append(False) # 没结束，所以是 False
                
                # **关键点 2: 开始【新的】一个动作周期**
                # 1. 记录新周期的起始状态
                last_decision_state = current_step_in_episode
                # 2. Agent 产生一个动作（显式调用）
                current_action = agent.take_action(last_decision_state)
                # 3. 重置周期奖励累加器
                cycle_accumulated_reward = 0.0
                print(f"步数 {current_step_in_episode}: 智能体决策，采取动作 {current_action}，开始新周期...")

            # --- 环境交互 ---
            # 无论是否是决策点，环境都正常运转
            next_state, reward, done = env.step(current_action)
            
            # 累积当前动作周期内的奖励
            cycle_accumulated_reward += gamma * reward

        # --- 回合结束处理 ---
        # **关键点 3: 存储【最后一个】不完整的动作周期的经验**
        # 循环结束后，最后一个动作周期因为 done=True 而中断，必须在这里手动存入
        if last_decision_state is not None:
            transition_dict['states'].append(last_decision_state)
            transition_dict['actions'].append(current_action)
            transition_dict['rewards'].append(cycle_accumulated_reward)
            transition_dict['next_states'].append(env.obs()) # 最后的 next_state 是环境的最终状态
            transition_dict['dones'].append(True)

        print(f"\n回合 {i_episode + 1} 结束。最终状态: {env.obs()}")
        print("采集到的经验池 transition_dict:")
        pprint.pprint(transition_dict)

        # 显式调用 agent.update
        agent.update(transition_dict)

        # 验证长度是否一致
        print("\n验证经验池中各个列表的长度：")
        for key, value in transition_dict.items():
            print(f"  - {key}: {len(value)}")
        assert len(transition_dict['states']) == len(transition_dict['next_states'])