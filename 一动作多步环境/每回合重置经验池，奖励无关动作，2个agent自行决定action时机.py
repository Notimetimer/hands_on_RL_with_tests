import random
import pprint # 使用 pprint 模块让字典输出更美观

# 1. 模拟一个极简环境 (无需改动)
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
            return self.current_step, 0, True

        self.current_step += 1
        reward = 1.0
        done = self.current_step >= self.max_steps
        next_state = self.current_step
        
        return next_state, reward, done

# 2. Agent 类用于决策 (无需改动)
class Agent:
    """
    一个简单的智能体，用于决定何时以及如何行动。
    - 决策间隔是随机的。
    """
    def __init__(self, name, min_interval=5, max_interval=12):
        self.name = name
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._interval = 0
        print(f"智能体 '{self.name}' 已创建，决策间隔将在 [{min_interval}, {max_interval}] 之间随机。")

    def is_time_to_act(self, current_interval):
        """
        根据距离上次决策的步数判断是否到了决策时刻。
        """
        if current_interval >= self._interval:
            interval = random.randint(self.min_interval, self.max_interval)
            self._interval = interval
            print(f"决策点判断: {self.name} 是。下次决策将在 {interval} 步之后")
            return True
        return False
    
    def take_action(self, state):
        """
        根据当前状态产生一个动作。
        """
        action = state
        return action

    def update(self, transition_dict):
        """
        使用收集到的经验来更新策略。
        """
        # 只有真正收集了经验的智能体才需要执行更新
        if transition_dict and transition_dict['states']:
            print(f"\n[{self.name} Update] 智能体 '{self.name}' 正在使用经验池进行更新... (此处省略具体实现)")
        else:
            print(f"\n[{self.name} Update] 没有为智能体 '{self.name}' 提供经验，跳过更新。")
        pass

    def reset(self):
        """在新回合开始时重置智能体的内部状态。"""
        self._interval = 0


# 3. 模拟训练主循环 (多智能体版本)
if __name__ == "__main__":
    
    env = SimpleEnv()
    # 创建两个独立的智能体实例，给它们不同的名字和决策间隔范围以示区别
    agent1 = Agent(name="Learner", min_interval=4, max_interval=8)
    agent2 = Agent(name="Partner", min_interval=5, max_interval=10)
    
    num_episodes = 2   # 运行 2 个回合进行演示
    gamma = 1

    for i_episode in range(num_episodes):
        
        # --- 为 Agent 1 (Learner) 初始化经验池 ---
        transition_dict_agent1 = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        
        # --- 重置环境和所有智能体 ---
        agent1.reset()
        agent2.reset()
        state = env.reset()
        done = False
        
        # --- 为每个智能体维护独立的周期信息 ---
        # Agent 1 的信息 (用于学习)
        last_decision_state_1 = None
        current_action_1 = None
        cycle_accumulated_reward_1 = 0.0
        interval_counter_1 = -1

        # Agent 2 的信息 (只用于行动)
        current_action_2 = None
        interval_counter_2 = -1

        # --- 回合主循环 ---
        while not done:
            # 每个环境步骤，两个智能体的内部间隔计数器都增加
            interval_counter_1 += 1
            interval_counter_2 += 1
            
            current_step_in_episode = env.obs()

            # --- 智能体 1 (Learner) 的决策 ---
            if agent1.is_time_to_act(interval_counter_1):
                # 如果是决策点，重置其内部间隔计数器
                interval_counter_1 = 0
                
                # 存储上一个周期的经验 (与单智能体逻辑相同)
                if current_step_in_episode > 0:
                    transition_dict_agent1['states'].append(last_decision_state_1)
                    transition_dict_agent1['actions'].append(current_action_1)
                    transition_dict_agent1['rewards'].append(cycle_accumulated_reward_1)
                    transition_dict_agent1['next_states'].append(current_step_in_episode)
                    transition_dict_agent1['dones'].append(False)
                
                # 开始新的周期
                last_decision_state_1 = current_step_in_episode
                current_action_1 = agent1.take_action(last_decision_state_1)
                cycle_accumulated_reward_1 = 0.0
                print(f"    步数 {current_step_in_episode}: 智能体 '{agent1.name}' 决策，采取动作 {current_action_1}，开始新周期...")

            # --- 智能体 2 (Actor_Only) 的决策 ---
            if agent2.is_time_to_act(interval_counter_2):
                # 如果是决策点，重置其内部间隔计数器
                interval_counter_2 = 0
                # 它只产生一个新动作，不关心经验存储
                current_action_2 = agent2.take_action(current_step_in_episode)
                print(f"    步数 {current_step_in_episode}: 智能体 '{agent2.name}' 决策，采取动作 {current_action_2} (仅行动，不存储)")

            # --- 环境交互 ---
            # 我们选择 agent1 的动作来驱动环境。
            # 在一个真实的多智能体环境中，step函数可能会接收一个动作字典，如 env.step({'agent1': current_action_1, 'agent2': current_action_2})
            action_to_env = current_action_1
            next_state, reward, done = env.step(action_to_env)
            
            # 只有为 agent1 累积奖励，因为它才是学习者
            cycle_accumulated_reward_1 += gamma * reward  # 如果考虑到智能体死亡的问题，就把agent.dead 也放在这里

        # --- 回合结束处理 (只为 agent1) ---
        if last_decision_state_1 is not None:
            transition_dict_agent1['states'].append(last_decision_state_1)
            transition_dict_agent1['actions'].append(current_action_1)
            transition_dict_agent1['rewards'].append(cycle_accumulated_reward_1)
            transition_dict_agent1['next_states'].append(env.obs())
            transition_dict_agent1['dones'].append(True)

        print(f"\n回合 {i_episode + 1} 结束。最终状态: {env.obs()}")
        print(f"为智能体 '{agent1.name}' 采集到的经验池:")
        pprint.pprint(transition_dict_agent1)

        # 显式调用两个 agent 的 update，但只有 agent1 会有数据
        agent1.update(transition_dict_agent1)
        agent2.update({}) # 传入空字典，表示没有经验给它

        # 验证 agent1 经验池的长度
        print(f"\n验证 '{agent1.name}' 经验池中各个列表的长度：")
        for key, value in transition_dict_agent1.items():
            print(f"  - {key}: {len(value)}")
        assert len(transition_dict_agent1['states']) == len(transition_dict_agent1['next_states'])