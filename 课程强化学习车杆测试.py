import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class CartPoleEnvWrapper(gym.Wrapper):
    def __init__(self, env, pole_length):
        super().__init__(env)
        self.env = env
        self.env.unwrapped.length = pole_length

def train_dqn(episode_count, pole_length, model, optimizer, epsilon_start=1.0):
    env = CartPoleEnvWrapper(gym.make('CartPole-v1'), pole_length)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    loss_fn = nn.MSELoss()

    epsilon = epsilon_start  # 初始探索概率
    epsilon_min = 0.01
    epsilon_decay = 0.975

    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0

        while not done:
            # env.render()
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.from_numpy(state).float()
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])

            target = reward + 0.95 * torch.max(model(torch.from_numpy(next_state).float())).item() * (not done)
            target_f = model(torch.from_numpy(state).float())
            target_f[0][action] = target
            loss = loss_fn(target_f, model(torch.from_numpy(state).float()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # if episode > episode_count - 10:
        #     epsilon = 0.0
        print(f'Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon}, Pole length: {pole_length}')

    env.close()
    return model, optimizer, epsilon

if __name__ == "__main__":
    state_size = 4  # CartPole state dimensions
    action_size = 2  # CartPole action dimensions
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters())

    lengths = [0.2, 0.4, 0.6, 0.8, 1.0]
    for length in lengths:
        epsilon_start = 1.0
        print(f"Training with pole length: {length}")
        model, optimizer, epsilon_start = train_dqn(100, length, model, optimizer, epsilon_start)