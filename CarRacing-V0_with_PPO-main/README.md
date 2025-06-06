# CarRacing-V0_with_PPO

## Introduction
Welcome to the CarRacing-V0 with PPO project! This project is aimed at training an autonomous car agent to navigate the CarRacing environment using Deep Reinforcement Learning (DRL). The Proximal Policy Optimization (PPO) algorithm is employed to optimize the car agent's policy, enabling it to efficiently race around the track in the OpenAI Gym toolkit environment.
For this project, you will train a car (agent) in a racing environment. 
<center><img src="https://github.com/rym-oualha/CarRacing-V0_with_PPO/blob/main/image/result.gif" width="600"></center>
Reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles in track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points. 

State consists of 96x96 pixels and there are some indicators shown at the bottom of the window for the state of RGB buffer.

The task is episodic and an episode finishes when all tiles are visited. 

CarRacing-v0 defines "solving" as getting average reward of 900 over 100 consecutive trials.

## Environment
The CarRacing-V0 environment is a part of the OpenAI Gym toolkit. You can find more information and documentation on this environment here.

## Getting Started

Clone the repository

```

git clone https://github.com/rym-oualha/CarRacing-V0_with_PPO.git
cd CarRacing-V0_with_PPO

```

To train your own agent (play with hyperparameters and episodes numbers) use *train.ipynb* and to test our agent use *test.ipynb*.
You can simply run them with Google Colab or Kaggle Notebooks.

## Train the agent 

In order to train the agent you have to:

  1. Initialize the agent

  2. Evaluate state and action space

  3. Train the agent using Proximal Policy Optimization (PPO).

  4. Iterate until agent reaches an average reward of 900 over 100 consecutive trials

You can train the agent following the instructions in the notebook *train.ipynb*.
