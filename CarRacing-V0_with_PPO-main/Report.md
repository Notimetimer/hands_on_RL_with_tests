# Introduction

For this project, you will train a car (agent) in a racing environment from the OpenAI Gym toolkit.
Gym is used for developing and comparing reinforcement learning algorithms.

Reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles in track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

State consists of 96x96 pixels and there are some indicators shown at the bottom of the window for the state of RGB buffer.

The task is episodic and an episode finishes when all tiles are visited.

CarRacing-v0 defines "solving" as getting average reward of 900 over 100 consecutive trials.

# Learning Algorithm

Proximal Policy Optimization (PPO) with clipped objective is a policy gradient method for reinforcement learning.

In our implementation, we maintain two policy networks. The first one is the current policy that we want to refine. And the second is the policy that we last used to collect samples.

We then compute a ratio between the new policy and the old policy to measure the difference between them.

We construct a new objective function to clip the estimated advantage function if the new policy is far away from the old policy.

If the probability ratio between the new policy and the old policy falls outside the range (1 — ε) and (1 + ε), the advantage function will be clipped.

Here is our algorithm :

<center><img src="https://github.com/rym-oualha/CarRacing-V0_with_PPO/blob/main/image/ppo-clip.png" width="600"></center>

# Model architecture

* 6 sequential conv2d layers
* 5 linear layers

# Hyperparameters

We have used the follwing hyperparameters in order to train the agent:
* GAMMA : the discount factor 0.99
* MAX_SIZE : max size of the buffer 2000 
* BATCH : the batch size of the training is 128 
* EPS : epsilon value is 0.1
* LEARNING_RATE : lr is 0.001 

# Results
After 1000 episodes, our agent was able to reach an average score of 66.45.
<center><img src="https://github.com/rym-oualha/CarRacing-V0_with_PPO/blob/main/image/episodes.png" width="600"></center>
<center><img src="https://github.com/rym-oualha/CarRacing-V0_with_PPO/blob/main/image/courbe.png" width="600"></center>

# Ideas for Future Work

We still can improve our results by:
* Training for more episodes
* Tuning the hyperparameters
