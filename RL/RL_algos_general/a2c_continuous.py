'''
The difference with continous action spaces is to have 2 output for every number of action outputs
This is because we have an output for the mean values and the variances.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque
from itertools import count


class ActorCriticContinuous(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        # NOTE: can share some layers of the actor and critic network as they have the same structure
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.mu = nn.Sequential(
            nn.Linear(int(hidden_size/2), num_actions),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(int(hidden_size/2), num_actions),
            nn.Softplus()
        )
        self.critic_head = nn.Linear(int(hidden_size/2), 1)  # estimated state value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # get action distribution
        mu = self.mu(x)
        var = self.var(x)
        # how 'good' is the current state?
        state_value = self.critic_head(x)
        return mu, var, state_value

# TRAIN LOOP -----------------------------------------------------------
def train():
    running_reward = [10 for _ in range(10)]
    for episode in range(num_episodes):
        state = env.reset()  # reset env
        ep_log_probs = []
        ep_rewards = []
        ep_state_values = []
        entropy = []
        for frame in count(1):  # will break out once done = True
            # For Loss Calculation: get action, state value (critic), and log probs of action under policy
            state = torch.from_numpy(state).float().unsqueeze(0).squeeze(-1).to(device)  # transform to work with torch
            mu, var, state_value = net.forward(state)
            ep_state_values.append(state_value.squeeze(dim=0))  # append state values

            # before multinom, now sample from normal --------
            normal_variable = Normal(mu, var)
            entropy.append(normal_variable.entropy())
            action = normal_variable.sample()
            log_prob = normal_variable.log_prob(action)
            # -------------------------------------------------

            ep_log_probs.append(log_prob)
            state, reward, done, _, = env.step(np.array(action))
            ep_rewards.append(reward)
            if done:
                break
        # once we went through the episode, update weights and print some stuff
        running_reward.append(sum(ep_rewards))
        running_reward = running_reward[1:]
        if episode % 10 == 0:
            print(f"Reward of Episode {episode}: {sum(ep_rewards)} \t "
                  f"Avg Reward of Last 10 Episodes: {sum(running_reward)/10}")

        # Optimization Step:
        disc_rewards = []
        R = 0
        for reward in ep_rewards[::-1]:
            R = reward + gamma*R
            disc_rewards.insert(0, R)
        disc_rewards = torch.Tensor(disc_rewards)
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 0.0001)  # normalize disc rewards
        ep_state_values = torch.cat(ep_state_values)
        ep_log_probs = torch.cat(ep_log_probs)

        advantage = disc_rewards - ep_state_values
        policy_loss = -(ep_log_probs*advantage).mean()
        value_loss = 0.5 * (advantage.pow(2)).mean()
        loss = policy_loss + value_loss - eps * torch.stack(entropy).mean()  # more exploration with higher entropy regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()




## PARAMS
env = gym.make('MountainCarContinuous-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gamma = 0.99
lr = 0.01
num_features = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
net = ActorCriticContinuous(num_features, num_actions, 32).to(device)
num_episodes = 3000

optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

eps = 1e-3

train()






