import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque
from itertools import count


class ActorCriticDiscrete(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        ### NOTE: can share some layers of the actor and critic network as they have the same structure
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))

        ### In the implementations I found online, the output of the critic is only 1-d
        ### i.e. kind of only the state value not the state action value???
        self.actor_head = nn.Linear(int(hidden_size/2), num_actions)
        self.critic_head = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # get action distribution
        action_probs = F.softmax(self.actor_head(x), dim=1)
        # how 'good' is the current state?
        state_value = self.critic_head(x)
        return action_probs, state_value

#### TRAIN LOOP
def train():
    for episode in range(num_episodes):
        state = env.reset()  # reset env
        ep_log_probs = []
        ep_rewards = []
        ep_state_values = []
        entropy = []
        running_reward = [10 for _ in range(10)]
        for frame in count(1):  # will break out once done = True
            # For Loss Calculation: get action, state value (critic), and log probs of action under policy
            state = torch.from_numpy(state).float().unsqueeze(0)  # transform to work with torch
            action_probs, state_value = net.forward(state)
            ep_state_values.append(state_value.squeeze(dim=0))  # append state values
            multinom_var = Categorical(action_probs)
            entropy.append(multinom_var.entropy())
            action = multinom_var.sample()
            log_prob = multinom_var.log_prob(action)
            ep_log_probs.append(log_prob)
            state, reward, done, _, = env.step(action.item())
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
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 0.0001)
        ep_state_values = torch.cat(ep_state_values)
        ep_log_probs = torch.cat(ep_log_probs)

        advantage = disc_rewards - ep_state_values
        policy_loss = -(ep_log_probs*advantage).mean()
        value_loss = 0.5 * (advantage.pow(2)).mean()
        loss = policy_loss + value_loss  # - eps * torch.stack(entropy).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()




## PARAMS
env = gym.make('CartPole-v1')
gamma = 0.99
lr = 0.005
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
net = ActorCriticDiscrete(num_features, num_actions, 128)
num_episodes = 3000

optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

eps = 1e-3

train()






