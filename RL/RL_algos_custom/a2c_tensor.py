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

from RL.RL_dev.custom_tensor_env import MyEnv


class ActorCriticDiscrete(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        ### NOTE: can share some layers of the actor and critic network as they have the same structure
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))


        self.actor_head = nn.Linear(int(hidden_size/2), num_actions)  # probabilistic mapping from states to actions
        self.critic_head = nn.Linear(int(hidden_size/2), 1)  # estimated state value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # get action distribution
        action_probs = F.softmax(self.actor_head(x), dim=1)
        # how 'good' is the current state?
        state_value = self.critic_head(x)
        return action_probs, state_value


########## TRAIN LOOP
def train():
    for episode in range(num_episodes):  # 1 episode is actually 1 sweep through the whole dataset!!!!!!
        state, done = env.reset()  # reset env
        ep_log_probs = []
        ep_rewards = []
        ep_state_values = []
        entropy = []
        running_reward = [10 for _ in range(10)]
        for frame in count(1):  # will break out once done = True
            # For Loss Calculation: get action, state value (critic), and log probs of action under policy
            shrinkage, scaled_volas, target = env.get_obs_space()
            obs = torch.cat([torch.Tensor(shrinkage.reshape(1,1)), torch.Tensor(scaled_volas)]).T


            action_probs, state_value = net.forward(obs)
            ep_state_values.append(state_value.squeeze(dim=0))  # append state values
            multinom_var = Categorical(action_probs)
            entropy.append(multinom_var.entropy())
            action = multinom_var.sample()
            log_prob = multinom_var.log_prob(action)
            ep_log_probs.append(log_prob)
            # as we have discrete random variables from 1 to 100 we divide action by 100
            # OR DO, SHRINKAGE + action.item() / 100
            state, reward, done, _, = env.step(action / num_actions, target)
            ep_rewards.append(reward)
            if done:
                break
        # once we went through the episode, update weights and print some stuff
        running_reward.append(sum(ep_rewards))
        running_reward = running_reward[1:]
        if episode % 30 == 0:
            print(f"Reward of Episode {episode}: {np.mean(ep_rewards)} \t "
                  f"Avg Reward of Last 10 Episodes: {sum(running_reward)/10}")

        # Optimization Step:

        disc_rewards = []
        R = 0
        #for reward in ep_rewards[::-1]:
        #    R = reward # + gamma*R
        #    disc_rewards.insert(0, R)
        disc_rewards = ep_rewards
        disc_rewards = torch.Tensor(disc_rewards)
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 0.0001)  # normalize
        ep_state_values = torch.cat(ep_state_values)
        ep_log_probs = torch.cat(ep_log_probs)

        advantage = disc_rewards - ep_state_values
        policy_loss = -(ep_log_probs*advantage).mean()
        value_loss = 0.5 * (advantage.pow(2)).mean()
        loss = policy_loss + value_loss  # - eps * torch.stack(entropy).mean()

        if episode % 30 == 0:
            print("break")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


###### INIT AND TRAIN
#env = gym.make('CartPole-v1')
# ONLY TRAIN ON SMALL SUBSET OF DATA
return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices"
pf_size = 100
env = MyEnv(return_data_path, pf_size)
gamma = 1  # since we actually only have 1 step transitions
lr = 0.05
num_features = pf_size+1
num_actions = 25
net = ActorCriticDiscrete(num_features, num_actions, 128)
num_episodes = 3000

optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

eps = 1e-3

train()