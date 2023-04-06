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


class Policy(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.hidden = 128
        self.fc1 = nn.Linear(num_features, self.hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(self.hidden, 2)

        # these store the log probs of the sampled actions and the rewards of a given episode
        # they are used for the calculation of the loss
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        action_scores = self.fc3(x)

        return F.softmax(action_scores, dim=1)  # my policy

    def act(self, state):
        pass

def sample_action(state):
    '''
    Samples an action from our policy by creating a multinomial random variable with parameters given by
    our network (= probabilities)
    Also calculates the log likelihood and saves it in the policy directly
    '''
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy.forward(state)
    multinom = Categorical(probs)
    action = multinom.sample()
    policy.saved_log_probs.append(multinom.log_prob(action))
    return action.item()



def finish_ep(gamma, eps):
    '''
    the actual training step at end of an epoch (after trajectory has been conducted)
    '''
    rew_to_go = 0
    policy_loss = []
    returns = deque()
    for rew in policy.rewards[::-1]:  # get all the "rewards to go" starting from behind
        rew_to_go = rew + gamma * rew_to_go  # as we get closer to timestep 0, later rewards are discounted more
        returns.appendleft(rew_to_go)  # append every "reward to go" to the returns queue
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)   # decrease variance, still unbiased i think
    for log_prob, rew_to_go in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob*rew_to_go)  # the actual loss, pytorch will take the gradient of it
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()  # check what this does, we have 1 loss for every step in the trajectory but we want to backprop only 1 so we take the sum (it's actually just the formula)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# PARAMS
gamma = 0.99  # discount factor
seed = 8312
eps = np.finfo(np.float32).eps.item()
env = gym.make('CartPole-v1')
num_features = env.observation_space.shape[0]
policy = Policy(num_features)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)


def main():
    running_reward = 10
    for episode in count(1):
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning! Should never exceed 10'000 steps in cartpole
            action = sample_action(state)
            state, reward, done, _, = env.step(action)
            policy.rewards.append(reward)  # reward of a step/frame
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_ep(gamma, eps)  # optimizes NN

        if episode % 10 == 0:
            print(f"Episode {episode}, Last Reward: {ep_reward}, Avg Reward (10 ep): {running_reward}")
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
        if running_reward > env.spec.reward_threshold:
            print(f"Good job! You solved the environmnet. The avg reward in the last 10 episodes is {running_reward}.")
            break
        scheduler.step()


if __name__ == '__main__':
    main()
