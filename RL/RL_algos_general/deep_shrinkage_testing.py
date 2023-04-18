import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque
import preprocessing_scripts.helper_functions as hf
from RL.RL_dev.RL_covariance_estimators import cov1Para

import pickle
base_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
# The rb indicates that the file is opened for READING in binary mode.
with open(rf"{base_path}\past_return_matrices_p30.pickle", 'rb') as f:
    past_ret_matrices = pickle.load(f)
with open(rf"{base_path}\future_return_matrices_p30.pickle", 'rb') as f:
    fut_ret_matrices = pickle.load(f)
with open(rf"{base_path}\rebalancing_days_full.pickle", 'rb') as f:
    rebalancing_days_full = pickle.load(f)
with open(rf"{base_path}\trading_days.pickle", 'rb') as f:
    trading_days = pickle.load(f)

print("loaded necessary data")


'''
Given a covariance matrix and a shrinkage target
'''


class Net(nn.Module):
    '''
    A Network to approximate the Q-function (= Q value for every state action pair)
    '''

    def __init__(self, num_features, num_actions):
        super(Net, self).__init__()
        self.state_space = num_features
        self.action_space = num_actions
        self.hidden = 100
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=True)
        self.l2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.l3 = nn.Linear(self.hidden, self.action_space, bias=True)  # output Q for every possible action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# init
model = Net()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# params
numepochs = 1000

for epoch in range(numepochs):
    loss = 0
