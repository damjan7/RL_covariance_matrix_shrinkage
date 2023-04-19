import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data.dataloader import DataLoader

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

    def __init__(self, num_stocks):
        super(Net, self).__init__()
        self.state_space = int( num_stocks * (num_stocks+1) / 2)
        self.action_space = 1  # discretized 100 actions btw 0 and 1, here will use continuous
        self.hidden = (num_stocks * num_stocks+1)
        self.hidden2 = int((num_stocks * num_stocks+1) / 8)
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=True)
        self.l2 = nn.Linear(self.hidden, self.hidden2, bias=True)
        self.l3 = nn.Linear(self.hidden2, self.action_space, bias=True)  # output Q for every possible action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.sigmoid(x)
        return x


# init
model = Net(num_stocks=30)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# params
numepochs = 1000

# let's just randomly pick indices for epoch as our samples are indepenedent anyway


def calculate_pf_return_std(shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat):
    estimator = shrinkage_intensitiy * target + (1-shrinkage_intensitiy) * sample
    pf_return_daily, pf_std_daily = hf.calc_pf_weights_returns_vars(estimator, reb_days, pas_ret_mat, fut_ret_mat)
    return pf_return_daily, pf_std_daily



for epoch in range(numepochs):
    epoch_loss = 0
    indices = np.random.choice(len(past_ret_matrices), len(past_ret_matrices))
    for idx in indices:
        sample, target = cov1Para(past_ret_matrices[idx])
        upper_indices = torch.triu_indices(30, 30, offset=0)  # get indices of upper triangular matrix
        covmat = torch.Tensor(sample.to_numpy())[upper_indices[0], upper_indices[1]]  # maybe need to flatten?
        shrinkage_intensity = model.forward(covmat)  # returns the shrinkage intensities, according to them, calc the weights and the return

        pf_return_daily, pf_std_daily = calculate_pf_return_std(
            shrinkage_intensity, target, sample, rebalancing_days_full.iloc[idx, ],
            past_ret_matrices.iloc[idx, ], fut_ret_matrices.iloc[idx, ]
        )

        # need to minimize pf_std_daily


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("k")
