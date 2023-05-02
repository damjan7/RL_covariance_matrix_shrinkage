"""
Testing shrinkage using NN's
input: shrinkage intensity and some other features, such as historical volatility,
output: shrinkage intensity
"""

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
from RL.RL_dev import RL_covariance_estimators as rl_shrkg_est
import preprocessing_scripts.helper_functions as hf
from sklearn import preprocessing

import pickle

class Net(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.out = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # get action distribution
        # x = self.out(x)
        x = F.sigmoid(self.out(x))
        return x

def get_obs(cur_id):
    past_return_data = past_return_matrices[cur_id]
    past_price_data = past_price_matrices[cur_id]

    # get shrinkage intensity obtained by some estimator
    shrinkage, target = rl_shrkg_est.get_shrinkage_cov1Para(past_return_data)
    # maybe also need the target, as it should be used in the step() function to calculate
    # the actual estimator [is needed for global min pf as an input]

    # let's use something like 60 day past volatility
    # IDEA ##################################################
    # maybe take mean vola or something like a stock basket?
    # or MEAN of scaled stock volas
    volas = hf.get_historical_vola(past_price_data, days=60)  # shape: (num_stocks, ); np.array
    scaled_volas = preprocessing.MaxAbsScaler().fit_transform(volas.reshape(-1, 1))
    return shrinkage, scaled_volas, target

def calculate_pf_return_std_TENSORTEST(shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat):
    estimator = shrinkage_intensitiy * target + (1-shrinkage_intensitiy) * sample
    pf_return_daily, pf_std_daily = hf.calc_pf_weights_returns_vars_TENSOR(estimator, reb_days, pas_ret_mat, fut_ret_mat)
    return pf_return_daily, pf_std_daily

# DATA
return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices"
pf_size = 100

with open(rf"{return_data_path}\past_return_matrices_p{pf_size}.pickle", 'rb') as f:
    past_return_matrices = pickle.load(f)

with open(rf"{return_data_path}\future_return_matrices_p{pf_size}.pickle", 'rb') as f:
    future_return_matrices = pickle.load(f)

with open(rf"{return_data_path}\rebalancing_days_full.pickle", 'rb') as f:
    rebalancing_days_full = pickle.load(f)

with open(rf"{return_data_path}\past_price_matrices_p{pf_size}.pickle", 'rb') as f:
    past_price_matrices = pickle.load(f)


# INITIALIZATION
pf_size = 100
num_actions = 25  # currently no impact
hidden_size = 32
model = Net(2, num_actions, hidden_size)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = None  # just minimize pf std
epochs = 100



# Train Loop
epoch_loss = []
actions_over_all_epochs = []
for epoch in range(1, epochs+1):
    cur_epoch_loss = []
    loss = []
    for i in range(len(past_price_matrices)): # just iterate through every elem and update after every elem
        shrinkage, scaled_volas, target = get_obs(i)
        shrinkage = torch.Tensor(shrinkage.reshape(1, 1))
        target = torch.Tensor(target)
        obs = torch.cat([shrinkage, (torch.Tensor(scaled_volas)).mean().view(1, 1) ]).T
        #obs = torch.cat([shrinkage, torch.Tensor(scaled_volas)]).T
        sample = torch.Tensor(past_return_matrices[i].values.T) @ torch.Tensor(past_return_matrices[i].values)

        action = model(obs)
        actions_over_all_epochs.append(action.item())
        # action = probabilities
        #action = torch.argmax(action) / (action.shape[1] - 1)


        pf_return_daily, pf_std_daily = [], []  # shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat
        #print(2/0)
        for j in range(100):
            res1, res2 = calculate_pf_return_std_TENSORTEST(
                shrinkage_intensitiy=action,
                target=target,
                sample=sample,
                reb_days=None,  # this is not needed
                pas_ret_mat=past_return_matrices[j],
                fut_ret_mat=future_return_matrices[j]
            )
            pf_return_daily.append(res1)
            pf_std_daily.append(res2)
            break

        cur_epoch_loss.append(pf_std_daily[0].item())

        # update

        loss.append(res2)
        if (i+1) % 32 == 0:
            optimizer.zero_grad()
            loss = torch.stack(loss).mean()
            loss.backward(retain_graph=True)
            optimizer.step()
            loss = []

    #print(0/0)
    epoch_loss.append(np.mean(cur_epoch_loss))
    if epoch % 15 == 0:
        print("cool, epoch loss:", epoch_loss)
        print("cool")
        print(1/0)







