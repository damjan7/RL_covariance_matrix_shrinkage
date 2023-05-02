"""
This file should contain a custom environment for testing RL algorithms
The idea of this environment is to supply the agent with a shrinkage intensity of some
covariance estimator (obtained by some shrinkage method)
and additional data, for example 60 day historical volatility
based on this the agent may or may not change the shrinkage intensity
"""

import numpy as np
import gym
from gym import spaces
from preprocessing_scripts import helper_functions as hf
import pickle
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque

# import shrinkage estimators
from RL.RL_dev import RL_covariance_estimators as rl_shrkg_est


# Q1: should the estimator and the shrinkage intensities be supplied to the agent or estimated within
        # --> I think supplied makes more sense as training may take too long otherwise

# Q2: should I also supply weights?
        # I don't think I should


class MyEnv(gym.Env):

    def __init__(self, return_data_path, pf_size):
        super().__init__()

        # observation space;
        # = shrinkage intensity, and other factors?
        # can package it inside a gym.spaces.Dict
        # maybe testing without 60_day vola first!
        # just use a concatenated vector as observation space, i.e. shrinkage intensity and vola in a vector
        self.observation_space = spaces.Dict(
            {
                "shrinkage_intensity": spaces.Discrete(100),  # actually continuous
                "60_day_vola": spaces.Discrete(100),  # would also be continuous actually
             }
        )
        # OR maybe easier, as it would rather work with existing algos, 1 discrete space of (2, 100)..

        # action space should be my shrinkage intensity
        # for now discrete as my RL algos are written for discrete cases only
        self.action_space = spaces.Discrete(100)


        # load the data into the environment
        # IDEA: could calculate all the intensities etc here so they do not to be caluclated over and over again

        self.return_data_path = return_data_path
        self.pf_size = pf_size
        with open(rf"{self.return_data_path}\past_return_matrices_p{self.pf_size}.pickle", 'rb') as f:
            self.past_return_matrices = pickle.load(f)

        with open(rf"{self.return_data_path}\future_return_matrices_p{self.pf_size}.pickle", 'rb') as f:
            self.future_return_matrices = pickle.load(f)

        with open(rf"{self.return_data_path}\rebalancing_days_full.pickle", 'rb') as f:
            self.rebalancing_days_full = pickle.load(f)


        with open(rf"{self.return_data_path}\past_price_matrices_p{self.pf_size}.pickle", 'rb') as f:
            self.past_price_matrices = pickle.load(f)


    def reset(self):
        '''
        This method should return the first state in our environment
        This could either be a random state, or just the first date in our dataset
        for now: first date in our dataset

        Also, it should create some stuff, such as rewards, weights, etc. to keep track of

        Finally, it should reset the environment
        '''
        self.state = 0  # only index --> denotes the first entry of rebalancing_dates['actual_reb_days']
                        # also makes a lot of sense since I always know "where I am"
                        # or alternatively return a tuple containing state idx, and other vars..

        self.rewards = []
        self.done = False

        return self.state, self.done  # or maybe do not return "done" but manually set it to False when resetting



    def step(self, action, target):  # for now target is also a input, will be changed later! [get_obs gives target]
        '''
        This method calculates new state and the reward obtained by the action
        new state will just be next date
        done = True if last date is reached

        action = shrinkage intensity
        '''
        # calculate weights and using them, calculate pf return and pf
        sample = self.past_return_matrices[self.state].values.T @ self.past_return_matrices[self.state].values
        sample = torch.Tensor(sample)
        target = torch.Tensor(target)
        sigmahat = action * target + (1 - action) * sample  # debug
        pfret, pfstd = hf.calc_pf_weights_returns_vars_TENSOR(
            sigmahat, None, self.past_return_matrices[self.state], self.future_return_matrices[self.state]
        )
        reward = - pfstd

        # advance in time, i.e., state increases by 1 timestep
        self.state += 1
        if self.state >= self.rebalancing_days_full.shape[0]-1:  # since we start at zero
            self.done = True  # we reached the end of the dataset

        return self.state, reward, self.done, None  # None = Info, first return value should be obs


    def get_obs_space(self):
        """
        returns observation space for a given state space ( =index)
        Should make use of it during the training loop
        """
        # This can all be written more efficiently, but want to keep it simple and clear as a first step
        date = self.rebalancing_days_full['actual_reb_day'].iloc[self.state]
        past_return_data = self.past_return_matrices[self.state]
        past_price_data = self.past_price_matrices[self.state]

        # calc the stuff I want my rl agent to have, i.e., shrinkage_intensity, vola, some factors
        # start small

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

        return shrinkage, scaled_volas, target  # shrinkage, volas are passed to the network



    ## train loop: take state --> return action
    # state is only index in my case: hence the step function


return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices"
pf_size = 100
myenv = MyEnv(return_data_path, pf_size)
myenv.reset()
shkrg = myenv.get_obs_space()[0]
tgt = myenv.get_obs_space()[2]
myenv.step(shkrg, tgt)

print("debug")

