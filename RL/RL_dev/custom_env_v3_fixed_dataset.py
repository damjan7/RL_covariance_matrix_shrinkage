"""
This script implements a custom environment that is intended to use with classical RL algorithms
the observation space / state space consists of some "opt" shrkg intensity and some additional
input parameters (such as historical volatility)
the action space consists of 10 discrete actions [corresponding to shrkg intensities between 0.1 and 0.9]
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
from collections import deque

# import shrinkage estimators
from RL.RL_dev import RL_covariance_estimators as rl_shrkg_est
from torch.nn.functional import mse_loss



class MyEnv(gym.Env):

    def __init__(self, shrk_data_path, pf_size):
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
        self.action_space = spaces.Discrete(21) # i.e. form 0 to 20


        # load the data into the environment
        # IDEA: could calculate all the intensities etc here so they do not to be caluclated over and over again

        self.shrk_data_path = shrk_data_path
        self.pf_size = pf_size
        with open(rf"{self.shrk_data_path}\fixed_shrkges_p{self.pf_size}.pickle", 'rb') as f:
            self.fixed_shrk_data = pickle.load(f)

        with open(rf"{self.shrk_data_path}\factor-1.0_p{self.pf_size}.pickle", 'rb') as f:
            self.optimal_shrk_data = pickle.load(f)



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



    def step(self, action):  # for now target is also a input, will be changed later! [get_obs gives target]
        '''
        This method calculates new state and the reward obtained by the action
        new state will just be next date
        done = True if last date is reached

        action = integer from 0 to 20 indicating shrinkage intensity
        --> shrinkage intensity is given by rounded number in np.linspace(0, 1, 21)
        all the corresponding rewards are stored, hence they do not need to be calculated
        '''
        shrkg = np.round(np.linspace(0, 1, 21), 2)[action]

        # action + 1 BECAUSE THE FIRST COLUMN IS THE HISTORICAL VOLATIEITIIEITOIEWP!!!!!!!!!!!!!!!!!!!!!!!!
        pf_std = self.fixed_shrk_data.iloc[self.state, action+1].astype(float)

        reward = -1 * (pf_std - self.fixed_shrk_data.iloc[self.state, 1:].astype(float).min()) ** 2

        # advance in time, i.e., state increases by 1 timestep
        self.state += 1
        if self.state >= self.fixed_shrk_data.shape[0]-1:  # since we start at zero
            self.done = True  # we reached the end of the dataset

        return self.state, reward, self.done, None  # None = Info, first return value should be obs


    def get_obs_space(self, cur_state):
        """
        given state, return opt shrk and hist vola [both are already calculated so easy]
        """
        hist_vola = self.optimal_shrk_data.iloc[cur_state, 1]  # col 1 is hist vola col
        shrk = self.optimal_shrk_data.iloc[cur_state, 0]  # col 0 is shrk col
        return hist_vola, shrk

    # train loop: take state --> return action
    # state is only index in my case: hence the step function

# shrk_data_path = r'C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets'

"""
pf_size = 100
myenv = MyEnv(shrk_data_path, pf_size)
myenv.reset() # returns next state, reward, done, None
#shkrg = myenv.get_obs_space()[0]
#tgt = myenv.get_obs_space()[2]
myenv.step(action)
"""

print("debug")

