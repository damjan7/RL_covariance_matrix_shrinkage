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
from RL.RL_dev.RL_covariance_estimators import cov1Para
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque


# Q1: should the estimator and the shrinkage intensities be supplied to the agent or estimated within
        # --> I think supplied makes more sense as training may take too long otherwise

# Q2: should I also supply weights?
        # I don't think I should


class MyEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # observation space;
        # = shrinkage intensity, and other factors?
        # can package it inside a gym.spaces.Dict


        # action space should be my shrinkage intensity
        # for now discrete as my RL algos are written for discrete cases only
        self.action_space = spaces.Discrete(100)
