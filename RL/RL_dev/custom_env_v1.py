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

# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
"""
The policy is a mapping from the current environment observation to a probability distribution of the actions to be
 taken. Within an agent, the policy is implemented by a function approximator with tunable parameters and a 
 specific approximation model, such as a deep neural network.
 --> Hence, my current environment observations must be my ?sample? covariance matrices or what?
"""


# need a custom environment that follows gym interface
# first idea is to build an RL agent using the p largest stocks and its past data
# also the 252 past trading days as for the other shrinkage methods?
class MyCustomEnv(gym.Env):
    def __init__(self, return_data_path, pf_size):
        super().__init__()

        # Define action and observation space
        # They must be gym.spaces objects    # Example when using discrete actions:
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,))  # i.e. shrinkage intensity?
        # BUT; according to stable-baselines3, I should normalize my actions between -1 and 1
        # because most RL algos rely on a gaussian distr. for cont. actions, hence:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        # let's first work with discretized action spaces so I do not need to
        # change any code
        self.action_space = spaces.Discrete(100)  # i.e. from 0 - 99, just map it to (0,1) later

        # should this be my cov mat estimator? I.e., -100, 100, shape = (p, p)
        # ..or even in -100'000 and 100'000 for safety
        # or just my weights?
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=
                        (100, 100), dtype=np.float64)  # let's do shape = 100,100 at first, i.e. 100 stocks
        # thats correct, but we don't actually have 100x100 params when looking at the covmat
        # rather we have n*(n+1)/2 free params
        self.observation_space = spaces.Box(low=-100, high=100, shape = (100*101/2), dtype=np.float64)

        # self.start_date = start_date
        # self.end_date = end_date
        # now could implement this nicely to create the datasets etc. but I already
        # have everything, so I will just import it and use it as is
        self.return_data_path = return_data_path
        self.pf_size = pf_size

        with open(rf"{self.return_data_path}\past_return_matrices_p{self.pf_size}.pickle", 'rb') as f:
            self.past_return_matrices = pickle.load(f)

        with open(rf"{self.return_data_path}\future_return_matrices_p{self.pf_size}.pickle", 'rb') as f:
            self.future_return_matrices = pickle.load(f)

        with open(rf"{self.return_data_path}\rebalancing_days_full.pickle", 'rb') as f:
            self.rebalancing_days_full = pickle.load(f)

        # need my shrinkage target somewhere so the RL agent only
        # has to determine the shrinkage intensity

        self.state = None
        self.reward = None
        self.done = False
        self.daily_returns_weighted = []  # store the daily future returns (NOT demeaned), weighted!
        self.weights = []

        self.start_reb_date = self.rebalancing_days_full['actual_reb_day'][0]
        self.end_reb_date = self.rebalancing_days_full['actual_reb_day'].iloc[-1]

        # self state should start at the first rebalancing date!?
        # or work with index, self.state = 0


    def step(self, action):
        # Execute one time step within the environment
        # any step doesn't change anything in the environment, it just changes the reward we get!
        # of course we need to store all the daily returns to calculate the average over the whole time frame

        '''
        The step method usually contains most of the logic of your environment.
        It accepts an action, computes the state of the environment after applying that action and
        returns the 4-tuple (observation, reward, done, info).
        Once the new state of the environment has been computed,
        we can check whether it is a terminal state, and we set done accordingly.
        :param action: the shrinkage inensitiy
        :return: next_state, reward, done, info [when calling env.step(action)]
        '''
        # i.e. takes an action --> shrinkage inensity
        # calculate cov mat estimator and future rewards, say for the next 21 days as done before
        # the reward we'll look at is the (negative) variance, as we want to maximize that, i.e.,
        # keep variance as small as possible

        X = self.past_return_matrices[self.state]  # the PAST return data for the corresponding rebalancing date
        sample, target = cov1Para(X)
        sigmahat = action * target + (1 - action) * sample.to_numpy()
        weights = hf.calc_global_min_variance_pf(sigmahat)

        # calculate the future reward using our new estimator
        # I have calculated the weights (using global min variance)
        # now i have to calc the variance of my portfolio in next 21 days
        # PF variance is given by weights.T @ CovMat @ weights
        # this is my reward!
        fut_ret = self.future_return_matrices[self.state].values  # return matrix
        fut_ret_demeaned = hf.demean_return_matrix(fut_ret)  # N x p
        fut_covmat = fut_ret_demeaned.T @ fut_ret_demeaned  # p x p covmat
        pf_var = weights.T @ fut_covmat @ weights
        # as we naturally maximize reward, our reward is the negative pf_variance
        # i hope this makes sense
        reward = - pf_var

        tmp_daily_returns_weighted = fut_ret @ weights
        self.daily_returns_weighted.append(tmp_daily_returns_weighted)
        self.weights.append(weights)

        # go one rebalancing date further
        self.state += 1  # our state is just the row-INDEX of the rebalancing_days dataframe
        if self.state >= self.rebalancing_days_full.shape[0]-1:  # since we start at zero
            self.done = True  # we reached the end of the dataset

        observation = np.ones((100, 100))

        # info contains some additional stuff I want to know
        info = {"daily_returns_weighted": tmp_daily_returns_weighted, "weights": weights}

        #return observation, reward, tmp_daily_returns_weighted, self.done, info
        return observation, reward, self.done, info




    def reset(self):
        # Reset the state of the environment to an initial state
        # for our environment this should mean just returning to the first rebalancing date?
        # Not sure how exactly I want to implement this
        """
        The reset method should return a tuple of the initial observation and some auxiliary information.
        We can use the methods _get_obs and _get_info that we implemented earlier for that:
        In my case this initial observation would be the first rebalancing date?
        :return:
        """
        self.state = 0  # should be start date or something [taken from rebalancing date full -> actual pred day]
        self.reward = None
        self.daily_returns_weighted = []
        self.weights = []
        self.done = False

        return np.ones((100, 100))  # just random [state/observation]

    def render(self, mode='human', close=False):
        pass
        # Render the environment to the screen

    def close(self):
        pass


return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
pf_size = 30
env = MyCustomEnv(return_data_path, pf_size)

daily_returns_all_episodes = []



from RL.RL_algos_general.DQN_custom import  main

main(env)













'''
for episode in range(1, episodes+1):
    state = env.reset()
    daily_weighted_returns_full = []  # in the end should have daily weighted returns for whole period here!
    # the daily returns are used to calculate the variance over the whole period


    while not env.done:
        action = env.action_space.sample()
        n_state, reward, daily_rets_weighted, env.done, info = env.step(action)
        daily_weighted_returns_full += list(daily_rets_weighted)
        if env.done:
            daily_returns_all_episodes.append(daily_weighted_returns_full)
'''