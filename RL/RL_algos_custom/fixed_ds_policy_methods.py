import pandas as pd
import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset

# for action probs
from torch.distributions import Categorical

# plotting
import matplotlib.pyplot as plt

from RL.RL_algos_custom import eval_funcs


'''
In this script, will try to implement some kind of policy method without looking at too many other examples
but rather from my understanding..

actually it shouldn't be too different from the other script with the fixed ds training
now we approximate the policy function (i.e. using a NN), then sample actions and update according to
the reinforce algorithm (for example)

Note, for example, REINFORCE, uses the *complete* return from time t (Gt) which includes all future rewards
again, I don't think this makes much sense as we can view everything as 1 step episodes

******Maybe read into contextual gradient bandits oder so?!

policy weights are parameterized by a NN --> NN outputs action probabilities
then we need to sample actions from a categorical though
--> the structure is the same as in the fixed ds training (where we estimate the q(s, a)) function
here we do NOT directly optimize the weights of the policy network according to rewards
but we sample policies from the network and according to them we update the network

--> NOTE; WE FIRST SAMPLE DISCRETELY, I.E. FROM THE 21 ACTIONS WE HAVE TO MAKE IT EASIER
'''

# IMPORT SHRK DATASETS
shrk_data_path = r'C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets'
pf_size = 100
with open(rf"{shrk_data_path}\fixed_shrkges_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{shrk_data_path}\factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\factor_data"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too
start_date = '1980-01-15'
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]


# define policy network
class PolicyNetwork(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))


        self.actor_head = nn.Linear(int(hidden_size/2), num_actions)  # probabilistic mapping from states to actions
        self.critic_head = nn.Linear(int(hidden_size/2), 1)  # estimated state value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # get action distribution
        action_probs = F.softmax(self.actor_head(x), dim=1)
        # how 'good' is the current state?, not needed here
        state_value = self.critic_head(x)
        return action_probs

# define custom cataset
class MyDataset(Dataset):

    def __init__(self, factors, fixed_shrk_data, optimal_shrk_data):
        self.factors = factors
        self.fixed_shrk_data = fixed_shrk_data
        self.optimal_shrk_data = optimal_shrk_data

    def __len__(self):
        return self.factors.shape[0]

    def __getitem__(self, idx):
        # inputs multiplied by 100 works better
        inp = torch.Tensor(np.append(self.factors.iloc[idx, :].values, self.optimal_shrk_data.iloc[idx, 1])) * 100
        labels = torch.Tensor(np.array(self.fixed_shrk_data.iloc[idx, 2:].values, dtype=float))
        # for labels: .view(1, -1) not nee


# init and train

# PARAMETERS:
num_epochs = 100
lr = 1e-4
num_features = factors.shape[1] + 1  # all 13 factors + opt shrk  [= INPUT DATA]
num_actions = fixed_shrk_data.shape[1] - 2  # since 1 col is dates, 1 col is hist vola
hidden_layer_size = 128
net = PolicyNetwork(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()

# MANUAL TRAIN LOOP


def train_manual():

    train_indices = (0, int(factors.shape[0] * 0.7))
    val_indices = (int(factors.shape[0] * 0.7), factors.shape[0])


    for epoch in range(1, num_epochs+1):
        epoch_loss=[]
        validation_loss = []
        val_preds = []
        actual_argmin_validationset = []
        train_pf_std = []

        for i in range(train_indices[1]):
            inp = torch.Tensor(np.append(factors.iloc[i, :].values, optimal_shrk_data.iloc[i, 1])) * 100
            out = net(inp.view(1, -1))  # instead of q(s, a), out is now to be interpreted as action probs

            multinom_dist = Categorical(out)
            action = multinom_dist.sample()  # this is the action, and according to it calculate loss
            # action is between 0 and 20

            # hacking solution --> need to solve the problem that cols/rows of my data are of dtype=object !
            pf_std = torch.Tensor(np.array(fixed_shrk_data.iloc[i, 2:].values, dtype=float))[action]

            train_pf_std.append(pf_std.item())

            # CALC LOSS AND BACKPROPAGATE
            # recall -> gradient descent so need to have a loss fct to minimize
            # pf_std should be minimal
            optimizer.zero_grad()
            log_prob =  multinom_dist.log_prob(action)  # log prob of policy dist evaluated at sampled action
            loss = pf_std * log_prob  # like policy gradient  [reward * ]
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        # end of epoch statistics
        print(f"Loss of Epoch {epoch} (mean and sd): {np.mean(epoch_loss)}, {np.std(epoch_loss)}")
        print(f"Training Set pf std of epoch {epoch}: {np.mean(train_pf_std)}, {np.std(train_pf_std)}" )
        if epoch % 10 == 0:
             print("break :-)")

        # validation
        net.eval()
        with torch.no_grad():
            for i in range(val_indices[0], val_indices[1]):
                inp = torch.Tensor(np.append(factors.iloc[i, :].values, optimal_shrk_data.iloc[i, 1])) * 100
                out = net(inp.view(1, -1))  # instead of q(s, a), out is now to be interpreted as action probs

                # now just predict with arg max!?
                action = torch.argmax(out)
                pf_std = torch.Tensor(np.array(fixed_shrk_data.iloc[i, 2:].values, dtype=float))[action]
                val_preds.append(action.item())
                validation_loss.append(pf_std.item())  # just pf std
                actual_min_action = np.argmin(fixed_shrk_data.iloc[i, 2:])
                actual_argmin_validationset.append(actual_min_action)


            print(f"Validation Loss of Epoch {epoch} (mean and sd): {np.mean(validation_loss)}, {np.std(validation_loss)}")
            print("cool")
        net.train()




def train_with_dataloader():

    # split dataset into train and test
    batch_size = 16
    total_num_batches = factors.shape[0] // batch_size
    len_train = int(total_num_batches * 0.7) * batch_size
    train_dataset = MyDataset(
        factors.iloc[:len_train, :],
        fixed_shrk_data.iloc[:len_train, :],
        optimal_shrk_data.iloc[:len_train, :]
    )
    val_dataset = MyDataset(
        factors.iloc[len_train:, :],
        fixed_shrk_data.iloc[len_train:, :],
        optimal_shrk_data.iloc[len_train:, :]
    )
    train_dataloader = DataLoader(train_dataset)
    val_dataloader = DataLoader(val_dataset)

    for epoch in range(1, num_epochs+1):
        train_preds = []
        val_preds = []
        actual_train_labels = []
        epoch_loss = []

        for i, data in enumerate(train_dataloader):
            inp, labels = data  # labels are actually the annualized pf standard deviations [= "reward"]

            out = net(inp.view(1, -1))  # instead of q(s, a), out is now to be interpreted as action probs
            multinom_dist = Categorical(out)
            action = multinom_dist.sample()  # this is the action, and according to it calculate loss
            # action is between 0 and 20

            # hacking solution --> need to solve the problem that cols/rows of my data are of dtype=object !
            pf_std = torch.Tensor(np.array(fixed_shrk_data.iloc[i, 2:].values, dtype=float))[action]  # but need this for all datapoints in batch

            pf_std = torch.Tensor(np.array(fixed_shrk_data.iloc[i, 2:].values, dtype=float))[action]

            actual_train_labels.append(torch.argmin(labels).item())


            out, _ = net(X.view(1, -1))
            train_preds.append(torch.argmin(out).item())
            # CALC LOSS AND BACKPROPAGATE
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        # end of epoch statistics
        print(f"Loss of Epoch {epoch} (mean and sd): {np.mean(epoch_loss)}, {np.std(epoch_loss)}")
        if epoch % 10 == 0:
            print("break :-)")
            # calc validation loss



# call train loop
train_manual()

train_with_dataloader()