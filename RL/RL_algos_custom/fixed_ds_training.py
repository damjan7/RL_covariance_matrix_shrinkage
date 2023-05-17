# will try to implement some easy variant of td learning
# not sure if I will use my custom gym environment, as it is not really necessary

import pandas as pd
import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn import preprocessing


##### CHECK IF THE DATES ACTUALLY CORRESPOND TO EACH OTHER!!!!!!!!!!!!!!!!!!!!!!!!!!
# they do according to a QUICK MANUAL CHECK..

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


# Even for a simple TD-Learning method, we need an estimate of the Q(s,a) or V(s) function
# As we have a continuous state space, we should use a NN to parametrize this

class ActorCritic(nn.Module):
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
        # action_probs = F.softmax(self.actor_head(x), dim=1)
        # I DO NOT NEED PROBABILITEIS NOW
        action_probs = self.actor_head(x)
        # how 'good' is the current state?
        state_value = self.critic_head(x)
        return action_probs, state_value

# Train Loop
# first idea: for every data point, compare my state value estimates to the actual rewards

# PARAMETERS:
num_epochs = 100
lr = 1e-3
num_features = factors.shape[1] + 1  # all 13 factors + opt shrk
num_actions = fixed_shrk_data.shape[1] - 2  # since 1 col is dates, 1 col is hist vola
hidden_layer_size = 128
net = ActorCritic(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr)

criterion = nn.MSELoss()

# TRAIN LOOP

for epoch in range(1, num_epochs+1):
    epoch_loss=[]
    for i in range(factors.shape[0]):
        inp = torch.Tensor(np.append(factors.iloc[i, :].values, optimal_shrk_data.iloc[i, 1])) * 100
        out, _ = net(inp.view(1, -1))
        # hacking solution --> need to solve the problem that cols/rows of my data are of dtype=object !
        labels = torch.Tensor(np.array(fixed_shrk_data.iloc[0, 2:].values, dtype=float)).view(1, -1)

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

print("done")

