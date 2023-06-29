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
from torch.distributions import Normal, Beta, LogNormal, Categorical



# plotting
import matplotlib.pyplot as plt

from RL.RL_algos_custom import eval_funcs
from RL.RL_dev import RL_covariance_estimators as rl_shrkg_est
from preprocessing_scripts import helper_functions as hf

from sklearn.preprocessing import StandardScaler, MinMaxScaler


'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
In this script I want to try discrete policy, but I look at the labels
calc loss via: loss = criterion(torch.min(labels)*100, pf_std*100) * log_prob
in this case we need no normalization of the labels

'''

# IMPORT SHRK DATASETS
shrk_data_path = r'C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets'
pf_size = 100

####################33
# CHANGE ESTIMATOR NAME DEPENDING ON WHICH ONE I WANT TO USE
estimator_name = 'cov2Para'

with open(rf"{shrk_data_path}\{estimator_name}_fixed_shrkges_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{shrk_data_path}\{estimator_name}_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)


# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\factor_data"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too
start_date = '1980-01-15'
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]

# IMPORT remaining datasets
pth = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
with open(rf"{pth}\future_return_matrices_p{pf_size}.pickle", 'rb') as f:
    future_matrices = pickle.load(f)
with open(rf"{pth}\past_return_matrices_p{pf_size}.pickle", 'rb') as f:
    past_matrices = pickle.load(f)



# define policy network
class PolicyNetwork(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))

        self.action_probs = nn.Linear(int(hidden_size/2), num_actions)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # get action distribution
        x = self.action_probs(x)
        x = F.softmax(x)

        # how 'good' is the current state?, not needed here
        return x



# init and train

# PARAMETERS:
num_epochs = 100
lr = 1e-4
weight_decay=1e-5  # for more regularization
num_features = factors.shape[1] + 1  # all 13 factors + opt shrk  [= INPUT DATA]
num_actions = fixed_shrk_data.shape[1] - 2   # first two cols are not actions
hidden_layer_size = 64
net = PolicyNetwork(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
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

            # choose softmax action [policy is softmax policy]
            # the NN is just a representation of the features w(s,a) for each state action tuple

            # actual pf sd's
            labels = torch.Tensor(np.array(fixed_shrk_data.iloc[i, 2:].values, dtype=float)).view(1, -1)

            multinom_dist = Categorical(out.squeeze())
            action = multinom_dist.sample()

            a = str(round(action.item()/100, 2))
            pf_std = torch.tensor(fixed_shrk_data.iloc[i, :].loc[a])

            loss = criterion(torch.min(labels)*100, pf_std*100)*1000

            optimizer.zero_grad()
            log_prob = multinom_dist.log_prob(action)  # log prob of policy dist evaluated at sampled action
            loss = loss * log_prob  # like policy gradient  [reward * ]
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        # end of epoch statistics
        print(f"Training Loss of Epoch {epoch} (mean and sd): {np.mean(epoch_loss)}, {np.std(epoch_loss)}")
        #print(f"Training Set pf std of epoch {epoch}: {np.mean(train_pf_std)}, {np.std(train_pf_std)}" )
        if epoch % 10 == 0:
             print("break :-)")

        # validation
        net.eval()
        with torch.no_grad():
            for i in range(val_indices[0], val_indices[1]):
                inp = torch.Tensor(np.append(factors.iloc[i, :].values, optimal_shrk_data.iloc[i, 1])) * 100
                out = net(inp.view(1, -1))  # instead of q(s, a), out is now to be interpreted as action probs

                # now just predict again by sampling? OR by argmax?
                #multinom_dist = Categorical(out.squeeze())
                #action = multinom_dist.sample()

                # LET'S TRY ARGMAX
                action = torch.argmax(out)


                val_preds.append(action.item())

                validation_loss.append(pf_std.item())  # just pf std
                actual_min_action = np.argmin(fixed_shrk_data.iloc[i, 2:])
                actual_argmin_validationset.append(actual_min_action)


            pfstd1, pfstd2, _, _ = eval_funcs.evaluate_preds(
                val_preds,
                optimal_shrk_data.iloc[val_indices[0]:val_indices[1], :],
                fixed_shrk_data.iloc[val_indices[0]:val_indices[1], :]
            )

            print(f"Validation Loss of Epoch {epoch} (mean and sd): {np.mean(validation_loss)}, {np.std(validation_loss)}")
            print(f"Validation PF std epoch {epoch} [network] (mean): {np.mean(pfstd1)}")
            print(f"Validation PF std epoch {epoch} [cov1para] (mean): {np.mean(pfstd2)}")
            print(f"Validation PF std epoch {epoch} [QIS] (mean): {0.10245195394691942}")
            print(f"Validation minimum attainable pf sd: ", 0.09259940834962073)

            y2 = optimal_shrk_data['shrk_factor'].iloc[val_indices[0]:val_indices[1]].values.tolist()
            if epoch == 5:
                print("done ")
            mapped_val_preds = list(map(eval_funcs.f2_map, val_preds))
            # eval_funcs.myplot(y2, mapped_val_preds)
        net.train()







# call train loop
train_manual()

