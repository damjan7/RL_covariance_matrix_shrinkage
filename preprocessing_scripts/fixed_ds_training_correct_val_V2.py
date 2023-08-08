# will try to implement some easy variant of td learning
# not sure if I will use my custom gym environment, as it is not really necessary
import wandb
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

# plotting
import matplotlib.pyplot as plt

from RL.RL_algos_custom import eval_funcs

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min



"""
Now, additionally use regularization (i.e. dropout) to improve generalization performance
"""


##### CHECK IF THE DATES ACTUALLY CORRESPOND TO EACH OTHER!!!!!!!!!!!!!!!!!!!!!!!!!!
# they do according to a QUICK MANUAL CHECK..

# IMPORT SHRK DATASETS
shrk_data_path = r'C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets'
pf_size = 225
# currently fixed_shrkges_.. is the dataset with cov1_para, may change this

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'

with open(rf"{shrk_data_path}\{fixed_shrk_name}_fixed_shrkges_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)

with open(rf"{shrk_data_path}\{opt_shrk_name}_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

# load the other optimal estimators for plotting purposes
#this is all for pf = 100
with open(rf"{shrk_data_path}\cov1para_factor-1.0_p100.pickle", 'rb') as f:
    cov1para_p100 = pickle.load(f)

with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets\cov1Para_p225.pickle", 'rb') as f:
    cov1para_p225 = pickle.load(f)

with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets\covDiag_p100.pickle", 'rb') as f:
    covdiag_p100 = pickle.load(f)

with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets\covCor_p100.pickle", 'rb') as f:
    covcor_p100 = pickle.load(f)

with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets\cov2para_p100.pickle", 'rb') as f:
    cov2para_p100 = pickle.load(f)

with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets\QIS_p225.pickle", 'rb') as f:
    qis_p225 = pickle.load(f)

#with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets\QIS_p100.pickle", 'rb') as f:
#    qis_p100 = pickle.load(f)

with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\rebalancing_days_full.pickle", 'rb') as f:
    rebalancing_days_full = pickle.load(f)


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

        self.state_action_head = nn.Linear(int(hidden_size/2), num_actions)  # probabilistic mapping from states to actions
        #self.critic_head = nn.Linear(int(hidden_size/2), 1)  # estimated state value, I don't use this for now

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # get action distribution
        # action_probs = F.softmax(self.state_action_head(x), dim=1)
        # I DO NOT NEED PROBABILITEIS NOW
        state_action_value = self.state_action_head(x)
        # how 'good' is the current state?
        #state_value = self.critic_head(x)
        return state_action_value


##### IMPLEMENTATION WITH DATALOADER
class MyDataset(Dataset):

    def __init__(self, factors, fixed_shrk_data, optimal_shrk_data, normalize=False):
        if normalize == True:  # for now only scale factors, I don't scale them actually
            self.factors_scaler = MinMaxScaler()
            self.factors = pd.DataFrame(self.factors_scaler.fit_transform(factors))
        else:
            self.factors = factors
        self.fixed_shrk_data = fixed_shrk_data
        self.optimal_shrk_data = optimal_shrk_data
        print("loaded")

    def __len__(self):
        return self.factors.shape[0]

    def __getitem__(self, idx):
        # inputs multiplied by 100 works better
        inp = torch.Tensor(np.append(self.factors.iloc[idx, :].values, self.optimal_shrk_data.iloc[idx, 1])) * 100

        #labels = np.array(self.fixed_shrk_data.iloc[idx, 2:].values, dtype=float)
        #labels = StandardScaler().fit_transform(labels.reshape(-1, 1))
        #labels = torch.Tensor(labels).squeeze()

        labels = torch.Tensor(np.array(self.fixed_shrk_data.iloc[idx, 2:].values, dtype=float))
        # for labels: .view(1, -1) not needed when working with Dataset and DataLoader
        return inp, labels

def train_with_dataloader(normalize=False):

    # split dataset into train and validation
    batch_size = 16
    total_num_batches = factors.shape[0] // batch_size
    len_train = int(total_num_batches * 0.7) * batch_size

    # last date to consider for training: 20050103
    end_train_idx = np.where(optimal_shrk_data['date'] == 20050103)[0][0]
    train_idxes = [i*21 for i in range(end_train_idx//21)]
    train_dataset = MyDataset(
        factors.iloc[train_idxes, :],
        fixed_shrk_data.iloc[train_idxes, :],
        optimal_shrk_data.iloc[train_idxes, :],
        normalize=normalize
    )

    # now for validation, do it correctly
    # 2005 01 03 + 21 days -> first date of val set, then every 21 days
    start_val_idx = np.where(factors.index == '2005-01-24')[0][0]
    val_idxes = [start_val_idx + i*21 for i in range((factors.shape[0]-start_val_idx) // 21)]
    val_dataset = MyDataset(
        factors.iloc[val_idxes, :],
        fixed_shrk_data.iloc[val_idxes, :],
        optimal_shrk_data.iloc[val_idxes, :],
        normalize=False,
    )
    if normalize == True:
        val_dataset.factors = pd.DataFrame(train_dataset.factors_scaler.transform(val_dataset.factors))

    train_dataloader = DataLoader(train_dataset)
    val_dataloader = DataLoader(val_dataset)

    validation_loss = []
    for epoch in range(1, num_epochs+1):
        train_preds = []
        val_preds = []
        actual_train_labels = []
        epoch_loss = []
        for i, data in enumerate(train_dataloader):
            X, labels = data  # labels are actually the annualized pf standard deviations [= "reward"]
            actual_train_labels.append(torch.argmin(labels).item())
            out = net(X.view(1, -1))
            train_preds.append(torch.argmin(out).item())
            opt_shrk = torch.tensor(optimal_shrk_data['shrk_factor'].iloc[i])
            out_shrk = torch.argmin(out) / 100
            # CALC LOSS AND BACKPROPAGATE
            optimizer.zero_grad()
            loss = criterion(out, labels)  # MSE between outputs of NN and pf std --> pf std can be interpreted
            # as value of taking action a in state s, hence want my network to learn this
            # loss += criterion(out_shrk, opt_shrk)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        print(f"Epoch {epoch} training done.")

        # end of epoch statistics
        print(f"Loss of Epoch {epoch} (mean and sd): {np.mean(epoch_loss)}, {np.std(epoch_loss)}")
        if epoch % 10 == 0:
            print("break :-)")
            # calc validation loss

        # log at end of epoch
        wandb.log({"train loss": np.mean(epoch_loss)})

        # validate at end of epoch
        # set model into evaluation mode and deactivate gradient collection
        net.eval()
        epoch_val_loss = []
        actual_argmin_validationset = []
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                X, labels = data
                out = net(X.view(1, -1))
                val_preds.append(torch.argmin(out).item())
                loss = criterion(out, labels)
                epoch_val_loss.append(loss.item())

                actual_argmin_validationset.append(torch.argmin(labels).item())

            # print mean and sd of val loss
        print(f"Validation Loss of Epoch {epoch} (mean and sd): {np.mean(epoch_val_loss)}, {np.std(epoch_val_loss)}")
        if True == False:
            actual_results = eval_funcs.correct_validationset_evaluation(val_preds, pf_size)
            print(actual_results)

        val_preds_shrkg = [i/100 for i in val_preds]
        # eval_funcs.myplot(val_preds_shrkg, cov2para_shrkges)

        if epoch == 5:
            print("f1")
        elif epoch == 10:
            print("f2")
        elif epoch == 20:
            print(f"f3")

        # model back to train
        net.train()


# PARAMETERS:
num_epochs = 20
lr = 1e-4
num_features = factors.shape[1] + 1  # all 13 factors + opt shrk
num_actions = fixed_shrk_data.shape[1] - 2  # since 1 col is dates, 1 col is hist vola
hidden_layer_size = 128
net = ActorCritic(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()


wandb.login()

run = wandb.init(
    project="fixed-ds-testing",
    entity="damjan-thesis",
    config={
        "architecture": net,
        "epochs": num_epochs,
        "learning_rate": lr,
        "hidden_layer_size": hidden_layer_size,
    }

)

train_with_dataloader(normalize=False)

print("done")


def myplot(*args):
    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(args[0]))
    for arg in args:
        ax.plot(x, arg)
    plt.legend()
    plt.show()

