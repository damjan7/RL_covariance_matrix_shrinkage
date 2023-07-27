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

fixed_shrk_data = fixed_shrk_data.iloc[2521:, :]
optimal_shrk_data = optimal_shrk_data.iloc[2521:, :]

# IMPORT FACTORS DATA AND PREPARE FOR FURTHER USE
factor_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\factor_data"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

vix = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\vix_daily_dat.csv"
vix = pd.read_csv(vix)
vix = vix.loc[:, ['Date', 'Close']].iloc[0:8131+1, :]
vix = vix.dropna()
scaler = StandardScaler()
vix.Close = scaler.fit_transform(np.reshape(vix.Close.values, (-1, 1)))

# as our shrk data starts from 1980-01-15 our factors data should too
# VIX starts at jan 02, 1990, so use this start date
start_date = '1990-01-02'
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]

scale = False
if scale == True:
    scaler = StandardScaler()
    factors = pd.DataFrame(scaler.fit_transform(factors))
    shrk_fac2 = scaler.fit_transform(optimal_shrk_data.shrk_factor.values.reshape(-1,1))
    optimal_shrk_data.shrk_factor = shrk_fac2


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

    def __init__(self, vix, fixed_shrk_data, optimal_shrk_data, normalize=False):
        if normalize == True:  # for now only scale vix, I don't scale them actually
            self.vix_scaler = MinMaxScaler()
            self.vix = pd.DataFrame(self.vix_scaler.fit_transform(vix))
        else:
            self.vix = vix
        self.fixed_shrk_data = fixed_shrk_data
        self.optimal_shrk_data = optimal_shrk_data
        print("loaded")

    def __len__(self):
        return self.vix.shape[0]

    def __getitem__(self, idx):
        # inputs multiplied by 100 works better
        try:
            inp = torch.Tensor(np.append(self.vix.iloc[idx, 1], self.optimal_shrk_data.iloc[idx, 1]))
        except:
            print("f")

        #labels = np.array(self.fixed_shrk_data.iloc[idx, 2:].values, dtype=float)
        #labels = StandardScaler().fit_transform(labels.reshape(-1, 1))
        #labels = torch.Tensor(labels).squeeze()

        labels = torch.Tensor(np.array(self.fixed_shrk_data.iloc[idx, 2:].values, dtype=float))
        # for labels: .view(1, -1) not needed when working with Dataset and DataLoader
        return inp, labels

def train_with_dataloader(normalize=False):

    # split dataset into train and validation
    batch_size = 16
    total_num_batches = vix.shape[0] // batch_size
    len_train = int(total_num_batches * 0.7) * batch_size

    train_dataset = MyDataset(
        vix.iloc[0:len_train, :],
        fixed_shrk_data.iloc[0:len_train, :],
        optimal_shrk_data.iloc[0:len_train, :],
        normalize=normalize
    )

    val_dataset = MyDataset(
        vix.iloc[len_train:, :],
        fixed_shrk_data.iloc[len_train:, :],
        optimal_shrk_data.iloc[len_train:, :],
        normalize=False,
    )
    if normalize == True:
        val_dataset.vix = pd.DataFrame(train_dataset.vix_scaler.transform(val_dataset.vix))

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
        # set model back into train mode

        act_argmin_shrgks = list(map(eval_funcs.f2_map, actual_argmin_validationset))
        y2 = val_dataset.optimal_shrk_data["shrk_factor"].values.tolist()

        val_indices = (int(vix.shape[0] * 0.7), vix.shape[0])
        val_ds = fixed_shrk_data.iloc[val_indices[0]:val_indices[1], 2:]
        mapped_shrkges = list(map(eval_funcs.f2_map, val_preds))

        if epoch == 5:
            print("f1")
        elif epoch == 7:
            print("f4")
        elif epoch == 10:
            print("f2")
        elif epoch == 20:
            print(f"f3")
        elif epoch == 27:
            print("f3")
        elif epoch == 35:
            print("f")
        elif epoch == 50:
            print("f")
        elif epoch == 65:
            print("f")
        elif epoch == 90:
            print("f")
        elif epoch == 115:
            print('f')
        elif epoch == 150:
            print("f")
        elif epoch == 200:
            print("f")
        net.train()
        path = rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"

'''
val_indices = val_dataset.optimal_shrk_data.index.to_list()
val_idxes_RESULTS = [val_indices[0] + 21*i for i in range( (val_indices[1] - val_indices[0]) // 21 )]
val_idxes_shrkges = [0 + 21*i for i in range( (val_indices[-1] - val_indices[0]) // 21 + 1 )]
mapped_shrkges_v2 = np.array(mapped_shrkges)[val_idxes_shrkges]
eval_funcs.myplot(mapped_shrkges_v2, optimal_shrk_data.shrk_factor.iloc[val_indices].values.tolist())

eval_funcs.myplot(mapped_shrkges, val_dataset.optimal_shrk_data.shrk_factor.values.tolist())



with open(rf"{path}\future_return_matrices_p{pf_size}.pickle", 'rb') as f:
    fut_ret_mats = pickle.load(f)
with open(rf'{path}\past_return_matrices_p{pf_size}.pickle', 'rb') as f:
    past_ret_mats = pickle.load(f)
with open(rf"{path}\rebalancing_days_full.pickle", 'rb') as f:
    reb_days = pickle.load(f)
    
eval_funcs.temp_eval_fct(mapped_shrkges_v2, fut_ret_mats, past_ret_mats, reb_days, val_idxes_v2)

'''




# PARAMETERS:
# SET SCALE AT TOP OF SCRIPT!
num_epochs = 200
lr = 1e-5
num_features = 1 + 1  # VIX + opt shrk
num_actions = fixed_shrk_data.shape[1] - 2  # since 1 col is dates, 1 col is hist vola
hidden_layer_size = 128
net = ActorCritic(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()

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

myplot(val_dataset.optimal_shrk_data["shrk_factor"].values.tolist(), mapped_shrkges)