
import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# plotting
import matplotlib.pyplot as plt

from RL.RL_algos_custom import eval_funcs

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

# IMPORT SHRK DATASETS
shrk_data_path = r'C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets'
pf_size = 500

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'
with open(rf"{shrk_data_path}\{fixed_shrk_name}_fixed_shrkges_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{shrk_data_path}\{opt_shrk_name}_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

# temporary
for i, shrk in enumerate(optimal_shrk_data.shrk_factor.values):
    if shrk > 0.6:
        run_mean = np.mean(optimal_shrk_data.shrk_factor.values[i-150:i+150])
        optimal_shrk_data.shrk_factor.iloc[i] = run_mean

# for actual, correct validation, need the future and past return matrices as well as the rebalancing days
with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\past_return_matrices_p{pf_size}.pickle", 'rb') as f:
    past_return_matrices = pickle.load(f)

with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\future_return_matrices_p{pf_size}.pickle", 'rb') as f:
    future_return_matrices = pickle.load(f)

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
# As we have a continuous state space, we discretize the action space
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


class MyDataset(Dataset):
    def __init__(self, fixed_shrk_data, optimal_shrk_data, normalize=False):
        self.fixed_shrk_data = fixed_shrk_data
        self.optimal_shrk_data = optimal_shrk_data
        print("loaded")

    def __len__(self):
        return self.fixed_shrk_data.shape[0]

    def __getitem__(self, idx):
        # inputs multiplied by 100 works better
        inp = torch.Tensor(np.array(self.optimal_shrk_data.iloc[idx, 1])) * 100

        labels = torch.Tensor(np.array(self.fixed_shrk_data.iloc[idx, 2:].values, dtype=float))
        # for labels: .view(1, -1) not needed when working with Dataset and DataLoader
        return inp, labels

def train_with_dataloader(normalize=False):
    # split dataset into train and validation
    batch_size = 16
    total_num_batches = factors.shape[0] // batch_size
    # tot len of dataset is 10374
    len_train = 5040
    # len_train = int(total_num_batches * 0.7) * batch_size
    train_dataset = MyDataset(
        fixed_shrk_data.iloc[:len_train, :],
        optimal_shrk_data.iloc[:len_train, :],
        normalize=normalize
    )

    val_dataset = MyDataset(
        fixed_shrk_data.iloc[len_train:, :],
        optimal_shrk_data.iloc[len_train:, :],
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

        # validate at end of epoch
        # set model into evaluation mode and deactivate gradient collection
        net.eval()
        epoch_val_loss = []
        actual_argmin_validationset = []
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                X, labels = data
                out = net(X.view(1, -1))
                dst = torch.tensor([i/100 for i in range(101)])
                dst = torch.abs(dst - X[-1]/100) * 0.05
                val_preds.append(torch.argmin(out+dst).item())
                loss = criterion(out, labels)
                epoch_val_loss.append(loss.item())

                actual_argmin_validationset.append(torch.argmin(labels).item())

            # print mean and sd of val loss
            print(f"Validation Loss of Epoch {epoch} (mean and sd): {np.mean(epoch_val_loss)}, {np.std(epoch_val_loss)}")

            # map predictions from 1 to 21 to shrinkage intensities
            mapped_shrkges = list(map(eval_funcs.f2_map, val_preds))

            #eval_funcs.simple_plot(val_preds, actual_argmin_validationset)
            #eval_funcs.simple_plot(val_preds, optimal_shrk_data["shrk_factor"], map1=True, map2=True)
            act_argmin_shrgks = list(map(eval_funcs.f2_map, actual_argmin_validationset))
            y2 = val_dataset.optimal_shrk_data["shrk_factor"].values.tolist()

            # eval_funcs.myplot(act_argmin_shrgks, mapped_shrkges, y2)
            # eval_funcs.myplot(mapped_shrkges, y2)
            # eval_funcs.myplot(mapped_shrkges, y2, cov1para_val_ds, covcor_shrk, covdiag_shrk)
            # print(f"Avg Shrk Intensity and Std Dev: {round(np.mean(mapped_shrkges), 2)}, {round(np.std(mapped_shrkges), 2)} ")

            '''
            eval_funcs.myplot(val_dataset.factors.iloc[:, 0].tolist(), val_dataset.factors.iloc[:, 1].tolist(), 
                              val_dataset.factors.iloc[:, 2].tolist(), val_dataset.factors.iloc[:, 3].tolist(), 
                              val_dataset.factors.iloc[:, 4].tolist(), val_dataset.factors.iloc[:, 4].tolist(), 
                              val_dataset.factors.iloc[:, 5].tolist(), val_dataset.factors.iloc[:, 6].tolist(), 
                              val_dataset.factors.iloc[:, 7].tolist(), y2)
            '''
            val_indices = (len_train, factors.shape[0])
            val_ds = fixed_shrk_data.iloc[val_indices[0]:val_indices[1], 2:]

            if epoch == 3:
                print("f")
            elif epoch == 5:
                print("f1")
            if epoch == 6:
                print("f")
            if epoch == 7:
                print("f1")
            if epoch == 8:
                print("f")
            elif epoch == 10:
                print("f2")
            elif epoch == 20:
                print(f"f3")

        path = rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
        #val_indices_correct = val_dataloader.dataset.optimal_shrk_data.index.values.tolist()
        #val_indices_correct = val_indices_correct[0:-400]
        #val_indices_correct = (4960, 8067)
        '''
val_indices_correct = (len_train, fixed_shrk_data.shape[0])
val_indices_results = [val_indices_correct[0] + 21*i for i in range( (val_indices_correct[-1] - val_indices_correct[0]) // 21)]
val_idxes_shrkges = [0 + 21*i for i in range( (val_indices[-1] - val_indices[0]) // 21 )]
mapped_shrkges_v2 = np.array(mapped_shrkges)[val_idxes_shrkges]
with open(rf"{path}\future_return_matrices_p{pf_size}.pickle", 'rb') as f:
    fut_ret_mats = pickle.load(f)
with open(rf'{path}\past_return_matrices_p{pf_size}.pickle', 'rb') as f:
    past_ret_mats = pickle.load(f)
with open(rf"{path}\rebalancing_days_full.pickle", 'rb') as f:
    reb_days = pickle.load(f)

res = eval_funcs.temp_eval_fct(mapped_shrkges_v2, fut_ret_mats, past_ret_mats, reb_days, val_indices_results)
'''

        net.train()

# PARAMETERS:
torch.manual_seed(31782)
num_epochs = 20
lr = 1e-4
num_features = 1  # all 13 factors + opt shrk
num_actions = fixed_shrk_data.shape[1] - 2  # since 1 col is dates, 1 col is hist vola
hidden_layer_size = 128
net = ActorCritic(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()

# start training
train_with_dataloader(normalize=False)

print("done")
