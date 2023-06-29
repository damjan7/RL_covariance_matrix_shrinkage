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
This file is more or less the same as V3 but will try some loss tuning

For example; also penalize if my shrinkage intensity is too far from the shrinkage intensity chosen by
the chosen model (i.e. cov2para)
"""


##### CHECK IF THE DATES ACTUALLY CORRESPOND TO EACH OTHER!!!!!!!!!!!!!!!!!!!!!!!!!!
# they do according to a QUICK MANUAL CHECK..

# IMPORT SHRK DATASETS
shrk_data_path = r'C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets'
pf_size = 100
# currently fixed_shrkges_.. is the dataset with cov1_para, may change this

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'

with open(rf"{shrk_data_path}\{fixed_shrk_name}_fixed_shrkges_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)
with open(rf"{shrk_data_path}\{opt_shrk_name}_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

# just used for plotting or something?
with open(rf"{shrk_data_path}\cov1para_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    cov1para = pickle.load(f)

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

# Train Loop
# first idea: for every data point, compare my state value estimates to the actual rewards



# TRAIN LOOP
def train_manual():
    for epoch in range(1, num_epochs+1):
        epoch_loss=[]
        for i in range(factors.shape[0]):
            inp = torch.Tensor(np.append(factors.iloc[i, :].values, optimal_shrk_data.iloc[i, 1])) * 100
            out = net(inp.view(1, -1))
            # hacking solution --> need to solve the problem that cols/rows of my data are of dtype=object !
            labels = torch.Tensor(np.array(fixed_shrk_data.iloc[i, 2:].values, dtype=float)).view(1, -1)

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

        # validate at end of each epoch

#train_manual()

##### IMPLEMENTATION WITH DATALOADER


class MyDataset(Dataset):

    def __init__(self, factors, fixed_shrk_data, optimal_shrk_data, normalize=False):
        if normalize == True:  # for now only scale factors
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

    # split dataset into train and test
    batch_size = 16
    total_num_batches = factors.shape[0] // batch_size
    len_train = int(total_num_batches * 0.7) * batch_size
    train_dataset = MyDataset(
        factors.iloc[:len_train, :],
        fixed_shrk_data.iloc[:len_train, :],
        optimal_shrk_data.iloc[:len_train, :],
        normalize=normalize
    )

    val_dataset = MyDataset(
        factors.iloc[len_train:, :],
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
            loss = criterion(out, labels)   # MSE between outputs of NN and pf std --> pf std can be interpreted
            # as value of taking action a in state s, hence want my network to learn this
            #### THIS BELOW DOESNT MAKE SENSE AND DOESNT WORK
            # loss += criterion(out_shrk, opt_shrk)
            # ADD something else to loss
            # loss += criterion(out_shrk, opt_shrk) * 1e15 # just to check
            #######

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())


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
            # set model back into train mode

            # log the val loss
            wandb.log({"val loss": np.mean(epoch_val_loss)})

            # return mean pf std of opt shrk estimator and shrk estimators chosen by my network
            pfstd1, pfstd2, pfstd1_sd, pfstd2_sd = eval_funcs.evaluate_preds(val_preds,
                                                       val_dataset.optimal_shrk_data,
                                                       val_dataset.fixed_shrk_data)

            print(f"Validation pf std with shrkges chosen by network: {pfstd1} \n"
                  f"Validation pf std with shrkges chosen by {opt_shrk_name}: {pfstd2}")
            print(f"Validation sd of pf std's [network]: ", pfstd1_sd)
            print(f"Validation sd of pf std's {opt_shrk_name}: ", pfstd2_sd)
            print(f"Validation PF std epoch {epoch} [QIS] (mean): {0.10245195394691942}")
            print(f"Validation minimum attainable pf sd: ", 0.09259940834962073)

            # log the pf std with shriges chosen by network vs by classical optimizer
            wandb.log({
                "pf sd - network estimator": pfstd1,
                "pf sd - closed form estimator": pfstd2
            })

            # map predictions from 1 to 21 to shrinkage intensities
            #mapped_shrkges = list(map(eval_funcs.f_map, val_preds))
            mapped_shrkges = list(map(eval_funcs.f2_map, val_preds))
            #wandb.log({
            #    "closed form shrinkages": val_dataset.optimal_shrk_data["shrk_factor"].values,
            #    "network shrinkages": mapped_shrkges
            # also plot argmin shrkg
            #})

            #eval_funcs.simple_plot(val_preds, actual_argmin_validationset)
            #eval_funcs.simple_plot(val_preds, optimal_shrk_data["shrk_factor"], map1=True, map2=True)
            act_argmin_shrgks = list(map(eval_funcs.f2_map, actual_argmin_validationset))
            y2 = val_dataset.optimal_shrk_data["shrk_factor"].values.tolist()

            cov1para_val_ds = cov1para.loc[list(val_dataset.optimal_shrk_data.index)]['shrk_factor'].values.tolist()
            # eval_funcs.myplot(act_argmin_shrgks, mapped_shrkges, y2)
            # eval_funcs.myplot(mapped_shrkges, y2)
            # eval_funcs.myplot(mapped_shrkges, y2, cov1para_val_ds)

            # plot pf stds
            # pf_sds_network = eval_funcs.get_pf_sds_daily(val_preds, val_dataset.fixed_shrk_data).tolist()
            # pf_sds_opt = val_dataset.optimal_shrk_data['pf_std'].values.tolist()
            # eval_funcs.myplot(pf_sds_network, pf_sds_opt)



            '''
            eval_funcs.myplot(val_dataset.factors.iloc[:, 0].tolist(), val_dataset.factors.iloc[:, 1].tolist(), 
                              val_dataset.factors.iloc[:, 2].tolist(), val_dataset.factors.iloc[:, 3].tolist(), 
                              val_dataset.factors.iloc[:, 4].tolist(), val_dataset.factors.iloc[:, 4].tolist(), 
                              val_dataset.factors.iloc[:, 5].tolist(), val_dataset.factors.iloc[:, 6].tolist(), 
                              val_dataset.factors.iloc[:, 7].tolist(), y2)
            '''
            val_indices = (int(factors.shape[0] * 0.7), factors.shape[0])
            val_ds = fixed_shrk_data.iloc[val_indices[0]:val_indices[1], 2:]

            if epoch == 5:
                 print("f1")
            elif epoch == 10:
                print("f2")


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

myplot(val_dataset.optimal_shrk_data["shrk_factor"].values.tolist(), mapped_shrkges)

