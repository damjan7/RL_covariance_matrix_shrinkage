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
from torch.utils.data import DataLoader, Dataset

# plotting
import matplotlib.pyplot as plt

from RL.RL_algos_custom import eval_funcs


"""
Explanation:
In this script, I work with the fixed dataset, i.e., for every day, we have 21 fixed shrinkage intensities. Using
all these 21 shrinkage intensities, the pf sd is calculated (21 day holding period as usual). Then we have a full
dataset of "actions" and "rewards"/labels.
I then train them in a "supervised" manner. The idea is to estimate the value function for each state, action pair.
Or RATHER (more precisely) just estimate the state-action value function (as I do it for "all" possible actions - 
not actually all as I discretize the action space).

This gives should give me a good approximation of the "true" action-value function. Using this it is easy
to pick the best action -- just pick the argmax of the outputs of this network (of the state-action value network).

Thoughts: This may seem like a supervised problem but it isn't actually. Of course, I use supervised learning
but this is to approximate the true state-action value function as closely as possible 
[supervised learning with NN's is always function approximation]. I could do the exactly same 
training in more "classical" RL ways: letting my network pick an action, then calculate the reward and then
adapt my state-action value function. But this only makes the training slower. And just logically thinking,
WHY would I want to use only 1 label as signal (i.e. 1 reward according to the action the network has chosen), instead
of all possible signals.....

As far as I understand it, I am doing off policy learning, actually there is no policy at all as we look at all
actions and the corresponding rewards.

What I could do now is also regularize the network and stuff like this to achieve more of an generalized
approximation (not only learning heavily on training data!) 

LOL; before I read the chapter 9 [part 2] of RL book, I implemented this, and, i.e., in ch9.3 they suggest
exactly this idea (but for state value functions)

Alternative idea? -> We could also create such a fixed dataset using a fixed policy. For example compose a linear func
of the optimal shrinkage intensity according to some estimator and the factor data
something like: shrk_new = shrk_opt * b1 + factors * W
and depending on the reward, optimize b1 and W to find a better shrk intensity

CAN I THINK OF MY IMPLEMENTATION AS POLICY GRADIENT FOLLOWING 21 POLICIES AT THE SAME TIME AND LOOKING JUST 1 
STEP AHEAD. I.E., OUTPUT OF NETWORK IS EXPECTED ONE STEP REWARD??????

Weiterer Punkt: ja, vielleicht ist die state-action value function einiges entfernt von derjenigen in den
out of sample daten, ABER, uns interessiert schlussendlich ja nur das RANKING zwischen den actions (welche action
schneidet am besten ab etc). Und da kann es ja trotzdem funktionieren. 


IMPORTANT NOTE; I MAY HAVE A SMALL OVERLAP OF TRAIN AND TEST SET AND THIS HAS TO BE RESOLVED FOR THE LATER 
IMPLEMENTATION
FOR NOW, I DON'T THINK THIS MAKES A BIG DIFFERENCE AS IT MAY ONLY BE ~21 DAYS
"""

q(s,a)

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
        self.critic_head = nn.Linear(int(hidden_size/2), 1)  # estimated state value, I don't use this for now

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
lr = 1e-4
num_features = factors.shape[1] + 1  # all 13 factors + opt shrk
num_actions = fixed_shrk_data.shape[1] - 2  # since 1 col is dates, 1 col is hist vola
hidden_layer_size = 128
net = ActorCritic(num_features, num_actions, hidden_layer_size)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()

# TRAIN LOOP
def train_manual():
    for epoch in range(1, num_epochs+1):
        epoch_loss=[]
        for i in range(factors.shape[0]):
            inp = torch.Tensor(np.append(factors.iloc[i, :].values, optimal_shrk_data.iloc[i, 1])) * 100
            out, _ = net(inp.view(1, -1))
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
        # for labels: .view(1, -1) not needed when working with Dataset and DataLoader
        return inp, labels

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


    validation_loss = []

    for epoch in range(1, num_epochs+1):
        train_preds = []
        val_preds = []
        actual_train_labels = []
        epoch_loss = []
        for i, data in enumerate(train_dataloader):
            X, labels = data  # labels are actually the annualized pf standard deviations [= "reward"]
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

        # validate at end of epoch
        # set model into evaluation mode and deactivate gradient collection
        net.eval()
        epoch_val_loss = []
        actual_argmin_validationset = []
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                X, labels = data
                out, _ = net(X.view(1, -1))
                val_preds.append(torch.argmin(out).item())
                loss = criterion(out, labels)
                epoch_val_loss.append(loss.item())

                actual_argmin_validationset.append(torch.argmin(labels).item())

            # print mean and sd of val loss
            print(f"Validation Loss of Epoch {epoch} (mean and sd): {np.mean(epoch_val_loss)}, {np.std(epoch_val_loss)}")
            # set model back into train mode

            # return mean pf std of opt shrk estimator and shrk estimators chosen by my network
            pfstd1, pfstd2 = eval_funcs.evaluate_preds(val_preds,
                                                       val_dataset.optimal_shrk_data,
                                                       val_dataset.fixed_shrk_data)
            print(f"pf std with shrkges chosen by network: {pfstd1} \n"
                  f"pf std with shrkges chosen by classical optimizer: {pfstd2}")

            eval_funcs.simple_plot(val_preds, actual_argmin_validationset)
            #eval_funcs.simple_plot(val_preds, optimal_shrk_data["shrk_factor"], map1=True, map2=True)


            net.train()



train_with_dataloader()

print("done")

