import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data.dataloader import DataLoader, Dataset

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque
import preprocessing_scripts.helper_functions as hf
from RL.RL_dev.RL_covariance_estimators import cov1Para

import pickle
base_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
# The rb indicates that the file is opened for READING in binary mode.
with open(rf"{base_path}\past_return_matrices_p30.pickle", 'rb') as f:
    past_ret_matrices = pickle.load(f)
with open(rf"{base_path}\future_return_matrices_p30.pickle", 'rb') as f:
    fut_ret_matrices = pickle.load(f)
with open(rf"{base_path}\rebalancing_days_full.pickle", 'rb') as f:
    rebalancing_days_full = pickle.load(f)
with open(rf"{base_path}\trading_days.pickle", 'rb') as f:
    trading_days = pickle.load(f)

print("loaded necessary data")




'''
Given a covariance matrix and a shrinkage target
'''


class Net(nn.Module):
    '''
    A Network to approximate the Q-function (= Q value for every state action pair)
    '''

    def __init__(self, num_stocks):
        super(Net, self).__init__()
        self.state_space = int( num_stocks * (num_stocks+1) / 2)
        self.action_space = 1  # discretized 100 actions btw 0 and 1, here will use continuous
        self.hidden = (num_stocks * num_stocks+1)
        self.hidden2 = int((num_stocks * num_stocks+1) / 8)
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=True)
        self.l2 = nn.Linear(self.hidden, self.hidden2, bias=True)
        self.l3 = nn.Linear(self.hidden2, self.action_space, bias=True)  # output Q for every possible action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.sigmoid(x)
        return x


# init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(num_stocks=30).to(device)
lr = 1
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss_fn = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# params
numepochs = 100

# let's just randomly pick indices for epoch as our samples are indepenedent anyway


def calculate_pf_return_std(shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat):
    estimator = shrinkage_intensitiy * target + (1-shrinkage_intensitiy) * sample
    pf_return_daily, pf_std_daily = hf.calc_pf_weights_returns_vars(estimator, reb_days, pas_ret_mat, fut_ret_mat)
    return pf_return_daily, pf_std_daily


def train_loop():
    for epoch in range(numepochs):
        epoch_loss = 0
        running_ep_loss = [1 for _ in range(10)]

        indices = np.random.choice(len(past_ret_matrices), len(past_ret_matrices))
        for idx in indices:
            sample, target = cov1Para(past_ret_matrices[idx])
            upper_indices = torch.triu_indices(30, 30, offset=0)  # get indices of upper triangular matrix
            covmat = torch.Tensor(sample.to_numpy())[upper_indices[0], upper_indices[1]].to(device)  # maybe need to flatten?
            shrinkage_intensity = model.forward(covmat).to("cpu")  # returns the shrinkage intensities, according to them, calc the weights and the return

            pf_return_daily, pf_std_daily = calculate_pf_return_std(
                shrinkage_intensity.item(), target, sample, rebalancing_days_full.iloc[idx,],
                past_ret_matrices[idx], fut_ret_matrices[idx]
            )
            epoch_loss += pf_std_daily

            # need to minimize pf_std_daily so this is already my loss and i can backpropagate it
            pf_std_daily = torch.tensor(pf_std_daily, requires_grad=True)
            optimizer.zero_grad()
            pf_std_daily.backward()
            optimizer.step()

        running_ep_loss.pop()
        running_ep_loss.insert(0, epoch_loss)


        if epoch % 10 == 0:
            print("running epoch loss (last 10 epochs):", np.mean(running_ep_loss))


def train_loop_v3():
    '''
    This train loop implements "batched" training on its own by always sampling indices of batch size
    instead of single indices
    '''
    running_ep_loss = [1 for _ in range(10)]
    for epoch in range(numepochs):
        epoch_loss = 0

        batch_size = 32
        indices = np.random.choice(len(past_ret_matrices), (int(len(past_ret_matrices) / batch_size), batch_size))
        for idx in indices:
            sample_target_lst = [cov1Para(past_ret_matrices[i]) for i in idx]  # stores the 32 sample and target matrices
            # sample is stored at index 0 of each element in list, target at index 1
            upper_indices = torch.triu_indices(30, 30, offset=0)  # get indices of upper triangular matrix
            #covmat = torch.Tensor(sample.to_numpy())[upper_indices[0], upper_indices[1]]  # maybe need to flatten?


            ## sample covariance matrix is normalized in the below line
            sample_upper_triu = [torch.Tensor(normalize_covmat(sample_target[0].values))[upper_indices[0], upper_indices[1]] for sample_target in sample_target_lst]
            sample_upper_triu = torch.stack(sample_upper_triu)
            shrinkage_intensities = model.forward(sample_upper_triu)  # returns the shrinkage intensities, according to them, calc the weights and the return

            pf_return_daily, pf_std_daily = [], []
            for i in range(batch_size):
                res1, res2 = calculate_pf_return_std(
                    shrinkage_intensities[i].item(),
                    sample_target_lst[0][1],
                    sample_target_lst[0][0],
                    rebalancing_days_full.iloc[i, ],
                    past_ret_matrices[i],
                    fut_ret_matrices[i]
                )
                pf_return_daily.append(res1)
                pf_std_daily.append(res2)

            pf_std_daily = torch.tensor(pf_std_daily, requires_grad=True)

            epoch_loss += pf_std_daily.sum()

            # need to minimize pf_std_daily so this is already my loss and i can backpropagate it
            optimizer.zero_grad()
            pf_std_daily.sum().backward()
            optimizer.step()

        running_ep_loss.pop()
        running_ep_loss.insert(0, epoch_loss.item())


        if epoch % 10 == 0:
            print("running epoch loss (last 10 epochs):", np.mean(running_ep_loss))
        if epoch % 50 == 0:
            print("now")

class MyDataset(Dataset):
    def __init__(self, past_dat, fut_dat):
        self.past_dat = past_dat
        self.fut_dat = fut_dat

    def __getitem__(self, idx):
        # ALSO WILL RETURN THE INDEX
        return torch.from_numpy(self.past_dat[idx].values), torch.from_numpy(self.fut_dat[idx].values), idx

    def __len__(self):
        return len(self.past_dat)


dataset = MyDataset(past_ret_matrices, fut_ret_matrices)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)


def normalize_covmat(covmat):
    # Get the diagonal of the covariance matrix, i.e. the standard deviatons
    diag = np.sqrt(np.diag(covmat))
    diag_matrix = np.diag(diag)
    diag_matrix_inv = np.linalg.inv(diag_matrix)
    #diag2_matrix_inv = np.zeros(diag_matrix_inv.shape)  # same entries as diag mat but on the other diagnoal
    #for i in range(diag2_matrix_inv.shape[0]):
    #    diag2_matrix_inv[diag2_matrix_inv.shape[0]-1, i] = diag[i]
    mat2_inverse = np.array(covmat.shape[0] * [diag**-1] )
    normalized_covmat = covmat @ diag_matrix @ mat2_inverse
    return normalized_covmat

def normalize_covmat(covmat):
    corrmat = np.zeros(covmat.shape)
    for i in range(covmat.shape[0] - 1):
        for j in range(i, covmat.shape[0] - 1):
            corrmat[i, j] = corrmat[j, i] = covmat[i, j] / (np.sqrt(covmat[i, i]) * np.sqrt(covmat[j, j]))
    return corrmat


#normalize_covmat()
#train_loop()
#train_loop_v2()
train_loop_v3()

print("training done")

