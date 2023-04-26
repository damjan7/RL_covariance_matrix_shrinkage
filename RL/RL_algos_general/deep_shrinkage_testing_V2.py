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


# LOADING DATA INTO MEMORY
import pickle
base_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
base_path_covmats = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL\covariance_matrices"

# DEPEND ON NUM STOCKS CONSIDERED --> TESTING WITH p = 30
with open(rf"{base_path}\past_return_matrices_p30.pickle", 'rb') as f:
    past_ret_matrices = pickle.load(f)
with open(rf"{base_path}\future_return_matrices_p30.pickle", 'rb') as f:
    fut_ret_matrices = pickle.load(f)

# DEPENDING SHRINKAGE METHOD
shrinkage_method = "cov1para"
with open(rf"{base_path_covmats}\covariance_correlation_data_p30_cov1para.pickle", 'rb') as f:
    covmats_dict = pickle.load(f)

sample_covmats = covmats_dict["sample_covmats"]
targets = covmats_dict["targets"]
upper_triu_sample_covmats = covmats_dict["upper_triu_sample_covmats"]
sample_corr_mats = covmats_dict["sample_corr_mats"]
upper_triu_sample_corrmats = covmats_dict["upper_triu_sample_corrmats"]

print("loaded necessary data")
############################

class Net(nn.Module):
    '''
    A Network to approximate the Q-function (= Q value for every state action pair)
    '''

    def __init__(self, num_stocks):
        super(Net, self).__init__()
        self.state_space = int(num_stocks * (num_stocks+1) / 2)
        self.action_space = 1  # discretized 100 actions btw 0 and 1, here will use continuous
        self.hidden = (num_stocks * num_stocks+1)
        self.hidden2 = int((num_stocks * num_stocks+1) / 8)
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=True)
        self.l2 = nn.Linear(self.hidden, self.hidden2, bias=True)
        self.l3 = nn.Linear(self.hidden2, self.action_space, bias=True)  # output Q for every possible action

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = self.l3(x)
        x = F.sigmoid(x)
        return x


class Net2(nn.Module):
    '''
    A Network to approximate the Q-function (= Q value for every state action pair)
    '''

    def __init__(self, num_stocks):
        super(Net2, self).__init__()
        v0 = int(num_stocks * (num_stocks+1) / 2)
        v1 = v0 * 3
        v2 = int(v1/2)
        v3 = int(v2/2)
        v4 = int(v3/2)

        self.l1 = nn.Linear(v0, v0, bias=True)
        self.l2 = nn.Linear(v0, v0, bias=True)
        self.l3 = nn.Linear(v2, v3, bias=True)
        self.l4 = nn.Linear(v3, v4, bias=True)
        self.l5 = nn.Linear(v0, 1, bias=True)  # output Q for every possible action

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        #x = F.leaky_relu(self.l2(x))
        #x = F.leaky_relu(self.l3(x))
        #x = F.leaky_relu(self.l4(x))
        x = self.l5(x)
        x = F.sigmoid(x)
        return x



def calculate_pf_return_std(shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat):
    estimator = shrinkage_intensitiy * target + (1-shrinkage_intensitiy) * sample
    pf_return_daily, pf_std_daily = hf.calc_pf_weights_returns_vars(estimator, reb_days, pas_ret_mat, fut_ret_mat)
    return pf_return_daily, pf_std_daily

def train_loop_batch():
    '''
    This train loop implements "batched" training on its own by always sampling indices of batch size
    instead of single indices
    '''
    running_ep_loss = [1 for _ in range(10)]
    for epoch in range(numepochs):
        epoch_loss = 0

        batch_size = 32
        all_indices = np.random.choice(len(past_ret_matrices), (int(len(past_ret_matrices) / batch_size), batch_size))
        for batch_indices in all_indices:

            X = torch.stack([torch.Tensor(upper_triu_sample_covmats[i]) for i in batch_indices])
            shrinkage_intensities = model.forward(X)  # returns the shrinkage intensities, according to them, calc the weights and the return

            pf_return_daily, pf_std_daily = [], []  # shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat
            for i in range(batch_size):
                res1, res2 = calculate_pf_return_std(
                    shrinkage_intensitiy=shrinkage_intensities[i].item(),
                    target=targets[batch_indices[i]],
                    sample=sample_covmats[batch_indices[i]],
                    reb_days=None,  # this is not needed
                    pas_ret_mat=past_ret_matrices[batch_indices[i]],
                    fut_ret_mat=fut_ret_matrices[batch_indices[i]]
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

        if (epoch+1) % 10 == 0:
            print(f"running epoch loss (epoch{epoch-8} - {epoch+1}):", np.mean(running_ep_loss))


def train_loop():
    '''
    This train loop implements "NON-batched" training on its own by always sampling indices of batch size
    instead of single indices
    '''
    running_ep_loss = [1 for _ in range(10)]
    for epoch in range(numepochs):
        epoch_loss = 0
        all_indices = np.random.choice(len(past_ret_matrices), len(past_ret_matrices))

        for curid in all_indices:
            X = torch.Tensor(upper_triu_sample_covmats[curid])
            shrinkage_intensity = model.forward(X)

            pf_return_daily, pf_std_daily = [], []
            res1, res2 = calculate_pf_return_std(
                shrinkage_intensitiy=shrinkage_intensity.item(),
                target=targets[curid],
                sample=sample_covmats[curid],
                reb_days=None,  # this is not needed
                pas_ret_mat=past_ret_matrices[curid],
                fut_ret_mat=fut_ret_matrices[curid]
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

        if (epoch+1) % 10 == 0:
            print(f"running epoch loss (epochs {epoch-8}-{epoch+1}):", np.mean(running_ep_loss))


#### WORK WITH TENSORS INSTEAD
def calculate_pf_return_std_TENSORTEST(shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat):
    estimator = shrinkage_intensitiy * target + (1-shrinkage_intensitiy) * sample
    pf_return_daily, pf_std_daily = hf.calc_pf_weights_returns_vars_TENSOR(estimator, reb_days, pas_ret_mat, fut_ret_mat)
    return pf_return_daily, pf_std_daily

def train_loop_batch_TENSORTEST():
    '''
    This train loop implements "batched" training on its own by always sampling indices of batch size
    instead of single indices
    '''

    running_ep_loss = [1 for _ in range(10)]
    for epoch in range(numepochs):
        epoch_loss = 0

        batch_size = 32
        all_indices = np.random.choice(len(past_ret_matrices), (int(len(past_ret_matrices) / batch_size), batch_size))
        for batch_indices in all_indices:

            X = torch.stack([upper_triu_sample_covmats[batch_idx] * 10 ** 4 for batch_idx in batch_indices])
            #X = torch.stack([upper_triu_sample_corrmats[i] * 10 for i in batch_indices])
            shrinkage_intensities = model.forward(X)  # returns the shrinkage intensities, according to them, calc the weights and the return

            pf_return_daily, pf_std_daily = [], []  # shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat
            for i in range(batch_size):
                res1, res2 = calculate_pf_return_std_TENSORTEST(
                    shrinkage_intensitiy=shrinkage_intensities[i],
                    target=targets[batch_indices[i]],
                    sample=sample_covmats[batch_indices[i]],
                    reb_days=None,  # this is not needed
                    pas_ret_mat=past_ret_matrices[batch_indices[i]],
                    fut_ret_mat=fut_ret_matrices[batch_indices[i]]
                )
                pf_return_daily.append(res1)
                pf_std_daily.append(res2)

            pf_std_daily = torch.stack(pf_std_daily)
            pf_return_daily = torch.stack(pf_return_daily)
            epoch_loss += pf_std_daily.sum()

            # need to minimize pf_std_daily so this is already my loss and i can backpropagate it
            optimizer.zero_grad()
            pf_std_daily.sum().backward()
            optimizer.step()

        running_ep_loss.pop()
        running_ep_loss.insert(0, epoch_loss.item())

        if (epoch+1) % 10 == 0:
            print(f"running epoch loss from the last 10 epochs ({epoch-8}-{epoch+1}):", np.mean(running_ep_loss))
            print(f"current shrinkage intensities: ", shrinkage_intensities.reshape((1, batch_size)))

############################


class NetDiag(nn.Module):
    '''
    A Network to approximate the Q-function (= Q value for every state action pair)
    '''

    def __init__(self, num_stocks):
        super(NetDiag, self).__init__()
        self.state_space = num_stocks

        hid1 = num_stocks*3
        hid2 = int(hid1 / 2)
        hid3 = int(hid2 / 6)
        self.l1 = nn.Linear(self.state_space, int(num_stocks/2), bias=True)
        self.l2 = nn.Linear(int(num_stocks/2), int(num_stocks/2), bias=True)
        self.l3 = nn.Linear(int(num_stocks/2), 1, bias=True)
        #self.l4 = nn.Linear(hid3, 1, bias=True)  # output Q for every possible action


    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        #x = F.leaky_relu(self.l3(x))
        x = self.l3(x)
        x = F.tanh(x)
        return x

def train_loop_batch_DIAGONAL():
    '''
    This train loop implements "batched" training on its own by always sampling indices of batch size
    instead of single indices
    '''
    running_ep_loss = [1 for _ in range(10)]
    for epoch in range(numepochs):
        epoch_loss = 0

        batch_size = 64
        all_indices = np.random.choice(len(past_ret_matrices), (int(len(past_ret_matrices) / batch_size), batch_size))
        for batch_indices in all_indices:

            X = torch.stack([torch.Tensor(np.diag(sample_covmats[i])) for i in batch_indices])
            shrinkage_intensities = model.forward(X)  # returns the shrinkage intensities, according to them, calc the weights and the return

            pf_return_daily, pf_std_daily = [], []  # shrinkage_intensitiy, target, sample, reb_days, pas_ret_mat, fut_ret_mat


            for i in range(batch_size):
                res1, res2 = calculate_pf_return_std(
                    shrinkage_intensitiy=shrinkage_intensities[i].item(),
                    target=targets[batch_indices[i]],
                    sample=sample_covmats[batch_indices[i]],
                    reb_days=None,  # this is not needed
                    pas_ret_mat=past_ret_matrices[batch_indices[i]],
                    fut_ret_mat=fut_ret_matrices[batch_indices[i]]
                )
                pf_return_daily.append(res1)
                pf_std_daily.append(res2)

            pf_std_daily = torch.tensor(pf_std_daily, requires_grad=True)
            epoch_loss += pf_std_daily.sum().item()


            # need to minimize pf_std_daily so this is already my loss and i can backpropagate it
            optimizer.zero_grad()
            pf_std_daily.sum().backward()
            optimizer.step()

        running_ep_loss.pop()
        running_ep_loss.insert(0, epoch_loss)

        if (epoch+1) % 10 == 0:
            print(f"running epoch loss (epochs {epoch-8}-{epoch+1}):", np.mean(running_ep_loss))
            print("shrinkage intensities:", shrinkage_intensities)



# init params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net2(num_stocks=30).to(device)
lr = 1e-4
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
numepochs = 1000

# TESTING of only passing DIAGONAL elems of covmat to net
#model = NetDiag(num_stocks=30).to(device)
'''
for name, param in model.named_parameters():
    if name.startswith('l'):
        param.data = torch.Tensor(np.random.normal(0, 2, param.shape))
'''
optimizer = optim.Adam(model.parameters(), lr=lr)

#train_loop_batch_DIAGONAL()
#train_loop_batch()
#train_loop()

# for tensortest need tensors for all data:
sample_covmats = torch.Tensor(np.array(covmats_dict["sample_covmats"]))
targets = torch.Tensor(np.array(covmats_dict["targets"]))
upper_triu_sample_covmats = torch.Tensor(np.array(covmats_dict["upper_triu_sample_covmats"]))
sample_corr_mats = torch.Tensor(np.array(covmats_dict["sample_corr_mats"]))
upper_triu_sample_corrmats = torch.Tensor(np.array(covmats_dict["upper_triu_sample_corrmats"]))

train_loop_batch_TENSORTEST()
