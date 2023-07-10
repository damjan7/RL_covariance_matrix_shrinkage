# this script analyzes the input data, i.e. to fixed_ds_training_V2, in some more detail
# the goal is to see if there is a linear relationship, and if it were enough only to model the lin relationship
# also look at input data with some exploratory data anylsis tools

# import packages
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eval_funcs



# import data
shrk_data_path = r'C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets'
pf_size = 225
# currently fixed_shrkges_.. is the dataset with cov1_para, may change this

fixed_shrk_name = 'cov2Para'
opt_shrk_name = 'cov2Para'

with open(rf"{shrk_data_path}\{fixed_shrk_name}_fixed_shrkges_p{pf_size}.pickle", 'rb') as f:
    fixed_shrk_data = pickle.load(f)

with open(rf"{shrk_data_path}\{opt_shrk_name}_factor-1.0_p{pf_size}.pickle", 'rb') as f:
    optimal_shrk_data = pickle.load(f)

factor_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\factor_data"
factors = pd.read_csv(factor_path + "/all_factors.csv")
factors = factors.pivot(index="date", columns="name", values="ret")

# as our shrk data starts from 1980-01-15 our factors data should too
start_date = '1980-01-15'
start_idx = np.where(factors.index == start_date)[0][0]
factors = factors.iloc[start_idx:start_idx+fixed_shrk_data.shape[0], :]

full_ds = 1

train_len = int(len(factors) * 0.7)
train_indices = (0, train_len)
val_indices = (train_len, len(factors))

train_input = np.concatenate((factors.iloc[train_indices[0]:train_indices[1], :].values,
                                 optimal_shrk_data.iloc[train_indices[0]:train_indices[1], 1].values.reshape(-1,1)),
                                 axis=1) * 100
train_input = train_input.astype(np.float64)
train_labels = np.array(fixed_shrk_data.iloc[train_indices[0]:train_indices[1], 2:].values, dtype=float)

train_labels_v2 = np.array(optimal_shrk_data.iloc[train_indices[0]:train_indices[1], :]['pf_std'])

pd_train_input = pd.DataFrame(train_input)
pd_train_input.columns = ['Factor 1','Factor 2','Factor 3','Factor 4','Factor 5','Factor 6','Factor 7',
                          'Factor 8','Factor 9','Factor 10','Factor 11', 'Factor 12', 'Factor 13', 'Opt Shrkg']

val_input = np.concatenate((factors.iloc[val_indices[0]:val_indices[1], :].values,
                                 optimal_shrk_data.iloc[val_indices[0]:val_indices[1], 1].values.reshape(-1,1)),
                                 axis=1) * 100
val_labels = np.array(fixed_shrk_data.iloc[val_indices[0]:val_indices[1], 2:].values, dtype=float)


print("data loaded")


# correlation between inputs
# should not be too large (multicollinearity). Actually looks good
plot = False
if plot:
    sns.heatmap(pd_train_input.corr(),annot=True)
    # can plot distributions of single input variables
    sns.displot(pd_train_input['Factor 1'])


# model
from sklearn.linear_model import Ridge, LinearRegression

ridge_model = Ridge()
ridge_model_v2 = Ridge()
lm = LinearRegression()

ridge_model.fit(train_input, train_labels)
ridge_model_v2.fit(train_input, train_labels_v2)
lm.fit(train_input, train_labels)

ridge_preds_pfsd = ridge_model.predict(val_input)
ridge_preds_pfsd_v2 = ridge_model_v2.predict(val_input)
lm_preds_pfsd = lm.predict(val_input)

# take arg min as prediction
ridge_preds = np.argmin(ridge_preds_pfsd, axis=1) / 100
ridge_preds_v2 = np.argmin(ridge_preds_pfsd_v2, axis=1) / 100
lm_preds = np.argmin(lm_preds_pfsd, axis=1) / 100

eval_funcs.myplot(ridge_preds, lm_preds)

print("training and predicting done")