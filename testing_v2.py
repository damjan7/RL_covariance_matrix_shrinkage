import pickle

import pandas as pd
import numpy as np

from preprocessing_scripts import helper_functions as hf
from covariance_estimators import cov1Para

base_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices"

# The rb indicates that the file is opened for READING in binary mode.
with open(rf"{base_path}\past_return_matrices_p100.pickle", 'rb') as f:
    past_return_matrices = pickle.load(f)

with open(rf"{base_path}\future_return_matrices_p100.pickle", 'rb') as f:
    future_return_matrices = pickle.load(f)

with open(rf"{base_path}\rebalancing_days_full.pickle", 'rb') as f:
    rebalancing_days_full = pickle.load(f)

print("loaded necessary data")

# now i have all the return matrices and the rebalancing date
# for every return matrix i will calculate a covariance matrix estimate (i.e. with shrinkage)
# and then calculate the weights of the portfolio (using the minimum variance estimator that only needs
# the covariance matrix or an estimator thereof)

# at the same time, we will use the covariance matrix to calculate the weights
estimates = []
weights = []
permno = []
monthly_portfolio_returns = []
monthly_portfolio_returns_with_permno = []

for idx, date in enumerate(rebalancing_days_full['actual_reb_day']):
    permno.append(past_return_matrices[idx].columns)  # p largest stocks we consider
    est = cov1Para(past_return_matrices[idx])
    estimates.append(est)
    weights.append(hf.calc_global_min_variance_pf(est))

    # for the monthly return data, I need the look 21 days in the future
    # note that the date list at the current index contains the returns for the past year
    # this was used to calculate the covariance matrices and determine the weights of the PF for the next 21 days
    monthly_portfolio_returns.append(hf.calc_monthly_return(future_return_matrices[idx]).values)  # values only
    monthly_portfolio_returns_with_permno.append(hf.calc_monthly_return(future_return_matrices[idx]))



weights = pd.DataFrame(weights, index=rebalancing_days_full['actual_reb_day'])
monthly_portfolio_returns = pd.DataFrame(monthly_portfolio_returns, index=rebalancing_days_full['actual_reb_day'])
monthly_portfolio_returns_with_permno = pd.DataFrame(monthly_portfolio_returns_with_permno, index=rebalancing_days_full['actual_reb_day'])
permno = pd.DataFrame(permno, index=rebalancing_days_full['actual_reb_day'])

# need to multiply these monthly returns for the p stocks of interest with the corresponding weights
# I chose on the rebalancing date to get the portfolio return for the month.


monthly_weighted_portfolio_returns = []
for i in range(weights.shape[0]):
    tmp = (weights.iloc[i, :] * monthly_portfolio_returns.iloc[i, :]).sum()
    monthly_weighted_portfolio_returns.append(tmp)

total_portfolio_return = np.prod(monthly_weighted_portfolio_returns)

print("done")

