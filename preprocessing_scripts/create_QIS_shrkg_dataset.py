import pandas as pd

import helper_functions_RL as hf_rl
import pickle
import numpy as np
from preprocessing_scripts import rl_covmat_ests_for_dataset as estimators
import helper_functions as hf

def create_benchmark_ds(out_path_dat, out_path_shrk, estimator, p):  # HERE QIS
                # just load data if they already exist
    with open(rf"{out_path_dat}\past_return_matrices_p{p}.pickle", 'rb') as f:
        past_return_matrices = pickle.load(f)

    with open(rf"{out_path_dat}\future_return_matrices_p{p}.pickle", 'rb') as f:
        future_return_matrices = pickle.load(f)

    with open(rf"{out_path_dat}\past_price_matrices_p{p}.pickle", 'rb') as f:
        past_price_matrices = pickle.load(f)


    # now using past return matrices, calculate shrinkage intensity using some estimators
    # also calculate pf std's using the "optimal" shrinkage intensity and the others from the list
    shrk_factors = [1]
    # loop through all factors and the past data
    # can add everything to 1 data
    for factor in shrk_factors:
        res = [["date", "shrk_factor", "hist_vola", "pf_return", "pf_std"]]
        for idx, past_return_matrix in enumerate(past_return_matrices):
            covmat_est = estimator(past_return_matrix)
            # based on covmat --> calc pf std (and maybe return)
            pf_ret, pf_std = hf.calc_pf_std_return(covmat_est, future_return_matrices[idx])
            # also want historical (can choose days) vola (and in future maybe different factors)
            hist_vola = hf.get_historical_vola(past_price_matrices[idx], days=60)
            date = past_return_matrix.index[0]
            res.append([date, 0, hist_vola, pf_ret, pf_std])
        # save to pandas dataframe and then to disk, for each factor separately
        df = pd.DataFrame(res)
        df.columns = df.iloc[0, :]
        df = df.drop(0)
        with open(rf"{out_path_shrk}\QIS_p{p}.pickle", 'wb') as pickle_file:
            pickle.dump(df, pickle_file)



out_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
out_path_shrk = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets"
estimator = estimators.QIS
pf_size=100


create_benchmark_ds(out_path, out_path_shrk, estimator, p=pf_size)

