import pandas as pd

import helper_functions_RL as hf_rl
import pickle
import numpy as np
from preprocessing_scripts import rl_covmat_ests_for_dataset as estimators
import helper_functions as hf

def create_fixed_shrk_datasets(path, end_date, p, out_pf_sample_period_length, estimation_window_length,
                         out_path_shrk, out_path_dat, estimator):
            # just load data if they already exist
    with open(rf"{out_path_dat}\past_return_matrices_p{p}.pickle", 'rb') as f:
        past_return_matrices = pickle.load(f)

    with open(rf"{out_path_dat}\future_return_matrices_p{p}.pickle", 'rb') as f:
        future_return_matrices = pickle.load(f)

    with open(rf"{out_path_dat}\past_price_matrices_p{p}.pickle", 'rb') as f:
        past_price_matrices = pickle.load(f)



    """
    Below: just 10 shrk intensities from 0.1 to 0.9 to have a general overview what good shrk intensities are
    """
    shrk_intensities_v2 = np.round(np.linspace(0, 1, 101), 2)
    colnames = list(shrk_intensities_v2.astype(str))
    res = [["date", "hist_vola"] + colnames]
    for idx, past_return_matrix in enumerate(past_return_matrices):
        shrk_res = []
        date = past_return_matrix.index[0]
        for shrk in shrk_intensities_v2:
            shrk_est, sample, target = estimator(past_return_matrix)
            new_shrk_est = shrk
            covmat_est = new_shrk_est * target + (1-new_shrk_est) * sample
            # based on covmat --> calc pf std (and maybe return)
            pf_ret, pf_std = hf.calc_pf_std_return(covmat_est, future_return_matrices[idx])
            shrk_res.append(pf_std)

        # also want historical (can choose days) vola (and in future maybe different factors)
        hist_vola = hf.get_historical_vola(past_price_matrices[idx], days=60)
        res.append([date, hist_vola] + shrk_res)
        # save to pandas dataframe and then to disk, for each factor separately
    df = pd.DataFrame(res)
    df.columns = df.iloc[0, :]
    df = df.drop(0)
    with open(rf"{out_path_shrk}\fixed_shrkges_p{p}.pickle", 'wb') as pickle_file:
        pickle.dump(df, pickle_file)



##### Let's call the function to create the necessary data frames
in_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"
end_date = 20220331  # create it for the full data set
estimation_window_length = -99
out_of_sample_period_length = -99
pf_size = 100  # [30, 50, 100, 225, 500]
return_data_path1 = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets"
return_data_path2 = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"
estimator = estimators.get_cov1Para

# in_path = None, if the necessary matrices already exist
create_fixed_shrk_datasets(path=in_path,
                     end_date=end_date,
                     p=pf_size,
                     out_pf_sample_period_length=out_of_sample_period_length,
                     estimation_window_length=estimation_window_length,
                     out_path_shrk=return_data_path1,
                     out_path_dat=return_data_path2,
                     estimator=estimator
                     )