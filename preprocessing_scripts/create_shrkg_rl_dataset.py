"""
This file should create a dataset containing
    - shrinkage intensity according to some shrinkage method
    - corresponding portfolio standard deviation (= reward/cost signal)
    - shrinkages from 0 to 1 (discretized) and corresponding pf std deviations
            - ALTERNATIVELY: consider deviation from optimal shrinkage intensity
    - additional params, such as historical vola for example

The idea predict (using RL/bandits) in what ways we should deviate from the "optimal" shrinkage intensity
according to some method
"""
import pandas as pd

import helper_functions_RL as hf_rl
import pickle
import numpy as np
from preprocessing_scripts import rl_covmat_ests_for_dataset as estimators
import helper_functions as hf


def create_data_matrices(path, end_date, p, out_pf_sample_period_length, estimation_window_length, out_path):
    """
    This function takes the path to the raw data, two other inputs, and saves all the necessary dataframes
    NOTE: past return matrices are DE-MEANED, future matrices are NOT
    :param path: path to the raw data
    :param end_date: end date we want to consider
    :param p: num of stocks considered for building the portfolio
    :return:
    """
    df, trading_days, rebalancing_days, start_date = hf_rl.load_preprocess(path=path, end_date=end_date,
                                                                           out_of_sample_period_length=out_pf_sample_period_length,
                                                                           estimation_window_length=estimation_window_length)

    # start_date is returned but nowhere used :-)

    rebalancing_days_full = hf_rl.get_full_rebalancing_dates_matrix(rebalancing_days)
    p_largest_stocks = hf_rl.get_p_largest_stocks_all_reb_dates_V2(df, rebalancing_days_full, p)

    # check if the "actual" rebalancing days are equal
    assert (rebalancing_days_full["actual_reb_day"].values == p_largest_stocks.index).all()

    # fill the remaining NaN's in the 'RET' column with zeros
    df['RET'] = df['RET'].fillna(0)


    past_return_matrices = []
    future_return_matrices = []
    past_price_matrices = []
    # future return matrices do not need to be demeaned of course!
    for idx, reb in enumerate(rebalancing_days_full['actual_reb_day']):
        assert reb == rebalancing_days_full['actual_reb_day'][idx]  # if this never failed, can just iterate through shape[0] of rebalancing_days_full
        tmp_mat = hf_rl.get_return_matrix(df, rebalancing_days_full['actual_reb_day'][idx], rebalancing_days_full['prev_reb_day'][idx], p_largest_stocks.loc[reb, :].values)
        tmp_mat = hf_rl.demean_return_matrix(tmp_mat)
        past_return_matrices.append(tmp_mat)

        tmp_mat = hf_rl.get_return_matrix(df, rebalancing_days_full['fut_reb_day'][idx], rebalancing_days_full['actual_reb_day'][idx], p_largest_stocks.loc[reb, :].values)
        future_return_matrices.append(tmp_mat)

        tmp_mat = hf_rl.get_price_matrix(df, rebalancing_days_full['actual_reb_day'][idx], rebalancing_days_full['prev_reb_day'][idx], p_largest_stocks.loc[reb, :].values)
        past_price_matrices.append(tmp_mat)

    # now using past return matrices, calculate shrinkage intensity using some estimators
    # also calculate pf std's using the "optimal" shrinkage intensity and the others from the list
    num_inensities = 11  # num shrinkage intensities considered
    shrk_factors = np.round(np.linspace(0, 2, 11), 3)
    # loop through all factors and the past data
    # can add everything to 1 data
    for factor in shrk_factors:
        res = [["shrk_factor", "hist_vola", "pf_return", "pf_std"]]
        for idx, past_return_matrix in enumerate(past_return_matrices):
            shrk_est, sample, target = estimators.get_cov1Para(past_return_matrix)
            new_shrk_est = factor * shrk_est
            covmat_est = new_shrk_est * target + (1-new_shrk_est) * sample
            # based on covmat --> calc pf std (and maybe return)
            pf_ret, pf_std = hf.calc_pf_std_return(covmat_est, future_return_matrices[idx])
            # also want historical (can choose days) vola (and in future maybe different factors)
            hist_vola = hf.get_historical_vola(past_price_matrices[idx], days=60)
            res.append([new_shrk_est, hist_vola, pf_ret, pf_std])
        # save to pandas dataframe and then to disk, for each factor separately
        df = pd.DataFrame(res)
        with open(rf"{out_path}\factor-{factor}_p{p}.pickle", 'wb') as pickle_file:
            pickle.dump(df, pickle_file)






"""
    # now let us save these return matrices to memory so we can use them all the time
    # The wb indicates that the file is opened for WRITING in binary mode.
    with open(rf"{out_path}\past_return_matrices_p{p}.pickle", 'wb') as pickle_file:
        pickle.dump(past_return_matrices, pickle_file)
    with open(rf"{out_path}\future_return_matrices_p{p}.pickle", 'wb') as pickle_file:
        pickle.dump(future_return_matrices, pickle_file)

    # save data frames containing rebalancing days and trading days
    with open(rf"{out_path}\rebalancing_days_full.pickle", 'wb') as pickle_file:
        pickle.dump(rebalancing_days_full, pickle_file)
    with open(rf"{out_path}\trading_days.pickle", 'wb') as pickle_file:
        pickle.dump(trading_days, pickle_file)

    # save past price data
    with open(rf"{out_path}\past_price_matrices_p{p}.pickle", 'wb') as pickle_file:
        pickle.dump(past_price_matrices, pickle_file)

    print("done")
"""



##### Let's call the function to create the necessary data frames

in_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"
end_date = 19901231
estimation_window_length = -99
out_of_sample_period_length = -99
pf_size = 100  # [30, 50, 100, 225, 500]
return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\shrk_datasets"

create_data_matrices(path=in_path,
                     end_date=end_date,
                     p=pf_size,
                     out_pf_sample_period_length=out_of_sample_period_length,
                     estimation_window_length=estimation_window_length,
                     out_path=return_data_path
                     )

