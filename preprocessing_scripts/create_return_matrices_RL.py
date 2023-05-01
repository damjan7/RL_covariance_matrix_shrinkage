import helper_functions_RL as hf_rl
import pickle


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

    rebalancing_days_full = hf_rl.get_full_rebalancing_dates_matrix(rebalancing_days, estimation_window_length)
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




##### Let's call the function to create the necessary data frames

in_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"
end_date = 19901231
estimation_window_length = -99
out_of_sample_period_length = -99
pf_size = 30  # [30, 50, 100, 225, 500]
return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"

create_data_matrices(path=in_path,
                     end_date=end_date,
                     p=pf_size,
                     out_pf_sample_period_length=out_of_sample_period_length,
                     estimation_window_length=estimation_window_length,
                     out_path=return_data_path
                     )

