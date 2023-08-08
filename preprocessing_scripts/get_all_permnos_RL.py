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

    # returns just every trading day, hence no estimation window length is needed
    rebalancing_days_full = hf_rl.get_full_rebalancing_dates_matrix(rebalancing_days)
    p_largest_stocks = hf_rl.get_p_largest_stocks_all_reb_dates_V2(df, rebalancing_days_full, p)

    with open(rf"{out_path}\permnos{p}.pickle", 'wb') as pickle_file:
        pickle.dump(p_largest_stocks, pickle_file)

    print("done")


##### Let's call the function to create the necessary data frames

in_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"
end_date = 20220302
estimation_window_length = -99
out_of_sample_period_length = -99
pf_size = 100  # [30, 50, 100, 225, 500]
return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"

create_data_matrices(path=in_path,
                     end_date=end_date,
                     p=pf_size,
                     out_pf_sample_period_length=out_of_sample_period_length,
                     estimation_window_length=estimation_window_length,
                     out_path=return_data_path
                     )

