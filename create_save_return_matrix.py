import helper_functions as hf
import pickle

# inputs
path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"
end_date = 20171231
p = 100


df, trading_days, rebalancing_days, start_date = hf.load_peprocess(path=path, end_date=end_date, out_of_sample_period_length=20,
                                                       estimation_window_length=1)
rebalancing_days_full = hf.get_full_rebalancing_dates_matrix(rebalancing_days)

p_largest_stocks = hf.get_p_largest_stocks_all_reb_dates(df, rebalancing_days, p)

# check if the "actual" rebalancing days are equal
assert (rebalancing_days_full["actual_reb_day"].values == p_largest_stocks.index).all()

return_mat_dict = {}
for idx, reb in enumerate(rebalancing_days_full['actual_reb_day']):
    tmp_mat = hf.get_return_matrix(df, reb, rebalancing_days_full['prev_reb_day'][idx], p_largest_stocks.loc[reb, :].values)
    tmp_mat = hf.demean_return_matrix(tmp_mat)
    return_mat_dict[str(reb)] = tmp_mat


# now let us save these return matrices to memory so we can use them all the time

# The wb indicates that the file is opened for WRITING in binary mode.
with open(r"/return_matrices/return_matrices_p100.pickle", 'wb') as pickle_file:
    pickle.dump(return_mat_dict, pickle_file)
