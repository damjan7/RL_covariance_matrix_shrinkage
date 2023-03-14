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

# fill the remaining NaN's in the 'RET' column with zeros
df['RET'] = df['RET'].fillna(0)


past_return_matrices = []
future_return_matrices = []
# future return matrices do not need to be demeaned of course!
for idx, reb in enumerate(rebalancing_days_full['actual_reb_day']):
    assert reb == rebalancing_days_full['actual_reb_day'][idx] # if this never failed, can just iterate through shape[0] of rebalancing_days_full
    tmp_mat = hf.get_return_matrix(df, rebalancing_days_full['actual_reb_day'][idx], rebalancing_days_full['prev_reb_day'][idx], p_largest_stocks.loc[reb, :].values)
    tmp_mat = hf.demean_return_matrix(tmp_mat)
    past_return_matrices.append(tmp_mat)

    tmp_mat = hf.get_return_matrix(df, rebalancing_days_full['fut_reb_day'][idx], rebalancing_days_full['actual_reb_day'][idx], p_largest_stocks.loc[reb, :].values)
    future_return_matrices.append(tmp_mat)


# now let us save these return matrices to memory so we can use them all the time
# The wb indicates that the file is opened for WRITING in binary mode.
path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices"
with open(rf"{path}\past_return_matrices_p100.pickle", 'wb') as pickle_file:
    pickle.dump(past_return_matrices, pickle_file)
with open(rf"{path}\future_return_matrices_p100.pickle", 'wb') as pickle_file:
    pickle.dump(future_return_matrices, pickle_file)

# save data frames containing rebalancing days and trading days
with open(rf"{path}\rebalancing_days_full.pickle", 'wb') as pickle_file:
    pickle.dump(rebalancing_days_full, pickle_file)
with open(rf"{path}\trading_days.pickle", 'wb') as pickle_file:
    pickle.dump(trading_days, pickle_file)

print("done")
