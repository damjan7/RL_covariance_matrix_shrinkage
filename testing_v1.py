import helper_functions as hf
import pandas as pd
import numpy as np

path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"
# actually period of interest begins in 1993, but need data until approx. 1 year before (12 * 21 trading days)
# such that cov mat can be calculated for the first date of interest
# may write this in a function as it is quite ugly like this


end_date = 20171231

df, trading_days, rebalancing_days, start_date = hf.load_peprocess(path=path, end_date=end_date, out_of_sample_period_length=20,
                                                       estimation_window_length=1)

#trading_dates_plus, rebalancing_dates = hf.get_trading_rebalancing_dates(df)

#df = df[df['date'] >= start_date]

#trading_dates, rebalancing_dates = hf.get_trading_rebalancing_dates(df)

p = 100
p_largest_stocks = hf.get_p_largest_stocks_all_reb_dates(df, rebalancing_days, p)

print("done")

# now we now at every date which p stocks we consider. Using these we calculate the weights of our portfolio
# at each rebalancing date
# for that we need this pivoted matrix (= return matrix)


# let's assume rebalancing_dates[0+12] is our first valid rebalancing date
# example: get_return_matrix(df, res['rebalancing_date'][12], res['rebalancing_date'][0], res.iloc[12, 1:])
# so it's easy to write a function that does this for every entry of output of get_p_largest_stocks_all_reb_dates



print("done")