import helper_functions as hf
import pandas as pd
import numpy as np

path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"
# actually period of interest begins in 1993, but need data until approx. 1 year before (12 * 21 trading days)
# such that cov mat can be calculated for the first date of interest
# may write this in a function as it is quite ugly like this

start_date_plus = 19920101
start_date = 19930101
end_date = 20171231

df = hf.load_peprocess(path, start_date_plus, end_date)
trading_dates_plus, rebalancing_dates = hf.get_trading_rebalancing_dates(df)

df = df[df['date'] >= start_date]

trading_dates, rebalancing_dates = hf.get_trading_rebalancing_dates(df)

p = 100
p_largest_stocks = hf.get_p_largest_stocks_all_reb_dates(df, rebalancing_dates, trading_dates_plus, p)

# now we now at every date which p stocks we consider. Using these we calculate the weights of our portfolio
# at each rebalancing date
# for that we need this pivoted matrix (= return matrix)



print("done")