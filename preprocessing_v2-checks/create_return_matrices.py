# load full crsp dataset
# for every date, calculate the p largest stocks
# for now just use stocks that have full past return and future return history
# idea: every date, take p*1.5 largest stocks, check if they have full history, if yes,
#DON'T demean them now
import pandas as pd
import numpy as np

pf_size = 100
crsp_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\CRSP_2022_03.csv"

data = pd.read_csv(crsp_path, dtype={'RET': np.float64}, na_values=['B', 'C'])
print("loaded data")
data = data.drop(["SHRCD", "EXCHCD"], axis=1)
data["MARKET_CAP"] = np.abs(data["PRC"]) * data["SHROUT"]

# there are some dates where we only have permno numbers but no observations [i.e. no PRC, RET, SHROUT]
# we will drop these
weird_dates = [19960219, 19921024, 20010911, 20121029, 19850927]
data = data.loc[~data['date'].isin(weird_dates), :]

subset = data.iloc[0:30, :]
print('col')


all_dates = sorted(np.unique(data['date'].values))
reb_days = all_dates[252:-21]

# get largest cap for every date
# first possible date is all_dates[252]
# largest possible is 21 days before end of data
# for each date, get a list of largest stocks, check if they have full history and future,
# if yes; take them
#pivoted_dataset = data.pivot(index='date', columns='PERMNO', values='MARKET_CAP')
pivoted_ds_SMALL = data.iloc[0:10000].pivot(index='date', columns='PERMNO', values='MARKET_CAP').fillna(0)
largest_stocks = {}
for idx, date in enumerate(reb_days):
    permno_list = pivoted_ds_SMALL.iloc[0, :].sort_values(ascending=False).index.tolist()[0: 1.5 * pf_size]
    for permno in permno_list:



