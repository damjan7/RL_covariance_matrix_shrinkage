path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL\rebalancing_days_full.pickle"

import pickle
import pandas as pd
import numpy as np

with open(path, 'rb') as f:
    reb_dates = pickle.load(f)

# first actual reb day is the one we have in our data + 4 years, i.e. 252*4 indices
# fut reb days stays same
# past reb days moves back automatically

#actual reb days
actual_reb_days = reb_dates['actual_reb_day'].iloc[1008:].values
fut_reb_days = reb_dates['fut_reb_day'].iloc[1008:].values
prev_reb_days = reb_dates['prev_reb_day'].iloc[:-1008].values
reb_dates_full = pd.DataFrame([actual_reb_days, prev_reb_days, fut_reb_days]).transpose()
reb_dates_full.columns = ['actual_reb_day', 'prev_reb_day', 'fut_reb_day']


print('col')
