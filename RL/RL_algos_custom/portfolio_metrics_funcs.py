import pandas as pd
import numpy as np

def avg_monthly_gross_leverage(weights):
    '''
    Given weights for every month, calculate the gross monthly leverage
    '''
    tot_sum = 0
    for w in weights:
        tot_sum += np.abs(w)
    res = tot_sum / len(weights)/21 # len(weights)/21 should be the number of months
    return res

def avg_monthly_turnover(weights):
    '''
    Given weights for every month, calculate the avg monthly turnover
    '''