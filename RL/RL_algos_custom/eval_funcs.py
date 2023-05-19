import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_preds(val_preds, opt_preds_ds, fixed_shrk_ds):
    """
    This function evaluates predictions, in this functions, DISCRETE shrinkages from 0 to x (20) which correspond
    to values between 0 and 1.
    The predictions are evaluated against some of the optimal predictions according to some shrkg estimator

    val preds = integers from 0 to 1
    opt preds = shrk intensities according to some shrk estimator
    """
    d1 = fixed_shrk_ds.iloc[:, 2:]
    pf_std_val = d1.values[np.arange(d1.shape[0]), val_preds]
    pf_opt_shrk = opt_preds_ds["pf_std"]

    return pf_std_val.mean(), pf_opt_shrk.mean()


def f_map(idx):
    """
    this function maps indices to shrinkages
    """
    shrkgs = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4',
              '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0']
    return shrkgs[idx]


def simple_plot(preds, actual_labels, map1=True, map2=True):
    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(preds))
    if map1 == True:
        y1 = list(map(f_map, preds))
    else:
        y1 = preds
    if map2 == True:
        y2 = list(map(f_map, actual_labels))
    else:
        y2 = actual_labels
    ax.plot(x, y1)
    ax.plot(x, y2)
    plt.legend()
    plt.show()