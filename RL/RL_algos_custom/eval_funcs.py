import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_preds(val_preds, opt_preds_ds, fixed_shrk_ds):
    """
    This function evaluates predictions, in this functions, DISCRETE shrinkages from 0 to x (20) which correspond
    to values between 0 and 1.
    The predictions are evaluated against some of the optimal predictions according to some shrkg estimator

    val preds = integers from 0 to x
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
    shrkgs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
              0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    return shrkgs[idx]

def f2_map(idx):
    """
    this function maps indices to shrinkages
    """
    shrkgs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
              0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35,
              0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53,
              0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71,
              0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
              0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
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


def myplot(*args):
    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(args[0]))
    for arg in args:
        ax.plot(x, arg)
    plt.legend()
    plt.show()
