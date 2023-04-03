import numpy as np
import pandas as pd
import math


# for RL only need the target and the sample covmat

def cov1Para(X, k=None):

    N, p = X.shape # sample size and matrix dimension
    # default setting
    if k is None or math.isnan(k):
        k = 1
    # vars
    n = N-k  # adjust effective sample size
    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(X.T.to_numpy(),X.to_numpy()))/n

    # compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar = sum(diag)/len(diag)
    target = meanvar*np.eye(p)


    return sample, target