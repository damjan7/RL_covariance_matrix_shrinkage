import numpy as np
import pandas as pd
import math

#####################################################
# for RL only need the target and the sample covmat #
# OR                                                #
# just want shrinkage intensity                     #
#####################################################
def cov1Para(X, k=None):

    N, p = X.shape # sample size and matrix dimension
    # default setting
    if k is None or math.isnan(k):
        k = 1
    # vars
    n = N-k  # adjust effective sample size
    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(X.T.to_numpy(), X.to_numpy()))/n

    # compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar = sum(diag)/len(diag)
    target = meanvar*np.eye(p)

    return sample, target

def get_shrinkage_cov1Para(X, k=None):

    N, p = X.shape # sample size and matrix dimension
    # default setting
    if k is None or math.isnan(k):
        k = 1
    # vars
    n = N-k  # adjust effective sample size
    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(X.T.to_numpy(), X.to_numpy()))/n

    # compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar = sum(diag)/len(diag)
    target = meanvar*np.eye(p)

    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    X2 = pd.DataFrame(np.multiply(X.to_numpy(), X.to_numpy()))
    sample2 = pd.DataFrame(np.matmul(X2.T.to_numpy(), X2.to_numpy()))/n  # sample covariance matrix of squared returns
    piMat = pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(), sample.to_numpy()))

    pihat = sum(piMat.sum())

    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2

    # diagonal part of the parameter that we call rho
    rho_diag = 0
    # off-diagonal part of the parameter that we call rho
    rho_off = 0

    # compute shrinkage intensity
    rhohat = rho_diag+rho_off
    kappahat = (pihat-rhohat)/gammahat
    shrinkage = max(0, min(1, kappahat/n))

    return shrinkage, target

