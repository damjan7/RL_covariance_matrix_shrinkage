from RL.RL_dev import RL_covariance_estimators as estimators
import pickle
import torch
import numpy as np

def normalize_covmat(covmat):
    """
    Given covariance matrix, normalizes it (= correlation matrix)
    """
    corrmat = np.zeros(covmat.shape)
    for i in range(covmat.shape[0] - 1):
        for j in range(i, covmat.shape[0] - 1):
            corrmat[i, j] = corrmat[j, i] = covmat[i, j] / (np.sqrt(covmat[i, i]) * np.sqrt(covmat[j, j]))
    return corrmat


def create_covmats(past_return_data_path, num_stocks, shrinkage_method: str="None"):
    # past_return_data_path = C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL [currently!]
    """
    Given the return matrices, this function creates and saves the covariance matrices as they are needed
    for determining the covariance matrix estimator (either using RL or classical shrinkage methods)
    Also creates and saves correlation matrices as they may be needed!

    --> SHOULD IT ALSO CALCULATE THE TARGET MATRIX AS IT HAS TO BE CALCULATED OVER AND OVER AGAIN OTHERWISE
    AND THIS MAKES THE WHOLE CODE SLOW

    input: past return data as a list, i.e. for every index of the list we need a covariance matrix
    """

    covariance_matrix_estimators = {
        "cov1para": estimators.cov1Para,
        #"sample": covariance_estimators.sample_covmat,
        #"cov2para": estimators_testing.cov2Para,
        #"cov_diag": estimators_testing.covDiag,
        #"cov_cor": estimators_testing.covCor,
        #"GIS": estimators_testing.GIS,
    }
    estimator = covariance_matrix_estimators[shrinkage_method]
    sample_covmats, targets = [], []
    upper_triu_sample_covmats = []
    corr_mats = []

    past_return_data = #pickle..

    for elem in past_return_data:
        sample, target = estimator(elem)
        triu_indices = torch.triu_indices(num_stocks, num_stocks, offset=0)
        # append all matrices
        sample_covmats.append(sample)
        targets.append(target)
        upper_triu_sample_covmats.append(sample[triu_indices[0], triu_indices[1]])
        corr_mats.append(normalize_covmat(sample))

    # save all to the out_path
    out_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL\covariance_matrices"



