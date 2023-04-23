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

    :inputs : past return data as a list, i.e. for every index of the list we need a covariance matrix
    :return : returns a dictionary containing 4 lists of values, see below
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

    # The rb indicates that the file is opened for READING in binary mode.
    with open(rf"{past_return_data_path}\past_return_matrices_p30.pickle", 'rb') as f:
        past_return_data = pickle.load(f)

    for elem in past_return_data:
        sample, target = estimator(elem)
        triu_indices = torch.triu_indices(num_stocks, num_stocks, offset=0)
        # append all matrices
        sample_covmats.append(sample.values)
        targets.append(target)
        upper_triu_sample_covmats.append(sample.values[triu_indices[0], triu_indices[1]])
        corr_mats.append(normalize_covmat(sample.values))

    # save all to the out_path
    out_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL\covariance_matrices"
    out = {
        "sample_covmats": sample_covmats,
        "targets": targets,
        "upper_triu_sample_covmats": upper_triu_sample_covmats,
        "sample_corr_mats": corr_mats
    }

    with open(rf"{out_path}\covariance_correlation_data_p{num_stocks}.pickle", 'wb') as pickle_file:
        pickle.dump(out, pickle_file)


create_covmats(
    past_return_data_path=r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL",
    num_stocks=30,
    shrinkage_method="cov1para"
)



