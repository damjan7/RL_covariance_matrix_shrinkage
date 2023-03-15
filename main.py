import numpy as np
import pandas as pd

import helper_functions as hf
from estimation import CovMatEstimation
import parameters
from parameters import params

end_date = 20171231
estimation_window_length = 1
out_of_sample_period_length = 20
pf_size = 100
estimator = None
raw_data_path = None
return_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices"
result_data_path = r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\results"


model1 = CovMatEstimation(end_date, estimation_window_length, out_of_sample_period_length, pf_size,
                          params["estimator"]["cov1para"],
                          raw_data_path, return_data_path, result_data_path)

model2 = CovMatEstimation(end_date, estimation_window_length, out_of_sample_period_length, pf_size,
                          params["estimator"]["cov2para"],
                          raw_data_path, return_data_path, result_data_path)

model3 = CovMatEstimation(end_date, estimation_window_length, out_of_sample_period_length, pf_size,
                          params["estimator"]["cov_diag"],
                          raw_data_path, return_data_path, result_data_path)

model4 = CovMatEstimation(end_date, estimation_window_length, out_of_sample_period_length, pf_size,
                          params["estimator"]["cov_cor"],
                          raw_data_path, return_data_path, result_data_path)

model5 = CovMatEstimation(end_date, estimation_window_length, out_of_sample_period_length, pf_size,
                          params["estimator"]["sample"],
                          raw_data_path, return_data_path, result_data_path)

print("done")