import numpy as np
import pandas as pd

import helper_functions as hf
from estimation import CovMatEstimation
import parameters
from parameters import params

end_date = 20171231
estimation_window_length = 1
out_of_sample_period_length = 20
pf_size = 500  # [30, 50, 100, 225, 500]
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

equal_weighted_res = model5.calc_equal_weighted_pf()

res_dict = {
    "Model 1": [model1.total_portfolio_return_V2, model1.total_pf_std_daily],
    "Model 2": [model2.total_portfolio_return_V2, model2.total_pf_std_daily],
    "Model 3": [model3.total_portfolio_return_V2, model3.total_pf_std_daily],
    "Model 4": [model4.total_portfolio_return_V2, model4.total_pf_std_daily],
    "Model 5": [model5.total_portfolio_return_V2, model5.total_pf_std_daily],
    "Model 1/N": [equal_weighted_res[5], equal_weighted_res[6]],
}

print("Results:\n", res_dict)

r_path = result_data_path + rf"\res_pfsize{pf_size}.csv"

res_df = pd.DataFrame(res_dict)
res_df.index = ['AV', 'SD']
res_df.to_csv(r_path, index=False)

print(f"done, saved to {r_path}")