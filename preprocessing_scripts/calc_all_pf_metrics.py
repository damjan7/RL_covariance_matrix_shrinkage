import pandas as pd
import numpy as np
import pickle
from RL.RL_algos_custom import eval_funcs
path = rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\RL"


pf_sizes = (30, 50, 100, 225, 500)
cvc_alphas = (0.16, 0.15, 0.21, 0.3, 0.47)
val_indices_correct = (5040, 10374)
val_indices_results = [val_indices_correct[0] + 21*i for i in range( (val_indices_correct[-1] - val_indices_correct[0]) // 21)]


fix_grid_results = []
pf_metrics_results = []
fix_grid_pf_metrics = []
for pf_size, cvc_alpha in zip(pf_sizes, cvc_alphas):
    with open(rf"{path}\future_return_matrices_p{pf_size}.pickle", 'rb') as f:
        fut_ret_mats = pickle.load(f)
    with open(rf'{path}\past_return_matrices_p{pf_size}.pickle', 'rb') as f:
        past_ret_mats = pickle.load(f)

    if pf_size == 50 or pf_size == 30:
        val_indices_results = val_indices_results[:-1]
    #res = eval_funcs.eval_oos_final(fut_ret_mats, past_ret_mats, val_indices_results, cvc_alpha)
    res2 = eval_funcs.grid_eval_fixed_shrkges(fut_ret_mats, past_ret_mats, val_indices_results)
    res3 = eval_funcs.get_pf_metrics(fut_ret_mats, past_ret_mats, val_indices_results, cvc_alpha)
    res4 = eval_funcs.grid_eval_fixed_shrkges_pf_metrics(fut_ret_mats, past_ret_mats, val_indices_results)
    fix_grid_results.append(res2)
    pf_metrics_results.append(res3)
    fix_grid_pf_metrics.append(res4)


print("done")
