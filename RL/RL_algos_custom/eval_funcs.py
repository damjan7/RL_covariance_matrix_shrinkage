import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing_scripts.rl_covmat_ests_for_dataset as estimators
from preprocessing_scripts import helper_functions as hf
import pickle

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

    return pf_std_val.mean(), pf_opt_shrk.mean(), pf_std_val.std(), pf_opt_shrk.std()

def temp_eval_fct(val_preds, fut_ret_mats, past_ret_mats, reb_days, val_indices):
    weighted_rets_model = []
    weighted_rets_cov2para = []
    weighted_rets_qis = []
    sample_covmat_only = []
    for i in range(len(val_indices)):
        shrk, sample, target = estimators.cov2Para(past_ret_mats[val_indices[i]])
        model_covmat_est = val_preds[i] * target + (1-val_preds[i]) * sample
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        sample_covmat = sample
        qis_covmat_est = estimators.QIS(past_ret_mats[val_indices[i]])

        weights_model = hf.calc_global_min_variance_pf(model_covmat_est)
        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)
        weights_sample = hf.calc_global_min_variance_pf(sample_covmat)

        weighted_rets_model += list(fut_ret_mats[val_indices[i]] @ weights_model)
        weighted_rets_cov2para += list(fut_ret_mats[val_indices[i]] @ weights_cov2para)
        weighted_rets_qis += list(fut_ret_mats[val_indices[i]] @ weights_qis)
        sample_covmat_only += list(fut_ret_mats[val_indices[i]] @ weights_sample)

    res = results = {
        "network pf return" : round(np.mean(weighted_rets_model) * 252 * 100, 2) ,
        "cov2para pf return" : round(np.mean(weighted_rets_cov2para) * 252 * 100, 2) ,
        "qis pf return" : round(np.mean(weighted_rets_qis) * 252 * 100, 2) ,
        "sample covmat return": round(np.mean(sample_covmat_only) * 252 * 100, 2) ,
        "network pf sd" : round(np.std(weighted_rets_model) * np.sqrt(252) *100, 2) ,
        "cov2para pf sd" : round(np.std(weighted_rets_cov2para) * np.sqrt(252) *100, 2) ,
        "qis pf sd" : round(np.std(weighted_rets_qis) * np.sqrt(252) *100, 2) ,
        "sample covmat sd": round(np.std(sample_covmat_only) * np.sqrt(252) *100, 2) ,
    }
    return res

def eval_oos_final(fut_ret_mats, past_ret_mats, val_indices, ALPHA_COV2PARA):
    weighted_rets_EW = []
    weighted_rets_cov1para = []
    weighted_rets_cov2para = []
    weighted_rets_cov2para_mean_weight = []
    weighted_rets_qis = []

    for i in range(len(val_indices)):
        shrk, sample, target = estimators.cov2Para(past_ret_mats[val_indices[i]])
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        cov2para_covmat_meanweight = ALPHA_COV2PARA * target + (1-ALPHA_COV2PARA) * sample

        shrk, sample, target = estimators.cov1Para(past_ret_mats[val_indices[i]])
        cov1para_covmat_est = shrk * target + (1-shrk) * sample
        qis_covmat_est = estimators.QIS(past_ret_mats[val_indices[i]])

        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_cov1para = hf.calc_global_min_variance_pf(cov1para_covmat_est)
        weigths_equal_pf = np.array([1 / fut_ret_mats[0].shape[1] for _ in range(fut_ret_mats[0].shape[1])])
        weights_cov2para_meanweight = hf.calc_global_min_variance_pf(cov2para_covmat_meanweight)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)

        weighted_rets_cov2para += list(fut_ret_mats[val_indices[i]] @ weights_cov2para)
        weighted_rets_cov1para += list(fut_ret_mats[val_indices[i]] @ weights_cov1para)
        weighted_rets_EW += list(fut_ret_mats[val_indices[i]] @ weigths_equal_pf)
        weighted_rets_cov2para_mean_weight += list(fut_ret_mats[val_indices[i]] @ weights_cov2para_meanweight)
        weighted_rets_qis += list(fut_ret_mats[val_indices[i]] @ weights_qis)

        AV = f"AV & {round(np.mean(weighted_rets_EW) * 252 * 100, 2)} & {round(np.mean(weighted_rets_cov1para) * 252 * 100, 2)}" \
             f" & {round(np.mean(weighted_rets_cov2para) * 252 * 100, 2)} & {round(np.mean(weighted_rets_cov2para_mean_weight) * 252 * 100, 2)}" \
             f" & {round(np.mean(weighted_rets_qis) * 252 * 100, 2)}"

        SD = f"SD & {round(np.std(weighted_rets_EW) * np.sqrt(252) * 100, 2)} & {round(np.std(weighted_rets_cov1para) * np.sqrt(252) * 100, 2)}" \
             f" & {round(np.std(weighted_rets_cov2para) * np.sqrt(252) * 100, 2)} & {round(np.std(weighted_rets_cov2para_mean_weight) * np.sqrt(252) * 100, 2)}" \
             f" & {round(np.std(weighted_rets_qis) * np.sqrt(252) * 100, 2)}"

        IR = f"IR & {round( (np.mean(weighted_rets_EW) * 252) / (np.std(weighted_rets_EW) * np.sqrt(252)), 2)} & " \
             f"{round( (np.mean(weighted_rets_cov1para) * 252) / (np.std(weighted_rets_cov1para) * np.sqrt(252)), 2)} & " \
             f"{round( (np.mean(weighted_rets_cov2para) * 252) / (np.std(weighted_rets_cov2para) * np.sqrt(252)), 2)} & " \
             f"{round( (np.mean(weighted_rets_cov2para_mean_weight) * 252) / (np.std(weighted_rets_cov2para_mean_weight) * np.sqrt(252)), 2)} & " \
             f"{round( (np.mean(weighted_rets_qis) * 252) / (np.std(weighted_rets_qis) * np.sqrt(252)) , 2)}"

    res = results = {

        "EW pf return": round(np.mean(weighted_rets_EW) * 252 * 100, 2),
        "cov1para pf return": round(np.mean(weighted_rets_cov1para) * 252 * 100, 2),
        "cov2para pf return" : round(np.mean(weighted_rets_cov2para) * 252 * 100, 2) ,
        "cov2para (mean ALPHA) pf return": round(np.mean(weighted_rets_cov2para_mean_weight) * 252 * 100, 2),
        "qis pf return" : round(np.mean(weighted_rets_qis) * 252 * 100, 2) ,

        "EW pf sd": round(np.std(weighted_rets_EW) * np.sqrt(252) * 100, 2),
        "cov1para pf sd": round(np.std(weighted_rets_cov1para) * np.sqrt(252) * 100, 2),
        "cov2para pf sd" : round(np.std(weighted_rets_cov2para) * np.sqrt(252) *100, 2) ,
        "cov2para (mean ALPHA) pf sd" : round(np.std(weighted_rets_cov2para_mean_weight) * np.sqrt(252) *100, 2) ,
        "qis pf sd" : round(np.std(weighted_rets_qis) * np.sqrt(252) *100, 2) ,
    }
    return (res, AV, SD, IR)


def get_pf_metrics(fut_ret_mats, past_ret_mats, val_indices, ALPHA_COV2PARA):
    weights_EW_full = []
    weights_cov1para_full = []
    weights_cov2para_full = []
    weights_cov2para_mean_weight_full = []
    weights_qis_full = []
    permnos = []

    weights_with_idx = []

    for i in range(len(val_indices)):
        shrk, sample, target = estimators.cov2Para(past_ret_mats[val_indices[i]])
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        cov2para_covmat_meanweight = ALPHA_COV2PARA * target + (1-ALPHA_COV2PARA) * sample

        shrk, sample, target = estimators.cov1Para(past_ret_mats[val_indices[i]])
        cov1para_covmat_est = shrk * target + (1-shrk) * sample
        qis_covmat_est = estimators.QIS(past_ret_mats[val_indices[i]])

        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_cov1para = hf.calc_global_min_variance_pf(cov1para_covmat_est)
        weights_equal_pf = np.array([1 / fut_ret_mats[0].shape[1] for _ in range(fut_ret_mats[0].shape[1])])
        weights_cov2para_meanweight = hf.calc_global_min_variance_pf(cov2para_covmat_meanweight)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)

        weights_cov2para_full.append(pd.DataFrame(weights_cov2para, index=fut_ret_mats[val_indices[i]].columns.tolist()))
        weights_cov1para_full.append(pd.DataFrame(weights_cov1para, index=fut_ret_mats[val_indices[i]].columns.tolist()))
        weights_EW_full.append(pd.DataFrame(weights_equal_pf, index=fut_ret_mats[val_indices[i]].columns.tolist()))
        weights_cov2para_mean_weight_full.append(pd.DataFrame(weights_cov2para_meanweight, index=fut_ret_mats[val_indices[i]].columns.tolist()))
        weights_qis_full.append(pd.DataFrame(weights_qis, index=fut_ret_mats[val_indices[i]].columns.tolist()))

    c2p = helper_pf_metrics(weights_cov2para_full, val_indices)
    c1p = helper_pf_metrics(weights_cov1para_full, val_indices)
    ew = helper_pf_metrics(weights_EW_full, val_indices)
    c2p_mean = helper_pf_metrics(weights_cov2para_mean_weight_full, val_indices)
    qis = helper_pf_metrics(weights_qis_full, val_indices)

    TO = f"TO & {ew[0]} & {c1p[0]} & {c2p[0]} & {c2p_mean[0]} & {qis[0]}"
    GL = f"GL & {ew[1]} & {c1p[1]} & {c2p[1]} & {c2p_mean[1]} & {qis[1]}"
    PL = f"PL & {ew[2]} & {c1p[2]} & {c2p[2]} & {c2p_mean[2]} & {qis[2]}"

    return (TO,GL,PL)


def helper_pf_metrics(weights_with_idx, val_indices):
    running_sum_turnover = 0
    running_sum_gross_leverage = np.sum(np.abs(weights_with_idx[-1].values))
    running_sum_prop_leverage = np.sum(weights_with_idx[-1].values < 0)
    for i in range(len(val_indices) - 1):
        df_tmp = pd.concat([weights_with_idx[i], weights_with_idx[i+1]], axis=1).fillna(0)
        df_tmp.columns = ['col1', 'col2']
        running_sum_turnover += np.sum(np.abs((df_tmp['col1'] - df_tmp['col2']).values))
        running_sum_gross_leverage += np.sum(np.abs(weights_with_idx[i].values))
        running_sum_prop_leverage += np.sum(weights_with_idx[i].values < 0)
    running_sum_turnover = np.round(running_sum_turnover / (len(val_indices) - 1), 2)
    running_sum_gross_leverage = np.round(running_sum_gross_leverage / len(val_indices), 2)
    running_sum_prop_leverage = np.round(running_sum_prop_leverage / ( len(val_indices) * weights_with_idx[-1].shape[0]), 2)
    return (running_sum_turnover, running_sum_gross_leverage, running_sum_prop_leverage)



def grid_eval_fixed_shrkges(fut_ret_mats, past_ret_mats, val_indices):
    GRID = [round(0 + 0.1 * i, 2) for i in range(11)]
    BASE_ESTIMATOR = estimators.cov2Para
    returns = {}
    sds = {}
    ret_str = f"AV & "
    sds_str = f"SD & "
    IR_str = f"IR & "
    for cur_shrkg in GRID:
        cur_weighted_returns = []
        for i in range(len(val_indices)):
            shrk, sample, target = BASE_ESTIMATOR(past_ret_mats[val_indices[i]])
            covmat = cur_shrkg * target + (1-cur_shrkg) * sample
            weights = hf.calc_global_min_variance_pf(covmat)
            cur_weighted_returns += list(fut_ret_mats[val_indices[i]] @ weights)
        ret = round(np.mean(cur_weighted_returns) * 252 * 100, 2)
        sd = round(np.std(cur_weighted_returns) * np.sqrt(252) * 100, 2)
        ir = round((np.mean(cur_weighted_returns) * 252) / (np.std(cur_weighted_returns) * np.sqrt(252)), 2)
        returns[f'shrk = {cur_shrkg}'] = ret
        sds[f'shrk = {cur_shrkg}'] = sd
        ret_str += f"{ret} & "
        sds_str += f"{sd} & "
        IR_str += f"{ir} & "

    return returns, sds, ret_str, sds_str, IR_str


def grid_eval_fixed_shrkges_pf_metrics(fut_ret_mats, past_ret_mats, val_indices):
    GRID = [round(0 + 0.1 * i, 2) for i in range(11)]
    BASE_ESTIMATOR = estimators.cov2Para

    TO = f"TO & "
    GL = f"GL & "
    PL = f"PL & "

    #weights_cov2para_full.append(pd.DataFrame(weights_cov2para, index=fut_ret_mats[val_indices[i]].columns.tolist()))
    for cur_shrkg in GRID:
        weights_matrix = []
        for i in range(len(val_indices)):
            shrk, sample, target = BASE_ESTIMATOR(past_ret_mats[val_indices[i]])
            covmat = cur_shrkg * target + (1-cur_shrkg) * sample
            weights = hf.calc_global_min_variance_pf(covmat)
            weights_matrix.append(pd.DataFrame(weights, index=fut_ret_mats[val_indices[i]].columns.tolist()))
        res = helper_pf_metrics(weights_matrix, val_indices)
        TO += f"{res[0]} & "
        GL += f"{res[1]} & "
        PL += f"{res[2]} & "
    return (TO, GL, PL)

def calc_pf_metrics_network_estimator(fut_ret_mats, past_ret_mats, shrkges, val_indices):
    BASE_ESTIMATOR = estimators.cov2Para
    TO = f"TO & "
    GL = f"GL & "
    PL = f"PL & "
    weights_matrix = []
    for i in range(len(val_indices)):
        shrk, sample, target = BASE_ESTIMATOR(past_ret_mats[val_indices[i]])
        covmat = shrkges[i] * target + (1-shrkges[i]) * sample
        weights = hf.calc_global_min_variance_pf(covmat)
        weights_matrix.append(pd.DataFrame(weights, index=fut_ret_mats[val_indices[i]].columns.tolist()))
    res = helper_pf_metrics(weights_matrix, val_indices)
    TO += f"{res[0]} & "
    GL += f"{res[1]} & "
    PL += f"{res[2]} & "
    return (TO, GL, PL)

def eval_cov1para_cov2para(fut_ret_mats, past_ret_mats, val_indices):
    weighted_rets_cov1para = []
    weighted_rets_cov2para = []
    for i in range(len(val_indices)):
        shrk, sample, target = estimators.cov2Para(past_ret_mats[val_indices[i]])
        cov2para_covmat_est = shrk * target + (1-shrk) * sample

        shrk, sample, target = estimators.cov1Para(past_ret_mats[val_indices[i]])
        cov1para_covmat_est = shrk * target + (1-shrk) * sample

        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_cov1para = hf.calc_global_min_variance_pf(cov1para_covmat_est)

        weighted_rets_cov2para += list(fut_ret_mats[val_indices[i]] @ weights_cov2para)
        weighted_rets_cov1para += list(fut_ret_mats[val_indices[i]] @ weights_cov1para)

    res = results = {
        "cov2para pf return" : round(np.mean(weighted_rets_cov2para) * 252 * 100, 2) ,
        "cov1para pf return" : round(np.mean(weighted_rets_cov1para) * 252 * 100, 2) ,
        "cov2para pf sd" : round(np.std(weighted_rets_cov2para) * np.sqrt(252) *100, 2) ,
        "cov1para pf sd" : round(np.std(weighted_rets_cov1para) * np.sqrt(252) *100, 2) ,
    }
    return res


def temp_eval_fct_returns_TESTING(val_preds, fut_ret_mats, past_ret_mats, reb_days, val_indices):
    weighted_rets_model = []
    weighted_rets_cov2para = []
    weighted_rets_qis = []
    sample_covmat_only = []
    for i in range(len(val_indices)):
        shrk, sample, target = estimators.cov2Para(past_ret_mats[val_indices[i]])
        model_covmat_est = 0 * target + (1) * sample
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        sample_covmat = sample
        qis_covmat_est = estimators.QIS(past_ret_mats[val_indices[i]])

        weights_model = hf.calc_global_min_variance_pf(model_covmat_est)
        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)
        weights_sample = hf.calc_global_min_variance_pf(sample_covmat)

        weighted_rets_model += list(fut_ret_mats[val_indices[i]] @ weights_model)
        weighted_rets_cov2para += list(fut_ret_mats[val_indices[i]] @ weights_cov2para)
        weighted_rets_qis += list(fut_ret_mats[val_indices[i]] @ weights_qis)
        sample_covmat_only += list(fut_ret_mats[val_indices[i]] @ weights_sample)

    res = results = {
        "network pf return" : round(np.mean(weighted_rets_model) * 252 * 100, 2) ,
        "cov2para pf return" : round(np.mean(weighted_rets_cov2para) * 252 * 100, 2) ,
        "qis pf return" : round(np.mean(weighted_rets_qis) * 252 * 100, 2) ,
        "sample covmat return": round(np.mean(sample_covmat_only) * 252 * 100, 2) ,
        "network pf sd" : round(np.std(weighted_rets_model) * np.sqrt(252) *100, 2) ,
        "cov2para pf sd" : round(np.std(weighted_rets_cov2para) * np.sqrt(252) *100, 2) ,
        "qis pf sd" : round(np.std(weighted_rets_qis) * np.sqrt(252) *100, 2) ,
        "sample covmat sd": round(np.std(sample_covmat_only) * np.sqrt(252) *100, 2) ,
    }
    return res


def temp_eval_fct_TEST(val_preds, fut_ret_mats, past_ret_mats, reb_days, val_indices):
    weighted_rets_cov2para = pd.DataFrame()
    weighted_rets_qis = []
    sample_covmat_only = []
    equal_pf = []
    for i in range(len(val_indices)):
        shrk, sample, target = estimators.cov2Para(past_ret_mats[val_indices[i]])
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        qis_covmat_est = estimators.QIS(past_ret_mats[val_indices[i]])

        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)
        weights_sample = hf.calc_global_min_variance_pf(sample)
        weigths_equal_pf = np.array([1 / fut_ret_mats[0].shape[1] for _ in range(fut_ret_mats[0].shape[1])])

        weighted_rets_cov2para += list(fut_ret_mats[val_indices[i]] @ weights_cov2para)
        weighted_rets_qis += list(fut_ret_mats[val_indices[i]] @ weights_qis)
        sample_covmat_only += list(fut_ret_mats[val_indices[i]] @ weights_sample)
        equal_pf += list(fut_ret_mats[val_indices[i]] @ weigths_equal_pf)


    res = results = {
        "cov2para pf return" : round(np.mean(weighted_rets_cov2para) * 252 * 100, 2) ,
        "qis pf return" : round(np.mean(weighted_rets_qis) * 252 * 100, 2) ,
        "sample covmat return": round(np.mean(sample_covmat_only) * 252 * 100, 2),
        "equal pf return": round(np.mean(equal_pf) * 252 * 100, 2),
        "cov2para pf sd" : round(np.std(weighted_rets_cov2para) * np.sqrt(252) *100, 2) ,
        "qis pf sd" : round(np.std(weighted_rets_qis) * np.sqrt(252) *100, 2) ,
        "sample covmat sd": round(np.std(sample_covmat_only) * np.sqrt(252) *100, 2),
        "equal pf sd": round(np.std(equal_pf) * np.sqrt(252) *100, 2)
    }
    return res


def correct_validationset_evaluation(val_preds, pf_size):
    """
    correctly evaluates prediction, using average of all results annualized as performance measure,
    as in estimation.py
    """
    # for actual, correct validation, need the future and past return matrices as well as the rebalancing days
    with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\past_return_matrices_p{pf_size}.pickle", 'rb') as f:
        past_return_matrices = pickle.load(f)

    with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\future_return_matrices_p{pf_size}.pickle", 'rb') as f:
        fut_return_matrices = pickle.load(f)

    with open(rf"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices\rebalancing_days_full.pickle", 'rb') as f:
        reb_days = pickle.load(f)

    model_shrkges = [num / 100 for num in val_preds]
    weighted_rets_model = []
    weighted_rets_cov2para = []
    weighted_rets_qis = []
    cov2para_shrkges = []
    # calculate shrinkage target
    for i in range(len(reb_days)):
        shrk, sample, target = estimators.cov2Para(past_return_matrices[i])
        model_covmat_est = model_shrkges[i] * target + (1-model_shrkges[i]) * sample
        cov2para_covmat_est = shrk * target + (1-shrk) * sample
        qis_covmat_est = estimators.QIS(past_return_matrices[i])
        weights_model = hf.calc_global_min_variance_pf(model_covmat_est)
        weights_cov2para = hf.calc_global_min_variance_pf(cov2para_covmat_est)
        weights_qis = hf.calc_global_min_variance_pf(qis_covmat_est)

        weighted_rets_model.append(fut_return_matrices[i] @ weights_model)
        weighted_rets_cov2para.append(fut_return_matrices[i] @ weights_cov2para)
        weighted_rets_qis.append(fut_return_matrices[i] @ weights_qis)

        cov2para_shrkges.append(shrk)

    results = {
        "network pf return" : np.mean(weighted_rets_model) * 252,
        "cov2para pf return" : np.mean(weighted_rets_cov2para) * 252,
        "qis pf return" : np.mean(weighted_rets_qis) * 252,
        "network pf sd" : np.std(weighted_rets_model) * np.sqrt(252),
        "cov2para pf sd" : np.std(weighted_rets_cov2para) * np.sqrt(252),
        "qis pf sd" : np.std(weighted_rets_qis) * np.sqrt(252),
    }
    print('hehe')
    return results, cov2para_shrkges

def get_pf_sds_daily(shrinkage_intensities_as_int, fixed_shrk_ds):
    """
    This function evaluates predictions, in this functions, DISCRETE shrinkages from 0 to x (20) which correspond
    to values between 0 and 1.
    The predictions are evaluated against some of the optimal predictions according to some shrkg estimator

    val preds = integers from 0 to x
    opt preds = shrk intensities according to some shrk estimator
    """

    d1 = fixed_shrk_ds.iloc[:, 2:]
    pf_std_val = d1.values[np.arange(d1.shape[0]), shrinkage_intensities_as_int]

    return pf_std_val


def evaluate_preds_v2(val_preds, opt_preds_ds, fixed_shrk_ds):
    """
    This function evaluates predictions, in this functions, DISCRETE shrinkages from 0 to x (20) which correspond
    to values between 0 and 1.
    The predictions are evaluated against some of the optimal predictions according to some shrkg estimator

    val preds = integers from 0 to x
    opt preds = shrk intensities according to some shrk estimator
    """

    d1 = fixed_shrk_ds.iloc[:, :]
    pf_std_val = d1.values[np.arange(d1.shape[0]), val_preds]
    pf_opt_shrk = opt_preds_ds["pf_std"]

    return pf_std_val.mean(), pf_opt_shrk.mean(), pf_std_val.std(), pf_opt_shrk.std()

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
