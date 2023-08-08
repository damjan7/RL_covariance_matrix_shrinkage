# from RL.RL_algos_custom.pf_size_30 import RL_CVC_p30_PF_metrics
from RL.RL_algos_custom.pf_size_50 import RL_CVC_p50_PF_metrics
from RL.RL_algos_custom.pf_size_100 import RL_CVC_p100_PF_metrics
from RL.RL_algos_custom.pf_size_225 import RL_CVC_p225_PF_metrics
from RL.RL_algos_custom.pf_size_500 import RL_CVC_p500_PF_metrics

# from RL.RL_algos_custom.pf_size_30 import RL_CVC_p30_PF_metrics_nofactors
from RL.RL_algos_custom.pf_size_50 import RL_CVC_p50_PF_metrics_nofactors
from RL.RL_algos_custom.pf_size_100 import RL_CVC_p100_PF_metrics_nofactors
from RL.RL_algos_custom.pf_size_225 import RL_CVC_p225_PF_metrics_nofactors
from RL.RL_algos_custom.pf_size_500 import RL_CVC_p500_PF_metrics_nofactors


res_pf_metrics_all = []
res_eval_all = []

'''
r1, r2 = RL_CVC_p30_PF_metrics.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)


r1, r2 = RL_CVC_p50_PF_metrics.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)


r1, r2 = RL_CVC_p100_PF_metrics.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)


r1, r2 = RL_CVC_p225_PF_metrics.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)

r1, r2 = RL_CVC_p500_PF_metrics.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)
'''


r1, r2 = RL_CVC_p50_PF_metrics_nofactors.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)


r1, r2 = RL_CVC_p100_PF_metrics_nofactors.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)


r1, r2 = RL_CVC_p225_PF_metrics_nofactors.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)

r1, r2 = RL_CVC_p500_PF_metrics_nofactors.train_with_dataloader(False)
res_pf_metrics_all.append(r1)
res_eval_all.append(r2)


print("cool")