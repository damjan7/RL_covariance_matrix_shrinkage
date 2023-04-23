import numpy as np
import pandas as pd
from preprocessing_scripts import helper_functions as hf
import pickle


class CovMatEstimation:

    def __init__(self, end_date, estimation_window_length, out_of_sample_period_length, pf_size, estimator,
                 raw_data_path, return_data_path, result_data_path):
        self.end_date = end_date
        self.estimation_window_length = estimation_window_length
        self.out_of_sample_period = out_of_sample_period_length
        self.pf_size = pf_size
        self.estimator = estimator
        self.create_data = return_data_path
        self.raw_data_path = raw_data_path
        self.return_data_path = return_data_path
        self.result_data_path = result_data_path
        self.past_return_matrices, self.future_return_matrices, self.rebalancing_days_full = self.load_data()

        self.weights, self.monthly_portfolio_returns, self.monthly_portfolio_returns_with_permno,\
            self.permno, self.total_portfolio_return, self.total_portfolio_return_V2,\
            self.total_pf_std_daily = self.calc_pf_weights_and_returns()

    def build_resultfile_name(self):
        pass

    def load_data(self):
        # The rb indicates that the file is opened for READING in binary mode.
        with open(rf"{self.return_data_path}\past_return_matrices_p{self.pf_size}.pickle", 'rb') as f:
            past_return_matrices = pickle.load(f)

        with open(rf"{self.return_data_path}\future_return_matrices_p{self.pf_size}.pickle", 'rb') as f:
            future_return_matrices = pickle.load(f)

        with open(rf"{self.return_data_path}\rebalancing_days_full.pickle", 'rb') as f:
            rebalancing_days_full = pickle.load(f)

        return past_return_matrices, future_return_matrices, rebalancing_days_full

    def calc_pf_weights_and_returns(self):
        """
        Calculates PF weights, covariance matrix estimates, monthly portfolio returns (using the calculated weights)
        Returns all these, and the "used" stocks (i.e., the permno numbers of the stocks)
        :return: Lists or pandas dataframes containing the above
        """
        # at the same time, we will use the covariance matrix to calculate the weights
        covmat_estimates = []
        weights = []
        permno = []
        monthly_portfolio_returns = []
        monthly_portfolio_returns_with_permno = []

        weighted_daily_returns = []
        daily_dates = []

        estimator = self.estimator

        # attention, past return matrices may already be de-meaned!
        for idx in range(self.rebalancing_days_full.shape[0]):  # iterate through all rebalancing days
            permno.append(self.past_return_matrices[idx].columns)  # p largest stocks we consider
            est = estimator(self.past_return_matrices[idx])  # calculate the covariance matrix estimator
            covmat_estimates.append(est)  # just append the covariance matrix estimator to the list
            weights.append(hf.calc_global_min_variance_pf(est))
            # for the monthly return data, I need the look 21 days in the future
            # note that the date list at the current index contains the returns for the past year
            # this was used to calculate the covariance matrices and determine the weights of the PF for the next 21 days
            monthly_portfolio_returns.append(
                hf.calc_monthly_return(self.future_return_matrices[idx]).values)  # values only
            monthly_portfolio_returns_with_permno.append(hf.calc_monthly_return(self.future_return_matrices[idx]))

            # this is the way gianluca implements it and I should use this as well
            # weighted daily returns [i] has the n daily returns for the pf made on rebalancing date i
            # need to average and annualize all these
            weighted_daily_returns += list(self.future_return_matrices[idx] @ weights[idx])
            daily_dates += list(self.future_return_matrices[idx].index)


        weights = pd.DataFrame(weights, index=self.rebalancing_days_full['actual_reb_day'])
        monthly_portfolio_returns = pd.DataFrame(monthly_portfolio_returns,
                                                 index=self.rebalancing_days_full['actual_reb_day'])
        monthly_portfolio_returns_with_permno = pd.DataFrame(monthly_portfolio_returns_with_permno,
                                                             index=self.rebalancing_days_full['actual_reb_day'])
        permno = pd.DataFrame(permno, index=self.rebalancing_days_full['actual_reb_day'])


        ### just for now
        monthly_weighted_portfolio_returns = []
        for i in range(weights.shape[0]):
            tmp = (weights.iloc[i, :] * monthly_portfolio_returns.iloc[i, :]).sum()
            monthly_weighted_portfolio_returns.append(tmp)
        total_portfolio_return = np.prod(monthly_weighted_portfolio_returns)
        ###

        # total return using method of gianluca in his paper
        total_portfolio_return_daily = np.mean(weighted_daily_returns) * 252
        total_pf_std_daily = np.std(weighted_daily_returns) * np.sqrt(252)

        return weights, monthly_portfolio_returns, monthly_portfolio_returns_with_permno, permno, \
               total_portfolio_return, total_portfolio_return_daily, total_pf_std_daily


    def calc_equal_weighted_pf(self):
        weights = []
        permno = []
        monthly_portfolio_returns = []
        monthly_portfolio_returns_with_permno = []

        weighted_daily_returns = []
        daily_dates = []

        for idx in range(self.rebalancing_days_full.shape[0]):  # iterate through all rebalancing days
            permno.append(self.past_return_matrices[idx].columns)  # p largest stocks we consider
            weights.append(np.ones(self.pf_size) / self.pf_size)
            # for the monthly return data, I need the look 21 days in the future
            # note that the date list at the current index contains the returns for the past year
            # this was used to calculate the covariance matrices and determine the weights of the PF for the next 21 days

            # this is not the way I use it anymore!
            monthly_portfolio_returns.append(
                hf.calc_monthly_return(self.future_return_matrices[idx]).values)  # values only
            monthly_portfolio_returns_with_permno.append(hf.calc_monthly_return(self.future_return_matrices[idx]))

            weighted_daily_returns += list(self.future_return_matrices[idx] @ weights[idx])
            daily_dates += list(self.future_return_matrices[idx].index)


        weights = pd.DataFrame(weights, index=self.rebalancing_days_full['actual_reb_day'])
        monthly_portfolio_returns = pd.DataFrame(monthly_portfolio_returns,
                                                 index=self.rebalancing_days_full['actual_reb_day'])
        monthly_portfolio_returns_with_permno = pd.DataFrame(monthly_portfolio_returns_with_permno,
                                                             index=self.rebalancing_days_full['actual_reb_day'])
        permno = pd.DataFrame(permno, index=self.rebalancing_days_full['actual_reb_day'])

        ### just for now
        monthly_weighted_portfolio_returns = []
        for i in range(weights.shape[0]):
            tmp = (weights.iloc[i, :] * monthly_portfolio_returns.iloc[i, :]).sum()
            monthly_weighted_portfolio_returns.append(tmp)
        total_portfolio_return = np.prod(monthly_weighted_portfolio_returns)
        ###

        # total return using method of gianluca in his paper
        total_portfolio_return_daily = np.mean(weighted_daily_returns) * 252
        total_pf_std_daily = np.mean(weighted_daily_returns) * np.sqrt(252)
        #####

        return weights, monthly_portfolio_returns, monthly_portfolio_returns_with_permno, permno, \
               total_portfolio_return, total_portfolio_return_daily, total_pf_std_daily
