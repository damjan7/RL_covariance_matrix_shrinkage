import numpy as np
import pandas as pd

def get_p_largest_stocks(df, rebalancing_date, rebalancing_date_12months_before, p):
    """
    This function returns the p largest stocks for some given rebalancing date.
    It should also include a filter for stocks that lack observations for the most recent 252 trading days.
    Something like at most 5% NaN values among the last 252 trading days should be appropriate.
    Not sure how efficient this code is yet....
    :param df: dataframe containing all stocks,..
    :param rebalancing_date: given the rebalancing date
    :param p: number of stocks considered
    :return: the largest p stocks (measured by market cap), at a given rebalancing date. May return only PERMNO number
    or a whole dataframe
    """
    tmp = df[df['date'] == rebalancing_date]
    tmp = tmp.sort_values("MARKET_CAP", ascending=False)
    tmp = tmp.iloc[0:2*p]
    permno = list(tmp['PERMNO'])  # contains double the needed number of stocks in case some need to be discarded

    # filter for stocks that lack observations for the most recent 252 trading days
    # at most 5% NaN values among the last 252 trading days
    # first filter dataframe for the last 252 trading days
    tmp_df = df[df['PERMNO'].isin(permno)]
    tmp_df = tmp_df[(tmp_df['date'] < rebalancing_date) & (tmp_df['date']) >= rebalancing_date_12months_before]

    #the temp df should contain roughly 252 observations --> check how many are NaN for each PERMNO
    tmp_df_wide = tmp_df.pivot(index='date', columns='PERMNO', values='RET')
    filter_idx = tmp_df_wide.isna().sum() / tmp_df_wide.shape[0] > 0.05  # are more than 5% values NaN's?
    filter_values = filter_idx[filter_idx == True].index.values

    # can add a check that if in the p PERMNO numbers we don't have too many missing values,
    # we can directly return them..

    for v in filter_values:
        permno.remove(v)
    assert len(permno) >= p  # if not we have a problem
    return permno[0:p]


def get_p_largest_stocks_all_reb_dates(df, rebalancing_dates, trading_dates_plus, p):
    """
    returns the p largest stocks for all rebalancing days for the whole considered dataset
    :param df:
    :param rebalancing_dates:
    :param start_date:
    :param end_date:
    :param p:
    :return: a dataframe containing all p largest stocks for all rebalancing dates
    """
    res = []
    tmp_idx = np.where(trading_dates_plus == rebalancing_dates[0])[0]  # [0] to access the value of the tuple
    for idx, reb_date in enumerate(rebalancing_dates):
        if idx < 12:  # in this case use trading_days_plus to get the previous 252 trading days
            reb_start = trading_dates_plus[tmp_idx - (12-idx) * 21][0]
        else:
            reb_start = rebalancing_dates[idx - 12]
        permno_nums = get_p_largest_stocks(df, reb_date, reb_start, p)
        res.append([reb_date] + permno_nums)  # need reb_date as a list

    res = pd.DataFrame(res, columns = ['rebalancing_date'] + ["stock " + str(i) for i in range(1, 101)])
    return res


def filter_years(df, start_date, end_date):
    """
    This function filters according to year and returns new dataframe.
    The column containing the dates is called 'date' and should stay the same
    End and start are inclusive!
    :param df: input dataframe
    :param start_date: starting date, in format YEAR/MONTH/DAY, 2001/01/01
    :param end_date: end date, in format YEAR/MONTH/DAY, 2001/01/01
    :return: new dataframe
    """
    df2 = df[(start_date <= df['date']) & (df['date'] <= end_date)]
    return df2

def load_peprocess(path, start_date, end_date):
    """
    Loads data
    Applies necessary preprocessing steps before working with the data.
    Returns the same dataframe, but correctly preprocessed.
    :param path: path to dataframe with columns ['PERMNO', 'date', 'SHRCD', 'EXCHCD', 'PRC', 'RET', 'SHROUT']
    :param start_date: start date which we consider in correct format! YYYY/MM/DD
    :param end_date: end date which we consider in correct format! YYYY/MM/DD
    :return: preprocessed dataframe; removed columns ['SHRCD', 'EXCHCD'], added column ['MARKET_CAP']
    """
    data = pd.read_csv(path, dtype={'RET': np.float64}, na_values=['B', 'C'])
    data = data.drop(["SHRCD", "EXCHCD"], axis=1)
    data["MARKET_CAP"] = np.abs(data["PRC"]) * data["SHROUT"]
    data = data[(start_date <= data['date']) & (data['date'] <= end_date)]
    return data

def get_trading_rebalancing_dates(df):
    """
    returns all trading and rebalancing dates for the whole dataset
    :param df: full (preprocessed) dataset
    :return: trading dates and rebalancing dates
    """
    trading_dates = df['date'].unique()
    reb_idx = [i for i in range(len(trading_dates)) if i % 21 == 0]
    rebalancing_dates = trading_dates[reb_idx]
    return trading_dates, rebalancing_dates


def get_return_matrix(df, rebalancing_date, rebalancing_date_12months_before, permno):
    """
    Given data input matrix, rebalancing date, and permno numbers of the p stocks of interest,
    returns the return matrix that is then used for the covariance matrix [last 252 trading days]
    Also, the remaining NaN's are filled with zeros
    :param df: full data matrix
    :param rebalancing_date: rebalancing date
    :param permno: p stocks with the largest market cap without more than 5% of NaN's in past 252 trading days
    :return: return matrix in wide format [n * p], dates on y axis, stocks on x axis
    """
    tmp_df = df[(rebalancing_date_12months_before <= df['date']) & (df['date'] < rebalancing_date)]
    tmp_df = tmp_df[tmp_df['PERMNO'].isin(permno)]
    tmp_df = tmp_df.pivot(index='date', columns='PERMNO', values='RET')
    tmp_df = tmp_df.fillna(0)
    return tmp_df


def demean_return_matrix(df):
    """
    given return matrix, returns de-meaned return matrix
    stocks are on x axis, dates on y axis
    :param df: return matrix
    :return: de-meaned return matrix
    """
    # assert df.shape[0] == 252
    df_demeaned = df - df.means()  # df.means() contains the means of each column (= each stock)
    return df_demeaned


def calc_global_min_variance_pf(covmat_estimator):
    """
    Calculates the global minimum portfolio WITHOUT SHORT SELLING CONSTRAINTS
    :param covmat_estimator: covariance matrix estimator of shape p x p
    :return: portfolio weights
    """
    vec_ones = np.ones((covmat_estimator.shape[0], 1))
    inv_covmat = np.linalg.inv(covmat_estimator)
    w = inv_covmat @ vec_ones @ np.linalg.inv(vec_ones.T @ inv_covmat @ vec_ones)
    return w

