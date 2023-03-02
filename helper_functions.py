def get_p_largest_stocks(df, rebalancing_date, rebalancing_date_12months_before, p):
    """
    This function returns the p largest stocks for some given rebalancing date.
    It should also include a filter for stocks that lack observations for the most recent 252 trading days.
    Something like at most 5% NaN values among the last 252 trading days should be appropriate.
    :param df: dataframe containing all stocks,..
    :param rebalancing_date: given the rebalancing date
    :param p: number of stocks considered
    :return: the largest p stocks (measured by market cap), at a given rebalancing date. May return only PERMNO number
    or a whole dataframe
    """
    tmp = df[df['date'] == rebalancing_date]
    tmp = tmp.sort_values("MARKET_CAP", ascending=False)
    tmp = tmp.iloc[0:2*p]
    permno = tmp['PERMNO']  # contains double the needed number of stocks in case some need to be discarded

    # filter for stocks that lack observations for the most recent 252 trading days
    # at most 5% NaN values among the last 252 trading days
    # first filter dataframe for the last 252 trading days
    tmp_df = df[df['PERMNO'].isin(permno)]
    tmp_df = tmp_df[(tmp_df['date'] < rebalancing_date) & (tmp_df['date']) >= rebalancing_date_12months_before]
    #the temp df should contain roughly 252 observations --> check how many are NaN for each PERMNO

    tmp_df_wide = tmp_df.pivot(index='date', columns='PERMNO', values='RET')
    filter_idx = tmp_df_wide.isna().sum() / tmp_df_wide.shape[0] > 0.05
    filter_values = filter_idx[filter_idx == True].index.values

    return


def filter_years(df, start_date, end_date):
    """
    This function filters according to year and returns new dataframe.
    The column containing the dates is called 'date' and should stay the same
    :param df: input dataframe
    :param start_date: starting date, in format YEAR/MONTH/DAY, 2001/01/01
    :param end_date: end date, in format YEAR/MONTH/DAY, 2001/01/01
    :return: new dataframe
    """
    df2 = df[(start_date <= df['date']) & (df['date'] <= end_date)]
    return df2

def preprocessing(df):
    """
    Applies necessary preprocessing steps before working with the data.
    Returns the same dataframe, but correctly preprocessed.
    :param df: dataframe with columns ['PERMNO', 'date', 'SHRCD', 'EXCHCD', 'PRC', 'RET', 'SHROUT']
    :return: preprocessed dataframe; removed columns ['SHRCD', 'EXCHCD'], added column ['MARKET_CAP']
    """

