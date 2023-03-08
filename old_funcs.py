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