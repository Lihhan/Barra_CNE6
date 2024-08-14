import os
import config
import calendar
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.tseries.offsets as toffsets
from itertools import dropwhile, chain, product
from functools import reduce, wraps
from dask import dataframe as dd
from dask.multiprocessing import get
#from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from ParallelCalFactor import ParallelCalFactor_pro
from globalVariables import *
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

def add_market(secucode):
    secucode_str = secucode.zfill(6)
    if secucode_str[0] == '3' or secucode_str[0] == '0':
        secucode_str += '.SZ'
    elif secucode_str[0] == '6':
        secucode_str += '.SH'
    return secucode_str

def day_factor_beta(logic_name, start_date, end_date, key, function_name):

    if logic_name == 'daily/size/size_vars':
        params_dict = {
            'start_date': start_date,                   # 因子计算开始日期
            'end_date': end_date,                       # 因子计算结束日期
            'data_key': [key],                   # 对应数据种类下想取的表名
            'rolling_window': 252,                   # 在提取开始数据之前还会提取多少数据
            'show_example_data': False,                 # 如果为True，则输出展示用数据
        }
        Pro = ParallelCalFactor_pro()
        res = Pro.process_factor(function_name, params=params_dict)
        
    return res

def draw_single_data(df, params):
    
    resdata = df[params['data_key'][0]]

    factor = pd.DataFrame(index=range(resdata.shape[1]), columns=['date', 'securityid'])
    factor['securityid'] = resdata.columns
    date = pd.to_datetime(resdata.index[-1].strftime('%F'))
    factor['date'] = date

    resdata = np.array(resdata)

    factor['data'] = resdata[-1]
    
    return factor

class MYCALFUNC():
    def __init__(self) -> None:
        pass
    
    def pivot(self, demo_df, arr):
        demo_df['indicator'] = arr
        return demo_df.pivot_table('indicator', columns='securityid', index='date')
    
    def getonefactor(self, indicator, df):
        factor = df.pivot_table(indicator, columns='securityid', index='date')
        return factor
    
    def add_market(self, df):
        df['securityid'] = df['securityid'].apply(lambda x: f"{int(x):06d}")
        df = df[df['securityid'].str[0].isin(['0', '3', '6'])] # 提取A股市场的股票
        def add_code(secucode):
            if secucode[0] == '3' or secucode[0] == '0':
                secucode = secucode+'.SZ'
            elif secucode[0] == '6':
                secucode = secucode+'.SH'
            return secucode
        df['securityid'] = df['securityid'].apply(add_code)
        return df

    def calcMIDCAP(self, LNCAP):
        cubed_LNCAP = np.power(LNCAP, 3)
        model = sm.OLS(cubed_LNCAP, sm.add_constant(LNCAP)).fit()
        residuals = model.resid
        quantiles = pd.Series(residuals).quantile([0.01, 0.99])
        winsorized_residuals = np.clip(pd.Series(residuals), quantiles.iloc[0], quantiles.iloc[1])
        standardized_residuals = (winsorized_residuals - winsorized_residuals.mean()) / winsorized_residuals.std()
        MIDCAP = standardized_residuals.values
        return MIDCAP
    
    def calcREG(self, row, reg_arr, window, half_life):
        decay = np.exp(np.log(0.5) / half_life)
        
        def get_weights(window):
            return np.array([decay ** (window - i) for i in range(window)])
        
        def fit_model(i):
            X = sm.add_constant(row.iloc[i:i+window])
            y = reg_arr[i:i+window]
            weights = get_weights(window)
            model = sm.WTS(y, X, weights=weights)
            results = model.fit()
            return (results.params[1], results.resid, results.params[0])
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fit_model, range(len(row) - window + 1)))

        return results
    
    def calc_RESIDUAL_VAR(self, series):
        return series.values.std()
    
    def calcLTREVRSL(self, df):
        lagged_df = df.shift(273)
        rolling_average = lagged_df.rolling(window=11).mean()
        result_df = -rolling_average
        return result_df
    
    def calcEANVAR(self, series, rolling_window):
        annual_data = series.resample('D').mean().rolling(window=rolling_window).agg(['std', 'mean'])
        annual_data = annual_data.reindex(series.index, method='ffill')
        return annual_data['std'] / annual_data['mean']
    
    def calcGROWTH(self, series, rolling_window):
        annual_mean = series.resample('D').mean().rolling(window=rolling_window).agg(['mean'])
        annual_mean = annual_mean.reindex(series.index, method='ffill')
        def get_slope(series):
            X = sm.add_constant(np.arange(len(series)))
            y = series.values
            model = sm.OLS(y, X, missing='drop')
            results = model.fit()
            return results.params[1]
        rolling_slope = series.rolling(window=rolling_window).apply(get_slope, raw=False)
        return -rolling_slope/annual_mean['mean']
    
    def fit(self, factor, neg, n):
        def Standardize(factor, neg):
            X_weighted_mean = np.nansum(factor * neg) / np.nansum(neg)
            S_std = np.nanstd(factor)
            return (factor - X_weighted_mean) / S_std
        def filter_extreme_MAD(vector, n):
            median = np.nanmedian(vector)
            abs_deviation = np.abs(vector - median)
            new_median = np.nanmedian(abs_deviation)
            dt_up = median + n * new_median
            dt_down = median - n * new_median
            return np.clip(vector, dt_down, dt_up)
        factor = Standardize(filter_extreme_MAD(factor, n), neg)
        return factor
    
    def ols2(self, x,y):
        top = (x - np.nanmean(x,axis=0))*(y - np.nanmean(y,axis=0))
        nan_sum_top = np.nanmean(top,axis=0)*np.count_nonzero(~np.isnan(top), axis=0)
        bot = np.square(x - np.nanmean(x,axis=0))
        nan_sum_bot = np.nanmean(bot,axis=0)*np.count_nonzero(~np.isnan(bot), axis=0)
        beta = nan_sum_top / nan_sum_bot
        alpha = np.nanmean(y, axis=0) - beta*np.nanmean(x,axis=0)
        y_pred = alpha + beta*x
        residual = y - y_pred
        rse = np.sqrt(np.nanmean(np.square(residual),axis=0)/(x.shape[0]-2))   # n-k residual standard error
        return beta, alpha, residual, rse
    
    def ols_with_halflife_2d(self, x, y, half_life):
        decay_factor = np.exp(-np.log(2) / half_life)
        weights = decay_factor ** np.arange(x.shape[0])[::-1]
        weighted_mean_x = np.nansum(weights[:, None] * x, axis=0) / np.sum(weights)
        weighted_mean_y = np.nansum(weights[:, None] * y, axis=0) / np.sum(weights)
        weighted_top = np.nansum(weights[:, None] * (x - weighted_mean_x) * (y - weighted_mean_y), axis=0)
        weighted_bot = np.nansum(weights[:, None] * np.square(x - weighted_mean_x), axis=0)
        beta = weighted_top / weighted_bot
        alpha = weighted_mean_y - beta * weighted_mean_x
        y_pred = alpha + beta * x
        residuals = y - y_pred
        rse = np.sqrt(np.nansum(weights[:, None] * np.square(residuals), axis=0) / (np.sum(weights)-2))
        return beta, alpha, rse
    
    def calc_growth_rate(self, y):
        x = np.arange(1, 21)
        x = np.repeat(np.array(x), y.shape[1]).reshape(x.shape[0], y.shape[1])
        beta = self.ols2(x, y)[0]
        return beta / np.nanmean(y, axis = 0)
    
    def weighted_log_return(log_returns, window_size = 1040, half_life = 260, lag_days = 273, average_window = 11):
        weights = np.exp(-np.log(2) / half_life * np.arange(window_size))
        weighted_returns = log_returns.rolling(window=window_size).apply(lambda x: np.dot(weights[:len(x)], x[::-1]), raw=True)
        lagged_returns = weighted_returns.shift(lag_days)
        non_lagged_average = bn.move_mean(lagged_returns, window=average_window, min_count=1, axis=0)
        final_result = -non_lagged_average
        return final_result
    
    def calc_resid(self, y, X):
        numerator = np.nansum((X - np.nanmean(X)) * (y - np.nanmean(y)))
        denominator = np.nansum((X - np.nanmean(X)) ** 2)
        beta = numerator / denominator
        alpha = np.nanmean(y) - beta * np.nanmean(X)
        residuals = y - (alpha + beta * X)
        return residuals
    
    def calc_resid_weighted(self, y, X, w):
        mean_X = np.nansum(w * X) / np.nansum(w)
        mean_y = np.nansum(w * y) / np.nansum(w)
        numerator = np.nansum(w * (X - mean_X) * (y - mean_y))
        denominator = np.nansum(w * (X - mean_X) ** 2)
        beta = numerator / denominator
        alpha = mean_y - beta * mean_X
        residuals = y - (alpha + beta * X)
        return residuals

def get_last_tradedate(date_loop_end):
    '''
    返回给定起始日期的交易日
    :param date_loop_start:
    :param date_loop_end: 不包括
    :return:
    '''
    trade_calendar = config.calendar_path
    calendar_date = pd.read_hdf(trade_calendar)
    trade_date = calendar_date[calendar_date['isOpen'] == 1]['calendarDate'].reset_index(drop=True)
    trade_date = pd.to_datetime(trade_date)  # all trade date series
    trade_date = trade_date.to_list()
    trade_date.sort()  # 所有交易日列表
    
    date_loop_end = str(date_loop_end)[:10]
    date_loop_end = datetime.strptime(date_loop_end, '%Y-%m-%d')  # 不包括 深度因子决定
    previous_date = trade_date[:trade_date.index(date_loop_end)][-1]  # 上一个日期

    return previous_date


def get_tradedate_calendar(date_loop_start,date_loop_end):
    '''
    返回给定起始日期的交易日序列
    :param date_loop_start:
    :param date_loop_end:
    :return:
    '''
    trade_calendar = config.calendar_path
    calendar_date = pd.read_hdf(trade_calendar)
    trade_date = calendar_date[calendar_date['isOpen'] == 1]['calendarDate'].reset_index(drop=True)
    trade_date = pd.to_datetime(trade_date)  # all trade date series
    trade_date = trade_date.to_list()
    trade_date.sort()  # 所有交易日列表

    date_loop_start = datetime.strptime(date_loop_start, '%Y-%m-%d')  # 包括
    date_loop_end = datetime.strptime(date_loop_end, '%Y-%m-%d')  # 不包括 深度因子决定
    date_loop_all = trade_date[trade_date.index(date_loop_start)+1:trade_date.index(date_loop_end)+1]

    return date_loop_all

def IndustryID2Dummy(industry_code):
    # 获取截面的行业因子值
    dummy_neutralsubID = pd.get_dummies(industry_code['SW_industry_name'])  # 这里是行业哑变量函数
    dummy_neutralsubID.index = industry_code['SECUCODE']
    return dummy_neutralsubID

def add_exchange(secucode):
    if secucode[0] == '3' or secucode[0] == '0':
        res = secucode+'.SZ'
    elif secucode[0] == '6':
        res = secucode+'.SH'
    return res

class IndustryPatition(object):
    '''
    当前股票，中信一级行业分类
    '''
    def __init__(self):
        pass

    def data_from_sql(self):
        industry_path = config.industry_index_path
        info_industry = pd.read_csv(industry_path,index_col=0)
        info_industry = pd.DataFrame(info_industry)

        latest_data = pd.DataFrame(info_industry.loc[info_industry.index.max(),:].values,columns=['SW_industry_name'])
        latest_data['SECUCODE'] = info_industry.loc[info_industry.index.max(),:].index
        latest_data.reset_index(drop=True,inplace=True)

        latest_data['SECUCODE'] = latest_data['SECUCODE'].apply(add_exchange)

        return latest_data

def GetDateIndustryData(startdate):
    # 获取A股的每日行业名称
    examp1 = IndustryPatition()
    industry_code = examp1.data_from_sql()[['SW_industry_name', 'SECUCODE']]
    industry_code['TRADINGDAY'] = pd.to_datetime(startdate)
    industry_code.index = industry_code['TRADINGDAY']
    return industry_code
    
    
# 线性插值
def get_fill_vals(nanidx, valid_vals):
    start, end = nanidx[0], nanidx[-1]
    before_val, after_val = valid_vals[start-1], valid_vals[end+1]
    diff = (after_val - before_val) / (1 + len(nanidx))
    fill_vals = [before_val + k * diff for k in range(1, len(nanidx) + 1)]
    return fill_vals


# 对Series对象进行线性插值
def linear_interpolate(series):
    vals = series.values
    valid_vals = list(dropwhile(lambda x: np.isnan(x), vals))
    idx = np.where(np.isnan(valid_vals))[0]
    start_idx = len(vals) - len(valid_vals)
    
    tmp = []
    for i, cur_num in enumerate(idx):
        try:
            next_num = idx[i+1]
        except IndexError:
            if cur_num < len(vals) - 1:
                try:
                    if tmp:
                        tmp.append(cur_num)
                        fill_vals = get_fill_vals(tmp, valid_vals)
                        for j in range(len(tmp)):
                            vals[start_idx + tmp[j]] = fill_vals[j]
                    else:
                        fill_val = 0.5 * (valid_vals[cur_num - 1] + valid_vals[cur_num + 1])
                        vals[start_idx + cur_num] = fill_val
                except IndexError:
                    break
                break
        else:
            if next_num - cur_num == 1:
                tmp.append(cur_num)
            else:
                if tmp:
                    tmp.append(cur_num)
                    fill_vals = get_fill_vals(tmp, valid_vals)
                    for j in range(len(tmp)):
                        vals[start_idx + tmp[j]] = fill_vals[j]
                    tmp = []
                else:
                    try:
                        fill_val = 0.5 * (valid_vals[cur_num - 1] + valid_vals[cur_num + 1])
                        vals[start_idx + cur_num] = fill_val
                    except IndexError:
                        break
    res = pd.Series(vals, index=series.index)
    return res

def merge3to1(factor1,factor2,factor3,weight):
    """
    将三个小因子合成一个大类因子,因子需已标准化
    factor 因子，为dataframe，index为日期，column为股票代码，要保证三个小因子日期和股票相同。
    weight 合并权重 list如weight = [0.1,0.2,0.7]
    返回：合并后的因子 为为dataframe，index为日期，column为股票代码
    注： dataframe可为1日的数据或多日的数据
    """
    noneDASTA = pd.isnull(factor1)*1
    noneCMRA = pd.isnull(factor2)*2
    noneHsigma = pd.isnull(factor3)*4
    
    nonesum = noneDASTA+noneCMRA+noneHsigma
    MergeFactor = weight[0]*factor1 + weight[1]*factor2 + weight[2]*factor3
    weight1 = [weight[1]/(weight[1]+weight[2]),weight[2]/(weight[1]+weight[2])]
    weight2 = [weight[0]/(weight[0]+weight[2]),weight[2]/(weight[0]+weight[2])]
    weight3 = [weight[0]/(weight[0]+weight[1]),weight[1]/(weight[0]+weight[1])]

    for index in MergeFactor.index:
        # print(index)
        for code in MergeFactor.columns:
            value = nonesum.loc[index,code]
            if value == 0:
                continue
            elif value == 1:
                MergeFactor.loc[index,code] = weight1[0]*factor2.loc[index,code] + weight1[1]*factor3.loc[index,code]
            elif value == 2:
                MergeFactor.loc[index,code] = weight2[0]*factor1.loc[index,code] + weight2[1]*factor3.loc[index,code]
            elif value == 3:
                MergeFactor.loc[index,code] = factor3.loc[index,code]
            elif value == 4:
                MergeFactor.loc[index,code] = weight3[0]*factor1.loc[index,code] + weight3[1]*factor2.loc[index,code]
            elif value == 5:
                MergeFactor.loc[index,code] = factor2.loc[index,code]
            elif value == 6:
                MergeFactor.loc[index,code] = factor1.loc[index,code]
        
    return MergeFactor


def merge2to1(factor1,factor2,weight):
    """
    将两个小因子合成一个大类因子,因子需已标准化
    factor 因子，为dataframe，index为日期，column为股票代码，要保证这两个小因子日期和股票相同。
    weight 合并权重 list如weight = [0.1,0.9]
    返回：合并后的因子 为为dataframe，index为日期，column为股票代码
    注： dataframe可为1日的数据或多日的数据
    """
    noneDASTA = pd.isnull(factor1)*1
    noneCMRA = pd.isnull(factor2)*2
    
    nonesum = noneDASTA+noneCMRA
    MergeFactor = weight[0]*factor1 + weight[1]*factor2

    for index in MergeFactor.index:
        # print(index)
        for code in MergeFactor.columns:
            value = nonesum.loc[index,code]
            if value == 0:
                continue
            elif value == 1:
                MergeFactor.loc[index,code] = factor2.loc[index,code]
            elif value == 2:
                MergeFactor.loc[index,code] = factor1.loc[index,code]
        
    return MergeFactor


def orthogonalize1(factor1,factor2):
    """
    factor1对factor2进行正交化
    （因子需已经标准化）
    factor1 = alpha + beta*factor2 +resids
    
    factor 因子，为dataframe，index为日期，column为股票代码，要保证这两个因子日期和股票相同。
    返回残差项resids，为dataframe，index为日期，column为股票代码
    注： dataframe可为1日的数据或多日的数据
    """
    datelist = factor1.index
    codelist = factor1.columns
    factor2 = factor2.loc[datelist,codelist]
    resid = pd.DataFrame(columns =codelist )
    for date in datelist:
        print(date)
        tmp = factor2.loc[date]
        x = sm.add_constant(tmp)
        x['y'] = factor1.loc[date]
        x = x.dropna()
        y = x['y']
        del x['y']
        x = x.astype(float)
        y = y.astype(float)
        
        est = sm.OLS(y,x).fit()
        res = est.resid
        res.name = date
        resid.loc[date] = res
    return resid


def orthogonalize2(factor1,factor2,factor3):
    """
    factor1对factor2、factor3进行正交化
    （因子需已经标准化）
    factor1 = alpha + beta1*factor2 + beta2*factor3 + resids
    
    factor 因子，为dataframe，index为日期，column为股票代码，要保证这三个因子日期和股票相同。
    返回残差项resids，为dataframe，index为日期，column为股票代码
    注： dataframe可为1日的数据或多日的数据
    """
    datelist = factor1.index
    codelist = factor1.columns
    factor2 = factor2.loc[datelist,codelist]
    factor3 = factor3.loc[datelist,codelist]
    resid = pd.DataFrame(columns =codelist )
    for date in datelist:
        print(date)
        tmp = factor2.loc[date]
        tmp.name = 'factor2'
        x = sm.add_constant(tmp)
        x['factor3'] = factor3.loc[date]
        x['y'] = factor1.loc[date]
        x = x.dropna()
        y = x['y']
        del x['y']
        x = x.astype(float)
        y = y.astype(float)        
        est = sm.OLS(y,x).fit()
        res = est.resid
        res.name = date
        resid.loc[date] = res
    return resid

def retain(df,min_periods=42):
    loc = pd.notnull(df)
    notnull = loc.sum(axis=0)
    a = (notnull>min_periods)*1
    a = a.replace(0,np.nan)
    a = sorted(a.dropna().index)
    return a
