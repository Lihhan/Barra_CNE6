import pandas as pd
import numpy as np
from datetime import datetime
from calcTools import *
import sys
import os
from ParallelCalFactor import ParallelCalFactor_pro
import globalVariables
Pro = ParallelCalFactor_pro()
import bottleneck as bn
from scipy.stats import skew,kurtosis
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy import stats
calculator = MYCALFUNC()

def short_time_daily_factors(df, params):
    # not longer than 252days
    mkt = df['basic/market_value']/1e8
    neg_mkt = df['basic/neg_market_value']/1e8
    close = df['basic/close_price_2']

    hs300_weight = df['basic/index_weight_000300'].iloc[-1]
    csi500_weight = df['basic/index_weight_000905'].iloc[-1]
    csi1000_weight = df['basic/index_weight_000852'].iloc[-1]

    factor = pd.DataFrame(index=range(neg_mkt.shape[1]), columns=['date', 'securityid'])
    factor['securityid'] = neg_mkt.columns
    date = pd.to_datetime(neg_mkt.index[-1].strftime('%F'))
    factor['date'] = date
    
    mkt = np.array(mkt)
    neg_mkt = np.array(neg_mkt)
    close = np.array(close)
    
    factor['mkt'] = mkt[-1]
    factor['neg_mkt'] = neg_mkt[-1]
    neg = mkt[-1]
    factor['close'] = close[-1]

    factor['SIZE'] = calculator.fit(np.log(mkt[-1]), neg, 5)
    
    y = np.power(factor['SIZE'], 3)
    x = factor['SIZE']
    numerator = np.nansum((x - np.nanmean(x))*(y - np.nanmean(y)))
    denominator = np.nansum(np.square(x - np.nanmean(x)))
    slope = numerator / denominator
    intercept = np.nanmean(y) - slope * np.nanmean(x)
    factor['MIDCAP'] = calculator.fit(np.power(factor['SIZE'], 3) - slope * factor['SIZE'] - intercept, neg, 5)

    MV = df['basic/neg_market_value']/1e8
    lnmv2 = np.square(np.log(np.array(MV)[-1]))
    factor['hs300_weight'] = calculator.fit(np.array(hs300_weight), neg, 5)
    factor['csi500_weight'] = calculator.fit(np.array(csi500_weight), neg, 5)
    factor['csi1000_weight'] = calculator.fit(np.array(csi1000_weight), neg, 5)
    def calc(index_weight):
        arr1 = np.array(index_weight)
        mask = ~np.isnan(arr1) & ~np.isnan(lnmv2)
        return lnmv2-np.nansum(arr1[mask] * lnmv2[mask])
    
    calc300 = calc(hs300_weight)
    calc500 = calc(csi500_weight)
    calc1000 = calc(csi1000_weight)
    
    factor['hs300_Relative_NLSize'] = calculator.fit(np.abs(calc300), neg, 5)
    factor['csi500_Relative_NLSize'] = calculator.fit(np.abs(calc500), neg, 5)
    factor['csi1000_Relative_NLSize'] = calculator.fit(np.abs(calc1000), neg, 5)
    
    factor['left_hs300_Relative_NLSize'] = calculator.fit(np.where(-calc300 >= 0, -calc300, 0), neg, 5)
    factor['left_csi500_Relative_NLSize'] = calculator.fit(np.where(-calc500 >= 0, -calc500, 0), neg, 5)
    factor['left_csi1000_Relative_NLSize'] = calculator.fit(np.where(-calc1000 >= 0, -calc1000, 0), neg, 5)
    
    factor['right_hs300_Relative_NLSize'] = calculator.fit(np.where(calc300 >= 0, calc300, 0), neg, 5)
    factor['right_csi500_Relative_NLSize'] = calculator.fit(np.where(calc500 >= 0, calc500, 0), neg, 5)
    factor['right_csi1000_Relative_NLSize'] = calculator.fit(np.where(calc1000 >= 0, calc1000, 0), neg, 5)
    
    index_ret = df['stock_index/basic/chg_pct_index'] # MktIdxdGet
    ret = df['basic/chg_pct']
    index_ret.columns = [str(x).rjust(6,'0') for x in index_ret.columns]
    benchmark = index_ret[globalVariables.BENCH]
    benchmark = np.repeat(np.array(benchmark), ret.shape[1]).reshape(benchmark.shape[0],ret.shape[1])
    factor['BETA'] = calculator.fit(calculator.ols2(np.array(benchmark[-252:]), np.array(ret[-252:]))[0], neg, 5)
    factor['RESVOL'] = calculator.fit(calculator.ols2(np.array(benchmark[-252:]), np.array(ret[-252:]))[3], neg, 5)
    
    PE = np.array(df['basic/PE_T'])[-1]
    PCF = np.array(df['basic/PCF_T'])[-1]
    EV_EBITDA = np.array(df['basic/EV_EBITDA'])[-1]
    factor['EP'] = calculator.fit(1/PE, neg, 5)
    factor['1/PCF'] = calculator.fit(1/PCF, neg, 5)
    factor['1/EBITDA'] = calculator.fit(1/EV_EBITDA, neg, 5)
    
    PB = np.array(df['basic/PB'])[-1]
    factor['BTOP'] = calculator.fit(1/PB, neg, 5)
    
    daily_turnover_rate = calculator.fit(df['basic/turnover_rate'], neg, 5)
    monthly_turnover_rate = calculator.fit(np.log(bn.move_mean(np.exp(daily_turnover_rate), window=21, min_count=1, axis=0)[-1]), neg, 5)
    seasonly_turnover_rate = calculator.fit(np.log(bn.move_mean(np.exp(daily_turnover_rate), window=63, min_count=1, axis=0)[-1]), neg, 5)
    yearly_turnover_rate = calculator.fit(np.log(bn.move_mean(np.exp(daily_turnover_rate), window=252, min_count=1, axis=0)[-1]), neg, 5)
    annualized_transaction_volume_rate = calculator.fit(np.array(daily_turnover_rate.ewm(span=252, adjust=True).sum())[-1], neg, 5)
    factor['LIQUIDTY'] = calculator.fit(1/4 * (monthly_turnover_rate + seasonly_turnover_rate + yearly_turnover_rate + annualized_transaction_volume_rate), neg, 5)
    
    factor['ChangePCT'] = calculator.fit(np.array(ret)[-1], neg, 5)
    
    t_liab = np.array(df['basic/fdmt/tLiab'])
    t_asset = np.array(df['basic/fdmt/tAssets'])
    liab_asset = calculator.fit(t_liab[-1] / t_asset[-1], neg, 5)
    PE = np.array(df['basic/fdmt/preferredStockE'])[-1]
    LD = np.array(df['basic/fdmt/tNcl'])[-1]
    BE = np.array(df['basic/fdmt/tShEquity'])[-1]-PE
    MLEV = calculator.fit(np.divide((neg_mkt[-1]+PE+LD), neg_mkt[-1]), neg, 5)
    BLEV = calculator.fit(np.divide((BE+PE+LD), neg_mkt[-1]), neg, 5)
    factor['LEVERAGE'] = calculator.fit((0.35 * liab_asset + 0.38 * MLEV + 0.27 * BLEV), neg, 5)
    
    DPS = np.array(df['basic/fdmt/perCashDiv'])[-1]
    factor['DPS'] = calculator.fit(DPS / close[-1], neg, 5)
    
    revenue = np.array(df['basic/fdmt/revenue_ttm'])[-1]
    cogs = np.array(df['basic/fdmt/COGS_ttm'])[-1]
    nincome = np.array(df['basic/fdmt/NIncome_ttm'])[-1]
    asset_turnover_rate = calculator.fit(np.divide(revenue, t_asset[-1]), neg, 5)
    gross_margin = calculator.fit(np.divide(revenue - cogs, t_asset[-1]), neg, 5)
    gross_sale = calculator.fit(np.divide(revenue - cogs, revenue), neg, 5)
    tasset_revenue = calculator.fit(np.divide(nincome, t_asset[-1]), neg, 5)
    factor['PROFIT'] = calculator.fit(1/4 * (asset_turnover_rate + gross_margin + gross_sale + tasset_revenue), neg, 5)
    
    logret = np.log(1+ret)
    logmkt = np.log(1+benchmark)
    excess_return = logret - logmkt
    neg_ewm_sum = -excess_return.ewm(halflife=10, min_periods=1).mean().rolling(window=63).sum()
    lagged_neg_ewm_sum = neg_ewm_sum.shift(1)
    factor['ShortTermReversal'] = calculator.fit(bn.move_mean(lagged_neg_ewm_sum, window=3, min_count=1, axis=0)[-1], neg, 5)
    
    def compute_industry_relative_strength(industry_df, relative_strength, market_caps):
        industry_bool = industry_df.astype(float)
        sqrt_market_caps = np.sqrt(market_caps)
        weighted_rs = sqrt_market_caps * relative_strength
        industry_sums = industry_bool.multiply(weighted_rs, axis=0)
        industry_weights = industry_bool.multiply(sqrt_market_caps, axis=0)
        weighted_avg = industry_sums.sum(axis=1) / industry_weights.sum(axis=1)
        return weighted_avg
    columns = [
        'filter/is_indus_compo_sw21_l1_Banking',
        'filter/is_indus_compo_sw21_l1_RealEstate',
        'filter/is_indus_compo_sw21_l1_Conglomerate',
        'filter/is_indus_compo_sw21_l1_Computer',
        'filter/is_indus_compo_sw21_l1_Retail',
        'filter/is_indus_compo_sw21_l1_MachineEquip',
        'filter/is_indus_compo_sw21_l1_PowerEquip',
        'filter/is_indus_compo_sw21_l1_ConstrDecor',
        'filter/is_indus_compo_sw21_l1_BuildMater',
        'filter/is_indus_compo_sw21_l1_BasicChemicals',
        'filter/is_indus_compo_sw21_l1_HomeAppliances',
        'filter/is_indus_compo_sw21_l1_TextileApparel',
        'filter/is_indus_compo_sw21_l1_Agriculture',
        'filter/is_indus_compo_sw21_l1_Electronics',
        'filter/is_indus_compo_sw21_l1_Transportation',
        'filter/is_indus_compo_sw21_l1_Utilities',
        'filter/is_indus_compo_sw21_l1_HealthCare',
        'filter/is_indus_compo_sw21_l1_Automobiles',
        'filter/is_indus_compo_sw21_l1_SocialServices',
        'filter/is_indus_compo_sw21_l1_NonFerrousMetals',
        'filter/is_indus_compo_sw21_l1_Telecoms',
        'filter/is_indus_compo_sw21_l1_Media',
        'filter/is_indus_compo_sw21_l1_NonbankFinan',
        'filter/is_indus_compo_sw21_l1_LightIndustry',
        'filter/is_indus_compo_sw21_l1_Defense',
        'filter/is_indus_compo_sw21_l1_FoodBeverages',
        'filter/is_indus_compo_sw21_l1_Steel',
    ]
    industry_df = pd.DataFrame()
    for col in columns:
        factor_name = col.split('_')[-1]
        industry_df[factor_name] = np.nan_to_num(np.array(df[col]), nan=0.0)[-1]
    result = []
    for i in range(-6, -3):
        relative_strength = np.array(-logret.ewm(span=126, adjust=True).sum())[i]
        result.append(np.array(compute_industry_relative_strength(industry_df, relative_strength, mkt[i])))
    factor['INDMOM'] = calculator.fit(np.mean(result, axis=0), neg, 5)
    
    return factor

def mid_time_daily_factors(df, params):
    # need at least 2yrs
    mkt = df['basic/market_value']/1e8
    neg_mkt = df['basic/neg_market_value']/1e8
    mkt = np.array(mkt)
    neg = mkt[-1]
    factor = pd.DataFrame(index=range(neg_mkt.shape[1]), columns=['date', 'securityid'])
    factor['securityid'] = neg_mkt.columns
    date = pd.to_datetime(neg_mkt.index[-1].strftime('%F'))
    factor['date'] = date
    factor['mkt'] = mkt[-1]
    
    index_ret = df['stock_index/basic/chg_pct_index'] # MktIdxdGet
    ret = df['basic/chg_pct']
    index_ret.columns = [str(x).rjust(6,'0') for x in index_ret.columns]
    benchmark = index_ret[globalVariables.BENCH]
    benchmark = np.repeat(np.array(benchmark), ret.shape[1]).reshape(benchmark.shape[0],ret.shape[1])
    HistoricalAlpha = calculator.fit(calculator.ols2(np.array(benchmark[-504:]), np.array(ret[-504:]))[1], neg, 5)
    st_HistoricalAlpha = calculator.fit(calculator.ols2(np.array(benchmark[-21:]), np.array(ret[-21:]))[1], neg, 5)
    factor['MOMENTUM'] = calculator.fit(HistoricalAlpha-st_HistoricalAlpha, neg, 5)
    
    t_liab = np.array(df['basic/fdmt/tLiab'])
    t_asset = np.array(df['basic/fdmt/tAssets'])
    DA = np.array(df['basic/fdmt/da'])
    cash = np.array(df['basic/fdmt/nChangeInCash_ttm'].drop_duplicates(subset=None, keep='first', inplace=False))
    intDebt = np.array(df['basic/fdmt/intDebt'].drop_duplicates(subset=None, keep='first', inplace=False))
    t_liab_t1 = np.array(df['basic/fdmt/tLiab_t4'])
    t_asset_t1 = np.array(df['basic/fdmt/tAssets_t4'])
    NOA = t_asset[-1] - cash[-1] - t_liab[-1] + intDebt[-1]
    NOA_t4 = t_asset_t1[-1] - cash[-2] - t_liab_t1[-1] + intDebt[-2]
    ACCR_BS = NOA - NOA_t4 - DA[-1]
    ABS = calculator.fit(-ACCR_BS / t_asset[-1], neg, 5)
    nincome = np.array(df['basic/fdmt/NIncome_ttm'])[-1]
    CFO = np.array(df['basic/fdmt/nCfOperateA'])[-1]
    CFI = np.array(df['basic/fdmt/nCfFrInvestA'])[-1]
    ACCR_CF = nincome - CFO - CFI + DA[-1]
    ACF = calculator.fit(-ACCR_CF / t_asset[-1], neg, 5)
    factor['EARNQLTY'] = calculator.fit(1/2 * (ABS + ACF), neg, 5)
    
    return factor

def long_time_daily_factors(df, params):
    # need at least 5 yrs
    neg_mkt = df['basic/neg_market_value']/1e8
    mkt = df['basic/market_value']/1e8
    mkt = np.array(mkt)
    neg = mkt[-1]
    factor = pd.DataFrame(index=range(neg_mkt.shape[1]), columns=['date', 'securityid'])
    factor['securityid'] = neg_mkt.columns
    date = pd.to_datetime(neg_mkt.index[-1].strftime('%F'))
    factor['date'] = date
    factor['mkt'] = mkt[-1]
    index_ret = df['stock_index/basic/chg_pct_index'] # MktIdxdGet
    ret = df['basic/chg_pct']
    index_ret.columns = [str(x).rjust(6,'0') for x in index_ret.columns]
    ret = df['basic/chg_pct']
    benchmark = index_ret[globalVariables.BENCH]
    benchmark = np.repeat(np.array(benchmark), ret.shape[1]).reshape(benchmark.shape[0],ret.shape[1])
    logret = np.log(1+ret)
    mean_longTimeHistoricalAlpha = calculator.fit(-np.nanmean([calculator.ols_with_halflife_2d(np.array(benchmark[-1040-i:-i]), np.array(ret[-1040-i:-i]), 260)[1] for i in range(262, 273)], axis=0), neg, 5)
    weighted_sums = calculator.fit(-np.nanmean([np.array(logret.ewm(span=1040, adjust=True).sum())[-1-i] for i in range(262, 273)], axis=0), neg, 5)
    factor['LTREVRSL'] = calculator.fit(1/2 * (mean_longTimeHistoricalAlpha + weighted_sums), neg, 5)
    
    EPS = df['basic/fdmt/eps'].iloc[::63, :]
    beta_EPS = calculator.calc_growth_rate(EPS.tail(20)) # 5年期
    RPS = df['basic/fdmt/revenue_ttm'].iloc[::63, :]
    beta_RPS = calculator.calc_growth_rate(RPS.tail(20)) # 5年期
    factor['beta_EPS'] = calculator.fit(beta_EPS, neg, 5)
    factor['beta_RPS'] = calculator.fit(beta_RPS, neg, 5)

    revenue_varate = bn.move_var(RPS, window=20, min_count=1, axis=0)[-1] / bn.move_mean(RPS, window=20, min_count=1, axis=0)[-1]
    factor['revenue_varate'] = calculator.fit(revenue_varate, neg, 5)
    n_income = df['basic/fdmt/NIncome_ttm'].iloc[::63, :]
    n_income_varate = bn.move_var(n_income, window=20, min_count=1, axis=0)[-1] / bn.move_mean(n_income, window=20, min_count=1, axis=0)[-1]
    factor['n_income_varate'] = calculator.fit(n_income_varate, neg, 5)
    nChangeInCash = df['basic/fdmt/nChangeInCash_ttm'].iloc[::63, :]
    nChangeInCash_varate = bn.move_var(nChangeInCash, window=20, min_count=1, axis=0)[-1] / bn.move_mean(nChangeInCash, window=20, min_count=1, axis=0)[-1]
    factor['nChangeInCash_varate'] = calculator.fit(nChangeInCash_varate, neg, 5)
    close = np.array(df['basic/close_price_2'])[-1]
    factor['close'] = calculator.fit(close, neg, 5)

    float_a = df['basic/float_a'].iloc[::63, :]
    beta_float_a = calculator.fit(-calculator.calc_growth_rate(float_a.tail(20)), neg, 5) # 5年期
    t_asset = df['basic/fdmt/tAssets'].iloc[::63, :]
    beta_t_asset = calculator.fit(-calculator.calc_growth_rate(t_asset.tail(20)), neg, 5) # 5年期
    purFixAssetsOth = df['basic/fdmt/purFixAssetsOth_ttm'].iloc[::63, :]
    dispFixAssetsOth = df['basic/fdmt/dispFixAssetsOth_ttm'].iloc[::63, :]
    t_capital_expenditure = purFixAssetsOth - dispFixAssetsOth
    beta_t_capital = calculator.fit(-calculator.calc_growth_rate(t_capital_expenditure.tail(20)), neg, 5) # 5年期
    factor['INVSQLTY'] = calculator.fit(1/3 * (beta_float_a + beta_t_asset + beta_t_capital), neg, 5)
    
    returns = []
    ret.index = pd.to_datetime(ret.index)
    last_date = ret.index[-1]
    for i in range(1, 6):
        start_date = last_date - pd.DateOffset(years=i)
        end_date = start_date + pd.DateOffset(months=1)
        daily_returns = ret.loc[(ret.index >= start_date) & (ret.index < end_date), :]
        returns.append(daily_returns)
    all_returns = np.vstack(returns)
    mean_returns = np.nanmean(all_returns, axis=0)
    std_returns = np.nanstd(all_returns, axis=0)
    factor['Seasonality'] = calculator.fit(mean_returns / std_returns, neg, 5)
    return factor


# 日度市场因子
def get_one_day_factors_CNE6(logic_name, start_date, end_date):

    if logic_name == 'daily/short_time_rolling_factors':
        params_dict = {
            'start_date': start_date,                   # 因子计算开始日期
            'end_date': end_date,                       # 因子计算结束日期
            'data_key': ['basic/close_price_2', 'basic/neg_market_value', 'stock_index/basic/chg_pct_index', 'basic/chg_pct', 'basic/PE_T', 'basic/market_value',\
                         'basic/turnover_rate', 'basic/PCF_T', 'basic/EV_EBITDA', 'basic/PB', 'basic/fdmt/tLiab', 'basic/fdmt/tAssets', 'basic/fdmt/preferredStockE', \
                         'basic/fdmt/tNcl', 'basic/fdmt/tShEquity', 'basic/fdmt/perCashDiv', 'basic/fdmt/revenue_ttm', 'basic/fdmt/COGS_ttm', 'basic/fdmt/NIncome_ttm',\
                         'basic/index_weight_000300', 'basic/index_weight_000905', 'basic/index_weight_000852',\
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Banking',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_RealEstate',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Coal',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Conglomerate',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Computer',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Retail',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_BeautyCare',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_MachineEquip',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_PowerEquip',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_ConstrDecor',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_BuildMater',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_BasicChemicals',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_HomeAppliances',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_TextileApparel',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Agriculture',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Electronics',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Transportation',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Utilities',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_HealthCare',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Automobiles',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_SocialServices',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_NonFerrousMetals',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Telecoms',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Media',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_NonbankFinan',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_LightIndustry',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Defense',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_FoodBeverages',
                        '[1day,stock]filter/is_indus_compo_sw21_l1_Steel'],                   # 对应数据种类下想取的表名
            'rolling_window': 252,                   # 在提取开始数据之前还会提取多少数据
            'show_example_data': False,                 # 如果为True，则输出展示用数据
        }
        Pro = ParallelCalFactor_pro()
        res = Pro.process_factor(short_time_daily_factors, params=params_dict)
        
    elif logic_name == 'daily/mid_time_rolling_factors':
        params_dict = {
            'start_date': start_date,                   # 因子计算开始日期
            'end_date': end_date,                       # 因子计算结束日期
            'data_key': ['basic/neg_market_value', 'stock_index/basic/chg_pct_index', 'basic/chg_pct', 'basic/fdmt/tLiab', 'basic/fdmt/tAssets', 'basic/fdmt/da', 'basic/market_value', 
                         'basic/fdmt/nChangeInCash_ttm', 'basic/fdmt/intDebt', 'basic/fdmt/tLiab_t4', 'basic/fdmt/tAssets_t4', 'basic/fdmt/NIncome_ttm', 'basic/fdmt/nCfOperateA', 'basic/fdmt/nCfFrInvestA'],                   # 对应数据种类下想取的表名
            'rolling_window': 504,                   # 在提取开始数据之前还会提取多少数据
            'show_example_data': False,                 # 如果为True，则输出展示用数据
        }
        Pro = ParallelCalFactor_pro()
        res = Pro.process_factor(mid_time_daily_factors, params=params_dict)

    elif logic_name == 'daily/long_time_rolling_factors':
        params_dict = {
            'start_date': start_date,                   # 因子计算开始日期
            'end_date': end_date,                       # 因子计算结束日期
            'data_key': ['basic/market_value', 'basic/neg_market_value', 'stock_index/basic/chg_pct_index', 'basic/chg_pct', 'basic/float_a', 'basic/fdmt/tAssets', 'basic/fdmt/purFixAssetsOth_ttm', 
                         'basic/fdmt/dispFixAssetsOth_ttm', 'basic/fdmt/eps', 'basic/fdmt/revenue_ttm', 'basic/fdmt/NIncome_ttm', 'basic/fdmt/nChangeInCash_ttm', 'basic/close_price_2'],                   # 对应数据种类下想取的表名
            'rolling_window': 1500,                   # 在提取开始数据之前还会提取多少数据
            'show_example_data': False,                 # 如果为True，则输出展示用数据
        }
        Pro = ParallelCalFactor_pro()
        res = Pro.process_factor(long_time_daily_factors, params=params_dict)

    return res

def process_con():
    con_forecast = pd.read_hdf('/mnt/data0/BasicData/StockDB/FundDB/raw/zyyx_data/ggoal_con_forecast_stk.h5')
    con_forecast_stk = con_forecast[['entrytime', 'stock_code', 'con_pe', 'con_eps', 'con_year', 'con_eps_type', 'con_np_yoy', 'con_np_type']].copy()
    con_forecast_stk = con_forecast_stk.sort_values(by='entrytime')
    con_forecast_stk.rename(columns={'stock_code': 'securityid'}, inplace=True)
    con_forecast_df = con_forecast_stk[con_forecast_stk['con_eps_type']==1.0] # 筛选出加权计算的结果
    con_forecast_df = con_forecast_df[con_forecast_df['con_np_type']==1.0]
    df_1y = con_forecast_df[con_forecast_df['con_year'] == con_forecast_df['entrytime'].dt.year]
    df_1y['entrytime'] = df_1y['entrytime'].apply(lambda x:pd.to_datetime(x).date())
    df_1y.sort_values(by=['entrytime', 'securityid'], inplace=True)
    result_df_1y = df_1y.groupby(['entrytime', 'securityid'])[['con_pe', 'con_eps', 'con_np_yoy']].mean().reset_index()
    result_df_1y.rename(columns={'con_np_yoy':'con_np_yoy_1y'}, inplace=True)
    
    # TODO 这里是导致先前出现大量NAN的问题所在，下面的min_periods=1是为了解决这个问题
    result_df_1y['con_eps_std'] = (
        result_df_1y.sort_values('entrytime')
        .groupby('securityid')['con_eps']
        .transform(lambda x: x.rolling(window=252, min_periods=1).std())
    )
    result_df_1y['con_eps_mean'] = (
        result_df_1y.sort_values('entrytime')
        .groupby('securityid')['con_eps']
        .transform(lambda x: x.rolling(window=252, min_periods=1).mean())
    )
    result_df_1y['entrytime'] = pd.to_datetime(result_df_1y['entrytime'])

    df_2y = con_forecast_df[con_forecast_df['con_year'] == con_forecast_df['entrytime'].dt.year+1]
    df_3y = con_forecast_df[con_forecast_df['con_year'] == con_forecast_df['entrytime'].dt.year+2]
    result_df_2y = df_2y.groupby(['entrytime', 'securityid'])[['con_np_yoy']].mean().reset_index()
    result_df_2y.rename(columns={'con_np_yoy':'con_np_yoy_2y'}, inplace=True)
    result_df_3y = df_3y.groupby(['entrytime', 'securityid'])[['con_np_yoy']].mean().reset_index()
    result_df_3y.rename(columns={'con_np_yoy':'con_np_yoy_3y'}, inplace=True)
    result_df_2y['entrytime'] = pd.to_datetime(result_df_2y['entrytime']).apply(lambda x: pd.to_datetime(x.date()))
    result_df_3y['entrytime'] = pd.to_datetime(result_df_3y['entrytime']).apply(lambda x: pd.to_datetime(x.date()))
    con_f = pd.merge(result_df_1y, result_df_3y, on=['entrytime', 'securityid'], how='inner')
    con_f = pd.merge(con_f, result_df_2y, on=['entrytime', 'securityid'], how='inner')
    con_f['con_npcgrate'] = ((1+result_df_1y['con_np_yoy_1y']) * (1+result_df_2y['con_np_yoy_2y']) * (1+result_df_3y['con_np_yoy_3y'])) ** (1/3) - 1

    def get_pe_month_diff(group):
        group['prev_entrytime'] = pd.to_datetime(group['entrytime'] - pd.DateOffset(months=1))
        group['prev_entrymonth'] = group['prev_entrytime'].dt.to_period('M')
        prev_month_stats = group.groupby('prev_entrymonth')['con_pe'].agg(np.nanmean).rename('con_pe_prev_mean')
        group['con_pe_prev_mean'] = group['prev_entrymonth'].map(prev_month_stats)
        group['con_pe_change'] = group['con_pe'] - group['con_pe_prev_mean']
        return group

    def get_eps_month_diff(group):
        group['prev_entrytime'] = pd.to_datetime(group['entrytime'] - pd.DateOffset(months=1))
        group['prev_entrymonth'] = group['prev_entrytime'].dt.to_period('M')
        prev_month_stats = group.groupby('prev_entrymonth')['con_eps'].agg(np.nanmean).rename('con_eps_prev_mean')
        group['con_eps_prev_mean'] = group['prev_entrymonth'].map(prev_month_stats)
        group['con_eps_change'] = group['con_eps'] - group['con_eps_prev_mean']
        return group

    result_con_forecast = con_f.groupby('securityid').apply(get_pe_month_diff).reset_index(drop=True)
    result_con_forecast = result_con_forecast.groupby('securityid').apply(get_eps_month_diff).reset_index(drop=True)
    result_con_forecast.sort_values(by=['entrytime', 'securityid'], inplace=True)
    result_con_forecast = result_con_forecast.reset_index()
    return result_con_forecast

def process_rpt():
    rpt_forecast = pd.read_hdf('/mnt/data0/BasicData/StockDB/FundDB/raw/zyyx_data/ggoal_rpt_forecast_stk.h5')
    rpt_forecast_stk = rpt_forecast[['entrytime', 'report_year', 'stock_code', 'forecast_dps', 'report_quarter']].copy()
    rpt_forecast_stk = rpt_forecast_stk.sort_values(by='entrytime')
    rpt_forecast_stk.rename(columns={'stock_code': 'securityid'}, inplace=True)
    rpt_forecast_df = rpt_forecast_stk[rpt_forecast_stk['report_quarter']==4.0] # 筛选出年报
    result_rpt_forecast = rpt_forecast_df[rpt_forecast_df['report_year'] == rpt_forecast_df['entrytime'].dt.year]
    result_rpt_forecast = result_rpt_forecast.groupby(['entrytime', 'securityid'])[['forecast_dps']].mean().reset_index()
    return result_rpt_forecast

def process_earnings():
    rpt_earnings = pd.read_hdf('/mnt/data0/BasicData/StockDB/FundDB/raw/zyyx_data/ggoal_rpt_earnings_adjust.h5')
    rpt_earnings_stk = rpt_earnings[['entrytime', 'stock_code', 'np_adjust_mark', 'current_create_date', 'previous_create_date']].copy()
    rpt_earnings_stk.rename(columns={'stock_code': 'securityid'}, inplace=True)
    rpt_earnings_stk.sort_values(by=['securityid', 'entrytime'], inplace=True)
    # 计算每行之前的上调和下调次数的累计值
    rpt_earnings_stk['up_count'] = (rpt_earnings_stk['np_adjust_mark'] == 2).astype(int)
    rpt_earnings_stk['down_count'] = (rpt_earnings_stk['np_adjust_mark'] == 3).astype(int)
    rpt_earnings_stk['cum_up_count'] = rpt_earnings_stk.groupby('securityid')['up_count'].cumsum()
    rpt_earnings_stk['cum_down_count'] = rpt_earnings_stk.groupby('securityid')['down_count'].cumsum()
    rpt_earnings_stk['cum_total_count'] = rpt_earnings_stk['cum_up_count'] + rpt_earnings_stk['cum_down_count']
    rpt_earnings_stk['net_adjustment'] = rpt_earnings_stk['cum_up_count'] - rpt_earnings_stk['cum_down_count']
    rpt_earnings_stk['adjustment_ratio'] = rpt_earnings_stk['net_adjustment'] / rpt_earnings_stk['cum_total_count'].replace(0, np.nan)

    def moving_average(series, window):
        return series.rolling(window=window, min_periods=21).mean()
    rpt_earnings_stk['moving_avg_adjustment_ratio'] = rpt_earnings_stk.groupby('securityid')['adjustment_ratio'].transform(
        lambda x: moving_average(x, 63)
    )
    rpt_earnings_stk = rpt_earnings_stk[['entrytime', 'securityid', 'moving_avg_adjustment_ratio']].groupby(['entrytime', 'securityid']).tail(1)
    rpt_earnings_stk.sort_values(by=['entrytime', 'securityid'], inplace=True)
    return rpt_earnings_stk
    
def fit_group(group, valuename):
    factor = group[valuename].values
    neg = group['mkt'].values
    standardized = calculator.fit(factor, neg, 5)
    group[valuename] = standardized
    return group

def process_df1(df_demo, result_con_forecast, result_rpt_forecast, rpt_earnings_stk):
    result_con_forecast['con_ep'] = 1 / result_con_forecast['con_pe']
    def prepare_demo_pivot(df):
        df_pivot = pd.pivot_table(df, index='entrytime', columns='securityid', values='mkt', fill_value=None)
        df_pivot.reset_index(inplace=True)
        df_pivot.rename(columns={'index': 'date'}, inplace=True)
        return df_pivot
    
    def merge_and_update(demo_pivot, result_df, value_col, chunk_size=30):
        num_chunks = (len(demo_pivot) + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            date_chunk = demo_pivot.iloc[i * chunk_size:(i + 1) * chunk_size]
            all_dates_stocks = date_chunk.assign(key=1).merge(
                pd.DataFrame({'securityid': demo_pivot.columns[1:], 'key': 1}),
                on='key').drop('key', axis=1)
            all_dates_stocks['entrytime'] = pd.to_datetime(all_dates_stocks['entrytime'])
            merged = pd.merge_asof(
                all_dates_stocks.sort_values('entrytime'),
                result_df,
                left_on='entrytime',
                right_on='entrytime',
                by='securityid',
                direction='backward'
            )
            pivot_result = merged.pivot(index='entrytime', columns='securityid', values=value_col)
            demo_pivot.update(pivot_result)
        demo_pivot.index = demo_pivot['entrytime']
        demo_pivot.drop('entrytime', axis=1, inplace=True)
        demo_pivot_reset = demo_pivot.reset_index().melt(id_vars=['entrytime'], var_name='securityid', value_name=value_col)
        return demo_pivot_reset

    def process_and_merge(df_demo, result_df, value_col):
        df_demo_sorted = df_demo.sort_values(by=['entrytime', 'securityid'])
        demo_pivot = prepare_demo_pivot(df_demo)
        demo_pivot_reset = merge_and_update(demo_pivot, result_df, value_col)
        merged_df = pd.merge(df_demo_sorted, demo_pivot_reset, how='left', on=['entrytime', 'securityid'])
        merged_df.index = merged_df['entrytime']
        merged_df.drop(columns=['entrytime'], inplace=True)
        merged_df.index.name = 'entrytime'
        return merged_df
    df_demo['securityid'] = df_demo['securityid'].astype(str).str.zfill(6)
    df_demo.rename(columns={'date': 'entrytime'}, inplace=True)

    merged_df_1 = process_and_merge(df_demo, result_con_forecast, 'con_ep')
    merged_df_2 = process_and_merge(df_demo, result_rpt_forecast, 'forecast_dps')
    merged_df_3 = process_and_merge(df_demo, result_con_forecast, 'con_pe_change')
    merged_df_4 = process_and_merge(df_demo, result_con_forecast, 'con_eps_change')
    merged_df_5 = process_and_merge(df_demo, rpt_earnings_stk, 'moving_avg_adjustment_ratio')

    df2_extracted = merged_df_2[['forecast_dps']]
    df3_extracted = merged_df_3[['con_pe_change']]
    df4_extracted = merged_df_4[['con_eps_change']]
    df5_extracted = merged_df_5[['moving_avg_adjustment_ratio']]
    merged_df = pd.concat([merged_df_1, df2_extracted, df3_extracted, df4_extracted, df5_extracted], axis=1)

    merged_df['EARNYILD'] = 1 / 4 * (merged_df['con_ep'] + merged_df['EP'] + merged_df['1/PCF'] + merged_df['1/EBITDA'])
    merged_df['DIVYILD'] = 1 / 2 * (merged_df['forecast_dps'] / merged_df['close'] + merged_df['DPS'])
    merged_df['AnalystSentiment'] = 1 / 3 * (merged_df['moving_avg_adjustment_ratio'] + merged_df['con_pe_change'] + merged_df['con_eps_change'])
    
    merged_df = merged_df.groupby('entrytime').apply(fit_group, 'EARNYILD')
    merged_df = merged_df.groupby('entrytime').apply(fit_group, 'DIVYILD')
    merged_df = merged_df.groupby('entrytime').apply(fit_group, 'AnalystSentiment')

    merged_df.drop(columns=['con_ep', 'EP', '1/PCF', '1/EBITDA', 'forecast_dps', 'close', 'DPS', 'moving_avg_adjustment_ratio', 'con_pe_change', 'con_eps_change'], inplace=True)
    return merged_df

def process_df3(df_demo, result_con_forecast):

    def prepare_demo_pivot(df, value_col):
        df_pivot = pd.pivot_table(df, index='entrytime', columns='securityid', values=value_col, fill_value=None)
        df_pivot.reset_index(inplace=True)
        df_pivot.rename(columns={'index': 'date'}, inplace=True)
        return df_pivot

    def merge_and_update(demo_pivot, result_df, value_col, chunk_size=30):
        num_chunks = (len(demo_pivot) + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            date_chunk = demo_pivot.iloc[i * chunk_size:(i + 1) * chunk_size]
            all_dates_stocks = date_chunk.assign(key=1).merge(
                pd.DataFrame({'securityid': demo_pivot.columns[1:], 'key': 1}),
                on='key').drop('key', axis=1)
            all_dates_stocks['entrytime'] = pd.to_datetime(all_dates_stocks['entrytime'])
            merged = pd.merge_asof(
                all_dates_stocks.sort_values('entrytime'),
                result_df,
                left_on='entrytime',
                right_on='entrytime',
                by='securityid',
                direction='backward'
            )
            pivot_result = merged.pivot(index='entrytime', columns='securityid', values=value_col)
            demo_pivot.update(pivot_result)
        demo_pivot.index = demo_pivot['entrytime']
        demo_pivot.drop('entrytime', axis=1, inplace=True)
        demo_pivot_reset = demo_pivot.reset_index().melt(id_vars=['entrytime'], var_name='securityid', value_name=value_col)
        return demo_pivot_reset

    def process_and_merge(df_demo, result_df, value_col):
        df_demo_sorted = df_demo.sort_values(by=['entrytime', 'securityid'])
        demo_pivot = prepare_demo_pivot(df_demo, 'LTREVRSL')
        demo_pivot_reset = merge_and_update(demo_pivot, result_df, value_col)
        merged_df = pd.merge(df_demo_sorted, demo_pivot_reset, how='left', on=['entrytime', 'securityid'])
        merged_df.index = merged_df['entrytime']
        merged_df.drop(columns=['entrytime'], inplace=True)
        merged_df.index.name = 'entrytime'
        return merged_df

    df_demo['securityid'] = df_demo['securityid'].astype(str).str.zfill(6)
    df_demo.rename(columns={'date': 'entrytime'}, inplace=True)

    merged_df_1 = process_and_merge(df_demo, result_con_forecast, 'con_npcgrate')
    merged_df_2 = process_and_merge(df_demo, result_con_forecast, 'con_eps_std')

    df2_extracted = merged_df_2[['con_eps_std']]
    merged_df = pd.concat([merged_df_1, df2_extracted], axis=1)

    merged_df['GROWTH'] = 1 / 3 * (merged_df['con_npcgrate'] + merged_df['beta_EPS'] + merged_df['beta_RPS'])
    merged_df['EARNVAR'] = 1 / 4 * (merged_df['revenue_varate'] + merged_df['n_income_varate'] + merged_df['nChangeInCash_varate'] + merged_df['con_eps_std'] / merged_df['close'])
    
    merged_df = merged_df.groupby('entrytime').apply(fit_group, 'GROWTH')
    merged_df = merged_df.groupby('entrytime').apply(fit_group, 'EARNVAR')

    merged_df.drop(columns=['con_npcgrate', 'beta_EPS', 'beta_RPS', 'revenue_varate', 'n_income_varate', 'nChangeInCash_varate', 'con_eps_std', 'close'], inplace=True)

    return merged_df

def get_pivot_factors(df1_temp, df2_temp, df3_temp, date):
    SIZE = pd.pivot(df1_temp, columns='date', index='securityid', values='SIZE')
    MIDCAP = pd.pivot(df1_temp, columns='date', index='securityid', values='MIDCAP')
    
    hs300_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='hs300_Relative_NLSize')
    csi500_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='csi500_Relative_NLSize')
    csi1000_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='csi1000_Relative_NLSize')
    
    left_hs300_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='left_hs300_Relative_NLSize')
    left_csi500_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='left_csi500_Relative_NLSize')
    left_csi1000_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='left_csi1000_Relative_NLSize')
    
    right_hs300_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='right_hs300_Relative_NLSize')
    right_csi500_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='right_csi500_Relative_NLSize')
    right_csi1000_Relative_NLSize = pd.pivot(df1_temp, columns='date', index='securityid', values='right_csi1000_Relative_NLSize')    
    
    BETA = pd.pivot(df1_temp, columns='date', index='securityid', values='BETA')
    RESVOL = pd.pivot(df1_temp, columns='date', index='securityid', values='RESVOL')
    EARNYILD = pd.pivot(df1_temp, columns='date', index='securityid', values='EARNYILD')
    
    BTOP = pd.pivot(df1_temp, columns='date', index='securityid', values='BTOP')
    LEVERAGE = pd.pivot(df1_temp, columns='date', index='securityid', values='LEVERAGE')
    LIQUIDTY = pd.pivot(df1_temp, columns='date', index='securityid', values='LIQUIDTY')
    DIVYILD = pd.pivot(df1_temp, columns='date', index='securityid', values='DIVYILD')
    PROFIT = pd.pivot(df1_temp, columns='date', index='securityid', values='PROFIT')
    MOMENTUM = pd.pivot(df2_temp, columns='date', index='securityid', values='MOMENTUM')
    EARNQLTY = pd.pivot(df2_temp, columns='date', index='securityid', values='EARNQLTY')
    LTREVRSL = pd.pivot(df3_temp, columns='date', index='securityid', values='LTREVRSL')
    GROWTH = pd.pivot(df3_temp, columns='date', index='securityid', values='GROWTH')
    EARNVAR = pd.pivot(df3_temp, columns='date', index='securityid', values='EARNVAR')
    INVSQLTY = pd.pivot(df3_temp, columns='date', index='securityid', values='INVSQLTY')
    AnalystSentiment = pd.pivot(df1_temp, columns='date', index='securityid', values='AnalystSentiment')
    Seasonality = pd.pivot(df3_temp, columns='date', index='securityid', values='Seasonality')
    ShortTermReversal = pd.pivot(df1_temp, columns='date', index='securityid', values='ShortTermReversal')
    
    INDMOM = pd.pivot(df1_temp, columns='date', index='securityid', values='INDMOM')
    
    INDUSTRY = GetDateIndustryData(date)
    
    return SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize, left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize,\
    right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize, BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, INDMOM,\
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY
    
