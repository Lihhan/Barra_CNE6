import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import statsmodels.api as sm
from os.path import expanduser
from calcFunc import *
from calcTools import *
from adjustments import *
from getFactorExpo import *
from writeFactorreturns import ProceeedOneDayFactorDf

calculator = MYCALFUNC()

def predictRisk(date, f, specific_R, df1_temp, df2_temp, df3_temp, neg_mkt_df_pivot, h=252, min_periods=40):
    # 在date收盘后预测下一期的风险
    date_str = datetime.strftime(pd.to_datetime(date),'%Y-%m-%d')
    
    SIZE, MIDCAP, BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, \
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY, ChangePCT = get_pivot_factors(df1_temp, df2_temp, df3_temp, date)

    print('factor values get in')

    location = list(f.index).index(date_str)
    factorreturn = f.iloc[max(0,location-h+1):location+1]  # max 返回[],注意这里每期的因子收益率该期收盘时该期实现的因子收益率
    nw_adjusted_cov,T = NW_adjustment(factorreturn)
    eigen_adjusted_cov = EigenfactorRiskAdjustment(nw_adjusted_cov,T)
    factor_return = f.iloc[:location+1]
    F = VolatilityRegimeAdjustment(eigen_adjusted_cov,factor_return,l=120,h=min(h,len(factor_return))) # h must align with the param h
    location2 = list(specific_R.index).index(date_str)
    sR = specific_R.iloc[max(0,location2-h+1):location2+1]  # 注意这里每期的特质收益率该期收盘时该期实现的特质收益率
    stocks = retain(sR,min_periods)

    # 当期的因子暴露,收盘后能得到该因子暴露数据

    X = ProceeedOneDayFactorDf(date, SIZE, MIDCAP, BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, \
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY)
    
    
    stocks = sorted(list(set(stocks).intersection(set(X.index))))
    X = X.loc[stocks]
    X.insert(0,'const',1)
    current = neg_mkt_df_pivot.loc[[datetime.datetime.strptime(date_str,'%Y-%m-%d')]].T # 当期流通市值
    current = current.loc[stocks]
    current.columns = ['CurrentValue']
    current = current**0.5
    V = current['CurrentValue']/current['CurrentValue'].sum()
    V = np.diag(V)

    sR = sR[stocks]
    sR = imputation(sR)
    sr_nw_cov,T = NW_adjustment(sR)
    sigma_u = StructuralModelAdjustment(sR,sr_nw_cov,X,V)
    sigma_sh = BayesianShrinkage(sigma_u,current['CurrentValue']**2)
    sigma_SH = sigma_sh[['SpecificVolatility']]
    specific_return = specific_R.iloc[:location2+1]
    specific_return = specific_return[sigma_sh.index]
    market_value = neg_mkt_df_pivot  #.iloc[:list(NegotiableMV.index).index(datetime.datetime.strptime(date_str,'%Y-%m-%d'))+1]
    market_value.index = [date_str]
    market_value = market_value[sigma_sh.index]
    sigma_VAR = Specific_Volatility_Regime_Adjustment(sigma_SH, specific_return, market_value, l=120, h=min(h,len(specific_return)))

    # 股票收益率的协方差矩阵
    delta = np.diag(sigma_VAR['SpecificVolatility'])
    sigma = np.dot(np.dot(X, F), X.T) + delta
    sigma = pd.DataFrame(sigma, columns=stocks, index=stocks)
    F = pd.DataFrame(F, columns=X.columns, index=X.columns)
    return F, sigma_VAR, sigma, X

def write_history_risk(factor_return, specific_return, start_date, end_date, h=120):
    date_series = get_tradedate_calendar(start_date, end_date)

    df1 = get_one_day_factors_CNE6('daily/short_time_rolling_factors', start_date, end_date)
    df2 = get_one_day_factors_CNE6('daily/mid_time_rolling_factors', start_date, end_date)
    df3 = get_one_day_factors_CNE6('daily/long_time_rolling_factors', start_date, end_date)
    
    result_con_forecast = process_con()
    result_rpt_forecast = process_rpt()
    rpt_earnings_stk =  process_earnings()
    df1 = process_df1(df1, result_con_forecast, result_rpt_forecast, rpt_earnings_stk)
    df3 = process_df3(df3, result_con_forecast)
    neg_mkt_df_pivot = df1.pivot_table('mkt', columns='securityid', index='date')

    specific_return.index = specific_return['securityid']
    specific_return.drop(['securityid'], axis=1, inplace=True)  # change format for calculation
    specific_return = specific_return.T

    risk_all_store = dict()
    iter = 0
    for date in date_series:
        risk_data = dict()
        print(date)
        df1_temp = df1[df1['date']==date]
        df2_temp = df2[df2['date']==date]
        df3_temp = df3[df3['date']==date]

        F, sigma_VAR, _, X = predictRisk(date, factor_return, specific_return, df1_temp, df2_temp, df3_temp, neg_mkt_df_pivot, h=h, min_periods=40)

        risk_data['factor_covar'] = F
        risk_data['specific_volatility'] = sigma_VAR
        risk_data['factor_expos'] = X

        print('expos shape', X.shape)
        risk_all_store[date] = risk_data
        iter += 1
        if iter % 250 == 0:
            with open(config.factor_return_saved_path+'factor_risk_all2.pk', 'wb') as f:
                pickle.dump(risk_all_store, f)  # 防止出错时应急使用


def update_risk(factor_return, specific_return, end_date, h=120):

    with open(config.factor_return_saved_path+'factor_risk_all.pk', 'rb') as f:
        existed_factor_risk = pickle.load(f)
        
    start_date = datetime.datetime.strftime(sorted(list(existed_factor_risk.keys()))[-1],'%Y-%m-%d')
    date_series = get_tradedate_calendar(start_date, end_date)

    specific_return.index = specific_return['SECUCODE']
    specific_return.drop(['SECUCODE'], axis=1, inplace=True)  # change format for calculation
    specific_return = specific_return.T

    if date_series == []:
        print('no date needs updating!')
    else:
        df1 = get_one_day_factors_CNE6('daily/short_time_rolling_factors', start_date, end_date)
        df2 = get_one_day_factors_CNE6('daily/mid_time_rolling_factors', start_date, end_date)
        df3 = get_one_day_factors_CNE6('daily/long_time_rolling_factors', start_date, end_date)
        
        result_con_forecast = process_con()
        result_rpt_forecast = process_rpt()
        rpt_earnings_stk =  process_earnings()
        df1 = process_df1(df1, result_con_forecast, result_rpt_forecast, rpt_earnings_stk)
        df3 = process_df3(df3, result_con_forecast)
        for date in date_series:
            risk_data = dict()
            print(date)
            # 数据准备  当天的因子数据
            df1_temp = df1[df1['date']==date]
            df2_temp = df2[df2['date']==date]
            df3_temp = df3[df3['date']==date]

            F, sigma_VAR, _, X = predictRisk(date, factor_return, specific_return, df1_temp, df2_temp, df3_temp, h=h, min_periods=40)

            risk_data['factor_covar'] = F
            risk_data['specific_volatility'] = sigma_VAR
            risk_data['factor_expos'] = X
            existed_factor_risk[date] = risk_data

        with open(config.factor_return_saved_path+'factor_risk_all.pk', 'wb') as f:
            pickle.dump(existed_factor_risk, f)  # 防止出错时应急使用


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    # end_date = "2021-10-25"
    #  读取相关数据
    # factor_return = pd.read_hdf(config.factor_return_saved_path+'Barra_factor_returns_CNE6.h5', key='factor_return')  # time series 计算协方差必须大于因子数目，否则矩阵奇异
    # specific_return = pd.read_hdf(config.factor_return_saved_path+'Barra_specific_returns_CNE6.h5', key='specific_return')
    factor_return = pd.read_hdf('factor_return.hdf')
    specific_return = pd.read_hdf('specific_return.hdf')
    start_date = '2017-12-29'  # todo:测试用，生成更长的历史数据
    end_date = '2023-06-02'
    write_history_risk(factor_return, specific_return, start_date, end_date)
    # update_risk(factor_return, specific_return, end_date, h=120)
