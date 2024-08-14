from calcTools import *
from datetime import datetime
import pandas as pd
import os
import sys,os

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path) # add current terminal path to sys.path

from functools import reduce
import config
# from predictRisk import *
import globalVariables
from getFactorExpo import *

def add_market(secucode):
    secucode_str = f"{int(secucode):06d}"
    if secucode_str[0] == '3' or secucode_str[0] == '0':
        secucode_str += '.SZ'
    elif secucode_str[0] == '6':
        secucode_str += '.SH'
    return secucode_str

def ProceeedOneDayFactorDf(date, SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize,\
    left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize, right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize,\
    BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, \
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY):
    print("start data!")
    BETA.columns = ['BETA']
    SIZE.columns = ['SIZE']
    MIDCAP.columns = ['MIDCAP']
    hs300_Relative_NLSize.columns = ['hs300_Relative_NLSize']
    csi500_Relative_NLSize.columns = ['csi500_Relative_NLSize']
    csi1000_Relative_NLSize.columns = ['csi1000_Relative_NLSize']
    left_hs300_Relative_NLSize.columns = ['left_hs300_Relative_NLSize']
    left_csi500_Relative_NLSize.columns = ['left_csi500_Relative_NLSize']
    left_csi1000_Relative_NLSize.columns = ['left_csi1000_Relative_NLSize']
    right_hs300_Relative_NLSize.columns = ['right_hs300_Relative_NLSize']
    right_csi500_Relative_NLSize.columns = ['right_csi500_Relative_NLSize']
    right_csi1000_Relative_NLSize.columns = ['right_csi1000_Relative_NLSize']
    RESVOL.columns = ['RESVOL']
    EARNYILD.columns = ['EARNYILD']
    GROWTH.columns = ['Growth']
    BTOP.columns = ['BTOP']
    LEVERAGE.columns = ['LEVERAGE']
    LIQUIDTY.columns = ['LIQUIDTY']
    DIVYILD.columns = ['DIVYILD']
    PROFIT.columns = ['PROFIT']
    MOMENTUM.columns = ['MOMENTUM']
    LTREVRSL.columns = ['LTREVRSL']
    EARNVAR.columns = ['EARNVAR']
    INVSQLTY.columns = ['INVSQLTY']
    EARNQLTY.columns = ['EARNQLTY']
    AnalystSentiment.columns= ['AnalystSentiment']
    Seasonality.columns = ['Seasonality']
    ShortTermReversal.columns = ['ShortTermReversal']
    
    for df in [BETA, SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize,\
        left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize, right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize,\
        EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, LTREVRSL, \
        EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal]:
        df.index = df.index.map(add_market)
    data = pd.concat([BETA, SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize,\
        left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize, right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize,\
        EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, LTREVRSL, \
        EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal], axis=1, join='inner')
    
    # TODO 暂做均值填充
    for column in list(data.columns[data.isnull().sum() > 0]):  # 只有因子值均值填充
        mean_val = data[column].mean()
        data[column].fillna(mean_val, inplace=True)
    data.fillna(data.mean(), inplace=True)
    print('datahere!')
    print(data)
    
    # data = data.dropna()
    print("data finish!")
    Industry0 = IndustryID2Dummy(INDUSTRY)
    Industry0.index.name = 'securityid'
    print(Industry0)
    res = pd.concat([Industry0, data], axis=1, join='inner')
    # res = res.dropna()
    return res

def BarraModel(previous_date, date1, stockreturn, SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize,\
    left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize, right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize,\
    BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, \
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY, NegotiableMV):
    
    factorvalue = ProceeedOneDayFactorDf(previous_date, SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize,\
    left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize, right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize,\
    BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, \
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY)
    
    print("getfactorvalue")
    NegotiableMV = NegotiableMV.rename(columns=add_market)
    marketvalue = np.sqrt(NegotiableMV.loc[previous_date, NegotiableMV.columns.intersection(factorvalue.index)])
    marketvalue.name = 'CurrentValue'
    marketvalue = pd.DataFrame(marketvalue)

    stockreturn = stockreturn.T.rename(columns=add_market)
    r1 = stockreturn.loc[date1, stockreturn.columns.intersection(factorvalue.index)]  # 股票日度收益 stockreturn
    r1.name = 'daily return'
    r1 = pd.DataFrame(r1)
    AllData = pd.concat([factorvalue, marketvalue, r1], axis=1, join='inner')
    AllData = sm.add_constant(AllData)
    X = AllData.iloc[:,0:AllData.columns.size-2]  # 国家，行业，风格因子的因子暴露数据汇总
    daily_return = pd.DataFrame(AllData.iloc[:,AllData.columns.size-1])  # 股票日收益
    returns=[]  # 股票日收益变成list形式
    for i in range(len(daily_return)):
        returns.append(float(daily_return.iloc[i,0]))
    # 回归权重矩阵
    marketvalue = AllData[marketvalue.columns]
    totalmarketvalue=marketvalue.sum()
    marketvaluepercent=marketvalue/totalmarketvalue
    V = np.multiply(np.eye(len(marketvaluepercent)), np.array(marketvaluepercent))
    #约束矩阵
    industrynum=globalVariables.INDUSTRYNUM
    x = np.array(np.square(marketvalue))
    x.shape = (len(x), 1)
    marketvalue2=np.multiply(np.array(X.iloc[:,1:industrynum+1]),x)#行业值乘以市值
    marketvaluesum1=sum(marketvalue2)#每个行业内相加
    marketvaluesumall = sum(marketvaluesum1)#返回所有行业总市值
    marketvaluepercent2=marketvaluesum1/marketvaluesumall#返回各个行业市值占比
    a1=np.eye(industrynum)
    a2=-marketvaluepercent2[0:industrynum-1] / marketvaluepercent2[industrynum-1]
    aa=np.vstack((a1,np.hstack((0,a2))))
    a=np.vstack((aa,np.zeros((X.columns.size - industrynum-1,industrynum))))
    a3=np.vstack((np.zeros((industrynum+1, X.columns.size - industrynum-1)),np.eye(X.columns.size-industrynum-1)))
    R=np.hstack((a,a3))
    #股票权重矩阵
    X=np.array(X)
    R = R.astype(np.float64)
    X = X.astype(np.float64)
    V = V.astype(np.float64)
    P=np.dot(np.dot(np.dot(np.dot(R,np.mat(np.dot(np.dot(np.dot(np.dot(R.T,X.T),V),X),R)).I),R.T),X.T),V)#直接引用带权重，带约束条件的最小二乘回归求解，得到纯因子组合的股票权重矩阵。
    # 因子收益
    FactorReturn=pd.DataFrame(np.dot(P,np.mat(returns).T),columns=[date1],index=AllData.columns[:-2])#因子收益
    print('FactorReturn:',FactorReturn)
    # 异质收益
    SpecificReturn = pd.DataFrame(np.mat(returns).T-np.dot(X,FactorReturn),columns=['SpecificReturn'],index = AllData.index)
    # 因子暴露
    X = pd.DataFrame(X,columns=AllData.columns[:-2],index = AllData.index)
    residstemp=AllData.loc[:,['CurrentValue']]# 期初市值
    return X,FactorReturn,SpecificReturn,residstemp # 期初因子暴露（上一期）因子收益，特质收益


def getBarraRes(previous_date, date, df1_temp, df2_temp, df3_temp, neg_mkt_df_pivot, ChangePCT):

    SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize,\
    left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize, right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize, \
    BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, \
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY = get_pivot_factors(df1_temp, df2_temp, df3_temp, previous_date)
    
    print('barra model getting in')

    X, FactorReturn, SpecificReturn, residstemp = BarraModel(previous_date, date, ChangePCT, SIZE, MIDCAP, hs300_Relative_NLSize, csi500_Relative_NLSize, csi1000_Relative_NLSize,\
    left_hs300_Relative_NLSize, left_csi500_Relative_NLSize, left_csi1000_Relative_NLSize, right_hs300_Relative_NLSize, right_csi500_Relative_NLSize, right_csi1000_Relative_NLSize,\
    BETA, RESVOL, EARNYILD, GROWTH, BTOP, LEVERAGE, LIQUIDTY, DIVYILD, PROFIT, MOMENTUM, \
    LTREVRSL, EARNVAR, INVSQLTY, EARNQLTY, AnalystSentiment, Seasonality, ShortTermReversal, INDUSTRY, neg_mkt_df_pivot)
    
    return X,FactorReturn,SpecificReturn,residstemp


def write_history_factor_return(start_date, end_date):
    
    date_series = get_tradedate_calendar(start_date, end_date)
    factor_return = pd.DataFrame()
    specific_return = []
    
    df1 = get_one_day_factors_CNE6('daily/short_time_rolling_factors', start_date, end_date)
    df2 = get_one_day_factors_CNE6('daily/mid_time_rolling_factors', start_date, end_date)
    df3 = get_one_day_factors_CNE6('daily/long_time_rolling_factors', start_date, end_date)
    
    result_con_forecast = process_con()
    result_rpt_forecast = process_rpt()
    rpt_earnings_stk = process_earnings()
    df1 = process_df1(df1, result_con_forecast, result_rpt_forecast, rpt_earnings_stk)
    df3 = process_df3(df3, result_con_forecast)
    print(df1, df3)
    df1.index = df1.index.set_names(['entrytime', 'temp1', 'temp2', 'temp3'])
    df3.index = df3.index.set_names(['entrytime', 'temp1', 'temp2'])
    df1 = df1.reset_index()
    df3 = df3.reset_index()
    df1.rename(columns={'entrytime':'date'},inplace=True)
    df3.rename(columns={'entrytime':'date'},inplace=True)
    df1.to_parquet('saved_files/df1.parquet')
    df2.to_parquet('saved_files/df2.parquet')
    df3.to_parquet('saved_files/df3.parquet')
    neg_mkt_df_pivot = df1.pivot_table('mkt', columns='securityid', index='date')
    ChangePCT = df1.pivot_table('ChangePCT', columns='date', index='securityid')
    for date in date_series:
        try:
            date_str = datetime.strftime(date,'%Y-%m-%d')
            previous_date = datetime.strftime(get_last_tradedate(date_str),'%Y-%m-%d')  # 小于date1的最近交易日期
            print(date_str, previous_date, "this time!")  # 当天的收益率和前一天的因子暴露回归得到当天的因子收益率
            
            df1_temp = df1[df1['date']==previous_date]
            df2_temp = df2[df2['date']==previous_date]
            df3_temp = df3[df3['date']==previous_date]


            X, FactorReturn, SpecificReturn, residstemp = getBarraRes(previous_date, date, df1_temp, df2_temp, df3_temp, neg_mkt_df_pivot, ChangePCT)
            
            tmp = FactorReturn.T
            tmp.index = [date_str]
            factor_return = pd.concat([factor_return, tmp])
            print(factor_return)
            print('SpecificReturn',SpecificReturn.shape)
            SpecificReturn.rename({'SpecificReturn':date_str},axis=1,inplace=True)
            SpecificReturn['securityid'] = SpecificReturn.index
            SpecificReturn.reset_index(drop=True,inplace=True)
            specific_return.append(SpecificReturn)
            
        except Exception as e:
            print(date,'error!',e)
    print('factor_return:',factor_return)
    factor_return.reset_index(drop=True)
    specific_R = reduce(lambda left, right: pd.merge(left, right, on=['securityid'],how='outer'), specific_return)  # inner会导致股票数量变少出错
    factor_return.to_hdf('saved_files/factor_return.hdf', key='factor_return')
    specific_R.to_hdf('saved_files/specific_R.hdf',key='specific_return')
    return 0

def update_factor_return(end_date):

    # existed_f_return = pd.read_hdf(config.factor_return_saved_path+'Barra_factor_returns_CNE6.h5') # TODO 存储路径待指定
    # existed_sp_return = pd.read_hdf(config.factor_return_saved_path+'Barra_specific_returns_CNE6.h5')
    existed_f_return = pd.read_hdf('saved_files/factor_return.hdf')
    existed_sp_return = pd.read_hdf('saved_files/specific_R.hdf')
    
    start_date = existed_f_return.index[-1]
    date_series = get_tradedate_calendar(start_date,end_date)  # 不包括最第一天，包括最后一天
    factor_return = pd.DataFrame()
    specific_return = []

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
        df1 = df1.reset_index()
        df3 = df3.reset_index()
        df1.rename(columns={'entrytime':'date'},inplace=True)
        df3.rename(columns={'entrytime':'date'},inplace=True)
        neg_mkt_df_pivot = df1.pivot_table('mkt', columns='securityid', index='date')
        ChangePCT = df1.pivot_table('ChangePCT', columns='date', index='securityid')
        for date in date_series:
            try:
                df1_temp = df1[df1['date']==previous_date]
                df2_temp = df2[df2['date']==previous_date]
                df3_temp = df3[df3['date']==previous_date]
                date_str = datetime.strftime(date,'%Y-%m-%d')
                previous_date = datetime.strftime(get_last_tradedate(date_str),'%Y-%m-%d')  # 小于date1的交易日期序列
                print(date_str, previous_date)  # 当天的收益率和前一天的因子暴露回归得到当天的因子收益率

                X, FactorReturn, SpecificReturn, residstemp = getBarraRes(previous_date, date, df1_temp, df2_temp, df3_temp, neg_mkt_df_pivot, ChangePCT)
                tmp = FactorReturn.T
                tmp.index = [date_str]
                factor_return = factor_return.append(tmp)
                print('SpecificReturn',SpecificReturn.shape)
                SpecificReturn.rename({'SpecificReturn':date_str},axis=1,inplace=True)
                SpecificReturn['SECUCODE'] = SpecificReturn.index
                SpecificReturn.reset_index(drop=True,inplace=True)
                specific_return.append(SpecificReturn)
            except Exception as e:
                print(date,'error!',e)

        factor_return.reset_index(drop=True)
        specific_R = reduce(lambda left, right: pd.merge(left, right, on=['SECUCODE'],how='inner'), specific_return)
        new_factor_return = existed_f_return.append(factor_return)
        new_specific_return = pd.merge(existed_sp_return,specific_R,on='SECUCODE',how='inner')

        new_factor_return.to_hdf('saved_files/factor_return.hdf', key='factor_return')
        new_specific_return.to_hdf('saved_files/specific_R.hdf',key='specific_return')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    # 今天的收盘后股票收益率以及昨天的因子暴露计算得到今天的因子收益率
    start_date = '2024-07-10'  # todo:测试用，生成更长的历史数据
    end_date = '2024-07-15'
    write_history_factor_return(start_date,end_date)
    # update_factor_return('2021-10-25')
