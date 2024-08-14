import numpy as np
import pandas as pd

def AutoCovariance_Calculation(TimeSeries,lags):
    """
    求解一个因子收益率矩阵在给定滞后期数下的AutoCovariance matrix
    """
    i = lags
    L = TimeSeries[i::]
    LL = TimeSeries[:-i:]
    average1=L.mean(axis=0)
    average2=LL.mean(axis=0)
    demean_L=np.array(L)-np.tile(average1,(L.shape[0],1))
    demean_LL=np.array(LL)-np.tile(average2,(LL.shape[0],1))
    AutoCovariance_temp=np.dot(demean_L.T,demean_LL)
    AutoCovariance=AutoCovariance_temp/len(TimeSeries)
    return AutoCovariance


def NW_Cal_AutoCovariance(TimeSeries,lags):
    """
    求解一个因子收益率矩阵在给定滞后期数下经过NW调整的AutoCovariance matrix
    """
    i = lags
    L = TimeSeries[i::]
    if i==0:
        LL = TimeSeries
    else:
        LL = TimeSeries[:-i:]
    average1 = L.mean(axis=0)
    average2 = LL.mean(axis=0)
    demean_L = np.array(L)-np.tile(average1,(L.shape[0],1))
    demean_LL = np.array(LL)-np.tile(average2,(LL.shape[0],1))
    AutoCovariance_temp = np.dot(demean_LL.T,demean_L)
    AutoCovariance = AutoCovariance_temp/len(TimeSeries)

    return AutoCovariance


def NW_Cal_AutoCovariance_EWMA(TimeSeries,lags,tao=90):
    """
    考虑数据加权的自协方差矩阵
    TimeSeries为因子收益率矩阵，行为日期，列为因子种类，值为因子收益率,建议TimeSeries的长度为252
    lags:为滞后期数
    tao：半衰期
    """
    i = lags
    L = TimeSeries[i::]
    if i==0:
        LL = TimeSeries
    else:
        LL = TimeSeries[:-i:]
    average1=L.mean(axis=0)
    average2=LL.mean(axis=0)
    demean_L=np.array(L)-np.tile(average1,(L.shape[0],1))
    demean_LL=np.array(LL)-np.tile(average2,(LL.shape[0],1))
    
    w_lambda = 0.5**(1/tao)
    t = len(L)
    weight = [w_lambda**(t-s) for s in range(t)]
    weight /= np.sum(weight)
    weight = np.diag(weight)
    AutoCovariance = np.dot(np.dot(demean_LL.T,weight),demean_L)

    return AutoCovariance

def imputation(df):
    df = df.copy()
    length = len(df)
    loc = pd.isnull(df)
    isnull = loc.sum(axis=0)
    isnull = isnull.replace(0,np.nan)
    isnull = isnull.dropna().index

    for column in isnull:
        notnull = df[column].dropna()
#        a = random.sample(list(notnull),length-len(notnull))
#        b = df[column].fillna(random.choice(list(notnull)))
        tmp = sorted(list(set(df.index).difference(set(notnull.index))))
        df[column].loc[tmp] = np.random.choice(list(notnull), length-len(notnull))
    return df
