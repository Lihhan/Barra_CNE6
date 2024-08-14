import math
import pandas as pd
import numpy as np
from calcFunc import *

def BayesianShrinkage(specific_volatility,marketvalue):
    """
    specific_volatility为每只股票的特异波动率，一列数组, 注意不能是含有columns的dataframe型，否则返回nan
    """
    cal_matrix=pd.DataFrame(specific_volatility,columns=['SpecificVolatility'],index = marketvalue.index)
    cal_matrix.insert(0,'SecuCode',marketvalue.index)
    cal_matrix.insert(1,'CurrentValue',marketvalue)#用来进行排序
    cal_matrix_order=cal_matrix.sort_values(by='CurrentValue',axis=0,ascending=True)
    cal_matrix_order=cal_matrix_order.reset_index(drop=True)#重置索引。  市值、特异波动率
    interval=np.arange(0,len(cal_matrix_order)+1,len(cal_matrix_order)/10)#分成十档
    shrinkage_intensity=[]#收缩系数
    shrinkaged_cov=[]#贝叶斯收缩后的结果
    coeficient_q = 1 #shrinkage_intensity系数
    for j in range(1,len(interval)):
        value_all=cal_matrix_order['CurrentValue'][int(interval[j - 1]):int(interval[j])].sum()
        weights=cal_matrix_order['CurrentValue'][int(interval[j - 1]):int(interval[j])]/value_all
        sigmal_ave=(cal_matrix_order['SpecificVolatility'][int(interval[j-1]):int(interval[j])] *weights).sum()#市值加权平均
        denominator1=(((cal_matrix_order['SpecificVolatility'][int(interval[j - 1]):int(interval[j])]-sigmal_ave)**2).sum()/len(weights))**(1/2)
        denominator2=coeficient_q*abs(cal_matrix_order['SpecificVolatility'][int(interval[j-1]):int(interval[j])]-sigmal_ave)
        ShrinkageIntensity = denominator2/(denominator2+denominator1)
        shrinkage_intensity.extend(ShrinkageIntensity)  # 收缩系数
        shrinkaged_cov.extend(ShrinkageIntensity*sigmal_ave+(1-ShrinkageIntensity)*(cal_matrix_order['SpecificVolatility'][int(interval[j-1]):int(interval[j])]))
    final_matrix_shrinkaged=cal_matrix_order.loc[:,['SecuCode','CurrentValue']]
    final_matrix_shrinkaged.insert(1,'SpecificVolatility',shrinkaged_cov)
    final_matrix_shrinkaged = final_matrix_shrinkaged.set_index('SecuCode',drop=True)
    final_matrix_shrinkaged = final_matrix_shrinkaged.sort_index()
    return final_matrix_shrinkaged


def Newey_West_adjustment(factorreturn,delays):
    """
    factorreturn：为字典，keys为日期，values[0]为各因子的因子收益率，
    delays:滞后阶数  自相关阶数
    """
    factorreturn_new=pd.DataFrame()
    time_series=list(factorreturn.keys())
    for j in range(len(factorreturn.keys())):
        factorreturn_new.insert(j,time_series[j],factorreturn[time_series[j]][0])#多期的因子收益率合成
    factorreturn_new2=factorreturn_new.transpose()
    lags_set=np.arange(1,delays)#参数对应最大滞后期。取5  ,注意delays的长度不宜选的过长，否则会使矩阵不是正定！！
    autocov_group=dict()
    for i in lags_set:
        autocov_group[str(i)] = AutoCovariance_Calculation(factorreturn_new2,i)
    temp_cov=pd.DataFrame(np.zeros(autocov_group['1'].shape))#初始化
    for i in lags_set:
#        temp_cov=temp_cov+(1-i/(delays-1+1))*(pd.DataFrame(autocov_group[str(i)])+pd.DataFrame(autocov_group[str(i)]).transpose())
        #修改
        temp_cov=temp_cov+(1-i/(delays+1))*(pd.DataFrame(autocov_group[str(i)])+pd.DataFrame(autocov_group[str(i)]).transpose())
    adjusted_cov=factorreturn_new2.cov() +temp_cov#barra模型中乘以22来得到月频的协方差
    T=factorreturn_new2.shape[0]#返回期数
    return adjusted_cov,T


def NW_adjustment(factorreturn,delays=2,tao=90):
    """
    factorreturn：dataframe,index为日期，columns为因子种类，value为因子收益率
    delays:滞后阶数  自相关阶数
    """
    lags_set=np.arange(1,delays+1)#参数对应最大滞后期。取5,注意delays的长度不宜选的过长，否则会使矩阵不是正定！！
    autocov_group=dict()
    for i in lags_set:
        autocov_group[str(i)]= NW_Cal_AutoCovariance_EWMA(factorreturn,i,tao)
    temp_cov=pd.DataFrame(np.zeros(autocov_group['1'].shape))#初始化
    for i in lags_set:
#        temp_cov=temp_cov+(1-i/(delays-1+1))*(pd.DataFrame(autocov_group[str(i)])+pd.DataFrame(autocov_group[str(i)]).transpose())
        #修改
        temp_cov=temp_cov+(1-i/(delays+1))*(pd.DataFrame(autocov_group[str(i)])+pd.DataFrame(autocov_group[str(i)]).transpose())
    adjusted_cov=factorreturn.cov().values +temp_cov#barra模型中乘以22来得到月频的协方差
    T=factorreturn.shape[0]#返回期数
    adjusted_cov.columns = factorreturn.columns
    adjusted_cov.index = factorreturn.columns
    return  adjusted_cov,T


def Specific_Volatility_Regime_Adjustment(sigma_SH,specific_return,market_value,l=252,h=252,tao=42):
    """
    对Δ做偏误调整
    specific_return: dataframe,index为日期，columns为股票代码，value为特质收益，长度应大于h+l,不含缺失值
    market_value: dataframe,index为日期，columns为股票代码,长度与specific_return相同，或至少大于等于h
    l:估计波动率用的样本长度
    h:波动率乘数样本长度
    tao：波动率乘数半衰期
    sigma_SH 为经过贝叶斯压缩后的特异风险
    返回经过偏误调整后的波动率
    """
    # 用之前252天的收益率预测该日的标准差,
    sigmatmp = specific_return.iloc[-(h+l):].rolling(l,min_periods=int(l/10)).std()
    sigma = sigmatmp.shift(1).iloc[-h:]

    Bias = specific_return.iloc[-h:]**2/sigma**2
    marketvalue1 = market_value.iloc[-h:]*pd.notnull(Bias)
    marketvalue1 = marketvalue1.replace(0,np.nan)
    w = marketvalue1.div(marketvalue1.sum(axis=1),axis='rows')#每行/每行和，获取股票每期的权重

    B_F_square = (Bias*w).sum(axis=1)

    w_lambda = 0.5**(1/tao)
    t = list(B_F_square.index).index(B_F_square.index[-1])
    weight = [w_lambda**(t-s) for s in range(t-h+1,t+1)]
    weight /= np.sum(weight)

    lambda_F = np.dot(B_F_square,weight)**0.5
    sigma_VAR = lambda_F*sigma_SH

    return sigma_VAR


def factor_volatility_multiplier(factor_return,l=60,h=252,tao=42):
    """
    factor_return: dataframe,index为日期，columns为因子种类，value为因子收益率
    l:估计波动率用的样本长度
    h:波动率乘数样本长度
    tao：波动率乘数半衰期

    """
    # 用之前60天的因子收益率预测该日的该因子标准差,
    sigmatmp = factor_return.iloc[-(h+l):].rolling(l).std().dropna()
    sigma = sigmatmp.shift(1).dropna()

    Bias = factor_return.iloc[-h:]**2/sigma**2
    B_F_square = Bias.sum(axis=1)/len(factor_return.columns)

    w_lambda = 0.5**(1/tao)
    t = list(B_F_square.index).index(B_F_square.index[-1])
    weight = [w_lambda**(t-s) for s in range(t-h+1,t+1)]
    weight /= np.sum(weight)

    lambda_F = np.dot(B_F_square,weight)**0.5

    return lambda_F


def VolatilityRegimeAdjustment(factor_cov,factor_return,l=252,h=252,tao=42):
    """
    factor_return: dataframe,index为日期，columns为因子种类，value为因子收益率
    l:估计波动率用的样本长度
    h:波动率乘数样本长度
    tao：波动率乘数半衰期

    """
    # 用之前252天的因子收益率预测该日的该因子标准差
    sigmatmp = factor_return.iloc[-(h+l):].rolling(l).std().dropna()
    sigma = sigmatmp.shift(1).dropna()

    Bias = factor_return.iloc[-h:]**2/sigma**2
    B_F_square = Bias.sum(axis=1)/len(factor_return.columns)

    w_lambda = 0.5**(1/tao)
    t = list(B_F_square.index).index(B_F_square.index[-1])
    weight = [w_lambda**(t-s) for s in range(t-h+1,t+1)]
    weight /= np.sum(weight)

    lambda_F = np.dot(B_F_square,weight)**0.5
    adjusted_cov = lambda_F**2*factor_cov
    return adjusted_cov


def eigenfactor_risk_adjustment(factcov,T,MC_numbers):#蒙特卡洛模拟次数，至少大于10000,输入先进行newey调整
    """
    针对因子协方差矩阵的eigenfactor risk调整
    (模拟风险偏差λ未调整)

    """
    D_zero,U_zero=np.linalg.eig(factcov)#每列对应特征向量
    iterr=0
    proportion=np.zeros((1,len(factcov)))
    while iterr<MC_numbers:
        simulated_b=np.zeros((len(factcov),T))
        for i in range(len(D_zero)):
            simulated_b[i,:]=np.random.normal(0,math.sqrt(D_zero[i]),T)#模拟,特征值是方差，这里是标准差
        esitimated_cov=(pd.DataFrame(np.dot(U_zero,simulated_b)).transpose()).cov()
        D_monte,U_monte=np.linalg.eig(esitimated_cov)
        D_monte_true=np.dot(np.dot(U_monte.T,factcov),U_monte)# the true FCM of the simulated eigenfactors, 注意此时的D并不是对角矩阵哦
        proportion=proportion+np.diag(D_monte_true)/D_monte#取出对角元素
        iterr+=1
    eigen_adjusted_cov=np.dot(np.dot(U_zero,np.diag(np.dot(np.diag((proportion / MC_numbers).tolist()[0]),D_zero))),U_zero.T)
    return eigen_adjusted_cov #得到调整完毕得因子收益协方差矩阵


def EigenfactorRiskAdjustment(factcov,T,MC_numbers=10000,alpha=1.2):#蒙特卡洛模拟次数，至少大于10000,输入先进行newey调整
    """
    针对因子协方差矩阵的eigenfactor risk调整
    (模拟风险偏差调整，参数为α)

    """
    D_zero,U_zero=np.linalg.eig(factcov)#每列对应特征向量
    iterr=0
    proportion=np.zeros((1,len(factcov)))
    while iterr<MC_numbers:
        simulated_b=np.zeros((len(factcov),T))
        for i in range(len(D_zero)):
            simulated_b[i,:]=np.random.normal(0,math.sqrt(abs(D_zero[i])),T)#模拟,特征值是方差，这里是标准差
        esitimated_cov=(pd.DataFrame(np.dot(U_zero,simulated_b)).transpose()).cov()
        D_monte,U_monte=np.linalg.eig(esitimated_cov)
        D_monte_true=np.dot(np.dot(U_monte.T,factcov),U_monte)# the true FCM of the simulated eigenfactors, 注意此时的D并不是对角矩阵哦
        proportion=proportion+np.diag(D_monte_true)/D_monte#取出对角元素
        iterr+=1
    
    proportion_adjust = np.square(alpha*(np.sqrt(abs(proportion/MC_numbers))-1)+1)
    eigen_adjusted_cov=np.dot(np.dot(U_zero,np.diag(np.dot(np.diag(proportion_adjust.tolist()[0]),D_zero))),U_zero.T)
    return eigen_adjusted_cov #得到调整完毕得因子收益协方差矩阵


def StructuralModelAdjustment(specific_return,sigma_NW,X,V,h=252):
    """
    将特异波动率经过Newey West自相关调整后，进行结构化模型调整
    specific_return: dataframe index日期，column为股票代码，值为异质收益, 应不含缺失值
    sigma_NW： 经过Newey West自相关调整后的异质收益协方差矩阵的对角元素。对角元素为各个股票的特异波动率
    X：dataframe，index='SecuCode', columns='FactorID'当期因子暴露，因子为模型中的因子，需已经标准化。
    V: 回归权重矩阵，n*n 市值权重,市值为最新期市值
    h：样本时间长度
    注意sigma_NW、X、V的股票代码顺序应一致
    返回sigma_u，为经过结构化模型调整后的特异波动率，dataframe，index为股票代码
    """

    w = pd.DataFrame(np.diag(V),index = specific_return.columns,columns = ['weight'])
    sigma_nw = pd.DataFrame(np.diag(sigma_NW),index = specific_return.columns,columns = ['sigmaNW'])


    robust_sigma_u = 1/1.35*(specific_return.quantile(q=0.75) - specific_return.quantile(q=0.25))

    specific_std = specific_return.std()
    Z_u = abs(specific_std/robust_sigma_u-1)

    tmp = 1-Z_u
    tmp[tmp>0] = 1
    gama = min(1,max(0,(h-60)/120))*tmp

    codelist = gama[gama == 1].index #用于回归的股票代码

    regress_X = X.loc[codelist]
    regress_Y = sigma_nw.loc[codelist]
    regress_w = w.loc[codelist]
    regress_w /= regress_w['weight'].sum()

    A = np.mat(regress_X)
    W = np.diag(regress_w['weight'])
    y = np.mat(regress_Y)
    #获取各因子（WLS回归）系数
    b = np.linalg.inv(A.T*W*A)*A.T*W*y

    E0 = 1.05
    sigma_STR = E0*np.exp(np.mat(X)*b)
    sigma_STR = pd.DataFrame(sigma_STR,columns = ['sigma_STR'],index = specific_return.columns)


#    sigma_u = np.multiply(gama,sigma_nw)+np.multiply((1-gama),sigma_STR)
    sigma_u = sigma_nw.mul(gama,axis='index')['sigmaNW'] + sigma_STR.mul((1-gama),axis='index')['sigma_STR']

    return sigma_u
