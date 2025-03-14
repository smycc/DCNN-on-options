from math import exp, sqrt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy.stats import norm
from pysabr import hagan_2002_lognormal_sabr as hagan2002
import matplotlib.pyplot as plt #绘图
from sklearn import preprocessing
from scipy import interpolate
from tqdm import tqdm

#根据sabr方法生成的sigma，再代入到bs中生成稀疏数据和稠密数据
def bs_option(S,K,T,r,sigma,option="call"):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    if option=='call':
        p=(S*norm.cdf(d1,0,1)-K*np.exp(-r*T)*norm.cdf(d2,0,1))
    elif option=='put':
        p=(K*np.exp(-r*T)*norm.cdf(-d2,0,1)-S*norm.cdf(-d1,0,1))
    else:
        return None
    return p

def find_volatility_put(market_price, S0, K, T, r=0.02, tol=1e-4, max_iter=100):
    sigma_min = 0.001  # 波动率范围下限调整为 0.01
    sigma_max = 1.0  # 波动率范围上限调整为 2.0
    iter_count = 0
    while iter_count<max_iter:
        sigma_mid = (sigma_min + sigma_max) / 2
        theoretical_price = bs_option(S0,K,T,r,sigma_mid,option="put")
        if abs(theoretical_price - market_price) <tol:
            return sigma_mid
        if theoretical_price < market_price:
            sigma_max = sigma_mid
        else:
            sigma_min = sigma_mid
        iter_count += 1
    return (sigma_min + sigma_max) / 2

#meshgrid
T=np.linspace(0.1,0.5,30)
M=np.linspace(0.9,1.15,30)
r=2.2938/100
alpha,beta,volvol,rho=0.2,1.0,0.6,0.8
BEFOREDATA=[]
f=1
for t in T:
    for k in M:
        s=np.exp(-r*t)
        iv = hagan2002.lognormal_vol(k + s, f + s, t, alpha, beta, rho, volvol)
        bs_call = bs_option(s, k, t, r, sigma=iv, option="call")
        bs_put = bs_option(s, k, t, r, sigma=iv, option="put")
        BEFOREDATA.append([t, k, iv, bs_call, bs_put])

BEFOREDATA=DataFrame(BEFOREDATA)
BEFOREDATA.columns=['T','M','IV','bs_call','bs_put']
BEFOREDATA.sort_values(["M","T"] , inplace=True, ascending=True)

#插值
T,M= np.meshgrid(T,M)
c0=BEFOREDATA['bs_call'].values.reshape(-1,30)
f1 = interpolate.interp2d(T, M, c0, kind='cubic')  # 由样本点生成三次样条插值
T1=np.linspace(0.1,0.5,100)
M1=np.linspace(0.9,1.15,100)
c1=f1(T1,M1)
C1=c1.reshape(-1)
#对插值求隐波
T1,M1=np.meshgrid(T1,M1)
T2,M2=T1.reshape(-1),M1.reshape(-1)
iv1=[]
for i in tqdm(range(len(T2))):
    c1=C1[i]
    t,k=T2[i],M2[i],
    s=exp(-r*t)
    v1=find_volatility_put(c1, s, k, t, r=0.022938, tol=1e-4, max_iter=100)
    iv1.append(v1)
#对插值求导数
import torch
from torch import log,sqrt,exp
from torch.distributions import Normal
normal_dist = Normal(loc=0.0, scale=1.0)
def calculate_theoretical_price(sigma, S, K, T, r):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * normal_dist.cdf(d1) - K * exp(-r * T) * normal_dist.cdf(d2)
    return call_price
m1, m2, m3 = 0.001, -0.01, 0.001
t = torch.tensor(T2, requires_grad=True)
k = torch.tensor(M2, requires_grad=True)
s = torch.tensor(np.exp(-r*T2), requires_grad=True)
sigma=torch.tensor(iv1,requires_grad=True)
c = calculate_theoretical_price(sigma, s, k, t, r=0.02)
dcdt = torch.autograd.grad(c, t,
                            grad_outputs=torch.ones(k.shape),  # 注意这里需要人为指定
                            create_graph=True,
                            retain_graph=True)[0]

dcdk = torch.autograd.grad(c, k,
                            grad_outputs=torch.ones(k.shape),  # 注意这里需要人为指定
                            create_graph=True,
                            retain_graph=True) [0] # 为计算二阶导保持计算图

dcdk2 = torch.autograd.grad(dcdk, k,
                            grad_outputs=torch.ones(k.shape),
                            create_graph=True)[0]  # 默认会自动销毁计算图

dcdt=torch.where(torch.isnan(dcdt), torch.full_like(dcdt, 1), dcdt)
dcdk = torch.where(torch.isnan(dcdk), torch.full_like(dcdk, -1), dcdk)
dcdk2 = torch.where(torch.isnan(dcdk2), torch.full_like(dcdk2, 1), dcdk2)
loss1 = torch.mean(m1*dcdt+m2*dcdk+m3*dcdk2)
print(loss1)


#对真实数据
test_data=[]
T1=np.linspace(0.1,0.5,100)
M1=np.linspace(0.9,1.15,100)
for t in T1:
    for m in M1:
        k=m
        s=np.exp(-r*t)
        iv=hagan2002.lognormal_vol(k+s,f+s,t,alpha,beta,rho,volvol)
        bs_price = bs_option(s, k, t, r, sigma=iv, option="call")
        test_data.append([k,t,iv,bs_price])
test_data=DataFrame(test_data)
test_data.columns=["M","T",'IV','bs_price']

print(np.mean((C1-test_data['bs_price'])**2))
print(np.mean((iv1-test_data['IV'])**2))




