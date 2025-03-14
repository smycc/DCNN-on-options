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
import torch
from torch import log,sqrt,exp
from torch.distributions import Normal

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

def find_volatility_call(market_price, S0, K, T, r=0.02, tol=1e-4, max_iter=100):
    sigma_min = 0.001  # 波动率范围下限调整为 0.01
    sigma_max = 1.0  # 波动率范围上限调整为 2.0
    iter_count = 0
    while iter_count<max_iter:
        sigma_mid = (sigma_min + sigma_max) / 2
        theoretical_price = bs_option(S0,K,T,r,sigma_mid,option="call")
        if abs(theoretical_price - market_price) <tol:
            return sigma_mid
        if theoretical_price > market_price:
            sigma_max = sigma_mid
        else:
            sigma_min = sigma_mid
        iter_count += 1
    return (sigma_min + sigma_max) / 2

def find_volatility_put(market_price, S0, K, T, r=0.02, tol=1e-4, max_iter=100):
    sigma_min = 0.001  # 波动率范围下限调整为 0.01
    sigma_max = 1.0  # 波动率范围上限调整为 2.0
    iter_count = 0
    while iter_count<max_iter:
        sigma_mid = (sigma_min + sigma_max) / 2
        theoretical_price = bs_option(S0,K,T,r,sigma_mid,option="put")
        if abs(theoretical_price - market_price) <tol:
            return sigma_mid
        if theoretical_price > market_price:
            sigma_max = sigma_mid
        else:
            sigma_min = sigma_mid
        iter_count += 1
    return (sigma_min + sigma_max) / 2

#meshgrid
T=np.linspace(0.1,0.5,30)
M=np.linspace(0.9,1.15,30)
r=2.2938/100
alpha,beta,rho,volvol=0.2,1.0,0.0,0.2
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

'''
#对样本数值画图
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter3D(BEFOREDATA['T'],BEFOREDATA['M'],BEFOREDATA['bs_call'],color='mediumpurple') # 通过修改camp修改曲面颜色
plt.title('call_scatter')
plt.xlabel('tau')
plt.ylabel('M')
ax = fig.add_subplot(122, projection='3d')
ax.scatter3D(BEFOREDATA['T'],BEFOREDATA['M'],BEFOREDATA['bs_put'],color='plum') # 通过修改camp修改曲面颜色
plt.title('put_scatter')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()
'''

'''
#图4-7
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(T1,M1, CV,color='mediumpurple', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('call_cubic_spline')
plt.xlabel('tau')
plt.ylabel('M')
ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(T1,M1, PV,color='plum', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('put_cubic_spline')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()
'''

#计算导数
#插值
T,M= np.meshgrid(T,M)
c0=(BEFOREDATA['bs_call']).values.reshape(-1,30)
p0=(BEFOREDATA['bs_put']).values.reshape(-1,30)
f = interpolate.interp2d(T, M, c0, kind='cubic')
g = interpolate.interp2d(T, M, p0, kind='cubic')
T1=np.linspace(0.1,0.5,100)
M1=np.linspace(0.9,1.15,100)
c1=f(T1,M1).transpose()
p1=g(T1,M1).transpose()
T1,M1= np.meshgrid(T1,M1)
C = np.asarray(c1).reshape((-1,100)).transpose()
P = np.asarray(p1).reshape((-1,100)).transpose()

#导数
normal_dist = Normal(loc=0.0, scale=1.0)
def calculate_theoretical_price(sigma, S, K, T, r):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * normal_dist.cdf(d1) - K * exp(-r * T) * normal_dist.cdf(d2)
    return call_price
T2,M2=T1.reshape(-1),M1.reshape(-1)
t = torch.tensor(T2, requires_grad=True)
k = torch.tensor(M2, requires_grad=True)
s = torch.tensor(np.exp(-r*T2), requires_grad=True)
CV,PV=[],[]
C0,P0,T0,M0=C.reshape(-1),P.reshape(-1),T1.reshape(-1),M1.reshape(-1)
for i in range(len(T2)):
    S0=np.exp(-r*T0[i])
    cv=find_volatility_call(C0[i], S0, M0[i], T0[i], r=0.022938, tol=1e-4, max_iter=100)
    pv=find_volatility_put(P0[i], S0, M0[i], T0[i], r=0.022938, tol=1e-4, max_iter=100)
    CV.append(cv)
    PV.append(pv)
CV1 = np.asarray(CV).reshape((-1,100)).transpose()
sigma=torch.tensor(CV,requires_grad=True)
print(sigma.size(),s.size(),k.size(),t.size())
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
dcdk= [loss.detach().numpy() for loss in dcdk]
dcdk2= [loss.detach().numpy() for loss in dcdk2]
dcdt= [loss.detach().numpy() for loss in dcdt]


#画图
fig=plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T1,M1, C,color='plum', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
for i in range(len(dcdk)):
    t=T2[i]
    if dcdk[i]>0 or dcdk[i]<-np.exp(-r*t) or dcdk2[i]<0 or dcdt[i]<0:
        ax.scatter3D(t,M2[i],0,color='red')
plt.title('call_cubic_spline')
plt.xlabel('tau')
plt.ylabel('M')
plt.title('call_cubic_spline')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()

fig=plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T1,M1, P,color='plum', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
for i in range(len(dcdk)):
    t=T2[i]
    if dcdk[i]>0 or dcdk[i]<-np.exp(-r*t) or dcdk2[i]<0 or dcdt[i]<0:
        ax.scatter3D(t,M2[i],0,color='red')
plt.title('put_cubic_spline')
plt.xlabel('tau')
plt.ylabel('M')
plt.title('put_cubic_spline')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()

#计算隐波 并画图4-8
fig=plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T1,M1, CV1,color='plum', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
for i in range(len(dcdk)):
    t=T2[i]
    if dcdk[i]>0 or dcdk[i]<-np.exp(-r*t) or dcdk2[i]<0 or dcdt[i]<0:
        ax.scatter3D(t,M2[i],0,color='red')
plt.title('IV_cubic_spline')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()


#图4-9
'''
T,M= np.meshgrid(T,M)
c0=(BEFOREDATA['bs_call']).values.reshape(-1,30)
f1 = interpolate.interp2d(T, M, c0, kind='linear')
f2 = interpolate.interp2d(T, M, c0, kind='cubic')
f3 = interpolate.interp2d(T, M, c0, kind='quintic')
T1=np.linspace(0.1,0.5,100)
M1=np.linspace(0.9,1.15,100)
c1=f1(T1,M1).transpose()
c2=f2(T1,M1).transpose()
c3=f3(T1,M1).transpose()
T1,M1= np.meshgrid(T1,M1)
C1 = np.asarray(c1).reshape((-1,100)).transpose()
C2 = np.asarray(c2).reshape((-1,100)).transpose()
C3 = np.asarray(c3).reshape((-1,100)).transpose()
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(131, projection='3d')
surf = ax.plot_surface(T1,M1, C1,color='mediumpurple', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('call_linear')
plt.xlabel('tau')
plt.ylabel('M')
ax = fig.add_subplot(132, projection='3d')
surf = ax.plot_surface(T1,M1, C2,color='plum', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('call_cubic')
plt.xlabel('tau')
plt.ylabel('M')
ax = fig.add_subplot(133, projection='3d')
surf = ax.plot_surface(T1,M1, C3,color='royalblue', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('call_quintic')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()
'''

