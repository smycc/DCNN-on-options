import numpy as np
from scipy.stats import norm

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

#只要给定tensor的s，t,m就能算出c
def find_volatility(market_price, S0, K, T, r=0.02, tol=1e-4, max_iter=100):
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

f,r=1,0.022938
alpha,v=0.2,0.2
def I1(tau,K):
    xishu=alpha/(2*np.pi*v*tau)
    zhishu1=(f-K)**2/(2*alpha**2*tau)
    zhishu2=(np.log(alpha))**2/(2*v**2*tau)
    return xishu*np.exp(-zhishu1-zhishu2)
def I2(tau,S_T,K):
    return np.exp(r*tau)*max(0,S_T-K)

#S是1列数据，tau_ex,K均为单个数据
def quad(S,tau_ex,K):
    sum1 = 0
    for j in range(len(S)):
        quad = I1(((j + 0.5) / 2 / 365)* (j + 0.5) / 365,K)
        sum1 += quad
    sum2 = 0
    for j in range(len(S)):
        s=S[j]
        high = I2((j + 0.5) / 365, s,K)
        wide = 1 / 365
        sum2 += high * wide
    c = np.exp(-r * tau_ex) * (max(f - K, 0) - sum1 + 0.5 * sum2)
    s = np.exp(-r * tau_ex)
    cv = find_volatility(c, s, K, tau_ex, r=0.022938, tol=1e-4, max_iter=100)
    return c

tau_ex,K=0.6,1.03

S=[1.2,1.34,1.3,2.1,0.9]
print(quad(S,tau_ex,K))
