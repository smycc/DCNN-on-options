import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch import log,sqrt,exp
from torch.distributions import Normal
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from pysabr import hagan_2002_lognormal_sabr as hagan2002
from torch.distributions import Normal
from sklearn.preprocessing import StandardScaler
from pylab import mpl

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

def find_volatility(market_price, S0, K, T, r, tol=1e-4, max_iter=100):
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

def tensor_iv(p, s, k, t,r):
    SIGMA=[]
    p,s,k,t,r=p.detach().numpy(),s.detach().numpy(),k.detach().numpy(),t.detach().numpy(),r.detach().numpy()
    for i in range(len(p)):
        p_,s_,k_,t_,r_=p[i],s[i],k[i],t[i],r[i]
        sigma=find_volatility(p_,s_,k_,t_,r_)
        SIGMA.append(sigma)
    SIGMA=torch.tensor(SIGMA)
    return SIGMA

normal_dist = Normal(loc=0.0, scale=1.0)
def calculate_theoretical_price(sigma, S, K, T, r):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * normal_dist.cdf(d1) - K * exp(-r * T) * normal_dist.cdf(d2)
    return call_price

#生成训练集和测试集
T0=np.linspace(0.1,0.5,200)
M0=np.linspace(0.8,1.25,200)
r=0.02
alpha,beta,rho,volvol=0.2,1.0,0.0,0.2
BEFOREDATA=[]
f=1
for t in T0:
    for k in M0:
        s=np.exp(-r*t)
        iv = hagan2002.lognormal_vol(k + s, f + s, t, alpha, beta, rho, volvol)
        bs_call = bs_option(s, k, t, r, sigma=iv, option="call")
        bs_put = bs_option(s, k, t, r, sigma=iv, option="put")
        BEFOREDATA.append([t, k, iv, bs_call, bs_put])
BEFOREDATA=DataFrame(BEFOREDATA)
BEFOREDATA.columns=['T','M','IV','bs_call','bs_put']
#加入噪声
def add_tabular_noise(df, noise_ratio=0.1):
    """
    表格数据加噪（数值型特征）
    :param df: pandas DataFrame
    :param noise_ratio: 噪声占特征值的比例
    :return: 加噪后的DataFrame
    """
    noisy_df = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        noise = np.random.normal(0, df[col].std() * noise_ratio, size=len(df))
        noisy_df[col] += noise
    return noisy_df
BEFOREDATA=add_tabular_noise((BEFOREDATA))
T1=np.linspace(0.1,0.5,100)
M1=np.linspace(0.9,1.15,100)
AFTERDATA=[]
for t in T1:
    for k in M1:
        s=np.exp(-r*t)
        iv = hagan2002.lognormal_vol(k + s, f + s, t, alpha, beta, rho, volvol)
        bs_call = bs_option(s, k, t, r, sigma=iv, option="call")
        bs_put = bs_option(s, k, t, r, sigma=iv, option="put")
        AFTERDATA.append([t, k, iv, bs_call, bs_put])
AFTERDATA=DataFrame(AFTERDATA)
AFTERDATA.columns=['T','M','IV','bs_call','bs_put']
# 转换为Tensor
X_train_tensor = torch.tensor(BEFOREDATA[['T','M']].to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(BEFOREDATA['bs_call'].to_numpy(), dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(AFTERDATA[['T','M']].to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(AFTERDATA['bs_call'].to_numpy(), dtype=torch.float32).view(-1, 1)

class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.sp1 = nn.Softplus()
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        #s, b= x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.sp1(x)
        x = self.linear1(x)
        return x

# 参数设置
model = LstmRNN(input_size=2, hidden_size=8, output_size=1, num_layers=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
Train_loss,Test_loss,sigma_loss,de_loss=[],[],[],[]
for epoch in tqdm(range(1000)):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(y_train_tensor, outputs)
    loss.backward()
    optimizer.step()
    Train_loss.append(loss)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(y_test_tensor, test_outputs)
    Test_loss.append(test_loss)

t = torch.tensor(AFTERDATA['T'].to_numpy(),requires_grad=True)
k = torch.tensor(AFTERDATA['M'].to_numpy(),requires_grad=True)
r=0.02*torch.ones(len(t))
s = exp(-r*t)
iv_m=tensor_iv(test_outputs[:,0], s, k, t,r)
Train_loss = [loss.detach().numpy() for loss in Train_loss]
Test_loss = [loss.detach().numpy() for loss in Test_loss]
IV= torch.tensor(AFTERDATA['IV'].to_numpy(), dtype=torch.float32).view(-1, 1)
sigma_loss=criterion(iv_m, IV)
print(Test_loss[-1],sigma_loss)

#导数部分
m1, m2, m3 = 0.001, -0.01, 0.001
c = calculate_theoretical_price(iv_m, s, k, t, r=0.02)
dcdt = torch.autograd.grad(c, t,
                               grad_outputs=torch.ones(k.shape),  # 注意这里需要人为指定
                               create_graph=True,
                               retain_graph=True)[0]
dcdk = torch.autograd.grad(c, k,
                               grad_outputs=torch.ones(k.shape),  # 注意这里需要人为指定
                               create_graph=True,
                               retain_graph=True)[0]  # 为计算二阶导保持计算图

dcdk2 = torch.autograd.grad(dcdk, k,
                                grad_outputs=torch.ones(k.shape),
                                create_graph=True)[0]  # 默认会自动销毁计算图

dcdt = torch.where(torch.isnan(dcdt), torch.full_like(dcdt, 1), dcdt)
dcdk = torch.where(torch.isnan(dcdk), torch.full_like(dcdk, -1), dcdk)
dcdk2 = torch.where(torch.isnan(dcdk2), torch.full_like(dcdk2, 1), dcdk2)
loss1 = torch.mean(m1 * dcdt + m2 * dcdk + m3 * dcdk2)
print(loss1)
'''
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["font.sans-serif"] = ["SimHei"]
Train_loss = [loss.detach().numpy() for loss in Train_loss]
Test_loss = [loss.detach().numpy() for loss in Test_loss]
x=range(1,1001)
plt.figure(figsize=(8,8))
plt.plot(x,Train_loss,label='train loss',color='black')
plt.plot(x,Test_loss,label='test loss',color='mediumpurple')
plt.xlabel('epoches')
plt.title('生成数据')
plt.legend()
plt.show()

mpl.rcParams["font.sans-serif"] = ["SimHei"]
t = torch.tensor(AFTERDATA['T'].to_numpy())
k = torch.tensor(AFTERDATA['M'].to_numpy())
r=0.02*torch.ones(len(t))
s = exp(-r*t)
call=np.asarray(test_outputs).reshape((-1,100)).transpose()
T,M=np.meshgrid(T1,M1)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T,M, call,color='plum', alpha=1.0, linewidth=1.0)
plt.xlabel('tau')
plt.ylabel('M')
plt.title('生成数据_看涨')
plt.legend()
plt.show()

call=np.asarray(test_outputs).reshape((-1,100)).transpose()
T,M=np.meshgrid(T1,M1)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T,M, call,color='royalblue', alpha=1.0, linewidth=1.0)
plt.xlabel('tau')
plt.ylabel('M')
plt.title('生成数据_隐波')
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
call=[loss.detach().numpy() for loss in test_outputs]
x=AFTERDATA['T']
y=AFTERDATA['M']
ax = plt.subplot(projection = '3d')
ax.scatter(x, y,call, c = 'mediumpurple')
plt.xlabel('tau')
plt.ylabel('M')
plt.title('真实数据_看涨')
plt.legend()
plt.show()
#隐波部分
t = torch.tensor(X_test_tensor[:, 0])
m = torch.tensor(X_test_tensor[:, 1])
s = torch.tensor(X_test['underlying_price'].to_numpy())
r = torch.tensor(X_test['r'].to_numpy())
k = torch.tensor(X_test['K'].to_numpy())
iv = torch.tensor(X_test['IMPLC_VOLATLTY'].to_numpy())
iv_m=tensor_iv(test_outputs[:,0], s, k, t,r)
sigma_loss.append(criterion(iv_m,iv))

print(Test_loss[-1],sigma_loss[-1])

#画图总结1
Train_loss = [loss.detach().numpy() for loss in Train_loss]
Test_loss = [loss.detach().numpy() for loss in Test_loss]
x=range(1,1001)
plt.figure(figsize=(8,8))
plt.plot(x,Train_loss,label='train loss',color='black')
plt.plot(x,Test_loss,label='test loss',color='mediumpurple')
plt.plot(x,sigma_loss,label='sigma loss',color='plum')
plt.xlabel('epoches')
plt.legend()
plt.show()

#画图总结2

T=np.linspace(0.1,0.5,100)
M=np.linspace(0.9,1.15,100)
alpha,beta,rho,volvol=0.2,1.0,0.0,0.2
f,r=1,0.02
Mesh_grid=[]
for t in T:
    for k in M:
        s = np.exp(-r * t)
        iv = hagan2002.lognormal_vol(k + s, f + s, t, alpha, beta, rho, volvol)
        bs_call= bs_option(s, k, t, r, sigma=iv, option="call")
        bs_put=bs_option(s, k, t, r, sigma=iv, option="put")
        cv = find_volatility(bs_call, s, k, t, r=0.02, tol=1e-4, max_iter=100)
        Mesh_grid.append([t, k, iv, bs_call, bs_put,cv])
Mesh_grid=DataFrame(Mesh_grid)
Mesh_grid.columns=['T','M','IV','bs_call','bs_put','cv']
Mesh_grid.to_csv(r'F:\smycc\codes\data\mesh_grid.csv')
Test=Mesh_grid[['T','M']]
Test = Test.to_numpy()
Test_tensor = torch.tensor(Test, dtype=torch.float32)
C_test=model(Test_tensor)
C=C_test.detach().numpy()
CV=[]
for i in range(len(Mesh_grid)):
    t,k=Mesh_grid['T'][i],Mesh_grid['M'][i]
    s = np.exp(-r * t)
    call=C[i]
    cv = find_volatility(call, s, k, t, r=0.02, tol=1e-4, max_iter=100)
    CV.append(cv)
CV = np.asarray(CV)
C=C.reshape(100,100)
CV=CV.reshape(100,100)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T,M, CV,color='mediumpurple', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('IV on MLP')
ax.set_xlabel('tau', rotation=20) # 设置标签角度
ax.set_ylabel('M', rotation=-45)
ax.set_zlabel('IV', rotation=0)
plt.show()
'''
