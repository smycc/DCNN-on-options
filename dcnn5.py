import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import log,sqrt,exp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pysabr import hagan_2002_lognormal_sabr as hagan2002
from pandas.core.frame import DataFrame
from bs_price import bs_option,find_volatility,tensor_iv,calculate_theoretical_price

#生成训练集和测试集
T0=np.linspace(0.1,0.5,30)
M0=np.linspace(0.8,1.25,30)
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
# 转换为Tensor
X_train_tensor = torch.tensor(BEFOREDATA[['T','M']].to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(BEFOREDATA['bs_call'].to_numpy(), dtype=torch.float32).view(-1, 1)

#lstm
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
lstm_model = LstmRNN(input_size=2, hidden_size=8, output_size=1, num_layers=5)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    lstm_model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# 训练模型
Loss1,Loss2,Total_loss,sigma_loss=[],[],[],[]
for epoch in tqdm(range(300)):
    optimizer.zero_grad()
    outputs =lstm_model(X_train_tensor)
    loss1 = criterion(y_train_tensor, outputs)
    Loss1.append(loss1)
    #隐波部分
    #meshgrid部分，也是测试部分
    r = 0.02
    c=outputs.reshape(-1)
    t = torch.tensor(X_train_tensor[:, 0],requires_grad=True)
    k = torch.tensor(X_train_tensor[:, 1],requires_grad=True)
    s = exp(-r*t)
    iv=tensor_iv(c, s, k, t)
    #导数部分
    m1, m2, m3 = 0.001, -0.01, -0.001
    c0 = calculate_theoretical_price(iv, s, k, t, r=0.02)
    dcdt = torch.autograd.grad(c0, t,
                               grad_outputs=torch.ones(k.shape),  # 注意这里需要人为指定
                               create_graph=True,
                               retain_graph=True)[0]
    dcdk = torch.autograd.grad(c0, k,
                               grad_outputs=torch.ones(k.shape),  # 注意这里需要人为指定
                               create_graph=True,
                               retain_graph=True)[0]  # 为计算二阶导保持计算图

    dcdk2 = torch.autograd.grad(dcdk, k,
                                grad_outputs=torch.ones(k.shape),
                                create_graph=True)[0]  # 默认会自动销毁计算图

    dcdt = torch.where(torch.isnan(dcdt), torch.full_like(dcdt, 1), dcdt)
    dcdk = torch.where(torch.isnan(dcdk), torch.full_like(dcdk, -1), dcdk)
    dcdk2 = torch.where(torch.isnan(dcdk2), torch.full_like(dcdk2, -1), dcdk2)
    loss2 = torch.mean(m1 * dcdt + m2 * dcdk + m3 * dcdk2)
    Loss2.append(loss2)
    loss=loss1+loss2
    Total_loss.append(loss)
    loss.backward()
    optimizer.step()


#画图总结1
Loss1 = [loss.detach().numpy() for loss in Loss1]
Loss2 = [loss.detach().numpy() for loss in Loss2]
Total_loss = [loss.detach().numpy() for loss in Total_loss]
iv_true=torch.tensor(BEFOREDATA['IV'].to_numpy(), dtype=torch.float32)
iv_loss = criterion(iv_true, iv)
print(Loss1,Loss2,Total_loss,iv_loss)
x=range(1,301)
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(6,6))
plt.plot(x,Loss1,label='预测损失',color='black')
plt.plot(x,Loss2,label='导数损失',color='mediumpurple')
plt.plot(x,Total_loss,label='总体损失',color='plum')
#plt.plot(x,sigma_loss,label='隐波损失',color='darkseagreen')
plt.xlabel('epoches')
plt.title('loss on dc_lstm generate data')
plt.legend()
plt.show()

#画图总结2 call\put
C=outputs.detach().numpy()
C = C.reshape((30,30)).transpose()
T,M=np.meshgrid(T0,M0)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T,M, C,color='mediumpurple', alpha=1.0, linewidth=1.0)
plt.title('put_dc_lstm on generate data')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()

#画图总结3 iv
IV=iv.detach().numpy()
IV = IV.reshape((30,30)).transpose()
T,M=np.meshgrid(T0,M0)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T,M, IV,color='mediumpurple', alpha=1.0, linewidth=1.0)
plt.title('生成数据 隐波 dc_lstm')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()

'''
print(c.size(),iv.size())
C=c.detach().numpy()
IV=iv.detach().numpy()
T1,M1=np.meshgrid(T,M)
T2,M2=T1.reshape(-1),M1.reshape(-1)
dcdk= [loss.detach().numpy() for loss in dcdk]
dcdk2= [loss.detach().numpy() for loss in dcdk2]
dcdt= [loss.detach().numpy() for loss in dcdt]
for i in range(len(dcdk)):
    t=T2[i]
    if dcdk[i]>0 or dcdk[i]<-np.exp(-r*t) or dcdk2[i]<0 or dcdt[i]<0:
        ax.scatter3D(t,M2[i],0,color='red')
'''
'''
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter3D(T2,M2,IV,color='plum') # 通过修改camp修改曲面颜色
for i in range(len(dcdk)):
    t=T2[i]
    if dcdk[i]>0 or dcdk[i]<-np.exp(-r*t) or dcdk2[i]<0 or dcdt[i]<0:
        ax.scatter3D(t,M2[i],0,color='red')
plt.title('iv_dc_mlp')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()

C=Mesh_grid['bs_call'].values.reshape(100,100)
fig=plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T,M, C,color='mediumpurple', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('call on DCMLP')
ax.set_xlabel('tau', rotation=20) # 设置标签角度
ax.set_ylabel('M', rotation=-45)
ax.set_zlabel('IV', rotation=0)
'''
'''
CV=[]
for i in range(len(Mesh_grid)):
    t,k=Mesh_grid['T'][i],Mesh_grid['M'][i]
    s = np.exp(-r * t)
    call=C[i]
    cv = find_volatility(call, s, k, t, r=0.02, tol=1e-4, max_iter=100)
    CV.append(cv)
CV = np.asarray(CV)
CV=CV.reshape(100,100)
fig=plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T,M, CV,color='mediumpurple', alpha=1.0, linewidth=1.0) # 通过修改camp修改曲面颜色
plt.title('IV on DCMLP')
ax.set_xlabel('tau', rotation=20) # 设置标签角度
ax.set_ylabel('M', rotation=-45)
ax.set_zlabel('IV', rotation=0)
plt.show()
'''