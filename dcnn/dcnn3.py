import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import log,sqrt,exp
from torch.distributions import Normal
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.distributions import Normal
from my_quad import quad,find_volatility
from sklearn import preprocessing

dataset = pd.read_csv(r"F:\smycc\codes\gan\data.csv")
dataset = dataset[dataset['M30'] > 0]
params = ['open', 'close', 'high', 'low', 'volume', 'turn', 'M30']
data = dataset[params].astype('float32')
data = data.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

# 得到训练集和测试集
# 用30天去预测31天
train = data[:854 - 30, :]
train_x, train_y = [], []
for i in range(len(train) - 30):
    train_x.append(train[i:i + 30, :])
    train_y.append(train[i + 30, :])
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
print(train_x_tensor.size(), train_y_tensor.size())

test = data[854 - 30:972, :]
test_x, test_y = [], []
for i in range(len(test) - 30):
    test_x.append(test[i:i + 30, :])
    test_y.append(test[i + 30, :])
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)
test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32)
print(test_x_tensor.size(), test_y_tensor.size())

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
        # s, b= x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.sp1(x)
        x = self.linear1(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(7, 72)
        self.ru1 = nn.ReLU()
        self.linear2 = nn.Linear(72, 100)
        self.ru2 = nn.ReLU()
        self.linear3 = nn.Linear(100, 10)
        self.ru3 = nn.ReLU()
        self.linear4 = nn.Linear(10, 1)
        self.sm4 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.ru1(x)
        x = self.linear2(x)
        x = self.ru2(x)
        x = self.linear3(x)
        x = self.ru3(x)
        x = self.linear4(x)
        x = self.sm4(x)
        return x

lstm_model = LstmRNN(input_size=7, hidden_size=8, output_size=7, num_layers=5)
mlp_model = MLP()
optimizer = torch.optim.Adam(list(lstm_model.parameters()) + list(mlp_model.parameters()), lr=1e-3)
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

def tensor_iv(p, s, k, t):
    SIGMA=[]
    p,s,k,t=p.detach().numpy(),s.detach().numpy(),k.detach().numpy(),t.detach().numpy()
    for i in range(len(p)):
        p_,s_,k_,t_=p[i],s[i],k[i],t[i]
        sigma=find_volatility(p_, s_, k_, t_)
        SIGMA.append(sigma)
    SIGMA=torch.tensor(SIGMA)
    return SIGMA

normal_dist = Normal(loc=0.0, scale=1.0)
def calculate_theoretical_price(sigma, S, K, T, r):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * normal_dist.cdf(d1) - K * exp(-r * T) * normal_dist.cdf(d2)
    return call_price

C_loss,CV_loss,D_loss,Total_loss,Loss=[],[],[],[],[]
for epoch in tqdm(range(300)):
    optimizer.zero_grad()
    output1 = lstm_model(train_x_tensor)[:, -1, :]  # 这里应该输出1*7,即为预测的第30天情况
    loss1 = criterion1(output1, train_y_tensor)
    y_tensor = torch.cat((torch.zeros(794, 30), torch.ones(794, 1)), 1)
    x_tensor = torch.cat((train_x_tensor, output1.reshape(794, 1, 7)), 1)
    output2 = mlp_model(x_tensor)[:, :, -1].reshape(794, 31)
    loss2 = criterion2(output2, y_tensor)
    loss = loss1 + 0.01 * loss2
    Loss.append(loss)

    #期权价格和导数
    t = np.linspace(0.1,0.5,30)
    k = np.linspace(0.9,1.15,30)
    r=torch.tensor(0.02)
    c,c0,p=torch.zeros(30),torch.zeros(30),torch.zeros(30)
    for i in range(30):
        c[i]= quad(output1[:,1], t[i], k[i])
        c0[i] = quad(train_y_tensor[:, 1], t[i], k[i])
    C_loss.append(criterion1(c0, c))
    t = torch.tensor(t, requires_grad=True)
    k = torch.tensor(k, requires_grad=True)
    s=exp(-r*t)
    iv = tensor_iv(c0, s, k, t)
    iv_g=tensor_iv(c, s, k, t)
    CV_loss.append(criterion1(iv_g, iv))
    # 导数部分
    m1, m2, m3 = 0.001, -0.01, -0.001
    c = calculate_theoretical_price(iv_g, exp(-r*t), k, t, r=0.02)
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
    dloss = torch.mean(m1 * dcdt + m2 * dcdk + m3 * dcdk2)
    D_loss.append(dloss)
    tloss = loss+dloss
    tloss.backward()
    optimizer.step()
    Total_loss.append(tloss)

Loss = [loss.detach().numpy() for loss in Loss]
D_loss = [loss.detach().numpy() for loss in D_loss]
Total_loss = [loss.detach().numpy() for loss in Total_loss]
CV_loss = [loss.detach().numpy() for loss in CV_loss]
print(C_loss[-1],D_loss[-1],Total_loss[-1],CV_loss[-1])

'''
x=range(1,301)
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(6,6))
plt.plot(x,Loss,label='预测损失',color='black')
plt.plot(x,D_loss,label='导数损失',color='mediumpurple')
plt.plot(x,Total_loss,label='总体损失',color='plum')
#plt.plot(x,sigma_loss,label='隐波损失',color='darkseagreen')
plt.xlabel('epoches')
plt.title('loss on dcgan true data')
plt.legend()
plt.show()

t = np.linspace(0.1,0.5,30)
k = np.linspace(0.9,1.15,30)
T0,M0=np.meshgrid(t,k)
T,M=T0.reshape(-1),M0.reshape(-1)
r=torch.tensor(0.02)
c,p=torch.zeros(900),torch.zeros(900)
for i in range(30):
    for j in range(30):
        c[30*i+j]= quad(output1[:,1], t[i], k[i])
        p[30*i+j] = c[i] + k[i] * exp(-r * t[i]) - exp(-r * t[i])
t = torch.tensor(T)
k = torch.tensor(M)
s=exp(-r*t)
iv_g=tensor_iv(c, s, k, t)

C=c.reshape((30,30))
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T0,M0, C,color='royalblue', alpha=1.0, linewidth=1.0)
plt.title("call on DCGAN true data")
ax.set_xlabel('tau', rotation=20) # 设置标签角度
ax.set_ylabel('M', rotation=-45)
ax.set_zlabel('c', rotation=0)
plt.show()

P=p.reshape((30,30))
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T0,M0, P,color='royalblue', alpha=1.0, linewidth=1.0)
plt.title("put on DCGAN true data")
ax.set_xlabel('tau', rotation=20) # 设置标签角度
ax.set_ylabel('M', rotation=-45)
ax.set_zlabel('p', rotation=0)
plt.show()

IV=iv_g.reshape((30,30))
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T0,M0, IV,color='royalblue', alpha=1.0, linewidth=1.0)
plt.title('IV on DCGAN true data')
ax.set_xlabel('tau', rotation=20) # 设置标签角度
ax.set_ylabel('M', rotation=-45)
ax.set_zlabel('IV', rotation=0)
plt.show()
'''