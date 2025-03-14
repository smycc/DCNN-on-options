import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pysabr import hagan_2002_lognormal_sabr as hagan2002
from pandas.core.frame import DataFrame
from bs_price import bs_option,find_volatility,tensor_iv,calculate_theoretical_price
from torch import log,sqrt,exp

#整理真实数据
df=pd.read_csv(r'F:\smycc\codes\data\all_data.csv',encoding='gbk')
data=df[(df['index']==510050)&(df['type']=='C')][:1000]
#使用留出法进行模型评估,80%训练，20%测试
params = ['tau', 'M','underlying_price','r','K','IMPLC_VOLATLTY']
X, target = data[params], data['option_price']
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2,
                                                    random_state=42, shuffle=True)
x_train = X_train.to_numpy()[:,0:2]
x_test = X_test[['tau','M']].to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)
y_train=y_train/max(y_train)
y_test=y_test/max(y_test)
# 转换为Tensor
X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#对生成的mesh_grid
T=np.linspace(0.1,0.5,20)
M=np.linspace(0.9,1.15,20)
r,f,alpha,beta,rho,v=0.02,1,0.2,1,0,0.2
mesh_grid=[]
for t in T:
    for k in M:
        s=np.exp(-r*t)
        iv=hagan2002.lognormal_vol(k+s,f+s,t,alpha,beta,rho,v)
        bs_price = bs_option(s, k, t, r, sigma=iv, option="call")
        mesh_grid.append([k, t, iv, bs_price])
mesh_grid = DataFrame(mesh_grid)
mesh_grid.columns = ["M", "T", 'IV', 'bs_price']
params = ['M', 'T']
mesh_data, mesh_y_data = mesh_grid[params], mesh_grid['bs_price']
mesh_data = mesh_data.to_numpy()
mesh_data_tensor=torch.tensor(mesh_data, dtype=torch.float32)

#mlp
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 72)
        self.ru1 = nn.ReLU()
        self.linear2 = nn.Linear(72, 100)
        self.ru2 = nn.ReLU()
        self.linear3 = nn.Linear(100, 10)
        self.ru3 = nn.ReLU()
        self.linear4  = nn.Linear(10, 1)
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

mlp_model=MLP()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    mlp_model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# 训练模型
Loss1,Loss2,Total_loss,sigma_loss=[],[],[],[]
for epoch in tqdm(range(300)):
    optimizer.zero_grad()
    outputs =mlp_model(X_train_tensor)
    loss1 = criterion(y_train_tensor, outputs)
    Loss1.append(loss1)
    #隐波部分
    #meshgrid部分，也是测试部分
    r = 0.02
    c=outputs.reshape(-1)
    t = torch.tensor(X_train_tensor[:, 0],requires_grad=True)
    m = torch.tensor(X_train_tensor[:, 1])
    s = torch.tensor(X_train['underlying_price'].to_numpy())
    r = torch.tensor(X_train['r'].to_numpy())
    k = torch.tensor(torch.tensor(X_train['K'].to_numpy(),dtype=torch.float32),requires_grad=True)
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
iv_true=torch.tensor(X_train['IMPLC_VOLATLTY'].to_numpy())
iv_loss = criterion(iv_true, iv)
print(Loss1[-1],Loss2[-1],Total_loss[-1],iv_loss)
x=range(1,301)
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(6,6))
plt.plot(x,Loss1,label='预测损失',color='black')
plt.plot(x,Loss2,label='导数损失',color='mediumpurple')
plt.plot(x,Total_loss,label='总体损失',color='plum')
#plt.plot(x,sigma_loss,label='隐波损失',color='darkseagreen')
plt.xlabel('epoches')
plt.title('loss on dcmlp true data')
plt.legend()
plt.show()

#画图总结2 call\put
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
c=c.detach().numpy()
surf = ax.scatter3D(x_train[:,0],x_train[:,1],c,color='plum') # 通过修改camp修改曲面颜色
plt.title('call_dc_mlp on true data')
plt.xlabel('tau')
plt.ylabel('M')
plt.show()

#画图总结3 iv
plt.figure(figsize=(6,6))
iv_m=[loss.detach().numpy() for loss in iv]
x=x_train[:,0]
y= x_train[:,1]
ax = plt.subplot(projection = '3d')
ax.scatter(x, y,iv_m,c = 'plum')
plt.xlabel('tau')
plt.ylabel('M')
plt.title('真实数据_隐波_dcmlp')
plt.legend()
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