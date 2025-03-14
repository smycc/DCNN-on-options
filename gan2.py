import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = pd.read_csv(r"F:\smycc\codes\gan\data.csv")
dataset=dataset[dataset['M30']>0]
params = ['open','close','high','low','volume','turn','M30']
data = dataset[params].astype('float32')
data = data.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
data= min_max_scaler.fit_transform(data)

#得到训练集和测试集
#用30天去预测31天
train=data[:854-30,:]
train_x,train_y=[],[]
for i in range(len(train)-30):
    train_x.append(train[i:i+30,:])
    train_y.append(train[i+30,:])
train_x=np.asarray(train_x)
train_y=np.asarray(train_y)
train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
print(train_x_tensor.size(),train_y_tensor.size())

test=data[854-30:972,:]
test_x,test_y=[],[]
for i in range(len(test)-30):
    test_x.append(test[i:i+30,:])
    test_y.append(test[i+30,:])
test_x=np.asarray(test_x)
test_y=np.asarray(test_y)
test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32)
print(test_x_tensor.size(),test_y_tensor.size())

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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(7, 72)
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

lstm_model = LstmRNN(input_size=7, hidden_size=8, output_size=7, num_layers=5)
mlp_model=MLP()
optimizer =torch.optim.Adam(list(lstm_model .parameters()) + list(mlp_model.parameters()),lr=1e-3)
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

for epoch in tqdm(range(3000)):
    output1 = lstm_model(train_x_tensor)[:,-1,:]  # 这里应该输出1*7,即为预测的第30天情况
    loss1 = criterion1(output1,train_y_tensor)
    y_tensor = torch.cat((torch.zeros(794,30), torch.ones(794, 1)), 1)
    x_tensor = torch.cat((train_x_tensor, output1.reshape(794,1,7)), 1)
    output2 = mlp_model(x_tensor)[:,:,-1].reshape(794,31)
    loss2 = criterion2(output2, y_tensor)
    loss = loss1 + 0.01 * loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = lstm_model(test_x_tensor)[:, -1, :]  # 这里应该输出1*7,即为预测的第30天情况
    loss1 = criterion1(pred, test_y_tensor)
    y_tensor = torch.cat((torch.zeros(118, 30), torch.ones(118, 1)), 1)
    x_tensor = torch.cat((test_x_tensor, pred.reshape(118, 1, 7)), 1)
    output2 = mlp_model(x_tensor)[:, :, -1].reshape(118, 31)
    loss2 = criterion2(output2, y_tensor)
    loss = loss1 + 0.01 * loss2


sns.set_style("darkgrid")
x=range(len(output1))
pred_price=output1[:,1]
true_price=train_y_tensor[:,1]
pred_price = pred_price.detach().numpy()
true_price = true_price.detach().numpy()
plt.figure(figsize=(14,8))
plt.plot(x, true_price, label='true', color='black')
plt.plot(x,pred_price,label='mlp',color='plum')
plt.show()

'''
x=dataset['date'][854-30:972-30]
pred_price=pred[:,1]
true_price=test_y_tensor[:,1]
pred_price = pred_price.detach().numpy()
true_price = true_price.detach().numpy()
plt.figure(figsize=(14,8))
plt.plot(x, true_price, label='true', color='black')
plt.plot(x,pred_price,label='lstm',color='plum')
plt.xticks(range(1,len(x),10),rotation=45)
plt.legend()
plt.show()


date=dataset['date'][854-30:972-30]
c={"date":date,"true":true_price,"pred1":pred_price}
data=pd.DataFrame(c)
data.to_csv(r'F:\smycc\codes\gan\gan.csv')
'''
