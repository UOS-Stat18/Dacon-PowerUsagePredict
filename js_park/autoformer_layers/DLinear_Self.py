import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm
tqdm.pandas
import warnings
warnings.filterwarnings('ignore')
import random

# 데이터 불러오기
test  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\test.csv")
train  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\train.csv")

train.head()

train = train[train["건물번호"] == 1]
new_train = train[["일시","기온(C)","풍속(m/s)","습도(%)","전력소비량(kWh)"]].set_index(keys='일시')
train_train = new_train[:-168]
train_test = new_train[-168:]

train_train_y = np.array(train_train["전력소비량(kWh)"]).reshape(1872,-1)
train_test_y = np.array(train_test["전력소비량(kWh)"]).reshape(168,-1)
train_train_x = train_train[["기온(C)","풍속(m/s)","습도(%)"]]
train_test_x = train_test[["기온(C)","풍속(m/s)","습도(%)"]]

train_train_y

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

train_train_y= min_max_scaler.fit_transform(train_train_y)
train_test_y = min_max_scaler.fit_transform(train_test_y)
train_train_x= min_max_scaler.fit_transform(train_train_x)
train_test_x = min_max_scaler.fit_transform(train_test_x)

#수정이 필요!
class moving_avg(torch.nn.Module): #커널사이즈 만큼 평균, stride만큼 이동, 패딩은 0
    def __init__(self, kernel_size, stride):# kernel_size는 열 개수
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x): 
        front = torch.tensor(x[0:1, :].repeat((self.kernel_size - 1) // 2, 1))
        # 시작을 kernel_size-1 // 2 만큼 늘려줌
        end = torch.tensor(x[-1:, :].repeat((self.kernel_size - 1) // 2, 1))
        # 끝을 kernel_size -1 // 2 만큼 늘려줌
        x = torch.cat([front, torch.tensor(x), end], dim=0)
        # cat에 dim = 1을 통해 [,*,]에서 *자리에 늘어나도록 합쳐줌
        x = self.avg(x.permute(1, 0))
        # permute()는 모든 차원들을 맞교환할 수 있다
        # 여기선 3번재, 2번째를 교환
        x = x.permute(1, 0)
        # 계산 후 다시 원래대로 차원을 돌려줌
        return x


class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = torch.tensor(x) - moving_mean
        return moving_mean, residual 

decomp = series_decomp(3)

temp_mean, temp_res = decomp(train_train_x)
temp_mean.shape

#window 생성

class windowDataset(Dataset):
    def __init__(self, x_mean, x_res, y, slide_size, num_features_x,num_features_y ,stride = 1):
        L = x_mean.shape[0]
        num_samples = (L - slide_size) // stride + 1

        X_M = np.zeros([slide_size, num_samples, num_features_x])
        X_R = np.zeros([slide_size, num_samples, num_features_x])
        Y = np.zeros([slide_size, num_samples, num_features_y])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + slide_size
            X_M[:,i,:] = x_mean[start_x:end_x]
            X_R[:,i,:] = x_res[start_x:end_x]
            Y[:,i,:] = y[start_x:end_x]
        
        X_M = X_M.transpose((1,0,2))
        X_R = X_R.transpose((1,0,2))
        Y = Y.transpose((1,0,2))

        self.x_m = X_M
        self.x_r = X_R
        self.y = Y

        self.len = len(X_M)

    def __getitem__(self, i):
        return self.x_m[i],self.x_r[i],self.y[i]
    def __len__(self):
        return self.len


slide_size = 96


train_dataset = windowDataset(temp_mean,temp_res,train_train_y, slide_size=slide_size,num_features_x=temp_res.shape[1],num_features_y=train_train_y.shape[1] ,stride=1)
train_loader = DataLoader(train_dataset, batch_size = 64)

train_train_y.shape

class LTSF_DLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LTSF_DLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.Linear_Trend = torch.nn.Linear(self.input_size, self.output_size) #한 개 층으로 이루어진 선형변환 통과
        self.Linear_Seasonal = torch.nn.Linear(self.input_size,  self.output_size)


    def forward(self, x_mean, x_res):
        trend_init, seasonal_init = x_mean, x_res
        #우선 decompsition을 통해 분해된 평균과 잔차를 추세와 계절로 받아줌
        trend_init, seasonal_init = trend_init.permute(0,2,1), seasonal_init.permute(0,2,1)
        #이후 계산의 편의를 위해 permute로 다시 차원 뒤집어줌
        trend_output = self.Linear_Trend(trend_init) # 추세에 대해 선형변환 통과
        seasonal_output = self.Linear_Seasonal(seasonal_init) # 계절에 대해 선형변환 통과
        x = seasonal_output + trend_output #분해되었던 추세(이동평균)와 계절(잔차)을 다시 더해줌
        return x.permute(0,2,1) # 추세와 계절이 모두 forward 시작할 때 permute된 상태이므로 다시 돌려주고 반환    
    

#학습
import torch.optim as optim

model = LTSF_DLinear(3,1)


learning_rate=0.01
epoch = 100
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.L1Loss()

best_valid=  float('inf')
patient=0
y.shape
from tqdm import tqdm

model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x_m,x_r,y in train_loader:
            optimizer.zero_grad()
            x_m = x_m.float()
            x_r = x_r.float()
            y = y.float()
            output = model(x_m, x_r)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))
        if loss<=best_valid:
            torch.save(model, r'C:\dacon\Dacon-PowerUsagePredict\js_park\best_AF.pth')
            patient=0
            best_valid=loss
        else:
            patient+=1
            if patient>=10:
                break

model = torch.load(r'C:\dacon\Dacon-PowerUsagePredict\js_park\best_AF.pth')

model.eval()


