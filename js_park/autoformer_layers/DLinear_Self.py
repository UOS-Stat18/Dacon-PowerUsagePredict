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

#window 생성

class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, num_features ,stride = 1):
        L = y.shape[0]
        num_samples = (L - input_window - output_window) // stride + 1

        X = np.zeros([input_window, num_samples, num_features])
        Y = np.zeros([output_window, num_samples, num_features])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i,:] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i,:] = y[start_y:end_y]

        
        X = X.transpose((1,0,2))
        Y = Y.transpose((1,0,2))

        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len


iw = 168 * 2
ow = 168


train_x_dataset = windowDataset(train_train_x, input_window=iw, output_window=ow,num_features=train_train_x.shape[1] ,stride=1)
train_y_dataset = windowDataset(train_train_y, input_window=iw, output_window=ow,num_features=train_train_y.shape[1] ,stride=1)
train_x_loader = DataLoader(train_x_dataset, batch_size=64)
train_y_loader = DataLoader(train_y_dataset, batch_size=64)


train_x_dataset.len
train_y_dataset.len
train_x_dataset[0][0].shape
train_y_dataset[0][0].shape

train_x_dataset[0][0][0].mean()
train_y_dataset[0].permute(0,2,1)


#수정이 필요!
class moving_avg(torch.nn.Module): #커널사이즈 만큼 평균, stride만큼 이동, 패딩은 0
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x): 
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # 시작을 kernel_size-1 // 2 만큼 늘려줌
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # 끝을 kernel_size -1 // 2 만큼 늘려줌
        x = torch.cat([front, x, end], dim=1)
        # cat에 dim = 1을 통해 [,*,]에서 *자리에 늘어나도록 합쳐줌
        x = self.avg(x.permute(0, 2, 1))
        # permute()는 모든 차원들을 맞교환할 수 있다
        # 여기선 3번재, 2번째를 교환
        x = x.permute(0, 2, 1)
        # 계산 후 다시 원래대로 차원을 돌려줌
        return x

temp = moving_avg(25,1)
temp(train_test_y)


def decompsition():




class LTSF_DLinear(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, feature_size):
        super(LTSF_DLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decompsition = kernel_size
        self.channels = feature_size

        self.Linear_Trend = torch.nn.Linear(self.input_size, self.hidden_size) #한 개 층으로 이루어진 선형변환 통과
        self.Linear_Seasonal = torch.nn.Linear(self.input_size,  self.hidden_size)


    def forward(self, x):
        trend_init, seasonal_init = self.decompsition(x)
        #우선 decompsition을 통해 분해된 평균과 잔차를 추세와 계절로 받아줌
        trend_init, seasonal_init = trend_init.permute(0,2,1), seasonal_init.permute(0,2,1)
        #이후 계산의 편의를 위해 permute로 다시 차원 뒤집어줌
        trend_output = self.Linear_Trend(trend_init) # 추세에 대해 선형변환 통과
        seasonal_output = self.Linear_Seasonal(seasonal_init) # 계절에 대해 선형변환 통과
        x = seasonal_output + trend_output #분해되었던 추세(이동평균)와 계절(잔차)을 다시 더해줌
        return x.permute(0,2,1) # 추세와 계절이 모두 forward 시작할 때 permute된 상태이므로 다시 돌려주고 반환

