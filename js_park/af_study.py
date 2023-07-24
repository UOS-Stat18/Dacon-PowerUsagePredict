import torch
import torch.nn as nn
import torch.nn.functional as F
from autoformer_layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from autoformer_layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from autoformer_layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
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

###########################
#여긴 뜯어볼때 실험하느라 필요한거 불러올라고 넣음
test  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\test.csv")
train  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\train.csv")
building  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\building_info.csv",
                        encoding="CP949")

train = train[train["건물번호"] == 1]
train.shape
train_df = train[['일시','전력소비량(kWh)','기온(C)','풍속(m/s)','습도(%)']]
test_df = train[['일시','전력소비량(kWh)','기온(C)','풍속(m/s)','습도(%)']]


window_size = 168*2
forcast_size= 168
batch_size = 32
targets = '전력소비량(kWh)'
date = '일시'
############################33


#우선 LSTF_DLinear 클래스를 통과하게 된다.
#LSTF_Linear, LSTF_DLinear, LSTF_Nlinear는 각각 서로 다른 특성을 가진 autoformer 모델
#난 우선 DLinear를 공부하기로 함
#논문에서 트랜스포머가 시간적 특성을 추출해내지 못하는 것을 보완하기 위해 추세와 계절로 분해하여 학습시키는 파트
#이 코드를 작성한 분의 설명에 따르면 Nlinear가 가장 좋은 성능을 냈다고 하긴 함
class LTSF_DLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, kernel_size, individual, feature_size):
        super(LTSF_DLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual #individual은 True 또는 False로 설정
        #아무래도 모델 변수 개수에 따라 true false로 설정하는듯
        self.channels = feature_size
        if self.individual: #True로 설정했을 경우
            #모듈리스트는  nn.Sequential과 마찬가지로 nn.Module의 list를 input으로 받는다.
            #이는 Python list와 마찬가지로, nn.Module을 저장하는 역할을 한다. index로 접근도 할 수 있다.
            # 하지만 nn.Sequential과 다르게 forward() method가 없다.
            #또한, 안에 담긴 module 간의 connection도 없다.
            #우리는 nn.ModuleList안에 Module들을 넣어 줌으로써 Module의 존재를 PyTorch에게 알려 주어야 한다.
            #만약 nn.ModuleList에 넣어 주지 않고, Python list에만 Module들을 넣어 준다면, PyTorch는 이들의 존재를 알지 못한다.
            # 이 경우 optimzier을 선언하고 model.parameter()로 parameter을 넘겨줄 때 "your model has no parameter" 와 같은 error을 받게 된다.
            #따라서 Module들을 Python list에 넣어 보관한다면, 꼭 마지막에 이들을 nn.ModuleList로 wrapping 해줘야 한다.
            #두 종류의 module이 받는 input이 서로 다르고, 여러 개를 반복적으로 정의해야 할 때 유용하게 사용할 수 있다.
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Trend[i].weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))
                self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Seasonal[i].weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))
        else: #FAlse로 설정한경우 -> 우리는 여기 해당될듯?
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forcast_size) #한 개 층으로 이루어진 선형변환 통과
            #window_size는 input size, forcast_size는 output size
            self.Linear_Trend.weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))
            # [output_size * input_size]크기의 1/input_size 행렬을 input으로 torch.nn.Parameter 실행
            # transpose해주는 이유는 Transpose를 취하면 backward() 를 할 때 효율이 좋다는 말이 있음
            # 1/input_size 역시 비슷한 이유로 추정
            self.Linear_Seasonal = torch.nn.Linear(self.window_size,  self.forcast_size)
            self.Linear_Seasonal.weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))

    def forward(self, x):
        trend_init, seasonal_init = self.decompsition(x)
        #우선 decompsition을 통해 분해된 평균과 잔차를 추세와 계절로 받아줌
        trend_init, seasonal_init = trend_init.permute(0,2,1), seasonal_init.permute(0,2,1)
        #이후 계산의 편의를 위해 permute로 다시 차원 뒤집어줌
        if self.individual: #변수 한 개일 경우 차원을 맞춰주는 과정
            # 현재 중요한 부분은 아니니 그냥 무시하겠음
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forcast_size], dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forcast_size], dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])                
        else: #우리가 진행할 부분
            trend_output = self.Linear_Trend(trend_init) # 추세에 대해 선형변환 통과
            seasonal_output = self.Linear_Seasonal(seasonal_init) # 계절에 대해 선형변환 통과
        x = seasonal_output + trend_output #분해되었던 추세(이동평균)와 계절(잔차)을 다시 더해줌
        return x.permute(0,2,1) # 추세와 계절이 모두 forward 시작할 때 permute된 상태이므로 다시 돌려주고 반환

window_size = 168*2
forcast_size= 168
batch_size = 32
targets = '전력소비량(kWh)'
date = '일시'

# 이 과정에서 series_decomp라는 클래스를 먼저 통과함
# 이 클래스가 self.decompsition를 결정함
# series_decomp의 역할을 알아보자
# 근데 이때 필요한게 moving_avg라는 클래스임
# 아무래도 시계열의 이동평균법을 가져온거 같다.
# 두 클래스에서 공통적으로 받고 있는데 kernel_size

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

t1 = torch.rand(30,30)
t1.shape
t1 = t1.reshape(-1,30,30)
t1.shape
t1[:,0:1,:].repeat(1,(11 - 1) // 2, 1).shape
t1 = t1[:,0:1,:].repeat(1,(11 - 1) // 2, 1) # [1,5,30]
front = t1[:,0:1,:].repeat(1,(11 - 1) // 2, 1) #[1,5,30]
end = t1[:,-1,:].repeat(1,(11 - 1) // 2, 1) #[1,5,30]

front.shape
end.shape
#repeat은 각 차원에 해당하는 숫자만큼 반복해서 늘려줌
#1면 그대로
# x.repeat(1,4,1)이면 4에 해당하는 차원만 4배 증가
a = torch.cat([front,t1,end], dim = 1) #[1,15,30]
a.permute(0,2,1) # [1,30,15]
# permute를 통해 3번째 자리와 2번째 자리의 차원 변경
avg = torch.nn.AvgPool1d(11, 2, 0)
a = avg(a.permute(0,2,1))
a = a.permute(0,2,1)
a.shape #[1,3,30] -> torch.nn.AvgPool1d를 통과해 3으로 작아짐


class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual 
    
# moving_mean에서 앞에 moving_avg의 stride=1인 값을 받아옴
# x서 moving_mean을 빼서 residual(잔차)를 구하고
# moving_mean이랑 residual을 return해줌
# 즉, decomposition은 평균과 잔차로의 분해라서 이런 표현을 사용



######이제 Dlinear 코드를 파악했음
#표준화와 time_slide를 어떻게 구성했는지 확인
#굳이 이렇게 해야하나 싶어서 논문에서 꼭 커스텀 사용해라 말하는지 확인할 필요 있음
#상관 없는 부분이면 원래 알던대로 표준화하고 돌릴 예정
def standardization(train_df, test_df, not_col, target):
    #target은 y! 우리는 '전력소비량(kWh)'
    train_df_ = train_df.copy() #train 카피
    test_df_ = test_df.copy() #test 카피
    col =  [col for col in list(train_df.columns) if col not in [not_col]]
    #정규화 하지 않을 col을 제외하고 col_list를 받아옴 
    mean_list = []
    std_list = []
    #col별로 평균 분산 구해서 x-mean(x)/std(x)로 표준화하고
    #list에 저장
    for x in col:
        mean, std = train_df_.agg(["mean", "std"]).loc[:,x]
        mean_list.append(mean)
        std_list.append(std)
        train_df_.loc[:, x] = (train_df_[x] - mean) / std
        test_df_.loc[:, x] = (test_df_[x] - mean) / std
    # 표준화된 train_df, 표준화된 test_df, 평균,표준편차(인덱스는 target에서 떼와서 붙임) 반환
    return train_df_, test_df_, mean_list[col.index(target)], std_list[col.index(target)]

#이것도 굳이 이렇게 해야되나 싶음
#원래 가지고 있던대로 가도 될듯?
def time_slide_df(df, window_size, forcast_size, date, target):
    df_ = df.copy()
    data_list = []
    dap_list = []
    date_list = []
    for idx in range(0, df_.shape[0]-window_size-forcast_size+1):
        x = df_.loc[idx:idx+window_size-1, target].values.reshape(window_size, 1)
        y = df_.loc[idx+window_size:idx+window_size+forcast_size-1, target].values
        date_ = df_.loc[idx+window_size:idx+window_size+forcast_size-1, date].values
        data_list.append(x)
        dap_list.append(y)
        date_list.append(date_)
    return np.array(data_list, dtype='float32'), np.array(dap_list, dtype='float32'), np.array(date_list)

#얘는 필요하다 싶으면 사용하면 되겠다.
class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self): #클래스 len계산하려고 넣은 거
        return len(self.Y)
    
    def __getitem__(self, idx): #마찬가지로 idx로 데이터 가져오려고 넣은거
        return self.X[idx], self.Y[idx]
    
