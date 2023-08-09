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

import torch.nn as nn

### encoder layer ###


## encoder input
# x_s, x_t = SeriesDecomp(x)
# x_t = AvgPool(Padding(x))
# x_s = x - x_t

# x_ens, x_ent = SeriesDecomp(x_en(I/2:I))

# x_en(0) = x
# S_en(l,1), _ = SeriesDecomp(Auto-Correlation(x_en(l-1)) + x_en(l-1))
# S_en(l,2), _ = SeriesDecomp(FeedForward(s_en(l-1) + s_en(l,1))


#1) SeriesDecomp(X)



#### #전처리

test  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\test.csv")
train  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\train.csv")
building  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\building_info.csv",encoding="CP949")

# Train Data Preprocessing
train = pd.merge(train,building, on="건물번호")
train.isna().sum() # 결측치 확인
train.drop(['일조(hr)','일사(MJ/m2)'], axis=1, inplace=True) # 일조, 일사 컬럼 제거
train.drop(['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'], axis=1, inplace=True) # 태양광용량, ESS저장용량, PCS용량 컬럼 제거
#train.drop(['num_date_time', '건물번호'], axis=1, inplace=True) # num_date_time, 건물번호 컬럼 제거 
train.drop(['num_date_time'], axis=1, inplace=True) # num_date_time 컬럼 제거 
train['강수량(mm)'].fillna(0.0, inplace=True) # 강수량 결측치 0으로 채우기
train['풍속(m/s)'].fillna(round(train['풍속(m/s)'].mean(),2), inplace=True) # 풍속 결측치 평균으로 채워서 반올림
train['습도(%)'].fillna(round(train['습도(%)'].mean(),2), inplace=True) # 습도 결측치 평균으로 채워서 반올림


strc_type = pd.get_dummies(train['건물유형']) # 건물유형 더미변수화
train = pd.concat([train, strc_type], axis=1)
train.drop(['건물유형'], axis=1, inplace=True)

train["일시"] = pd.to_datetime(train["일시"])

print(train["일시"].min(), train["일시"].max())

train["time"] = train["일시"].dt.hour
train["month"] = train["일시"].dt.month
train["dayofweek"] = train["일시"].dt.dayofweek
train["day"] = train["일시"].dt.day


def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))

train['time_sin'] = sin_transform(train['time'])
train['time_cos'] = cos_transform(train['time'])
train['dayofweek_sin'] = sin_transform(train['dayofweek'])
train['dayofweek_cos'] = cos_transform(train['dayofweek'])
train['month_sin'] = sin_transform(train['month'])
train['month_cos'] = cos_transform(train['month'])
train['day_sin'] = sin_transform(train['day'])
train['day_cos'] = cos_transform(train['day'])

train = train.drop(['time', 'month', 'dayofweek', 'day'], axis=1)

train_1 = train[train['건물번호'] == 1]
train_1 = train_1.set_index(['일시'])

train_train = train_1.iloc[:-168*4,:]
train_valid = train_1.iloc[-168*4:]


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

train_train[['기온(C)','강수량(mm)','풍속(m/s)','습도(%)','연면적(m2)','냉방면적(m2)']] = min_max_scaler.fit_transform(train_train[['기온(C)','강수량(mm)','풍속(m/s)','습도(%)','연면적(m2)','냉방면적(m2)']])
train_valid[['기온(C)','강수량(mm)','풍속(m/s)','습도(%)','연면적(m2)','냉방면적(m2)']] = min_max_scaler.fit_transform(train_valid[['기온(C)','강수량(mm)','풍속(m/s)','습도(%)','연면적(m2)','냉방면적(m2)']])

tt_x = train_train.drop(["전력소비량(kWh)"], axis = 1)
tt_x.shape
tt_y = train_train[['건물번호','전력소비량(kWh)']]
tt_y[['전력소비량(kWh)']] = min_max_scaler.fit_transform(tt_y[['전력소비량(kWh)']])
tt_y = tt_y.drop(['건물번호'],axis=1)

tv_x = train_valid.drop(["전력소비량(kWh)"], axis = 1)
tv_y = train_valid[['건물번호','전력소비량(kWh)']]
tv_y[['전력소비량(kWh)']] = min_max_scaler.fit_transform(tv_y[['전력소비량(kWh)']])
tv_y = tv_y.drop(['건물번호'],axis=1)

tt_df = pd.concat([tt_x,tt_y], axis = 1)
tv_df = pd.concat([tv_x,tv_y], axis = 1)

pd.DataFrame(tt_x[0:3].mean()).T
###### 데이터 로더 생성 ###########

class windowDataset(Dataset):
    def __init__(self, x, y, input_window, output_window, num_features ,stride = 1):
        L = y.shape[0]
        num_samples = (L - input_window - output_window) // stride + 1

        X = np.zeros([input_window, num_samples, num_features])
        Y = np.zeros([output_window, num_samples, 1])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i,:] = x[start_x:end_x]

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


iw = 168*2
ow = 168

train_dataset = windowDataset(tt_x, tt_y, input_window=iw, output_window=ow,num_features=tt_x.shape[1] ,stride=1)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle= False)

valid_dataset = windowDataset(tv_x, tv_y, input_window=iw, output_window=ow,num_features=tv_x.shape[1] ,stride=1)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle= False)

import torch.nn as nn

class moving_avg(torch.nn.Module): #커널사이즈 만큼 평균, stride만큼 이동, 패딩은 0
    def __init__(self, kernel_size, stride):# kernel_size는 열 개수
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x): 
        front = torch.tensor(x[:,0:1, :].repeat(1,(self.kernel_size - 1) // 2, 1))
        # 시작을 kernel_size-1 // 2 만큼 늘려줌
        end = torch.tensor(x[:,-1:, :].repeat(1,(self.kernel_size - 1) // 2, 1))
        # 끝을 kernel_size -1 // 2 만큼 늘려줌
        x = torch.cat([front, torch.tensor(x), end], dim=1)
        # cat에 dim = 1을 통해 [,*,]에서 *자리에 늘어나도록 합쳐줌
        x = self.avg(x.permute(0, 2, 1))
        # permute()는 모든 차원들을 맞교환할 수 있다
        # 여기선 3번재, 2번째를 교환
        x = x.permute(0, 2, 1)
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
    

class LTSF_NLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, target_size):
        super(LTSF_NLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.target_size = target_size

        self.Linear_Trend = torch.nn.Linear(self.input_size, self.output_size) #한 개 층으로 이루어진 선형변환 통과
        self.Linear_Seasonal = torch.nn.Linear(self.input_size,  self.output_size)
        self.last_linear = torch.nn.Linear(27,self.target_size)



    def forward(self, x_mean, x_res):
        trend_init, seasonal_init = x_mean, x_res

        t_last = trend_init[:,-1:,:].detach()
        s_last = seasonal_init[:,-1:,:].detach()       

        trend_output, seasonal_output = trend_init - t_last, seasonal_init - s_last 
        trend_output, seasonal_output = trend_init.permute(0,2,1), seasonal_init.permute(0,2,1)
        trend_output = self.Linear_Trend(trend_output).permute(0,2,1)
        seasonal_output = self.Linear_Seasonal(seasonal_output).permute(0,2,1)


        trend_output = trend_output + t_last
        seasonal_output = seasonal_output + s_last

        x = seasonal_output + trend_output #분해되었던 추세(이동평균)와 계절(잔차)을 다시 더해줌
        output= self.last_linear(x)
        return output # 추세와 계절이 모두 forward 시작할 때 permute된 상태이므로 다시 돌려주고 반환  


'''
a,b = decomp(i[0])
t = a+b
t = t.permute(0,2,1)
t.shape
t_l = t[:,-1,:].detach()
t_l.shape
t = t - t_l
t.shape
t.permute(0,2,1).shape
t_lin = nn.Linear(336,168)
t.permute(0,2,1).shape
t= t_lin(t.float())
t_lin(t_l.float()).shape
z = t+t_lin(t_l.float())
z.shape
l_lin = nn.Linear(27,1)
l_lin(z.permute(0,2,1)).shape
'''

import torch.optim as optim

model = LTSF_NLinear(336,168,1)
decomp = series_decomp(27)
learning_rate=0.001
epoch = 1000
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

best_valid=  float('inf')
patient=0


from tqdm import tqdm

with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        total_valid_loss=0
        for x,y in train_loader:
            model.train()
            optimizer.zero_grad()
            x = x.float()
            y = y.float()
            x_m, x_r = decomp(x)
            output = model(x_m, x_r)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        for x,y in valid_loader:
            model.eval()
            with torch.no_grad():
                x = x.float()
                y = y.float()
                x_m, x_r = decomp(x)
                output = model(x_m, x_r)
            valid_loss = criterion(output, y)
            total_valid_loss += valid_loss.item()

        tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))
        if valid_loss<=best_valid:
            torch.save(model, r'C:\dacon\Dacon-PowerUsagePredict\js_park\best_AF_N.pth')
            patient=0
            best_valid=loss
        else:
            patient+=1
            if patient>=400:
                break

model = torch.load(r'C:\dacon\Dacon-PowerUsagePredict\js_park\best_AF_N.pth')


tv_y.index
output.shape
predict = output[8,:,:].detach().numpy()
predict = min_max_scaler.inverse_transform(predict)
real = y[8,:,:].detach().numpy()
real = min_max_scaler.inverse_transform(real)

final = pd.DataFrame({'predict' : predict.flatten(), 'real' : real.flatten()})

plt.figure(figsize=(10,5))
plt.plot(final['real'], label="real")
plt.plot(final['predict'], label="predict")

plt.title("Prediction")
plt.legend()
plt.show()

# Define SMAPE loss function
def SMAPE(true, pred):
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) * 100

SMAPE(real, predict)

#################test
test  = pd.read_csv(r"C:\dacon\Dacon-PowerUsagePredict\dataset\test.csv")

