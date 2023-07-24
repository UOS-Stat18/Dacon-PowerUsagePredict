import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import plotly.express as px

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm
tqdm.pandas

import warnings
warnings.filterwarnings('ignore')

import random

#smape는 이거 변환해서 사용하면 될듯
"""
SMAPE() 함수는 데이터를 입력하면 SMAPE 값을 반환하는 사용자 정의 함수입니다.
입력으로 들어가는 data는 Dataframe의 형태이며 실제 정답 값은 Real 이라는 Column으로 예측 값은 Prediction 이라는 Column으로 저장되어 있습니다.
"""

def SMAPE(data):
    data["Symmetric Absolute Percentage Error"] = abs((data["Real"] - data["Prediction"]))/((abs(data["Real"]) + abs(data["Prediction"])) / 2) * 100
    smape = data["Symmetric Absolute Percentage Error"].mean()
    return smape

# data check

test  = pd.read_csv(r"C:\daycon\Dacon-PowerUsagePredict\dataset\test.csv")
train  = pd.read_csv(r"C:\daycon\Dacon-PowerUsagePredict\dataset\train.csv")
building  = pd.read_csv(r"C:\daycon\Dacon-PowerUsagePredict\dataset\building_info.csv",
                        encoding="CP949")

test.head()
train.head()
building.head()

# building na 확인
# 건물유형은 요일처럼 태깅해서 특성으로 바꿔주고
# 연면적이랑 냉방면적을 특성으로 활용
# 전체 행 개수가 100개라 의미있게 사용이 가능할까?
building[building['태양광용량(kW)'] == '-'].shape[0] #64
building[building['ESS저장용량(kWh)'] == '-'].shape[0] #95
building[building['PCS용량(kW)'] == '-'].shape[0] #95

# 건물번호 1에 대해 간단한 모델 생성
# test_1(건물번호 1번 생성)
train_1 = train[train['건물번호'] == 1]
test_1 = test[test['건물번호'] == 1]
test_1.shape

#데이터 대략적인 모양 확인
print(train_1.shape)
train_1.describe()


plt.figure(figsize=(20,5))
plt.plot(range(len(train_1)), train_1["전력소비량(kWh)"])

# 시작 및 종료 일시 확인
train_1.head() # 시작일시 : 20220601 00
train_1.tail() # 종료일시 : 20220824 23

# 예측해야하는 test 시작 종료 확인
test.head() # 시작일시 : 20220825 00
test.tail() # 종료일시 : 20220831 23

##데이터 전처리
# 일시 -> 연,월,주,일 로 분해하여 feature 생성
train_1["일시"] = pd.to_datetime(train_1["일시"])

train_1["year"] = train_1["일시"].dt.year
train_1["month"] = train_1["일시"].dt.month
train_1["dayofweek"] = train_1["일시"].dt.dayofweek
train_1["day"] = train_1["일시"].dt.day

#체크
train_1.head() # 잘 변경되었음

# 사인/코사인 변환
def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))

train_1['dayofweek_sin'] = sin_transform(train_1['dayofweek'])
train_1['dayofweek_cos'] = cos_transform(train_1['dayofweek'])
train_1['month_sin'] = sin_transform(train_1['month'])
train_1['month_cos'] = cos_transform(train_1['month'])
train_1['day_sin'] = sin_transform(train_1['day'])
train_1['day_cos'] = cos_transform(train_1['day'])

#변환 후 원래 변수들 삭제
train_1 = train_1.drop(['year', 'month', 'dayofweek', 'day'], axis=1)

#체크ㅋ
train_1.head() # 잘 변경되었음


#train_1을 test에 맞게 일주일치 빼서 정리
train_train = train_1[:-168]
train_test = train_1[-168:]

#일단 연습이니까 결측치 없는 애들만 다루자
train_train = train_train.drop(['num_date_time','일시','건물번호','강수량(mm)','일조(hr)','일사(MJ/m2)'], axis=1)
train_test = train_test.drop(['num_date_time','일시','건물번호','강수량(mm)','일조(hr)','일사(MJ/m2)'], axis=1)


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

train_train[['기온(C)',"풍속(m/s)","습도(%)","전력소비량(kWh)"]] = min_max_scaler.fit_transform(train_train[['기온(C)',"풍속(m/s)","습도(%)","전력소비량(kWh)"]])
train_test[['기온(C)',"풍속(m/s)","습도(%)","전력소비량(kWh)"]] = min_max_scaler.fit_transform(train_test[['기온(C)',"풍속(m/s)","습도(%)","전력소비량(kWh)"]])


train_train

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

train_dataset = windowDataset(train_train, input_window=iw, output_window=ow,num_features=train_train.shape[1] ,stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)

import torch.nn as nn
# Lstm encoder
class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden



class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.view([-1,1,10]), encoder_hidden_states)

        output = self.linear(lstm_out)

        return output, self.hidden



class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size 

        self.encoder = lstm_encoder(input_size= input_size, hidden_size= hidden_size)
        self.decoder = lstm_decoder(input_size= input_size, hidden_size= hidden_size)
    
    def forward(self, inputs, targets, target_len, teacher_forching_ratio):
        batch_size = inputs.shape[0] 
        input_size = inputs.shape[2] 

        outputs = torch.zeros(batch_size, target_len, input_size)

        _,hidden = self.encoder(inputs) 
        decoder_input = inputs[:, -1, :] 


        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)

            if random.random() < teacher_forching_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:, t, :] = out
        
        return outputs


    def predict(self, inputs, target_len):
        self.eval()
        inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0] 
        input_size = inputs.shape[2] 
        outputs = torch.zeros(batch_size, target_len, input_size) 
    
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1,:]
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)
            decoder_input = out
            outputs[:, t, :] = out
        return outputs

# Train
import torch.optim as optim

model = lstm_encoder_decoder(input_size=10, hidden_size=64)

learning_rate=0.01
epoch = 100
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

best_valid=  float('inf')
patient=0

from tqdm import tqdm

model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x,y in train_loader:
            optimizer.zero_grad()
            x = x.float()
            y = y.float()
            output = model(x, y, ow, 0.6)
            ####바뀐 부분#######
            a = output[-1][:,0:4].detach().numpy()
            a = a[:,3]
            b = y[-1][:,0:4].detach().numpy()
            b = b[:,3]
            ####################
            loss = criterion(torch.from_numpy(a), torch.from_numpy(b))
            ###추가###
            loss.requires_grad_(True)
            #########
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))
        if loss<=best_valid:
            torch.save(model, r'C:\daycon\Dacon-PowerUsagePredict\js_park\best_seq1.pth')
            patient=0
            best_valid=loss
        else:
            patient+=1
            if patient>=10:
                break


model = torch.load(r'C:\daycon\Dacon-PowerUsagePredict\js_park\best_seq1.pth')
model.eval()

pred_input = torch.tensor(train_train.iloc[-168:].values).float()
predict = model.predict(pred_input, target_len=ow)[-1][:,0:4].detach().numpy()

real = train_test.iloc[:,0:4].to_numpy()

predict = min_max_scaler.inverse_transform(predict)
real = min_max_scaler.inverse_transform(real)

predict = predict[:,3]
real = pd.DataFrame(real).iloc[:,3].to_numpy()

final = pd.DataFrame({'predict' : predict, 'real' : real})
final

predict.min()
predict.max()
real.min()
real.max()


plt.figure(figsize=(10,5))
plt.plot(final['real'], label="real")
plt.plot(final['predict'], label="predict")

plt.title("Prediction")
plt.legend()
plt.show()


def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPEval(predict,real)
