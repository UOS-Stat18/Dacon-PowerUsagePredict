import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Import Module
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import tqdm
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm


# fixed random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42) # Seed 고정


# check cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Load Data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')
train.head()
train.info()
train.shape

building = pd.read_csv('./data/building_info.csv')
building.head()

train = pd.merge(train, building, on='건물번호') # 건물번호를 기준으로 두 데이터 병합
train.head()
train.shape

test = pd.merge(test, building, on='건물번호') 
test.head()
test.shape


# Train Data Preprocessing
train.isna().sum() # 결측치 확인
train.drop(['일조(hr)','일사(MJ/m2)'], axis=1, inplace=True) # 일조, 일사 컬럼 제거
train.drop(['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'], axis=1, inplace=True) # 태양광용량, ESS저장용량, PCS용량 컬럼 제거
train.drop(['num_date_time', '건물번호'], axis=1, inplace=True) # num_date_time, 건물번호 컬럼 제거 
train['강수량(mm)'].fillna(0.0, inplace=True) # 강수량 결측치 0으로 채우기
train['풍속(m/s)'].fillna(round(train['풍속(m/s)'].mean(),2), inplace=True) # 풍속 결측치 평균으로 채워서 반올림
train['습도(%)'].fillna(round(train['습도(%)'].mean(),2), inplace=True) # 습도 결측치 평균으로 채워서 반올림

strc_type = pd.get_dummies(train['건물유형']) # 건물유형 더미변수화
train = pd.concat([train, strc_type], axis=1)
train.drop(['건물유형'], axis=1, inplace=True)

# date 컬럼 분해 (year는 의미가 없으므로 생략)
train['month'] = train['일시'].apply(lambda x : float(x[4:6]))
train['day'] = train['일시'].apply(lambda x : float(x[6:8]))
train['time'] = train['일시'].apply(lambda x : float(x[9:11]))
train.drop(['일시'], axis=1, inplace=True)

# 시간 주기 컬럼 생성
month_periodic = (train['month'] - 1) / 12.0
day_periodic = (train['day'] - 1) / 31.0
time_periodic = train['time'] / 23.0

def sin_transform(values):
    return np.sin(2 * np.pi * values)
def cos_transform(values):
    return np.cos(2 * np.pi * values)

train['month_sin'] = sin_transform(month_periodic)
train['month_cos'] = cos_transform(month_periodic)
train['day_sin'] = sin_transform(day_periodic)
train['day_cos'] = cos_transform(day_periodic)
train['time_sin'] = sin_transform(time_periodic)
train['time_cos'] = cos_transform(time_periodic)

# 순서 재배치
train = train[train.columns[:4].to_list() + train.columns[5:].to_list() + train.columns[4:5].to_list()]
train.head()
train.columns


# Test Data Preprocessing
test.isna().sum() # 결측치 확인
test.drop(['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'], axis=1, inplace=True) # 태양광용량, ESS저장용량, PCS용량 컬럼 제거
test.drop(['num_date_time', '건물번호'], axis=1, inplace=True) # num_date_time, 건물번호 컬럼 제거 

strc_type = pd.get_dummies(test['건물유형']) # 건물유형 더미변수화
test = pd.concat([test, strc_type], axis=1)
test.drop(['건물유형'], axis=1, inplace=True)

# date 컬럼 분해 (year는 의미가 없으므로 생략)
test['month'] = test['일시'].apply(lambda x : float(x[4:6]))
test['day'] = test['일시'].apply(lambda x : float(x[6:8]))
test['time'] = test['일시'].apply(lambda x : float(x[9:11]))
test.drop(['일시'], axis=1, inplace=True)

# 시간 주기 컬럼 생성
month_periodic = (test['month'] - 1) / 12.0
day_periodic = (test['day'] - 1) / 31.0
time_periodic = test['time'] / 23.0

def sin_transform(values):
    return np.sin(2 * np.pi * values)
def cos_transform(values):
    return np.cos(2 * np.pi * values)

test['month_sin'] = sin_transform(month_periodic)
test['month_cos'] = cos_transform(month_periodic)
test['day_sin'] = sin_transform(day_periodic)
test['day_cos'] = cos_transform(day_periodic)
test['time_sin'] = sin_transform(time_periodic)
test['time_cos'] = cos_transform(time_periodic)


# Normalization
scaler = MinMaxScaler()
continuous_columns = ['기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '연면적(m2)', '냉방면적(m2)']
train[continuous_columns] = scaler.fit_transform(train[continuous_columns])
test[continuous_columns] = scaler.transform(test[continuous_columns])


# Sliding Window Dataset
class windowDataset(Dataset):
    def __init__(self, data, input_window, output_window, num_features, stride=1):
        
        L = data.shape[0]
        
        num_samples = (L - input_window - output_window) // stride + 1
        # stride씩 움직일 때 생기는 총 sample 개수
        # (204000 - 100 - 1) // 1 + 1 = 203900

        X = np.zeros([input_window, num_samples, num_features]) # 100, 203900, 28
        Y = np.zeros([output_window, num_samples, num_features]) # 1, 203900, 28

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = data[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = data[start_y:end_y]

        X = X.transpose((1,0,2))
        Y = Y.transpose((1,0,2))
        self.x = X
        self.y = Y
        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len

input_window = 100 # 100개의 데이터로
output_window = 1 # 1개의 데이터 예측
num_features = train.shape[1]
stride = 1

train_dataset = windowDataset(data=train, input_window=input_window, output_window=output_window, 
                              num_features = num_features, stride=stride)
train_loader = DataLoader(train_dataset, batch_size=64)


# Modeling
class lstm_encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden


for x, y in train_loader:
    break
x.shape
y.shape
input_size = x.shape[2]
hidden_size = 16
batch_size = x.shape[0]
target_len = output_window
encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
out, hidden = encoder(x.float())
out.shape
hidden[0].shape # hidden state
hidden[1].shape # cell state
x[:,-1,:].shape
    

class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out)
        return output, self.hidden


decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)
out, hidden = decoder(x[:,-1,:].float(), hidden)
out.shape
hidden[0].shape
hidden[1].shape
out.squeeze(1).shape
out.squeeze(1)[:,-1].unsqueeze(-1).shape

output = torch.zeros(batch_size, target_len, 1)
output[:,0,:] = out.squeeze(1)[:,-1].unsqueeze(-1)
output.shape


class lstm_encoder_decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(batch_size, target_len, 1)

        _, hidden = self.encoder(inputs)
        
        decoder_input = inputs[:, -1, :] # shape : (64,28)

        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1) 

            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out

            outputs[:,t,:] = out[:,-1].unsqueeze(-1)

        return outputs

    def predict(self, inputs, target_len):
        self.eval()
        
        inputs = inputs.unsqueeze(0)

        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(batch_size, target_len, input_size)

        _, hidden = self.encoder(inputs)

        decoder_input = inputs[:, -1, :]

        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            decoder_input = out
            outputs[:,t,:] = out

        return outputs[0,:,:].detach().numpy()


# Train
input_size = train.shape[1]
hidden_size = 16
model = lstm_encoder_decoder(input_size=input_size, hidden_size=hidden_size)
learning_rate=0.01
epoch = 1000
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()
best_valid = float('inf')
patient = 0

model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x, y in train_loader:

            optimizer.zero_grad()
            
            x = x.float()
            y = y.float()
            y = y[:,:,-1].unsqueeze(-1)

            output = model(inputs=x, targets=y, target_len=output_window, 
                           teacher_forcing_ratio=0.6)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        tr.set_postfix(loss = "{0:.5f}".format(total_loss/len(train_loader)))

        if loss <= best_valid:
            torch.save(model, 'best.pth') # pytorch에서는 모델을 저장할 때 .pt 또는 .pth 확장자를 사용
            patient += 0
            best_valid = loss
        else:
            patient += 1
            if patient >= 30:
                break

model = torch.load("best.pth", map_location=device)
pred = torch.tensor(train.iloc[-input_window:].values).float()
predict = model.predict(pred, target_len=output_window)
real = test.iloc[:, :-6].to_numpy()

predict = scaler.inverse_transform(predict)
real = scaler.inverse_transform(real)

predict = predict[: ,0]
real = real[: ,0]


# Visualization

plt.figure(figsize=(20,5))
plt.plot(range(365), real, label="real")
plt.plot(range(365), predict, label="predict")

plt.title("Test Set")
plt.legend()
plt.show()

def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPEval(predict, real)

