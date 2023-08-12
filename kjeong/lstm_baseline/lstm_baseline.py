import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import
import pandas as pd
import numpy as np
import random
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

from tqdm.auto import tqdm


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


# data load
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')
train_df.head()
train_df.info()
train_df.shape


# train data preprocessing
# 일조, 일사 열 제거
train_df = train_df.drop(['일조(hr)','일사(MJ/m2)'], axis=1)
train_df.head()

# 결측치 확인
train_df.isna().sum()

# 강수량 결측치 0.0으로 채우기
train_df['강수량(mm)'].fillna(0.0, inplace=True)

# 풍속, 습도 결측치 평균으로 채우고 반올림하기
train_df['풍속(m/s)'].fillna(round(train_df['풍속(m/s)'].mean(),2), inplace=True)
train_df['습도(%)'].fillna(round(train_df['습도(%)'].mean(),2), inplace=True)

# date 컬럼 분해 (year는 의미가 없으므로 생략)
train_df['month'] = train_df['일시'].apply(lambda x : float(x[4:6]))
train_df['day'] = train_df['일시'].apply(lambda x : float(x[6:8]))
train_df['time'] = train_df['일시'].apply(lambda x : float(x[9:11]))

# 순서 재배치
train_df = train_df[train_df.columns[:7].to_list() + train_df.columns[8:].to_list() + train_df.columns[7:8].to_list()]
train_df.head()
train_df.shape


# hyperparameter setting
input_size = 8  # feature의 개수
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 5
window_size = 24  # 예측에 사용될 시간 윈도우 크기
batch_size = 64
learning_rate = 0.001


# dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size):
        # 데이터셋의 전처리를 해주는 부분
        self.df = df
        self.window_size = window_size

    def __len__(self):
        # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return len(self.df) - self.window_size
    
    def __getitem__(self, idx):
        # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
        x = torch.tensor(self.df[idx:idx+self.window_size, :], dtype=torch.float)
        if self.df.shape[1] > 1:
            y = torch.tensor(self.df[idx+self.window_size, -1], dtype=torch.float)
        else:
            y = None
        return x, y
    
def create_data_loader(df, window_size, batch_size):
    dataset = TimeSeriesDataset(df, window_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

# normalization
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_df.drop(['num_date_time', '건물번호', '일시'], axis=1).values)
train_loader = create_data_loader(train_data, window_size, batch_size)


# model define
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"current device: {device}")

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# train
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.unsqueeze(1).to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 300 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))


# test data preprocessing
# 학습 데이터에서 마지막 행 가져오기
last_train_data = train_df.drop(['num_date_time', '건물번호', '일시',], axis=1).loc[204000-24:,:]

# 실수형 데이터로 변환
test_df['습도(%)'] = test_df['습도(%)'].astype('float64')

# 날짜 데이터 추가
test_df['month'] = test_df['일시'].apply(lambda x : float(x[4:6]))
test_df['day'] = test_df['일시'].apply(lambda x : float(x[6:8]))
test_df['time'] = test_df['일시'].apply(lambda x : float(x[9:11]))

# 전력소비량 열 생성
final_df = pd.concat((test_df.drop(['num_date_time', '건물번호', '일시',], axis=1), pd.DataFrame(np.zeros(test_df.shape[0]))),axis=1)
final_df = final_df.rename({0:'전력소비량(kWh)'},axis=1)


# test dataset
test_df = pd.concat((last_train_data, final_df)).reset_index(drop=True)
test_data = scaler.transform(test_df.values) # train과 동일하게 scaling
test_data.shape

# Dataset & DataLoader
test_dataset = TimeSeriesDataset(test_data, window_size)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# inference
model.eval()
test_predictions = []
with torch.no_grad():
    for i in range(test_data.shape[0] - window_size):
        x = torch.Tensor(test_data[i:i+window_size,:]).to(device)
        new_x = model(x.view(1,window_size,-1))
        
        test_data[i+window_size,-1] = new_x # 입력 업데이트
        test_predictions.append(new_x.detach().cpu().numpy().item()) # 예측 결과 저장


# submit
predictions = scaler.inverse_transform(test_data)[24:,-1] # 원래 scale로 복구
sample_submission['answer'] = predictions
sample_submission.head()
sample_submission.to_csv('lstm_baseline_submission.csv', index=False)