import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"using PyTorch version:  {torch.__version__}, Device : {DEVICE}")

FEATURE_NUMS = 4        # 입력층으로 들어가는 데이터 개수 feature
SEQ_LENGTH = 5          # 정답을 만들기 위해 필요한 시점 개수 time step
HIDDEN_SIZE = 4         # 한 hidden_state의 크기. 차원 개수.
NUM_LAYERS = 1          # RNN 계열 계층이 몇겹으로 쌓여있는지.
LEARNING_RATE = 1e-3    # 학습율
BATCH_SIZE = 20         # 학습을 위한 배치 하나당 몇개의 원소가 있는지.

EPOCHS = 200

df = fdr.DataReader('005930', '2020-01-01', '2024-06-30')
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]
df.head(10)

# train : test - 70 : 30 분리

SPLIT = int(0.7*len(df))  # train : test = 7 : 3

train_df = df[ :SPLIT ]
test_df = df[ SPLIT: ]


scaler_x = MinMaxScaler()  # feature scaling
train_df.iloc[:, :-1] = scaler_x.fit_transform(train_df.iloc[:, :-1])
test_df.iloc[:, :-1] = scaler_x.transform(test_df.iloc[:, :-1])
## train만 fit을 해서 test, 즉 미래 정보가 학습하는데에 쓰이지 않게함.
## test에도 fit_transform을 해버리면 미래 주식정보의 min/max를 알게되서 그 min/max에 맞게 scaling 되어버림

scaler_y = MinMaxScaler()  # label scaling
train_df.iloc[ : , -1 ] = scaler_y.fit_transform(train_df.iloc[ : , [-1] ])
test_df.iloc[ : , -1 ] = scaler_y.transform(test_df.iloc[ : , [-1] ])
## x와 y에 서로 다른 두 scaler를 쓴 이유는 scaler 객체가 자기 안에 학습한 min/max 정보를 저장하기 때문에,
## x에 쓴 scaler를 y에 다시 쓰면 y의 min/max를 기억하는 scaler가 되어버림.
## 그래서 나중에 다른 x데이터가 들어오면 그걸 예전 기준에 맞춰서 scaling해야했을때, x를 scaling했던 scaler는 y의 min/max만 기억하고 있기때문에 처리 할수 없음.



## 데이터 정리해서 준비하기
def MakeSeqNumpyData(data, seq_length):

    x_seq_list = []
    y_seq_list = []

    for i in range(len(data) - seq_length):
        x_seq_list.append(data[ i:i+seq_length, :-1 ])
        y_seq_list.append(data[ i+seq_length, [-1] ])

    x_seq_numpy = np.array(x_seq_list)
    y_seq_numpy = np.array(y_seq_list)

    return x_seq_numpy, y_seq_numpy

    
## 정리한 데이터 확인하기
x_train_data, y_train_data = MakeSeqNumpyData(np.array(train_df), SEQ_LENGTH)
x_test_data, y_test_data = MakeSeqNumpyData(np.array(test_df), SEQ_LENGTH)

print(x_train_data.shape, y_train_data.shape)
print(x_test_data.shape, y_test_data.shape)


## DataLoader 세팅
x_train_tensor = torch.FloatTensor(x_train_data).to(DEVICE)
y_train_tensor = torch.FloatTensor(y_train_data).to(DEVICE)

x_test_tensor = torch.FloatTensor(x_test_data).to(DEVICE)
y_test_tensor = torch.FloatTensor(y_test_data).to(DEVICE)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)



class MyLSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, data):
        h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)

        outputs, (h_n, c_n) = self.lstm(data, (h0, c0)) # lstm의 리턴값은 마지막 레이어의 모든 timestep의 hidden state를 모아놓은 outputs -> shape은 [B, T, H × D]
                                               # 그리고 모든 layer/direction에서 마지막에 남은 hidden state들을 가지는 h_n -> shape은 [L × D, B, H]
                                               # 그리고 모든 layer/direction에서 마지막에 남은 cell state들을 가지는 c_n -> shape은 [L × D, B, H]
        last_hs = outputs[:, -1, :] # 5일 sequence → 다음 날 Close 하나를 예측하기 때문에 sequence 하나당 출력 하나가 필요, 즉 sequence를 다 읽고 난 뒤 맨 마지막 hidden state를 가져옴
        prediction = self.fc(last_hs) # 가져온 마지막 hidden state를 Linear층 통과 시켜서 예측값

        return prediction


model = MyLSTMModel(FEATURE_NUMS, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    ## 모든 배치의 loss 합과 지금까지 확인한 전체 sequence 개수 초기화
    total_loss_sum = 0
    total_sequences = 0

    for sequences, targets in dataloader:

        ## sequences에는 5일치 Open, High, Low, Volume 데이터가 들어있음
        ## shape: [batch_size, sequence_length, input_size]
        ## 현재 설정에서는 보통 [20, 5, 4]
        x_train = sequences.to(DEVICE)

        ## targets에는 각 5일 sequence 다음 날의 Close 정답이 들어있음
        ## shape: [batch_size, 1]
        ## 현재 설정에서는 보통 [20, 1]
        y_train = targets.to(DEVICE)

        ## 5일짜리 sequence를 LSTM에 입력하여 다음 날 Close 예측
        ## outputs shape: [batch_size, 1]
        ## 현재 설정에서는 보통 [20, 1]
        outputs = model(x_train)

        ## 모델이 예측한 scaled Close와 실제 scaled Close 사이의 MSE loss 계산
        ## nn.MSELoss()의 기본 reduction='mean'이므로
        ## loss는 현재 배치에 속한 sequence들의 평균 loss
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## loss.item()은 현재 batch의 평균 loss
        ## 여기에 현재 batch의 sequence 개수를 곱해서
        ## 현재 batch의 loss 합으로 바꾼 뒤 epoch 전체에 누적
        total_loss_sum += loss.item() * y_train.size(0)

        ## 현재 batch에 들어있는 sequence 개수 누적
        ## 보통 20개지만 마지막 batch는 더 작을 수 있음
        total_sequences += y_train.size(0)

    ## epoch 전체 sequence에 대한 평균 loss
    train_avg_loss = total_loss_sum / total_sequences

    return train_avg_loss


def model_evaluate(dataloader, model, loss_function):

    model.eval()

    total_loss_sum = 0
    total_sequences = 0

    prediction_list = []
    target_list = []

    with torch.no_grad():

        for sequences, targets in dataloader:

            sequences = sequences.to(DEVICE)
            targets = targets.to(DEVICE)
            ## 각 5일 sequence에 대한 다음 날 Close 예측
            ## shape: [batch_size, 1]
            predictions = model(sequences)

            ## 예측값과 실제 Close 사이의 MSE loss
            loss = loss_function(predictions, targets)

            ## epoch 전체 평균 loss 계산을 위한 누적
            total_loss_sum += loss.item() * targets.size(0)
            total_sequences += targets.size(0)

            ## 그래프 및 최종 평가를 위해 batch별 결과 저장
            prediction_list.append(predictions)
            target_list.append(targets)

    test_avg_loss = total_loss_sum / total_sequences

    ## 여러 batch의 결과를 다시 전체 test 데이터 순서로 결합
    test_pred_tensor = torch.cat(prediction_list, dim=0)
    test_target_tensor = torch.cat(target_list, dim=0)

    return test_avg_loss, test_pred_tensor, test_target_tensor


## 모델 학습
train_loss_list = []

start_time = datetime.now()

for epoch in range(EPOCHS):

    train_avg_loss = model_train(
        train_loader,
        model,
        loss_function,
        optimizer
    )

    train_loss_list.append(train_avg_loss)

    ## 10 epoch마다 현재 평균 loss 확인
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Train Loss: {train_avg_loss:.6f}"
        )

end_time = datetime.now()

print(f"Elapsed Time: {end_time - start_time}")


## 학습된 모델을 test data로 평가
test_avg_loss, test_pred_tensor, test_target_tensor = model_evaluate(
    test_loader,
    model,
    loss_function
)

print(f"Test Loss: {test_avg_loss:.6f}")

## scaling된 test 예측값 tensor를 NumPy 배열로 변환. scaler로 다시 복원하려고 inverse_transform을 해야하는데 NumPy를 인자로 받기 때문.
test_pred_numpy = test_pred_tensor.cpu().numpy()

## scaling된 실제 test 정답 tensor를 NumPy 배열로 변환
test_target_numpy = test_target_tensor.cpu().numpy()

## scaling된 값을 원래 실제 주가 단위로 복원
pred_inverse = scaler_y.inverse_transform(test_pred_numpy)
y_test_inverse = scaler_y.inverse_transform(test_target_numpy)


## 실제 test Close와 모델이 예측한 Close를 같은 그래프에 표시
## x축: test sequence의 순서
## y축: inverse_transform으로 복원한 실제 주가
plt.plot(y_test_inverse, label="Actual")
plt.plot(pred_inverse, label="Prediction")

plt.xlabel("Test Sequence")
plt.ylabel("Close Price")
plt.title("Samsung Electronics Stock Price Prediction (LSTM)")

plt.grid()
plt.legend()

plt.show()
