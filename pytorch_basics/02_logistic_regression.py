import numpy as np
import torch
from pathlib import Path
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "logistic_regression_data.csv"

loaded_data = np.loadtxt(data_path, delimiter = ",")

x_train_np = loaded_data[:, 0:-1]
y_train_np = loaded_data[:, [-1]]

x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)

class MyLogisticRegressionModel(nn.Module):

    def __init__(self, input_nodes):
        super().__init__()
        self.logistic_stack = nn.Sequential(
            nn.Linear(input_nodes, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.logistic_stack(data)
        return prediction

model = MyLogisticRegressionModel(x_train.shape[1])

for param in model.parameters():
    print(param)

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

train_loss_list = []
train_accuracy_list = []
nums_epoch = 5000

for epoch in range(nums_epoch + 1):

    outputs = model(x_train)
    loss = loss_function(outputs, y_train)

    train_loss_list.append(loss.item())

    prediction = outputs > 0.5 ## 여기서 outputs 텐서의 값들 중 0.5 이상인 값을 true, 이하인 값을 false로 바꿈
    correct = prediction.float() == y_train ## y_train이 0과 1이기 때문에 true, false 를 1과 0으로 바꾸고 y_train과 비교함 -> 다시 boolean 텐서인 correct
    accuracy = correct.sum().item() / len(correct) ## correct텐서의 true, false를 sum()으로 자동으로 0과 1로 더함. 그 합 나누기 전체 데이터 개수로 정확도 계산
    ## 헷갈리지 말아야 할게, correct텐서는 맞게 예측한 개수가 아니라 각 데이터가 맞았는지를 저장한 텐서임. true, false만 가지고 있음

    train_accuracy_list.append(accuracy)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item(), ' accuracy = ', accuracy)
