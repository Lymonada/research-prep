import numpy as np
import torch
from pathlib import Path
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "linear_regression_data.csv"

loaded_data = np.loadtxt(data_path, delimiter = ",")

x_train_np = loaded_data[:, 0:-1]
y_train_np = loaded_data[:, [-1]]

x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)

class MyLinearRegressionModel(nn.Module):

    def __init__(self, input_nodes):
        super.__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_nodes, 1)
        )

    def forward(self, data):
        prediction = self.linear_stack(data)
        return prediction

model = MyLinearRegressionModel(x_train.shape[1])

for name, child in model.named_children():
    for param in child.parameters():
        print(name, param)

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

loss_list = []
nums_epoch = 2000

for epoch in range(nums_epoch + 1):

    prediction = model(x_train)
    loss = loss_function(prediction, y_train)

    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item())
