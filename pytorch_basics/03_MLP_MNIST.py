import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
     
train_dataset = datasets.MNIST(root='MNIST_data/', train=True,  # 학습 데이터
                               transform=transforms.ToTensor(), # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                               download=True)

test_dataset = datasets.MNIST(root='MNIST_data/', train=False,  # 테스트 데이터
                              transform=transforms.ToTensor(),  # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                              download=True)

### 학습데이터를 0.85 : 0.15 비율로 나눠서 0.15는 검증데이터로
train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)
train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

class MyDeepLearningModel(nn.Module):
    
    def __init__(self):
        super().__init__
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, data):
        data = self.flatten(data)
        data = self.fc1(data)
        data = self.ReLU(data)
        data = self.dropout(data)
        logits = self.fc2(data)

        return logits
    
### DataLoader 정의
BATCH_SIZE = 32

train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = MyDeepLearningModel()
loss_function = nn.CrossEntropyLoss() ## 여기에 Softmax 함수가 포함되어 있음
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    ## 모든 배치의 누적 loss, 맞게 예측한 이미지 개수, 지금까지 확인한 전체 이미지 수 초기화
    total_loss_sum = total_correct = total_images = 0
    ## 전체 배치의 개수는 dataloder의 길이
    total_train_batch = len(dataloader)

    for images, labels in dataloder:

        x_train = images
        y_val = labels

        outputs = model(x_train) ## outputs는 모델이 예측한 확률을 가지는 텐서. 10개의 확률을 가지고 가장 큰 확률의 인덱스가 모델이 예측한 정답
        loss = loss_function(outputs, y_val)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() ## loss.item()은 배치 하나의 평균 loss. 그래서 배치마다 loss를 누적해서 더한다

        total_images += y_val.size(0) ## y_train은 이미지 32장의 실제 정답을 가진 텐서 ex) tensor([7, 2, 0, ..]). 그래서 y_train의 0번째 차원의 크기는 숫자들의 개수
        predictions = torch.argmax(outputs, dim = 1) ## outputs 텐서에서 가장 큰 수의 인덱스, 즉 예측한 숫자.
        ## dim = 1인 이유는, outputs 텐서의 shape이 [batch_size, 10]이기 때문에, dim = 1로 해야 각 이미지마다 10개의 클래스 중 가장 큰 값의 인덱스를 찾음
        ## dim = 0으로 하면, 각 숫자 클래스에 대해 현재 배치의 어떤 이미지가 가장 높은 점수를 받았는지를 찾음
        correct = predictions == y_val ## 예측한 숫자가 정답과 같으면 True 다르면 False로 해서 correct라는 텐서에 저장 ex) tensor([True, False, True, ...])
        total_correct += correct.sum().item() ## True는 1, False는 0으로 해서 더한값이 총 맞춘 개수

    train_avg_loss = total_loss_sum / total_train_batch ## 배치 하나당 평균 loss를 더한거를 전체 배치로 나눠서 평균 오차
    train_avg_accuracy = 100 * total_correct / total_images ## 맞게 예측한 이미지 개수를 전체 이미지 개수로 나눠서 평균 정확도

    return (train_avg_loss, train_avg_accuracy)

## model_evaluate()은 model.train()과 대부분 같고, 역전파 부분만 없음
def model_evaluate(dataloader, model, loss_function, optimizer):

    model.eval()

    with torch.no_grad(): ## 미분하지 않겠다

        total_loss_sum = total_correct = total_images = 0
        total_val_batch = len(dataloader)

        for images, labels in dataloder:

            x_val = images
            y_val = labels

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            total_loss_sum += loss.item()

            total_images += y_val.size(0)
            predictions = torch.argmax(outputs, dim = 1)
            correct = predictions == y_val
            total_correct += correct.sum().item()

        val_avg_loss = total_loss_sum / total_val_batch
        val_avg_accuracy = 100 * total_correct / total_images 

        return (val_avg_loss, val_avg_accuracy)



