import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import csv
from pathlib import Path
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") ## GPU를 사용할 수 있으면 cuda사용
print(f"Device : {DEVICE}")

## 결과를 저장할 폴더와 파일 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "MNIST"
RESULT_DIR = PROJECT_ROOT / "results" / "mnist"
PLOT_DIR = PROJECT_ROOT / "plots" / "mnist"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


## 시드를 설정해서 이후에 똑같이 재현할 수 있게 함 
SEED = 42 
torch.manual_seed(SEED) 

train_dataset = datasets.MNIST(root=DATA_DIR, train=True,  # 학습 데이터
                               transform=transforms.ToTensor(), # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                               download=True)

test_dataset = datasets.MNIST(root=DATA_DIR, train=False,  # 테스트 데이터
                              transform=transforms.ToTensor(),  # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                              download=True)


### 학습데이터를 0.85 : 0.15 비율로 나눠서 0.15는 검증데이터로
train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)


# train/validation 분할을 동일하게 재현하기 위한 generator -> 코드를 다시 실행해도 동일한 MNIST 이미지들이 train과 validation에 들어가도록 함.
split_generator = torch.Generator().manual_seed(SEED)

train_dataset, validation_dataset = random_split(
    train_dataset,
    [train_dataset_size, validation_dataset_size],
    generator=split_generator
)


class MyCNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.pooling = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout25 = nn.Dropout(0.25)
        self.dropout50 = nn.Dropout(0.5)

    def forward(self, data):
        data = self.conv1(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = self.conv2(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = data.view(-1, 7 * 7 * 64)

        data = self.fc1(data)
        data = torch.relu(data)
        data= self.dropout50(data)

        logits = self.fc2(data)

        return logits

### DataLoader 정의
BATCH_SIZE = 32

train_loader_generator = torch.Generator().manual_seed(SEED) ## train_dataset_loader는 shuffle=true이기 때문에 같은 데이터 순서로 학습하도록 시드 설정
train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=train_loader_generator)

validation_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

LEARNING_RATE = 1e-3
model = MyCNNModel().to_(DEVICE)
loss_function = nn.CrossEntropyLoss() ## 여기에 Softmax 함수가 포함되어 있음
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)



def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    ## 모든 배치의 누적 loss, 맞게 예측한 이미지 개수, 지금까지 확인한 전체 이미지 수 초기화
    total_loss_sum = total_correct = total_images = 0

    for images, labels in dataloader:

        x_train = images.to(DEVICE) # [32, 1, 28, 28]
        y_train = labels.to(DEVICE) # [32]

        outputs = model(x_train) 
        ## outputs의 각 행에는 이미지 한장에 대한 10개의 숫자 클래스의 logits가 저장됨 
        ## 가장 큰 logit을 가진 클래스가 모델의 예측값임
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * y_train.size(0) ## loss.item()은 배치 하나의 평균 loss. 거기에 배치 하나의 이미지 수를 곱해서 배치 하나의 loss 합을 구한 후 누적

        total_images += y_train.size(0) ## y_train은 이미지 32장의 실제 정답을 가진 텐서 ex) tensor([7, 2, 0, ..]). 그래서 y_train의 0번째 차원의 크기는 숫자들의 개수
        predictions = torch.argmax(outputs, dim = 1) ## outputs 텐서에서 가장 큰 수의 인덱스, 즉 예측한 숫자.
        ## dim = 1인 이유는, outputs 텐서의 shape이 [batch_size, 10]이기 때문에, dim = 1로 해야 각 이미지마다 10개의 클래스 중 가장 큰 값의 인덱스를 찾음
        ## dim = 0으로 하면, 각 숫자 클래스에 대해 현재 배치의 어떤 이미지가 가장 높은 점수를 받았는지를 찾음
        correct = predictions == y_train ## 예측한 숫자가 정답과 같으면 True 다르면 False로 해서 correct라는 텐서에 저장 ex) tensor([True, False, True, ...])
        total_correct += correct.sum().item() ## True는 1, False는 0으로 해서 더한값이 총 맞춘 개수

    train_avg_loss = total_loss_sum / total_images ## 모든 이미지의 loss합을 전체 이미지 개수로 나눠서 epoch 전체 이미지에 대한 평균 loss
    train_avg_accuracy = 100 * total_correct / total_images ## 맞게 예측한 이미지 개수를 전체 이미지 개수로 나눠서 epoch 전체 이미지에 대한 정확도

    return (train_avg_loss, train_avg_accuracy)

## 평가에서는 모델을 eval 모드로 전환해 Dropout을 하지않고, gradient 계산과 가중치 업데이트를 수행 X
def model_evaluate(dataloader, model, loss_function):

    model.eval()

    with torch.no_grad():

        total_loss_sum = total_correct = total_images = 0

        for images, labels in dataloader:

            x_val = images.to(DEVICE)
            y_val = labels.to(DEVICE)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            total_loss_sum += loss.item() * y_val.size(0)

            total_images += y_val.size(0)
            predictions = torch.argmax(outputs, dim = 1)
            correct = predictions == y_val
            total_correct += correct.sum().item()

        val_avg_loss = total_loss_sum / total_images
        val_avg_accuracy = 100 * total_correct / total_images 

        return (val_avg_loss, val_avg_accuracy)



train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []


## 데이터로 학습
EPOCHS = 20
for epoch in range(EPOCHS): 
    
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)
    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)

    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function)
    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)



## 테스트 데이터로 테스트
test_avg_loss, test_avg_accuracy = model_evaluate(
    test_dataset_loader,
    model,
    loss_function
)

print(f"Test Accuracy: {test_avg_accuracy:.2f}%")
print(f"Test Loss: {test_avg_loss:.4f}")



## 학습 가능한 총 파라미터 수
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)



## 어떤걸 결과에 저장할지 결정
result = {
    "model": "CNN",
    "dataset": "MNIST",
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "parameters": trainable_parameters,
    "test_loss": test_avg_loss,
    "test_accuracy": test_avg_accuracy
}

fieldnames = [
    "model",
    "dataset",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "parameters",
    "test_loss",
    "test_accuracy"
]



## CSV 파일에 결과 기록
log_path = RESULT_DIR / "model_comparison.csv"
write_header = not log_path.exists()

with log_path.open("a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    if write_header:
        writer.writeheader()

    writer.writerow(result)



epoch_list = range(1, EPOCHS + 1)
loss_plot_path = PLOT_DIR / "CNN_loss.png"
accuracy_plot_path = PLOT_DIR / "CNN_accuracy.png"

## Loss 그래프 그리기
plt.figure()

plt.plot(epoch_list, train_loss_list, label="Train Loss")
plt.plot(epoch_list, val_loss_list, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN MNIST Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(loss_plot_path)
plt.show()
plt.close()

## Accuracy 그래프 그리기
plt.figure()

plt.plot(epoch_list, train_accuracy_list, label="Train Accuracy")
plt.plot(epoch_list, val_accuracy_list, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN MNIST Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(accuracy_plot_path)
plt.show()
plt.close()
