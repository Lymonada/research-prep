## Experiment Question

이 실험의 목적은 MNIST 이미지 분류에서 **모델 구조와 학습 설정이 성능에 미치는 영향**을 비교하는 것이다.

구체적으로 다음 질문에 답하고자 한다.

1. 동일한 optimizer와 learning rate를 사용했을 때 CNN이 MLP보다 높은 분류 성능을 보이는가?
2. 동일한 모델에서 `SGD, lr=0.01`과 `Adam, lr=0.001`을 사용할 때 학습 과정과 최종 성능이 어떻게 달라지는가?
3. MLP와 CNN의 성능 차이가 모델 구조에서 주로 발생하는지, 아니면 optimizer와 learning rate를 포함한 학습 설정에서도 크게 영향을 받는지 확인한다.
4. 각 실험의 validation loss와 validation accuracy 변화를 비교하여 수렴 속도와 학습 안정성의 차이를 확인한다.

이번 실험에서는 optimizer와 learning rate가 함께 변경되므로, 두 설정의 차이를 단순히 optimizer만의 효과라고 해석하지 않고 **학습 설정 전체의 효과**로 해석한다.  

## Controlled Variables

네 가지 실험에서 다음 조건은 동일하게 유지한다.

- Dataset: MNIST
- 입력 전처리: `transforms.ToTensor()`
- 학습·검증 데이터 분할 비율: 85% / 15%
- Test dataset: MNIST 공식 test dataset
- Random seed: 42
- Batch size: 32
- Epochs: 20
- Loss function: `nn.CrossEntropyLoss()`
- Train/validation/test loss 계산 방식
- Train/validation/test accuracy 계산 방식
- 동일한 train/validation 데이터 분할
- 동일한 학습 데이터 shuffle 순서
- 동일한 실행 장치
- 동일한 성능 기록 및 그래프 저장 방식

동일한 seed와 generator를 사용하여 코드를 다시 실행해도 같은 MNIST 이미지가 train set과 validation set에 포함되고, 같은 순서로 학습 데이터가 제공되도록 한다.

모든 실험에서 epoch별 train loss, validation loss, train accuracy, validation accuracy를 동일한 방식으로 계산하고 기록한다. 최종 평가는 동일한 test dataset을 사용하여 수행한다. 

단, 다음 요소는 실험에서 비교하기 위해 변경한다.

- Model architecture: MLP 또는 CNN
- Optimizer: SGD 또는 Adam
- Learning rate: 0.01 또는 0.001

## Experiment Settings

### Model Architecture

#### MLP

MLP는 MNIST 이미지의 `[1, 28, 28]` 형태를 784개의 값으로 펼친 뒤 fully connected layer를 통과하도록 구성한다.

- Flatten
- Linear: 784 → 256
- ReLU
- Dropout: 0.3
- Linear: 256 → 10

#### CNN

CNN은 두 개의 convolution layer를 사용하여 이미지의 공간적 특징을 추출한 뒤 fully connected layer를 통해 10개 클래스를 분류한다.

- Conv2d: 1 → 32, kernel size 3, padding 1
- ReLU
- MaxPool2d: kernel size 2, stride 2
- Dropout: 0.25
- Conv2d: 32 → 64, kernel size 3, padding 1
- ReLU
- MaxPool2d: kernel size 2, stride 2
- Dropout: 0.25
- Flatten: `64 × 7 × 7`
- Linear: 3136 → 256
- ReLU
- Dropout: 0.5
- Linear: 256 → 10

:contentReference[oaicite:3]{index=3}

### Training Combinations

| Experiment ID | Model | Optimizer | Learning Rate | Purpose |
|---|---|---|---:|---|
| `MLP_SGD_lr0.01` | MLP | SGD | 0.01 | 기존 MLP 학습 설정 |
| `MLP_Adam_lr0.001` | MLP | Adam | 0.001 | MLP에서 학습 설정 변경 효과 확인 |
| `CNN_SGD_lr0.01` | CNN | SGD | 0.01 | CNN에서 학습 설정 변경 효과 확인 |
| `CNN_Adam_lr0.001` | CNN | Adam | 0.001 | 기존 CNN 학습 설정 |

### Main Comparisons

#### 동일한 모델에서 학습 설정 비교

- `MLP_SGD_lr0.01` vs. `MLP_Adam_lr0.001`
- `CNN_SGD_lr0.01` vs. `CNN_Adam_lr0.001`

이를 통해 동일한 모델에서 학습 설정이 수렴 속도와 최종 성능에 미치는 영향을 확인한다.

#### 동일한 학습 설정에서 모델 구조 비교

- `MLP_SGD_lr0.01` vs. `CNN_SGD_lr0.01`
- `MLP_Adam_lr0.001` vs. `CNN_Adam_lr0.001`

이를 통해 optimizer와 learning rate를 동일하게 유지했을 때 MLP와 CNN의 구조적 차이가 성능에 미치는 영향을 확인한다.

## Final Results

## Learning Curve Comparison

## Findings
1. Effect of training setting on MLP
2. Effect of training setting on CNN
3. MLP vs CNN under SGD
4. MLP vs CNN under Adam

## Limitations
