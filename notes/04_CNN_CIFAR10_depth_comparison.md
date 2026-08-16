# CIFAR-10 CNN Depth Comparison

## 1. Experiment Purpose

CIFAR-10 이미지 분류에서 기본 CNN보다 convolution layer를 더 깊게 구성했을 때 분류 성능이 어떻게 달라지는지 확인한다.

이번 실험에서는 데이터셋과 학습 설정은 동일하게 유지하고, CNN의 구조를 변경하여 비교했다.

---

## 2. Dataset

* Dataset: CIFAR-10
* Image shape: `[3, 32, 32]`
* Number of classes: 10
* Train / Validation split: 85% / 15%
* Test set: CIFAR-10 official test set
* Transform: `transforms.ToTensor()`

MNIST와 달리 CIFAR-10은 RGB 3채널의 실제 사물 이미지로 구성되어 있어 이미지의 색상, 배경, 형태가 더 다양하다.

---

## 3. Common Training Settings

두 모델에서 다음 학습 조건을 동일하게 유지했다.

* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 64
* Epochs: 100
* Seed: 42
* Loss function: `nn.CrossEntropyLoss()`

따라서 이번 비교에서는 모델 구조의 변화에 따른 성능 차이를 중심으로 확인한다.

---

## 4. Model Architecture

### CNN

기본 CNN은 2개의 convolution layer를 사용한다.

```text
Input: 3 × 32 × 32

Conv 3 → 32
ReLU
MaxPool
Dropout

Conv 32 → 64
ReLU
MaxPool
Dropout

Flatten: 64 × 8 × 8 = 4096

Linear 4096 → 128
ReLU
Dropout

Linear 128 → 10
```

* Convolution layers: 2
* Trainable parameters: 545,098

### DeepCNN

DeepCNN은 convolution layer를 7개로 늘리고, feature channel을 점차 증가시키면서 여러 번의 MaxPooling을 통해 공간 크기를 줄였다.

```text
Input: 3 × 32 × 32

Conv 3 → 32
Conv 32 → 32
MaxPool
→ 32 × 16 × 16

Conv 32 → 64
Conv 64 → 64
MaxPool
→ 64 × 8 × 8

Conv 64 → 128
MaxPool
→ 128 × 4 × 4

Conv 128 → 128
MaxPool
→ 128 × 2 × 2

Conv 128 → 256
MaxPool
→ 256 × 1 × 1

Flatten: 256

Linear 256 → 128
ReLU
Dropout

Linear 128 → 10
```

* Convolution layers: 7
* Trainable parameters: 616,362

DeepCNN에서는 공간 크기가 `32 × 32 → 1 × 1`로 감소하는 대신 feature channel은 `3 → 32 → 64 → 128 → 256`으로 증가한다.

마지막 `256 × 1 × 1` feature map은 공간적인 위치 정보는 크게 압축되어 있지만, 이미지 전체에서 추출한 256개의 feature를 나타내는 representation으로 볼 수 있다.

---

## 5. Results

| Model   | Train Acc | Val Acc | Test Acc | Val Loss | Test Loss | Parameters |  Train Time |
| ------- | --------: | ------: | -------: | -------: | --------: | ---------: | ----------: |
| CNN     |    78.36% |  74.55% |   74.66% |   0.7674 |    0.7751 |    545,098 |  932.00 sec |
| DeepCNN |    81.46% |  80.97% |   80.90% |   0.5774 |    0.5970 |    616,362 | 1235.22 sec |

### Performance Change

DeepCNN은 기본 CNN과 비교하여:

* Train accuracy: `78.36% → 81.46%` (+3.10%p)
* Validation accuracy: `74.55% → 80.97%` (+6.42%p)
* Test accuracy: `74.66% → 80.90%` (+6.24%p)

Validation loss와 test loss도 모두 감소했다.

반면:

* Parameters: `545,098 → 616,362`
* Training time: `932.00 sec → 1235.22 sec`

으로 모델 규모와 계산 비용은 증가했다.

---

## 6. Observation

기본 CNN보다 convolution layer를 깊게 구성한 DeepCNN에서 validation accuracy와 test accuracy가 모두 향상되었다.

특히 test accuracy가 `74.66%`에서 `80.90%`로 약 6.24%p 증가하여, 이번 설정에서는 더 깊은 convolution 구조가 CIFAR-10 이미지의 특징을 표현하는 데 효과적이었다.

DeepCNN에서는 convolution layer를 여러 단계 거치면서 단순한 local feature를 점차 더 복잡한 feature representation으로 변환하고, MaxPooling을 통해 공간 크기를 줄여 최종적으로 `256 × 1 × 1`의 compact한 representation을 만든다.

또한 DeepCNN의 최종 train accuracy는 81.46%, validation accuracy는 80.97%, test accuracy는 80.90%로 서로 비교적 가까웠다. 따라서 최종 결과만 보면 training data에만 성능이 크게 치우친 모습은 나타나지 않았다.

한편 convolution layer는 2개에서 7개로 크게 증가했지만 전체 parameter 수는 약 13%만 증가했다. DeepCNN에서는 반복적인 pooling으로 마지막 feature map을 `256 × 1 × 1`까지 줄였기 때문에 fully connected layer의 입력 크기가 기본 CNN의 4096에서 256으로 크게 감소했기 때문이다.

대신 더 많은 convolution 연산이 필요하므로 학습 시간은 약 32% 증가했다.

---

## 7. Limitations

이번 비교에서 변경된 것은 단순히 convolution layer의 개수만은 아니다.

DeepCNN에서는 다음 요소들도 함께 변경되었다.

* Convolution layer의 깊이
* Feature channel 수
* MaxPooling 횟수
* 마지막 feature map의 크기

따라서 성능 향상을 단순히 **depth 하나의 효과**라고 단정하기보다는, 더 깊어진 전체 CNN architecture의 효과로 해석하는 것이 적절하다.

또한 이번 실험은 seed 42를 사용한 한 번의 실행 결과이므로, 여러 seed에서 반복 실험한 결과는 아니다.

---

## 8. Conclusion

CIFAR-10 분류에서 기본 2-layer CNN을 7-layer DeepCNN으로 확장한 결과 test accuracy가 `74.66%`에서 `80.90%`로 향상되었다.

이번 실험을 통해 이미지의 공간 크기를 점차 줄이는 동시에 feature channel을 증가시키고 convolution layer를 깊게 쌓는 CNN 구조가 더 복잡한 이미지 특징을 표현하는 데 활용될 수 있음을 확인했다.

동시에 모델을 깊게 만드는 것은 더 높은 성능을 얻을 가능성이 있지만, 학습 시간과 계산량 증가라는 trade-off가 있다는 점도 확인했다.
