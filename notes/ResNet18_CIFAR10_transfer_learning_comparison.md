# CIFAR-10 ResNet18 Transfer Learning Comparison

## 1. Experiment Purpose

CIFAR-10 이미지 분류에서 ImageNet으로 pretrained된 ResNet18을 이용해 transfer learning을 수행하고, 학습 가능한 범위에 따라 성능이 어떻게 달라지는지 비교한다.

두 가지 방식을 실험했다.

1. **Frozen ResNet18**: pretrained parameters를 모두 freeze하고 마지막 `fc` layer만 학습
2. **Fine-tuned ResNet18**: 마지막 `fc` layer와 ResNet의 마지막 block인 `layer4`를 함께 학습

이를 통해 pretrained feature를 그대로 사용하는 방식과 일부 feature extractor까지 새로운 데이터셋에 맞게 조정하는 fine-tuning 방식의 차이를 확인한다.

CIFAR-10을 사용한 또 다른 이유는 이전에 직접 학습한 CNN과 DeepCNN 실험과 연결해서 결과를 살펴보기 위해서이다.

Transfer learning 자체만 확인한다면 별도의 custom dataset을 사용할 수도 있지만, 동일한 CIFAR-10 classification task를 사용함으로써 scratch에서 학습한 CNN 계열 모델과 pretrained ResNet18을 활용한 모델의 결과를 함께 관찰할 수 있다.

따라서 이번 실험 결과도 이전 CNN / DeepCNN 결과와 동일한 results/CIFAR10/run_summary.csv에 기록했다.

---

## 2. Dataset and Preprocessing

* Dataset: CIFAR-10
* Number of classes: 10
* Train / Validation split: 85% / 15%
* Test set: CIFAR-10 official test set
* Seed: 42

Pretrained ResNet18이 ImageNet에서 학습할 때 사용한 입력 형태에 맞추기 위해 `ResNet18_Weights.DEFAULT.transforms()`를 사용했다.

따라서 CIFAR-10의 `32 × 32` 이미지는 ResNet18의 pretrained weights에 맞는 크기와 normalization으로 변환되어 입력된다.

---

## 3. Common Training Settings

두 실험에서 다음 학습 조건을 동일하게 유지했다.

* Model: pretrained ResNet18
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 64
* Epochs: 20
* Seed: 42
* Loss function: `nn.CrossEntropyLoss()`

두 모델 모두 기존 ImageNet용 `fc` layer를 CIFAR-10의 10개 class에 맞게 새로 교체했다.

---

## 4. Transfer Learning Strategy

### ResNet18 Frozen

pretrained ResNet18의 모든 parameter를 freeze한 뒤 새로운 `fc` layer만 학습했다.

```text
Pretrained ResNet18

Feature Extractor
[ Frozen ]

        ↓

FC: 512 → 10
[ Trainable ]
```

* Trainable parameters: **5,130**

이 방식에서는 ImageNet에서 학습된 feature extractor는 그대로 유지하고, 그 feature를 이용해 CIFAR-10을 분류하는 마지막 classifier만 새롭게 학습한다.

### ResNet18 Fine-Tuning

처음에는 pretrained parameters를 freeze한 뒤, 마지막 ResNet block인 `layer4`와 새로운 `fc` layer를 학습 가능하게 설정했다.

```text
Early ResNet Layers
[ Frozen ]

        ↓

layer4
[ Trainable ]

        ↓

FC: 512 → 10
[ Trainable ]
```

* Trainable parameters: **8,398,858**

따라서 pretrained feature를 그대로 사용하는 것뿐만 아니라, ResNet의 마지막 단계에서 추출되는 high-level feature도 CIFAR-10에 맞게 조정한다.

---

## 5. Results

| Model                | Train Acc | Val Acc | Test Acc | Val Loss | Test Loss | Trainable Parameters |  Train Time |
| -------------------- | --------: | ------: | -------: | -------: | --------: | -------------------: | ----------: |
| ResNet18 Frozen      |    79.03% |  77.63% |   78.09% |   0.6382 |    0.6383 |                5,130 | 3410.64 sec |
| ResNet18 Fine-Tuning |    98.87% |  89.43% |   89.54% |   0.5349 |    0.5514 |            8,398,858 | 3581.13 sec |

### Performance Change

Fine-tuning은 Frozen 방식과 비교하여:

* Train accuracy: `79.03% → 98.87%` (+19.84%p)
* Validation accuracy: `77.63% → 89.43%` (+11.80%p)
* Test accuracy: `78.09% → 89.54%` (+11.45%p)

Validation loss와 test loss도 모두 감소했다.

반면 학습 가능한 parameter 수는:

* `5,130 → 8,398,858`

로 크게 증가했다.

총 학습 시간은:

* `3410.64 sec → 3581.13 sec`

로 이번 실행에서는 약 170초 증가했다.

| Model                | Training Strategy        | Epochs | Test Accuracy |
| -------------------- | ------------------------ | -----: | ------------: |
| CNN                  | From scratch             |    100 |        74.66% |
| DeepCNN              | From scratch             |    100 |        80.90% |
| ResNet18 Frozen      | Pretrained + FC only     |     20 |        78.09% |
| ResNet18 Fine-Tuning | Pretrained + layer4 + FC |     20 |    **89.54%** |

이전 CIFAR-10 실험까지 함께 보면, 직접 구성한 CNN은 74.66%, 더 깊게 구성한 DeepCNN은 80.90%의 test accuracy를 기록했다. 
Pretrained ResNet18에서는 feature extractor를 모두 freeze한 경우 78.09%, 마지막 ResNet block까지 fine-tuning한 경우 89.54%를 기록했다.

다만 이 결과들은 동일한 CIFAR-10 dataset을 사용했지만 preprocessing, epoch 수, architecture 및 pretrained weights 사용 여부가 다르므로 
엄밀한 controlled experiment로 직접 비교할 수는 없다. 여기서는 지금까지의 CIFAR-10 학습 과정에서 모델과 학습 방식이 확장되면서 나타난 결과를 참고하는 용도로 사용한다.

---

## 6. Observation

모든 pretrained feature를 고정하고 마지막 classifier만 학습한 Frozen ResNet18에서도 test accuracy **78.09%**를 얻었다.

이는 ImageNet에서 미리 학습된 ResNet18의 feature extractor가 CIFAR-10에서도 어느 정도 유용한 image feature를 제공할 수 있음을 보여준다.

그러나 마지막 ResNet block인 `layer4`까지 학습 가능하게 만든 Fine-tuning 모델에서는 test accuracy가 **89.54%**까지 향상되었다.

즉, pretrained feature를 그대로 사용하는 것보다 일부 high-level feature를 CIFAR-10에 맞게 다시 조정하는 것이 이번 실험에서는 훨씬 높은 분류 성능으로 이어졌다.

Fine-tuning 모델의 train accuracy는 **98.87%**, validation accuracy는 **89.43%**로 약 9.44%p의 차이가 나타났다. 따라서 Frozen 모델보다 학습 데이터에 훨씬 강하게 fitting되었으며, 어느 정도 overfitting의 가능성도 확인할 수 있다.

---

## 7. Limitations

이번 실험에서는 Frozen과 Fine-tuning의 차이를 확인하기 위해 동일한 learning rate `0.001`을 사용했다.

하지만 실제 fine-tuning에서는 pretrained weights가 크게 변하지 않도록 더 작은 learning rate를 사용하거나, pretrained layer와 새로 추가한 `fc` layer에 서로 다른 learning rate를 적용하기도 한다.

또한 이번 결과는 seed 42를 사용한 한 번의 실행 결과이므로 여러 seed에서 반복한 평균 성능은 아니다.

따라서 이번 실험의 목적은 최적의 CIFAR-10 성능을 찾는 것보다는 **pretrained model의 freeze와 partial fine-tuning이 실제 학습 과정과 결과에 어떤 차이를 만드는지 경험하는 것**에 있다.

---

## 8. Conclusion

Pretrained ResNet18에서 마지막 `fc` layer만 학습한 결과 test accuracy는 **78.09%**였으며, 마지막 ResNet block인 `layer4`까지 fine-tuning하자 **89.54%**로 향상되었다.

이번 실험을 통해 transfer learning에서는 pretrained model 전체를 처음부터 다시 학습하지 않고도 기존에 학습된 feature를 새로운 문제에 활용할 수 있음을 확인했다.

또한 target dataset에 맞게 pretrained model의 일부 layer를 추가로 학습하는 fine-tuning을 통해 feature representation을 조정하고 성능을 크게 향상시킬 수 있음을 직접 확인했다.
