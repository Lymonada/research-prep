# CIFAR-10 ResNet18 Transfer Learning Comparison

## 1. Experiment Purpose

CIFAR-10 이미지 분류에서 ImageNet으로 pretrained된 ResNet18을 이용해 transfer learning을 수행하고, 학습 가능한 범위에 따라 성능이 어떻게 달라지는지 비교한다.

두 가지 방식을 실험했다.

1. **Frozen ResNet18**: pretrained parameters를 모두 freeze하고 마지막 `fc` layer만 학습
2. **Fine-tuned ResNet18**: 마지막 `fc` layer와 ResNet의 마지막 block인 `layer4`를 함께 학습

이를 통해 pretrained feature를 그대로 사용하는 방식과 일부 feature extractor까지 새로운 데이터셋에 맞게 조정하는 partial fine-tuning 방식의 차이를 확인한다.

CIFAR-10을 사용한 또 다른 이유는 이전에 직접 학습한 CNN과 DeepCNN 실험과 연결해서 결과를 살펴보기 위해서이다.

Transfer learning 자체만 경험하는 것이 목적이라면 별도의 custom dataset을 사용할 수도 있지만, 동일한 CIFAR-10 classification task를 사용함으로써 scratch에서 학습한 CNN 계열 모델과 pretrained ResNet18을 활용한 모델의 결과를 함께 관찰할 수 있다.

따라서 이번 실험 결과도 이전 CNN / DeepCNN 결과와 동일한 `results/CIFAR10/run_summary.csv`에 기록했다.

다만 이전 CNN 실험과 이번 ResNet18 실험은 학습 조건이 동일하지 않으므로, 네 모델 전체의 정확도를 엄밀한 controlled experiment로 직접 비교하지는 않는다.

---

## 2. Dataset and Preprocessing

* Dataset: CIFAR-10
* Number of classes: 10
* Train / Validation split: 85% / 15%
* Test set: CIFAR-10 official test set
* Seed: 42

Pretrained ResNet18이 ImageNet에서 학습할 때 사용한 입력 형태에 맞추기 위해 `ResNet18_Weights.DEFAULT.transforms()`를 사용했다.

이를 통해 CIFAR-10의 원본 `3 × 32 × 32` 이미지를 pretrained ResNet18에 적합한 크기로 resize / crop하고, ImageNet에 맞는 normalization을 적용했다.

이전 CNN / DeepCNN 실험에서는 CIFAR-10의 `32 × 32` 이미지를 `transforms.ToTensor()`를 통해 그대로 사용했기 때문에, 이전 실험과 이번 ResNet18 실험은 preprocessing 방식에서도 차이가 있다.

---

## 3. Common Training Settings

Frozen과 Fine-tuning 실험에서는 다음 학습 조건을 동일하게 유지했다.

* Model: pretrained ResNet18
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 64
* Epochs: 20
* Seed: 42
* Loss function: `nn.CrossEntropyLoss()`

두 모델 모두 기존 ImageNet의 1000개 class를 분류하던 `fc` layer를 CIFAR-10의 10개 class에 맞게 새로 교체했다.

따라서 **Frozen vs Fine-tuning 비교에서는 학습 가능한 parameter 범위가 주요 차이점**이다.

---

## 4. Transfer Learning Strategy

### ResNet18 Frozen

Pretrained ResNet18의 모든 parameter를 freeze한 뒤 새로운 `fc` layer만 학습했다.

```text
Pretrained ResNet18

Feature Extractor
[ Frozen ]

        ↓

FC: 512 → 10
[ Trainable ]
```

* Trainable parameters: **5,130**

이 방식에서는 ImageNet에서 학습된 feature extractor를 그대로 유지하고, 그 feature를 이용해 CIFAR-10을 분류하는 마지막 classifier만 새롭게 학습한다.

즉, pretrained model을 새로운 classification task의 **feature extractor**로 사용하는 방식이다.

### ResNet18 Fine-Tuning

Pretrained parameters를 먼저 freeze한 뒤, 마지막 ResNet block인 `layer4`와 새로운 `fc` layer를 학습 가능하게 설정했다.

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

이 방식에서는 ImageNet에서 학습된 초기 feature는 유지하면서, 마지막 단계의 high-level feature를 CIFAR-10에 맞게 추가로 조정한다.

---

## 5. Frozen vs Fine-Tuning Results

| Model                | Train Acc | Val Acc |   Test Acc | Val Loss | Test Loss | Trainable Parameters |  Train Time |
| -------------------- | --------: | ------: | ---------: | -------: | --------: | -------------------: | ----------: |
| ResNet18 Frozen      |    79.03% |  77.63% |     78.09% |   0.6382 |    0.6383 |                5,130 | 3410.64 sec |
| ResNet18 Fine-Tuning |    98.87% |  89.43% | **89.54%** |   0.5349 |    0.5514 |            8,398,858 | 3581.13 sec |

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

---

## 6. Comparison with Previous CIFAR-10 Experiments

이전에 직접 구성한 CNN / DeepCNN 결과까지 함께 기록하면 다음과 같다.

| Model                | Training Strategy          | Epochs | Test Accuracy |
| -------------------- | -------------------------- | -----: | ------------: |
| CNN                  | From scratch               |    100 |        74.66% |
| DeepCNN              | From scratch               |    100 |        80.90% |
| ResNet18 Frozen      | Pretrained + FC only       |     20 |        78.09% |
| ResNet18 Fine-Tuning | Pretrained + `layer4` + FC |     20 |    **89.54%** |

직접 구성한 기본 CNN은 **74.66%**, 더 깊게 구성한 DeepCNN은 **80.90%**의 test accuracy를 기록했다.

이후 pretrained ResNet18을 이용한 transfer learning에서는 feature extractor 전체를 freeze하고 `fc`만 학습했을 때 **78.09%**, 마지막 ResNet block인 `layer4`까지 fine-tuning했을 때 **89.54%**를 기록했다.

이 결과들을 통해 지금까지의 CIFAR-10 실험이

```text
CNN from scratch
        ↓
Deeper CNN from scratch
        ↓
Pretrained feature extractor
        ↓
Partial fine-tuning
```

으로 확장되는 과정을 한 번에 확인할 수 있다.

다만 이 표는 네 모델의 성능을 동일 조건에서 공정하게 비교하기 위한 benchmark가 아니라, **동일한 CIFAR-10 classification task에서 서로 다른 학습 방식을 경험하며 얻은 결과를 함께 기록하기 위한 참고 비교**이다.

---

## 7. Observation

모든 pretrained feature를 고정하고 마지막 classifier만 학습한 Frozen ResNet18에서도 test accuracy **78.09%**를 얻었다.

이는 ImageNet에서 미리 학습된 ResNet18의 feature extractor가 CIFAR-10에서도 어느 정도 유용한 image feature를 제공할 수 있음을 보여준다.

하지만 Frozen ResNet18의 test accuracy는 이전에 직접 학습한 DeepCNN의 **80.90%**보다 약간 낮았다.

따라서 pretrained model을 사용한다고 해서 마지막 classifier만 교체하는 것만으로 항상 더 높은 성능을 얻는 것은 아니며, pretrained feature가 target dataset에 얼마나 잘 맞는지도 중요하다는 것을 확인할 수 있었다.

반면 마지막 ResNet block인 `layer4`까지 학습 가능하게 만든 Fine-tuning 모델에서는 test accuracy가 **89.54%**까지 향상되었다.

즉, pretrained feature를 그대로 사용하는 것보다 일부 high-level feature를 CIFAR-10에 맞게 다시 조정하는 것이 이번 실험에서는 훨씬 높은 분류 성능으로 이어졌다.

Fine-tuning 모델의 train accuracy는 **98.87%**, validation accuracy는 **89.43%**로 약 9.44%p의 차이가 나타났다.

따라서 Frozen 모델보다 training data에 훨씬 강하게 fitting되었으며, 성능 향상과 함께 어느 정도 overfitting의 가능성도 나타났다.

---

## 8. Comparison Notes and Limitations

### CNN / DeepCNN vs ResNet18

이전 CNN / DeepCNN 결과와 이번 ResNet18 결과는 모두 CIFAR-10을 사용했지만, 학습 조건이 다르기 때문에 네 모델의 test accuracy를 엄밀한 controlled comparison으로 해석해서는 안 된다.

주요 차이는 다음과 같다.

| Setting             | CNN / DeepCNN               | ResNet18 Frozen / Fine-Tuning         |
| ------------------- | --------------------------- | ------------------------------------- |
| Initialization      | Random initialization       | ImageNet pretrained weights           |
| Original image size | `3 × 32 × 32`               | `3 × 32 × 32`                         |
| Input preprocessing | `transforms.ToTensor()`     | Pretrained ResNet18 transforms        |
| Model input         | `32 × 32` image             | Resize / crop된 larger image           |
| Epochs              | 100                         | 20                                    |
| Training range      | Entire model                | FC only / `layer4` + FC               |
| Main purpose        | CNN architecture comparison | Transfer learning strategy comparison |

따라서 CNN / DeepCNN과 ResNet18 사이의 성능 차이에는 단순한 architecture 차이뿐 아니라 다음 요소들이 함께 포함되어 있다.

* Pretraining 여부
* Input preprocessing
* Epoch 수
* Model architecture
* 학습 가능한 parameter 범위

따라서 `89.54% > 80.90%`라는 결과만으로 ResNet18 architecture 자체가 DeepCNN보다 정확히 그만큼 우수하다고 결론내릴 수는 없다.

이번 기록에서는 네 모델의 결과를 **scratch에서 직접 CNN을 학습하는 단계에서 pretrained model과 fine-tuning을 활용하는 단계까지 학습 방법이 확장되는 과정의 참고 결과**로 해석한다.

### Fine-Tuning Settings

Frozen과 Fine-tuning의 차이를 확인하기 위해 두 ResNet18 실험에서는 동일한 learning rate `0.001`을 사용했다.

하지만 실제 fine-tuning에서는 pretrained weights가 크게 변하지 않도록 더 작은 learning rate를 사용하거나, pretrained layer와 새롭게 추가한 `fc` layer에 서로 다른 learning rate를 적용할 수도 있다.

또한 이번 결과는 seed 42를 사용한 한 번의 실행 결과이므로 여러 seed에서 반복 실험한 평균 성능은 아니다.

따라서 이번 실험의 목적은 최적의 CIFAR-10 성능을 찾는 것보다는 **pretrained model의 freeze와 partial fine-tuning이 실제 학습 과정과 결과에 어떤 차이를 만드는지 경험하는 것**에 있다.

---

## 9. Conclusion

Pretrained ResNet18에서 마지막 `fc` layer만 학습한 결과 test accuracy는 **78.09%**였으며, 마지막 ResNet block인 `layer4`까지 fine-tuning하자 **89.54%**로 향상되었다.

이번 실험을 통해 transfer learning에서는 pretrained model 전체를 처음부터 다시 학습하지 않고도 기존에 학습된 feature를 새로운 문제에 활용할 수 있음을 확인했다.

또한 pretrained feature를 그대로 사용하는 것에서 더 나아가 target dataset에 맞게 일부 layer를 fine-tuning함으로써 feature representation을 조정하고 성능을 향상시킬 수 있음을 직접 확인했다.

이전 CNN / DeepCNN 실험과 동일한 CIFAR-10을 사용함으로써 scratch training에서 pretrained transfer learning으로 학습 방법이 확장되는 과정도 함께 기록할 수 있었다.

다만 이전 CNN 실험과 ResNet18 실험은 preprocessing, epoch 수, initialization, 학습 범위 등 여러 조건이 다르므로, 이 결과들은 직접적인 architecture 성능 비교보다는 **지금까지 진행한 CIFAR-10 학습 과정의 연속적인 실험 기록**으로 해석하는 것이 적절하다.
