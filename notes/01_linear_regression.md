
# Linear Regression

## 1. 학습 목적

선형회귀가 입력값으로부터 연속적인 값을 예측하는 원리를 이해하고, PyTorch를 이용해 모델의 가중치와 편향을 학습한다.

이번 학습에서는 다음 두 가지를 함께 정리한다.

- **Simple Linear Regression**: 하나의 입력 특성을 사용
- **Multiple Linear Regression**: 여러 개의 입력 특성을 사용

두 모델은 입력 특성의 개수에는 차이가 있지만, 예측값을 계산하고 손실을 줄이는 학습 원리는 동일하다.

---

## 2. What is Linear Regression?

선형회귀는 입력값과 출력값 사이의 선형적인 관계를 학습해 연속적인 값을 예측하는 모델이다.

예를 들면 다음과 같은 문제에 사용할 수 있다.

- 공부 시간을 이용한 시험 점수 예측
- 집의 크기, 방 개수, 위치 등을 이용한 집값 예측
- 기온과 습도를 이용한 전력 사용량 예측

선형회귀는 합격 또는 불합격처럼 특정 클래스를 예측하는 것이 아니라, 시험 점수나 집값처럼 연속적인 숫자를 예측한다.

---

## 3. Simple Linear Regression

단순 선형회귀는 하나의 입력 특성을 사용한다.

$$
\hat{y} = wx + b
$$

- $x$: 입력값
- $w$: 가중치
- $b$: 편향
- $\hat{y}$: 모델이 예측한 값
- $y$: 실제 정답값

예를 들어 공부 시간 $x$를 이용해 시험 점수 $y$를 예측할 수 있다.

$$
\text{score} = w \times \text{study time} + b
$$

가중치 $w$는 입력값이 한 단위 증가할 때 예측값이 얼마나 변하는지를 나타낸다.

편향 $b$는 입력값이 0일 때 모델이 기본적으로 가지는 출력값이다.

PyTorch에서는 다음과 같이 표현할 수 있다.

```python
model = nn.Linear(
    in_features=1,
    out_features=1
)
```

입력 특성이 하나이고 출력값도 하나이므로 `nn.Linear(1, 1)`을 사용한다.

---

## 4. Multiple Linear Regression

실제 문제에서는 하나의 입력만으로 출력값을 충분히 설명하기 어려운 경우가 많다.

예를 들어 시험 점수를 예측할 때 다음과 같은 여러 특성을 함께 사용할 수 있다.

- 공부 시간
- 출석률
- 과제 점수

다중선형회귀는 여러 개의 입력 특성을 사용해 하나의 연속적인 값을 예측한다.

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

각 입력 특성에는 서로 다른 가중치가 대응한다.

- $x_1$: 첫 번째 입력 특성
- $w_1$: 첫 번째 특성의 가중치
- $x_2$: 두 번째 입력 특성
- $w_2$: 두 번째 특성의 가중치
- $b$: 편향

입력 특성이 세 개라면 다음과 같이 표현할 수 있다.

$$
\hat{y} = w_1x_1 + w_2x_2 + w_3x_3 + b
$$

PyTorch에서는 다음과 같이 표현한다.

```python
model = nn.Linear(
    in_features=3,
    out_features=1
)
```

입력 특성이 세 개이고 예측할 출력값은 하나이므로 `nn.Linear(3, 1)`을 사용한다.

---

## 5. Simple and Multiple Linear Regression Comparison

두 모델의 가장 중요한 차이는 입력 특성의 개수이다.

| 구분 | 단순 선형회귀 | 다중선형회귀 |
|---|---|---|
| 입력 특성 개수 | 1개 | 2개 이상 |
| 출력 개수 | 1개 | 1개 |
| 모델 | $wx+b$ | $w_1x_1+\cdots+w_nx_n+b$ |
| PyTorch 예시 | `nn.Linear(1, 1)` | `nn.Linear(3, 1)` |
| 대표 손실함수 | MSE | MSE |
| 학습 방식 | 경사하강법 | 경사하강법 |

다중선형회귀는 완전히 새로운 학습 방법이 아니라, 단순 선형회귀에서 입력과 가중치의 개수가 여러 개로 확장된 형태이다.

---

## 6. Matrix Representation

선형회귀는 입력 특성의 개수와 관계없이 다음과 같은 행렬 형태로 표현할 수 있다.

$$
\hat{Y} = XW + b
$$

- $X$: 여러 sample과 feature를 포함한 입력 행렬
- $W$: 각 feature에 대응하는 가중치
- $b$: 편향
- $\hat{Y}$: 모델의 예측값

### 다중선형회귀의 shape

sample이 5개이고 feature가 3개라면 입력 tensor의 shape은 다음과 같다.

```text
X.shape = [5, 3]
```

개념적으로 가중치와 편향의 shape은 다음과 같이 생각할 수 있다.

```text
W.shape = [3, 1]
b.shape = [1]
```

행렬곱을 수행하면 다음과 같은 출력이 만들어진다.

```text
[5, 3] @ [3, 1] = [5, 1]
```

따라서 5개의 sample 각각에 대해 하나의 예측값이 생성된다.

### 단순 선형회귀의 shape

단순 선형회귀도 같은 행렬 연산 구조를 사용한다.

```text
X.shape = [5, 1]
W.shape = [1, 1]

[5, 1] @ [1, 1] = [5, 1]
```

결국 단순 선형회귀와 다중선형회귀는 입력 feature 차원의 크기만 다를 뿐 같은 연산 구조를 가진다.

> 참고: PyTorch의 `nn.Linear` 내부에서 저장되는 `weight`의 실제 shape은  
> `[out_features, in_features]`이다.  
> 예를 들어 `nn.Linear(3, 1)`의 `weight.shape`은 `[1, 3]`이다.  
> 하지만 forward 연산에서는 입력과 전치된 weight가 곱해져 결과적으로 `XW + b`와 같은 계산이 수행된다.

---

## 7. Prediction

모델에 입력값을 전달하면 현재 가중치와 편향을 사용해 예측값을 계산한다.

```python
prediction = model(x)
```

학습이 시작되기 전의 가중치와 편향은 아직 적절한 값이 아니므로 처음 예측값은 실제 정답과 차이가 클 수 있다.

학습의 목적은 예측값과 정답값의 차이가 작아지도록 가중치와 편향을 반복해서 수정하는 것이다.

---

## 8. Loss Function

모델의 예측값과 실제 정답값의 차이를 측정하기 위해 손실함수를 사용한다.

선형회귀에서는 주로 **Mean Squared Error, MSE**를 사용한다.

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

PyTorch에서는 다음과 같이 사용할 수 있다.

```python
loss_function = nn.MSELoss()

prediction = model(x)
loss = loss_function(prediction, y)
```

예측값과 정답값의 차이를 제곱하는 이유는 다음과 같다.

1. 양수 오차와 음수 오차가 서로 상쇄되는 것을 막는다.
2. 큰 오차에 더 큰 손실을 부여한다.
3. 미분할 수 있기 때문에 경사하강법을 적용할 수 있다.

손실값이 작을수록 모델의 예측값이 실제 정답값에 가깝다는 뜻이다.

단순 선형회귀와 다중선형회귀 모두 같은 방식으로 MSE를 사용할 수 있다.

---

## 9. Gradient Descent

모델은 손실값이 감소하는 방향으로 가중치와 편향을 수정한다.

가중치와 편향은 다음과 같은 방식으로 업데이트된다.

$$
w := w - \eta \frac{\partial L}{\partial w}
$$

$$
b := b - \eta \frac{\partial L}{\partial b}
$$

- $L$: 손실함수
- $\eta$: 학습률
- $\frac{\partial L}{\partial w}$: 가중치에 대한 손실의 기울기
- $\frac{\partial L}{\partial b}$: 편향에 대한 손실의 기울기

gradient는 현재 위치에서 파라미터를 어느 방향으로 움직였을 때 손실이 증가하는지를 나타낸다.

따라서 gradient의 반대 방향으로 이동하면 손실을 감소시킬 수 있다.

- gradient가 양수이면 파라미터를 감소시킨다.
- gradient가 음수이면 파라미터를 증가시킨다.

다중선형회귀에서는 각 가중치가 따로 업데이트된다.

$$
w_j := w_j - \eta \frac{\partial L}{\partial w_j}
$$

즉, 모델은 각 입력 특성이 예측에 얼마나 기여해야 하는지를 각각의 가중치를 통해 학습한다.

---

## 10. Optimizer

이번 예제에서는 SGD optimizer를 사용할 수 있다.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

- `model.parameters()`: 학습할 가중치와 편향
- `lr`: 한 번의 업데이트에서 파라미터를 변경하는 크기

학습률이 너무 크면 손실의 최솟값을 지나치거나 학습이 불안정해질 수 있다.

학습률이 너무 작으면 파라미터가 조금씩만 바뀌기 때문에 학습 속도가 지나치게 느려질 수 있다.

---

## 11. Training Process

선형회귀의 학습 과정은 다음 순서로 진행된다.

1. 입력값으로 예측값을 계산한다.
2. 예측값과 정답값을 비교해 손실을 계산한다.
3. 역전파를 통해 gradient를 계산한다.
4. optimizer가 가중치와 편향을 업데이트한다.
5. 이 과정을 여러 epoch 동안 반복한다.

```python
for epoch in range(num_epochs):
    prediction = model(x)
    loss = loss_function(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### `optimizer.zero_grad()`

이전 학습 단계에서 계산된 gradient를 초기화한다.

PyTorch에서는 gradient가 기본적으로 누적되기 때문에 각 학습 단계가 시작될 때 초기화해야 한다.

### `loss.backward()`

손실값을 기준으로 각 가중치와 편향의 gradient를 계산한다.

단순 선형회귀에서는 하나의 입력 특성에 대응하는 가중치의 gradient를 계산하고, 다중선형회귀에서는 여러 가중치 각각의 gradient를 계산한다.  

$$
\frac{\partial L}{\partial w}
$$

### `optimizer.step()`

계산된 gradient를 이용해 가중치와 편향을 실제로 업데이트한다.  

$$
w_j := w_j - \eta \frac{\partial L}{\partial w_j}
$$


---

## 12. Tensor Shape Comparison

### Simple Linear Regression

sample이 5개이고 각 sample의 feature가 1개인 경우:

```python
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])
```

```text
x.shape = [5, 1]
y.shape = [5, 1]
```

```python
model = nn.Linear(1, 1)
```

### Multiple Linear Regression

sample이 5개이고 각 sample의 feature가 3개인 경우:

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0]
])
```

```text
x.shape = [5, 3]
y.shape = [5, 1]
```

```python
model = nn.Linear(3, 1)
```

`nn.Linear`의 `in_features`는 sample 개수가 아니라 각 sample이 가지고 있는 feature의 개수를 의미한다.

- 첫 번째 차원: sample 개수
- 두 번째 차원: feature 개수

---

## 13. 핵심 비교

두 모델에서 달라지는 부분은 다음과 같다.

- 입력 feature 개수
- 학습되는 weight 개수
- 입력 tensor의 shape
- `nn.Linear`의 `in_features`

반면 다음 내용은 동일하다.

- 연속적인 값 예측
- `XW + b` 형태의 선형 결합
- MSE를 이용한 손실 계산
- `loss.backward()`를 이용한 gradient 계산
- optimizer를 이용한 파라미터 업데이트
- training loop의 전체 구조

```text
입력 데이터
    ↓
모델의 예측
    ↓
정답과 비교
    ↓
손실 계산
    ↓
gradient 계산
    ↓
파라미터 업데이트
    ↓
더 나은 예측
```

---

## 14. What I Learned

이번 학습을 통해 다음 내용을 확인했다.

- 선형회귀는 연속적인 값을 예측하는 모델이다.
- 단순 선형회귀는 하나의 입력 특성을 사용한다.
- 다중선형회귀는 여러 개의 입력 특성을 사용한다.
- 다중선형회귀는 단순 선형회귀와 다른 학습 방법이 아니라 입력 차원이 확장된 형태이다.
- 두 모델 모두 MSE와 경사하강법을 사용할 수 있다.
- `nn.Linear`는 내부적으로 선형 결합을 계산한다.
- `in_features`는 sample 개수가 아니라 입력 feature의 개수이다.
- 입력 feature가 증가하면 학습해야 할 가중치의 수도 증가한다.
- `loss.backward()`는 모든 가중치와 편향의 gradient를 계산한다.
- `optimizer.step()`은 계산된 gradient를 이용해 파라미터를 업데이트한다.

---

## 15. Difficulties and Clarifications

처음에는 단순 선형회귀와 다중선형회귀를 서로 다른 종류의 모델이라고 생각했다.

하지만 두 모델은 모두 다음과 같은 공통 구조를 사용한다.

$$
\hat{Y} = XW + b
$$

단순 선형회귀에서는 입력 feature의 개수가 1이고, 다중선형회귀에서는 입력 feature의 개수가 여러 개라는 차이가 있다.

또한 `nn.Linear(3, 1)`에서 숫자 3은 전체 데이터의 개수가 아니라 한 sample이 가지고 있는 입력 feature의 개수이다.

sample 개수는 입력 tensor의 첫 번째 차원에 나타나고 feature 개수는 마지막 차원에 나타난다.

---

## 16. Connection to the Next Step

선형회귀는 $XW+b$의 결과를 그대로 예측값으로 사용한다.

하지만 분류 문제에서는 선형 결합의 결과를 클래스 또는 확률과 연결해야 한다.

다음 단계에서는 선형 결합으로 계산된 값을 sigmoid 함수에 통과시켜 두 클래스 중 하나를 예측하는 **Binary Logistic Regression**을 학습한다.
```
