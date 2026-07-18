
# Binary Logistic Regression

## 1. 학습 목적

로지스틱 회귀가 입력값으로부터 두 클래스 중 하나를 예측하는 원리를 이해하고, PyTorch를 이용해 모델의 가중치와 편향을 학습한다.

이번 학습에서는 다음 내용을 정리한다.

-   이진 분류 문제의 의미
    
-   선형회귀와 로지스틱 회귀의 차이
    
-   선형 결합과 logit
    
-   sigmoid 함수를 이용한 확률 계산
    
-   Binary Cross Entropy
    
-   로지스틱 회귀의 학습 과정
    
-   예측 확률을 클래스로 변환하는 방법
    

로지스틱 회귀는 이름에 `Regression`이 포함되어 있지만, 주로 두 개의 클래스를 구분하는 **이진 분류 문제**에 사용된다.

----------

## 2. What is Binary Classification?

이진 분류는 입력값을 두 개의 클래스 중 하나로 분류하는 문제이다.

예를 들면 다음과 같은 문제에 사용할 수 있다.

-   이메일이 스팸인지 아닌지 분류
    
-   환자가 특정 질병을 가지고 있는지 분류
    
-   시험에 합격할지 불합격할지 분류
    
-   거래가 정상 거래인지 이상 거래인지 분류
    

일반적으로 두 클래스는 `0`과 `1`로 표현한다.

예를 들어 합격 여부를 예측하는 문제에서는 다음과 같이 정의할 수 있다.

```text
불합격: class 0
합격: class 1

```

여기서 `0`과 `1`은 클래스의 이름이다.

어떤 상태를 클래스 0으로 정하고 어떤 상태를 클래스 1로 정할지는 문제를 정의하는 사람이 결정한다.

----------

## 3. Why Not Linear Regression?

선형회귀는 다음과 같은 선형 결합을 사용한다.

$$  
\hat{y} = wx+b  
$$

그러나 선형회귀의 출력값에는 범위 제한이 없다.

```text
-3.7
0.4
1.8
12.5

```

이진 분류에서는 입력값이 클래스 1에 속할 확률을 예측하고자 한다.

확률은 반드시 `0`과 `1` 사이의 값이어야 한다.

$$  
0 \leq P(y=1 \mid x) \leq 1  
$$

그러나 선형회귀의 출력은 음수가 되거나 1보다 커질 수 있으므로 그대로 확률로 해석하기 어렵다.

로지스틱 회귀는 선형 결합의 결과를 **sigmoid 함수**에 통과시켜 `0`과 `1` 사이의 값으로 변환한다.

```text
선형 결합
    ↓
sigmoid
    ↓
0과 1 사이의 확률

```

----------

## 4. Linear Combination and Logit

로지스틱 회귀도 먼저 입력값과 가중치를 이용해 선형 결합을 계산한다.

입력 특성이 하나라면 다음과 같다.

$$  
z = wx+b  
$$

입력 특성이 여러 개라면 다음과 같다.

$$  
z = w_1x_1+w_2x_2+\cdots+w_nx_n+b  
$$

행렬 형태로 표현하면 다음과 같다.

$$  
Z = XW+b  
$$

여기서 $z$는 sigmoid 함수에 입력되기 전의 선형 결합 결과이다.

로지스틱 회귀에서는 이 값을 **logit**이라고 부른다.

-   $x$: 입력값
    
-   $w$: 가중치
    
-   $b$: 편향
    
-   $z$: 선형 결합의 결과인 logit
    

logit은 아직 확률이 아니므로 범위 제한이 없다.

```text
z = -2.4
z = 0.0
z = 3.1

```

로지스틱 회귀에서는 이 logit을 sigmoid 함수에 통과시켜 확률로 변환한다.

----------

## 5. Sigmoid Function

sigmoid 함수는 모든 실수를 `0`과 `1` 사이의 값으로 변환한다.

$$  
\sigma(z)=\frac{1}{1+e^{-z}}  
$$

로지스틱 회귀의 최종 출력은 다음과 같다.

$$  
\hat{y}=\sigma(z)  
$$

$z=wx+b$를 대입하면 다음과 같다.

$$  
\hat{y}=\sigma(wx+b)  
$$

여기서 $\hat{y}$는 입력값이 클래스 1에 속할 확률로 해석할 수 있다.

$$  
\hat{y}=P(y=1 \mid x)  
$$

예를 들어 sigmoid 함수의 출력이 `0.8`이라면 다음과 같이 해석할 수 있다.

```text
클래스 1일 확률: 0.8
클래스 0일 확률: 0.2

```

클래스 0일 확률은 다음과 같이 계산한다.

$$  
P(y=0 \mid x)=1-\hat{y}  
$$

따라서 $\hat{y}=0.8$이라면 클래스 0일 확률은 다음과 같다.

$$  
1-0.8=0.2  
$$

----------

## 6. Behavior of the Sigmoid Function

sigmoid 함수의 출력은 logit $z$의 크기에 따라 달라진다.

### $z$가 매우 큰 양수인 경우

$z$가 큰 양수이면 sigmoid 출력은 `1`에 가까워진다.

```text
z = 5.0
sigmoid(z) ≈ 0.993

```

따라서 모델은 해당 입력이 클래스 1에 속할 가능성이 높다고 판단한다.

### $z$가 0인 경우

$z=0$이면 sigmoid 출력은 정확히 `0.5`이다.

$$  
\sigma(0)=\frac{1}{1+e^0}=\frac{1}{2}=0.5  
$$

이 값은 클래스 0과 클래스 1 중 어느 한쪽으로도 강하게 기울지 않은 상태이다.

### $z$가 매우 큰 음수인 경우

$z$가 큰 음수이면 sigmoid 출력은 `0`에 가까워진다.

```text
z = -5.0
sigmoid(z) ≈ 0.007

```

따라서 모델은 해당 입력이 클래스 0에 속할 가능성이 높다고 판단한다.

전체 흐름은 다음과 같다.

```text
z가 큰 음수
    ↓
sigmoid 출력이 0에 가까움
    ↓
클래스 0에 가깝다고 예측

```

```text
z가 큰 양수
    ↓
sigmoid 출력이 1에 가까움
    ↓
클래스 1에 가깝다고 예측

```

----------

## 7. Classification Threshold

sigmoid의 출력은 확률이므로 실제 클래스를 결정하려면 기준값이 필요하다.

이 기준값을 **threshold**라고 한다.

이진 분류에서는 일반적으로 `0.5`를 threshold로 사용한다.

-   예측 확률이 `0.5` 이상이면 클래스 1
    
-   예측 확률이 `0.5` 미만이면 클래스 0
    

PyTorch에서는 다음과 같이 표현할 수 있다.

```python
predicted_class = (probability >= 0.5).float()

```

예를 들면 다음과 같다.

```text
probability = 0.82 → predicted class = 1
probability = 0.31 → predicted class = 0

```

threshold는 주로 모델의 성능을 평가하거나 실제 예측 결과를 만들 때 사용한다.

학습 과정에서는 확률을 바로 `0`이나 `1`로 변환하지 않는다.

모델이 정답에서 얼마나 멀리 떨어져 있는지를 계산하려면 연속적인 확률값이 필요하기 때문이다.

예를 들어 실제 정답이 1일 때 다음 두 예측은 모두 threshold를 적용하면 클래스 1이 된다.

```text
예측 확률 0.51 → class 1
예측 확률 0.99 → class 1

```

하지만 `0.99`는 `0.51`보다 실제 정답 1에 훨씬 가까운 예측이다.

손실함수는 이러한 확률의 차이까지 반영해 모델을 학습한다.

----------

## 8. Binary Cross Entropy

이진 분류에서는 주로 **Binary Cross Entropy, BCE**를 손실함수로 사용한다.

하나의 sample에 대한 BCE는 다음과 같다.

$$  
L=  
-\left[  
y\log(\hat{y})  
+  
(1-y)\log(1-\hat{y})  
\right]  
$$

-   $y$: 실제 정답
    
-   $\hat{y}$: 모델이 예측한 클래스 1의 확률
    

실제 정답 $y$는 `0` 또는 `1`이므로 두 경우로 나누어 생각할 수 있다.

### 실제 정답이 1인 경우

$y=1$을 BCE 식에 대입하면 다음과 같다.

$$  
L=  
-\left[  
\log(\hat{y})  
+  
0  
\right]  
$$

따라서 다음과 같이 정리된다.

$$  
L=-\log(\hat{y})  
$$

모델이 클래스 1일 확률을 높게 예측할수록 손실이 작아진다.

```text
실제 정답: 1

예측 확률 0.9 → 작은 손실
예측 확률 0.1 → 큰 손실

```

### 실제 정답이 0인 경우

$y=0$을 BCE 식에 대입하면 다음과 같다.

$$  
L=  
-\left[  
0  
+  
\log(1-\hat{y})  
\right]  
$$

따라서 다음과 같이 정리된다.

$$  
L=-\log(1-\hat{y})  
$$

모델이 클래스 1일 확률을 낮게 예측할수록 손실이 작아진다.

```text
실제 정답: 0

예측 확률 0.1 → 작은 손실
예측 확률 0.9 → 큰 손실

```

결국 BCE는 모델이 실제 정답 클래스에 높은 확률을 부여하도록 만든다.

----------

## 9. BCE for Multiple Samples

여러 sample이 하나의 batch에 포함된 경우에는 각 sample의 BCE를 계산한 뒤 평균을 구한다.

$$  
BCE=  
-\frac{1}{N}  
\sum_{i=1}^{N}  
\left[  
y_i\log(\hat{y}_i)  
+  
(1-y_i)\log(1-\hat{y}_i)  
\right]  
$$

예를 들어 batch에 4개의 sample이 포함되어 있다면 다음과 같이 계산된다.

```text
sample 1의 loss
sample 2의 loss
sample 3의 loss
sample 4의 loss
        ↓
     평균 loss

```

이렇게 계산된 평균 손실값을 기준으로 `loss.backward()`가 모델의 가중치와 편향에 대한 gradient를 계산한다.

----------

## 10. Logistic Regression Model

입력 특성이 3개인 이진 분류 문제를 생각해볼 수 있다.

```python
model = nn.Sequential(
    nn.Linear(
        in_features=3,
        out_features=1
    ),
    nn.Sigmoid()
)

```

입력 특성이 3개이므로 `in_features=3`을 사용한다.

이진 분류에는 클래스가 두 개 있지만 모델의 출력값은 하나만 사용한다.

모델의 출력값 하나를 클래스 1일 확률로 해석할 수 있고, 클래스 0일 확률은 `1 - probability`로 계산할 수 있기 때문이다.

```text
모델 출력값
    ↓
클래스 1일 확률
    ↓
클래스 0일 확률 = 1 - 클래스 1일 확률

```

예를 들어 모델의 출력이 `0.7`이라면 다음과 같다.

```text
클래스 1일 확률: 0.7
클래스 0일 확률: 0.3

```

----------

## 11. Gradient of Logistic Regression

로지스틱 회귀에서는 다음과 같은 순서로 예측값과 손실을 계산한다.

$$  
z=wx+b  
$$

$$  
\hat{y}=\sigma(z)  
$$

$$  
L=  
-\left[  
y\log(\hat{y})  
+  
(1-y)\log(1-\hat{y})  
\right]  
$$

sigmoid와 BCE를 함께 미분하면 logit $z$에 대한 손실의 gradient는 다음과 같이 정리된다.

$$  
\frac{\partial L}{\partial z}=\hat{y}-y  
$$

이 값은 현재 예측 확률과 실제 정답의 차이를 나타낸다.

### 실제 정답이 1인 경우

예측 확률이 `0.8`이고 실제 정답이 `1`이라면 다음과 같다.

$$  
\hat{y}-y=0.8-1=-0.2  
$$

gradient가 음수라는 것은 현재 $z$가 더 커지면 손실이 감소할 수 있다는 의미이다.

$z$가 커지면 sigmoid 출력도 커진다.

```text
z 증가
    ↓
sigmoid 출력 증가
    ↓
예측 확률이 1에 가까워짐

```

하지만 optimizer가 $z$ 자체를 직접 수정하는 것은 아니다.

$z$는 다음 식으로 계산되는 중간값이다.

$$  
z=wx+b  
$$

optimizer는 가중치 $w$와 편향 $b$를 수정한다.

그 결과 다음 forward 연산에서 계산되는 $z$가 달라진다.

```text
w와 b 업데이트
    ↓
다음 forward에서 새로운 z 계산
    ↓
새로운 sigmoid 출력 계산

```

### 실제 정답이 0인 경우

예측 확률이 `0.8`이고 실제 정답이 `0`이라면 다음과 같다.

$$  
\hat{y}-y=0.8-0=0.8  
$$

gradient가 양수라는 것은 현재 $z$가 더 작아지면 손실이 감소할 수 있다는 의미이다.

$z$가 작아지면 sigmoid 출력도 작아진다.

```text
z 감소
    ↓
sigmoid 출력 감소
    ↓
예측 확률이 0에 가까워짐

```

이 경우에도 optimizer는 $z$를 직접 수정하지 않는다.

가중치와 편향을 수정해 다음 forward에서 계산되는 $z$가 작아지는 방향으로 학습한다.

----------

## 12. Weight and Bias Gradient

logit은 다음과 같이 계산된다.

$$  
z=wx+b  
$$

가중치 $w$에 대한 손실의 gradient는 chain rule을 이용해 계산할 수 있다.

$$  
\frac{\partial L}{\partial w}
\frac{\partial L}{\partial z}  
\frac{\partial z}{\partial w}  
$$

앞에서 다음 식을 확인했다.

$$  
\frac{\partial L}{\partial z}=\hat{y}-y  
$$

또한 $z=wx+b$이므로 다음과 같다.

$$  
\frac{\partial z}{\partial w}=x  
$$

따라서 가중치에 대한 gradient는 다음과 같다.

$$  
\frac{\partial L}{\partial w}
(\hat{y}-y)x  
$$

편향에 대한 gradient는 다음과 같다.

$$  
\frac{\partial L}{\partial b}
\hat{y}-y  
$$

입력 특성이 여러 개인 경우 각 가중치에 대한 gradient는 다음과 같다.

$$  
\frac{\partial L}{\partial w_j}
(\hat{y}-y)x_j  
$$

여기서 중요한 점은 가중치 gradient의 부호가 단순히 $\hat{y}-y$만으로 결정되지 않는다는 것이다.

가중치 gradient에는 입력값 $x_j$도 곱해진다.

따라서 입력값이 양수인지 음수인지에 따라 각 가중치가 업데이트되는 방향은 달라질 수 있다.

반면 편향의 gradient는 다음과 같이 예측 확률과 실제 정답의 차이로 계산된다.

$$  
\frac{\partial L}{\partial b}
\hat{y}-y  
$$

----------

## 15. Optimizer

로지스틱 회귀에서도 선형회귀와 마찬가지로 optimizer를 사용해 가중치와 편향을 업데이트한다.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

```

-   `model.parameters()`: 모델이 학습할 가중치와 편향
    
-   `lr`: 한 번의 업데이트에서 parameter를 변경하는 크기
    

가중치는 다음과 같이 업데이트된다.

$$  
w_j  
:=  
w_j-\eta\frac{\partial L}{\partial w_j}  
$$

편향은 다음과 같이 업데이트된다.

$$  
b  
:=  
b-\eta\frac{\partial L}{\partial b}  
$$

업데이트되는 대상은 모델의 parameter인 가중치와 편향이다.

logit $z$와 예측 확률 $\hat{y}$는 parameter가 아니라 forward 연산 과정에서 계산되는 중간값이다.

```text
업데이트되는 값
- weight
- bias

forward에서 다시 계산되는 값
- logit
- probability
- loss

```

경사하강법은 선형회귀에서만 사용하는 방법이 아니다.

손실함수를 미분할 수 있다면 로지스틱 회귀와 인공신경망을 포함한 다양한 모델에서 사용할 수 있다.

----------

## 16. Training Process

로지스틱 회귀의 학습 과정은 다음 순서로 진행된다.

1.  입력값을 모델에 전달한다.
    
2.  선형 결합을 통해 logit을 계산한다.
    
3.  sigmoid를 통해 클래스 1일 확률을 계산한다.
    
4.  예측 확률과 실제 정답을 이용해 BCE loss를 계산한다.
    
5.  역전파를 통해 가중치와 편향의 gradient를 계산한다.
    
6.  optimizer가 가중치와 편향을 업데이트한다.
    
7.  이 과정을 여러 epoch 동안 반복한다.
    

```python
for epoch in range(num_epochs):
    probability = model(x)
    loss = loss_function(probability, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

### `probability = model(x)`

현재 가중치와 편향을 이용해 logit을 계산하고 sigmoid 함수를 적용한다.

```text
입력 x
    ↓
nn.Linear
    ↓
logit
    ↓
nn.Sigmoid
    ↓
probability

```

### `loss = loss_function(probability, y)`

모델이 예측한 확률과 실제 정답을 비교해 BCE loss를 계산한다.

### `optimizer.zero_grad()`

이전 학습 단계에서 계산된 gradient를 초기화한다.

PyTorch에서는 gradient가 기본적으로 누적되기 때문에 각 학습 단계마다 초기화해야 한다.

### `loss.backward()`

손실값을 기준으로 가중치와 편향에 대한 gradient를 계산한다.

계산된 gradient는 각 parameter의 `.grad` 속성에 저장된다.

```python
print(model[0].weight.grad)
print(model[0].bias.grad)

```

모델이 `nn.Sequential`로 정의되어 있으므로 첫 번째 계층인 `nn.Linear`에 접근하기 위해 `model[0]`을 사용한다.

### `optimizer.step()`

각 parameter의 `.grad`에 저장된 gradient를 이용해 가중치와 편향을 실제로 업데이트한다.

----------

## 17. Prediction and Accuracy

학습이 끝난 후 모델의 분류 결과를 확인하려면 확률에 threshold를 적용한다.

```python
with torch.no_grad():
    probabilities = model(x)
    predictions = (probabilities >= 0.5).float()

```

`torch.no_grad()`는 평가 과정에서 gradient 계산을 비활성화한다.

평가 과정에서는 역전파가 필요하지 않으므로 불필요한 연산과 메모리 사용을 줄일 수 있다.

정확도는 예측 클래스와 실제 정답이 같은 sample의 비율로 계산한다.

```python
correct = (predictions == y).sum().item()
accuracy = correct / len(y)

```

수식으로 표현하면 다음과 같다.

$$  
Accuracy
\frac{\text{올바르게 예측한 sample 수}}  
{\text{전체 sample 수}}  
$$

예를 들어 10개의 sample 중 8개를 올바르게 분류했다면 정확도는 다음과 같다.

```text
accuracy = 8 / 10 = 0.8

```

즉, 정확도는 80%이다.

손실과 정확도는 서로 다른 정보를 나타낸다.

-   loss: 모델이 정답에 얼마나 적절한 확률을 부여했는지 측정
    
-   accuracy: threshold를 적용한 후 몇 개의 클래스를 맞혔는지 측정
    

----------

## 18. Linear Regression and Logistic Regression Comparison

선형회귀와 로지스틱 회귀는 모두 먼저 선형 결합을 계산한다.

$$  
Z=XW+b  
$$

하지만 선형 결합의 결과를 사용하는 방법이 다르다.

### Linear Regression

```text
입력 X
    ↓
XW + b
    ↓
연속적인 예측값

```

선형회귀에서는 선형 결합의 결과를 최종 예측값으로 사용한다.

예측값의 범위에는 제한이 없다.

대표적인 손실함수로 MSE를 사용한다.

### Logistic Regression

```text
입력 X
    ↓
XW + b
    ↓
logit
    ↓
sigmoid
    ↓
클래스 1일 확률

```

로지스틱 회귀에서는 선형 결합의 결과를 sigmoid 함수에 통과시킨다.

sigmoid 출력은 `0`과 `1` 사이의 값이며 클래스 1일 확률로 해석한다.

대표적인 손실함수로 BCE를 사용한다.

----------


## 19. Core Training Flow

로지스틱 회귀의 전체 학습 흐름은 다음과 같다.

```text
입력 데이터 X
    ↓
nn.Linear
    ↓
선형 결합 XW + b
    ↓
logit
    ↓
nn.Sigmoid
    ↓
클래스 1일 확률
    ↓
nn.BCELoss
    ↓
loss 계산
    ↓
loss.backward()
    ↓
gradient 계산
    ↓
optimizer.step()
    ↓
가중치와 편향 업데이트

```

가중치와 편향이 업데이트되면 다음 forward에서 새로운 logit과 확률이 계산된다.

```text
weight와 bias 업데이트
    ↓
새로운 logit 계산
    ↓
새로운 확률 계산
    ↓
새로운 loss 계산

```

평가 과정은 다음과 같다.

```text
입력 데이터 X
    ↓
model(X)
    ↓
probability
    ↓
threshold 적용
    ↓
class 0 또는 class 1

```

----------

## 20. What I Learned

이번 학습을 통해 다음 내용을 확인했다.

-   로지스틱 회귀는 주로 이진 분류 문제에 사용한다.
    
-   클래스는 일반적으로 `0`과 `1`로 표현한다.
    
-   어떤 상태를 클래스 0과 클래스 1로 정의할지는 문제에 따라 결정된다.
    
-   로지스틱 회귀도 먼저 $XW+b$ 형태의 선형 결합을 계산한다.
    
-   선형 결합의 결과를 logit이라고 부른다.
    
-   logit은 아직 확률이 아니며 범위 제한이 없다.
    
-   sigmoid는 logit을 `0`과 `1` 사이의 확률로 변환한다.
    
-   sigmoid 출력은 클래스 1일 확률로 해석할 수 있다.
    
-   클래스 0일 확률은 `1 - probability`로 계산할 수 있다.
    
-   BCE는 예측 확률과 실제 정답 사이의 차이를 측정한다.
    
-   실제 정답 클래스에 낮은 확률을 부여할수록 BCE loss가 커진다.
    
-   이진 분류에서는 하나의 출력값으로 두 클래스의 확률을 표현할 수 있다.
    
-   `nn.Linear(in_features, 1)`은 각 sample에 대해 하나의 logit을 출력한다.
    
-   `nn.Sigmoid()`는 logit을 클래스 1일 확률로 변환한다.
    
-   `nn.BCELoss()`는 예측 확률과 실제 정답을 비교한다.
    
-   optimizer는 가중치와 편향을 업데이트한다.
    
-   가중치와 편향이 바뀌면 다음 forward에서 새로운 logit이 계산된다.
    
-   가중치에 대한 gradient는 예측 오차뿐 아니라 입력값의 영향도 받는다.
    
-   경사하강법은 선형회귀뿐 아니라 로지스틱 회귀에도 적용된다.
