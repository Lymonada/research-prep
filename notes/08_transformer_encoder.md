# Transformer Encoder Block

이 문서는 Multi-Head Self-Attention을 이해한 이후, 그 output이 실제 **Transformer Encoder Block** 안에서 어떻게 사용되는지를 정리한 노트이다.

이전 Attention 노트에서 다음 흐름까지 학습했다.

```text
Input X
↓
Q / K / V Projection
↓
Multi-Head Self-Attention
↓
Concatenate Heads
↓
Output Projection W_O
↓
MHA(X)
```

이제 여기서부터

- Output Projection $W_O$
- Residual Connection
- Residual이 gradient 전달에 도움이 되는 이유
- Layer Normalization
- Feed Forward Network
- 두 번째 Residual + LayerNorm
- Transformer Encoder Block 전체 구조
- Encoder Block의 shape이 유지되는 이유

를 하나의 흐름으로 연결한다.

---

## 1. Starting Point: Multi-Head Attention Output

기준으로 다음 값을 사용한다.

$$
d_{model}=512,\quad
h=8,\quad
d_{head}=64,\quad
d_{ff}=2048
$$

입력 sequence representation은

$$
X:[B,T,512]
$$

이다.

Multi-Head Self-Attention을 거치면 각 head가 서로 다른 projection space에서 sequence 내 token 간 관계를 계산한다.

각 head:

$$
head_i =
softmax
\left(
\frac{Q_iK_i^T}{\sqrt{d_{head}}}
\right)V_i
$$

여러 head를 concatenate한 뒤 $W_O$를 적용하면 최종 Multi-Head Attention output을 얻는다.

$$
MHA(X) =
Concat(head_1,\dots,head_h)W_O
$$

shape은

$$
MHA(X):[B,T,512]
$$

이다.

즉 Multi-Head Attention을 지나도 입력과 output의 shape이 동일하다.

```text
Input X
[B,T,512]

        ↓ MHA

MHA(X)
[B,T,512]
```

이 shape이 유지되는 것은 이후 Residual Connection을 사용할 수 있게 해주고, Encoder Block을 여러 층 연속해서 쌓을 수 있게 하는 데 중요하다.

---

## 2. What Does $W_O$ Do?

각 head의 output은

$$
[B,T,64]
$$

이고, 8개의 head를 concatenate하면

$$
[B,T,512]
$$

가 된다.

token 하나만 보면 다음과 같은 형태이다.

$$
Concat
(
z_i^{(1)},
z_i^{(2)},
\dots,
z_i^{(8)}
)
$$

즉,

```text
head 1 information
|
head 2 information
|
head 3 information
|
...
|
head 8 information
```

을 단순히 옆으로 붙여놓은 상태라고 볼 수 있다.

여기에

$$
W_O:[512,512]
$$

를 적용한다.

$$
Concat(head_1,\dots,head_8)W_O
$$

이 projection을 통해 여러 head에서 얻은 feature가 다시 서로 섞인다.

따라서 $W_O$는

> 여러 head가 서로 다른 관점에서 가져온 정보를 하나의 통합된 token representation으로 다시 조합하는 output projection

으로 이해할 수 있다.

---

## 3. Why Not Send $MHA(X)$ Directly to the Next Layer?

Multi-Head Attention만 사용한다면

```text
X
↓
MHA
↓
MHA(X)
```

처럼 끝낼 수도 있다.

하지만 Transformer Encoder에서는 원래 입력 $X$를 다시 더한다.

$$
X + MHA(X)
$$

이 구조가 **Residual Connection**, 또는 **Skip Connection**이다.

입력과 MHA output은 모두

$$
[B,T,512]
$$

이므로 element-wise addition이 가능하다.

결과 shape도

$$
[B,T,512]
$$

이다.

---

## 4. Residual Connection

Residual Connection의 기본 형태는

$$
y=x+F(x)
$$

이다.

Transformer의 첫 번째 sublayer에서는

$$
F(x)=MHA(x)
$$

라고 보면 된다.

따라서

$$
y=x+MHA(x)
$$

이다.

---

## 5. Residual Connection: Forward View

forward 계산 관점에서 Residual Connection은 직관적으로

> **기존 representation + 새롭게 계산한 정보**

라고 볼 수 있다.

Attention은 token이 다른 token을 참고해 새로운 contextual information을 만든다.

그런데 기존 token representation을 완전히 버리고 Attention output으로 덮어쓰는 대신,

$$
x_i + MHA(x_i)
$$

처럼 기존 representation을 직접 남겨둔다.

여기서 $MHA(x_i)$라는 표현은 단순히 token $i$ 하나만 계산했다는 뜻은 아니다. 실제로는 token $i$가 sequence의 다른 token들을 참고해서 만들어진 Multi-Head Attention output을 의미한다.

따라서 직관적으로는,

```text
기존 token representation
+
다른 token들을 참고해서 얻은 contextual update
```

라고 생각할 수 있다.

---

## 6. Residual Connection: Backward View

Residual Connection에는 forward에서 정보를 보존하는 역할뿐 아니라, 깊은 network를 학습할 때 gradient 전달을 안정화하는 효과도 있다.

여기서 흔히

> "gradient가 identity path를 통해 직접 흐른다"

라고 표현한다.

하지만 이 표현은 비유적이므로, 실제 의미를 chain rule 관점에서 이해하는 것이 중요하다.

---

## 7. What Does "Gradient Flows" Mean?

gradient가 실제 정보 덩어리처럼 network 안을 이동하는 것은 아니다.

"gradient가 흐른다"는 표현은 정확히는

> 계산 그래프를 역방향으로 따라가면서 **chain rule에 의해 derivative를 곱하고, 여러 경로의 gradient contribution을 더하는 과정**

을 의미한다.

즉 gradient flow는 backpropagation에서 이루어지는 chain rule 계산을 직관적으로 표현한 말이다.

---

## 8. Without a Residual Connection

먼저

$$
y=F(x)
$$

라고 하자.

Loss를 $L$이라고 하면,

$$
\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
$$

이고,

$$
y=F(x)
$$

이므로,

$$
\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial y}
F'(x)
$$

이다.

즉 위쪽 layer에서 계산된 upstream gradient

$$
\frac{\partial L}{\partial y}
$$

가 이전 쪽으로 전달될 때 반드시

$$
F'(x)
$$

와 곱해진다.

---

## 9. Why Can This Become a Problem?

문제는 $F(x)$라는 함수를 사람이 미분하기 어렵다는 것이 아니다.

PyTorch에서는 autograd가 derivative를 계산한다.

핵심 문제는 **여러 layer의 derivative가 chain rule에 의해 반복적으로 곱해진다는 것**이다.

단순한 scalar 예시로 각 layer의 derivative가

$$
0.5
$$

라고 하자.

10개의 transformation을 지나면,

$$
0.5^{10}
\approx
0.00098
$$

이 된다.

따라서 앞쪽 layer로 갈수록 gradient가 매우 작아질 수 있다.

실제 neural network에서는 scalar derivative 대신 Jacobian이 등장하므로 더 복잡하지만, 핵심은 동일하다.

> 깊은 network에서는 여러 transformation의 derivative가 계속 곱해지기 때문에 gradient가 지나치게 작아지거나 커질 수 있다.

---

## 10. With a Residual Connection

이번에는

$$
y=x+F(x)
$$

이다.

이를 $x$에 대해 미분하면,

$$
\frac{\partial y}{\partial x} =
1+F'(x)
$$

이다.

따라서,

$$
\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial y}
\left(
1+F'(x)
\right)
$$

이고 이를 풀어 쓰면,

$$
\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial y}
+
\frac{\partial L}{\partial y}F'(x)
$$

이다.

즉 두 contribution이 존재한다.

### Identity Path

$$
\frac{\partial L}{\partial y}
\times 1
$$

### Transformation Path

$$
\frac{\partial L}{\partial y}
\times F'(x)
$$

그리고 이 두 contribution을 더한다.

---

## 11. What Does the "Derivative 1 Path" Mean?

Residual Connection에서 $x$가 addition node로 직접 연결되어 있다.

forward:

```text
                 F(x)
               ↗
x ───────────────→ +
                   ↓
                   y
```

addition은

$$
y=a+b
$$

일 때,

$$
\frac{\partial y}{\partial a}=1,
\quad
\frac{\partial y}{\partial b}=1
$$

이다.

따라서 backpropagation에서는 upstream gradient가

- 한쪽에서는 $F'(x)$의 영향을 받고
- 다른 한쪽에서는 $1$과 곱해져

$x$에 대한 gradient 계산에 함께 기여한다.

즉,

> "gradient가 1을 통해 흐른다"

는 말은

> **upstream gradient가 $\times 1$ 된 항으로 그대로 이전 representation의 gradient 계산에 포함된다**

는 뜻이다.

정보가 따로 이동하는 것이 아니다.

---

## 12. Both Paths Are Used

Residual Connection이 있다고 해서 모든 gradient가 identity path만 사용하는 것은 아니다.

두 경로를 **모두 사용한다.**

$$
\frac{\partial L}{\partial x} =
\underbrace{
\frac{\partial L}{\partial y}
}_{identity}
+
\underbrace{
\frac{\partial L}{\partial y}F'(x)
}_{transformation}
$$

Residual은 $F'(x)$ 경로를 제거하지 않는다.

대신,

> $F'(x)$ 경로 **외에도** upstream gradient가 직접 기여할 수 있는 identity term을 추가한다.

따라서 transformation의 derivative가 작더라도 이전 layer가 받을 gradient가 반드시 그 값만큼 작아질 필요는 없다.

---

## 13. Simple Numerical Example

upstream gradient가

$$
\frac{\partial L}{\partial y}=0.8
$$

이고,

$$
F'(x)=0.05
$$

라고 하자.

Residual이 없으면,

$$
\frac{\partial L}{\partial x} =
0.8\times0.05 =
0.04
$$

이다.

Residual이 있으면,

$$
\frac{\partial L}{\partial x} =
0.8\times1
+
0.8\times0.05 =
0.8+0.04 =
0.84
$$

이다.

즉 $F'(x)$를 통한 gradient contribution도 그대로 계산하지만, 동시에 identity path를 통한 contribution도 존재한다.

---

## 14. Residual Does Not Completely Eliminate Vanishing Gradient

Residual Connection이 있다고 해서 vanishing gradient가 절대 발생하지 않는다는 뜻은 아니다.

실제 neural network에서는 vector와 matrix가 사용되기 때문에

$$
1+F'(x)
$$

라는 scalar 표현보다는

$$
I + J_F(x)
$$

형태의 Jacobian으로 생각하는 것이 더 정확하다.

여러 방향의 gradient가 서로 상쇄될 수도 있고, network의 다른 부분에서 optimization 문제가 발생할 수도 있다.

따라서 Residual Connection은

> vanishing gradient를 완전히 제거하는 장치

라기보다,

> **깊은 network에서 gradient 전달과 optimization을 훨씬 안정적으로 만들어주는 구조**

라고 이해하는 것이 정확하다.

---

## 15. Connection to RNN / LSTM / GRU

Residual Connection을 배우면 RNN의 vanishing gradient 문제와 연결되는 의문이 생길 수 있다.

Vanilla RNN은

$$
h_t=F(h_{t-1},x_t)
$$

와 같은 형태이고, 시간축을 따라

```text
h1 → h2 → h3 → ... → hT
```

많은 recurrent transformation을 거친다.

backpropagation에서는 여러 timestep의 derivative가 계속 곱해지므로 vanishing / exploding gradient 문제가 발생할 수 있다.

---

## 16. Could Residual Connections Be Used in RNNs?

가능하다.

예를 들어 개념적으로,

$$
h_t =
h_{t-1}
+
F(h_{t-1},x_t)
$$

같은 recurrent residual 구조를 생각할 수 있고, 실제로 residual / skip connection을 사용하는 recurrent architecture도 존재한다.

하지만 단순 residual 하나만으로 RNN의 장기 dependency 문제 전체가 해결되지는 않는다.

RNN에서는

- 무엇을 기억할지
- 무엇을 버릴지
- 무엇을 새롭게 반영할지

를 시간축에서 조절하는 것도 중요하다.

---

## 17. Relation to LSTM

LSTM의 cell-state update는

$$
c_t =
f_t\odot c_{t-1}
+
i_t\odot\tilde{c}_t
$$

형태이다.

이를 보면,

```text
기존 cell state를 보존하는 경로
+
새로운 information
```

라는 additive structure가 있다.

특히

$$
f_t\approx1
$$

이라면 이전 cell state가 다음 timestep으로 거의 그대로 전달될 수 있다.

따라서 LSTM의 cell state는 Residual Connection과 완전히 동일한 구조는 아니지만,

> **이전 state를 직접 보존할 수 있는 additive path를 제공하여 information과 gradient 전달을 쉽게 한다**

는 철학적인 공통점이 있다.

---

## 18. Relation to GRU

GRU의 update도 대략

$$
h_t =
(1-z_t)\odot h_{t-1}
+
z_t\odot\tilde{h}_t
$$

와 같이 볼 수 있다.

즉

```text
기존 hidden state
+
새롭게 계산된 candidate state
```

를 gate로 조절하여 결합한다.

따라서 Residual Connection, LSTM cell-state update, GRU hidden-state update는 동일한 architecture는 아니지만,

> 이전 representation을 완전히 덮어쓰기보다 직접 보존할 수 있는 additive path를 둔다

는 공통된 intuition을 가지고 있다.

---

## 19. Layer Normalization

Residual Connection 이후에는 Layer Normalization을 적용한다.

원래 Transformer의 Post-Norm 형태를 기준으로 하면,

$$
Y =
LayerNorm
\left(
X+MHA(X)
\right)
$$

이다.

shape은

$$
Y:[B,T,512]
$$

로 유지된다.

---

## 20. What Does LayerNorm Normalize?

입력 shape이

$$
[B,T,512]
$$

라고 하자.

LayerNorm은 **각 token의 512개 feature**를 기준으로 평균과 분산을 계산한다.

즉,

```text
Token 1: [512 features] → normalize
Token 2: [512 features] → normalize
Token 3: [512 features] → normalize
...
```

이다.

token끼리 평균을 내는 것이 아니다.

각 token representation 내부의 feature dimension을 normalize한다.

---

## 21. LayerNorm Formula

한 token의 representation을

$$
x =
[x_1,x_2,\dots,x_{512}]
$$

라고 하자.

평균:

$$
\mu =
\frac{1}{512}
\sum_{j=1}^{512}x_j
$$

분산:

$$
\sigma^2 =
\frac{1}{512}
\sum_{j=1}^{512}
(x_j-\mu)^2
$$

normalize:

$$
\hat{x}_j =
\frac{x_j-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

이후 학습 가능한 parameter를 이용해 다시 조정한다.

$$
y_j =
\gamma_j\hat{x}_j+\beta_j
$$

여기서

$$
\gamma,\beta
$$

는 학습 가능한 parameter이다.

---

## 22. Is LayerNorm "Flattening" the Values?

LayerNorm을 단순히 값을 평탄하게 만드는 연산으로 이해하는 것은 조금 부정확하다.

예를 들어,

$$
[1,2,3]
$$

과

$$
[100,200,300]
$$

은 scale이 매우 다르다.

하지만 각각 자신의 평균과 표준편차를 기준으로 normalize하면 비슷한 normalized pattern을 가질 수 있다.

LayerNorm은

> 큰 값을 모두 작게 만들거나 feature 차이를 없애는 것

이 아니라,

> **token 내부 feature들의 상대적 pattern을 유지하면서 전체적인 중심과 scale을 일정하게 정돈하는 것**

에 가깝다.

따라서 "평탄화"보다

> **representation의 눈금과 중심을 맞춘다**

는 intuition이 더 적절하다.

---

## 23. Why Learnable $\gamma$ and $\beta$?

normalize한 뒤 항상 평균 0, 분산 1인 representation만 강제하면 모델이 필요한 표현 범위를 제한할 수 있다.

따라서

$$
\gamma
$$

와

$$
\beta
$$

를 학습하게 하여,

> 먼저 안정적인 기준으로 normalize하고, 필요한 scale과 위치는 모델이 다시 학습하도록 한다.

---

## 24. What LayerNorm Does Not Do

LayerNorm은 token끼리 정보를 교환하지 않는다.

즉,

```text
Token 1
Token 2
Token 3
```

사이의 관계를 계산하는 것은 Attention의 역할이다.

LayerNorm은 각각의 token 안에서

```text
[feature 1, feature 2, ..., feature 512]
```

의 distribution을 normalize한다.

따라서,

> Attention = token 간 information exchange  
> LayerNorm = token 내부 feature scale stabilization

으로 구분할 수 있다.

---

## 25. First Add & Norm

지금까지를 합치면 Encoder Block의 첫 번째 절반은 다음과 같다.

```text
X
│
├─────────────────────────────┐
│                             │
▼                             │
Multi-Head Self-Attention     │
│                             │
▼                             │
MHA(X)                        │
│                             │
└────────────── + ◀───────────┘
                │
                ▼
           LayerNorm
                │
                ▼
                Y
```

수식:

$$
Y =
LayerNorm
\left(
X+MHA(X)
\right)
$$

shape:

$$
Y:[B,T,512]
$$

이 구조를 흔히 **Add & Norm**이라고 부른다.

- Add = Residual Connection
- Norm = Layer Normalization

---

## 26. Feed Forward Network

첫 번째 Add & Norm 이후에는 **Position-wise Feed Forward Network (FFN)**가 등장한다.

Transformer의 기본 FFN은 작은 MLP라고 볼 수 있다.

원래 Transformer에서는 다음과 같은 형태이다.

$$
FFN(x) =
ReLU(xW_1+b_1)W_2+b_2
$$

기준:

$$
d_{model}=512
$$

$$
d_{ff}=2048
$$

이면,

```text
512
↓
Linear
↓
2048
↓
ReLU
↓
Linear
↓
512
```

구조이다.

---

## 27. FFN Shape Flow

입력:

$$
Y:[B,T,512]
$$

첫 번째 Linear:

$$
W_1:[512,2048]
$$

따라서,

$$
[B,T,512]
\rightarrow
[B,T,2048]
$$

Activation 이후에도,

$$
[B,T,2048]
$$

이다.

두 번째 Linear:

$$
W_2:[2048,512]
$$

이므로,

$$
[B,T,2048]
\rightarrow
[B,T,512]
$$

최종적으로,

$$
FFN(Y):[B,T,512]
$$

가 된다.

---

## 28. Attention vs FFN

Transformer Encoder에서 Attention과 FFN은 서로 다른 역할을 한다.

### Attention

Attention은 token들 사이에서 정보를 교환한다.

token $i$의 새로운 representation을 만들 때 다른 token들을 참고한다.

```text
token 1 ─┐
token 2 ─┼──→ token i
token 3 ─┤
...      │
token T ─┘
```

즉 **token mixing**을 수행한다.

### FFN

FFN은 각 token을 독립적으로 처리한다.

```text
token 1 [512] → FFN → [512]
token 2 [512] → FFN → [512]
token 3 [512] → FFN → [512]
```

token 1의 FFN 계산에서 token 2를 직접 참고하지 않는다.

대신 한 token의 representation 안에 있는 feature들을 복잡하게 변환한다.

즉 **feature processing / feature mixing**을 수행한다.

---

## 29. A Useful Mental Model

Attention과 FFN의 역할을 다음처럼 구분할 수 있다.

> **Attention:**  
> "다른 token들 중 어떤 정보를 가져와야 하는가?"

> **FFN:**  
> "그렇게 모은 정보를 이 token 내부에서 어떻게 가공할 것인가?"

Attention을 거친 token은 이미 다른 token들의 문맥 정보를 포함하고 있다.

FFN은 그 contextual representation을 받아 각 token 내부 feature를 더 복잡하고 유용한 representation으로 바꾼다.

---

## 30. Why Expand from 512 to 2048?

FFN은

$$
512\rightarrow2048\rightarrow512
$$

처럼 중간 hidden dimension을 크게 확장한다.

첫 번째 Linear layer는 기존 512개의 feature를 조합해 더 넓은 intermediate feature space를 만든다.

직관적으로는,

```text
512-dimensional representation
↓
many different feature combinations
↓
2048-dimensional intermediate representation
```

이라고 볼 수 있다.

더 넓은 hidden dimension은 다양한 intermediate feature combination을 표현할 capacity를 제공한다.

---

## 31. Why Is Nonlinearity Important?

만약 activation function이 없다면,

$$
xW_1W_2
$$

가 된다.

이때

$$
W_1W_2=W
$$

라고 합칠 수 있으므로,

$$
xW
$$

라는 하나의 Linear transformation과 본질적으로 같아진다.

즉 단순히

$$
512\rightarrow2048\rightarrow512
$$

로 차원을 넓혔다 줄이는 것만으로는 충분하지 않다.

중간의

$$
ReLU
$$

같은 nonlinear activation 때문에 두 Linear layer를 하나의 Linear transformation으로 합칠 수 없게 된다.

따라서,

> **넓은 intermediate space + nonlinearity**

가 FFN이 복잡한 representation transformation을 학습할 수 있게 하는 핵심이다.

---

## 32. What Is the FFN Actually Doing?

Attention을 거친 token representation 안에는 이미

- 현재 token 자체의 정보
- 주변 token에서 가져온 정보
- 멀리 떨어진 token과의 관계
- 여러 attention head가 수집한 contextual information

등이 복잡하게 포함되어 있을 수 있다.

FFN의 첫 번째 Linear는 이런 feature들을 여러 방식으로 조합한다.

개념적인 예로,

```text
feature A = combination of several existing features
feature B = another combination
feature C = another combination
...
```

와 같은 intermediate feature를 만들 수 있다.

Activation은 여기에 nonlinearity를 추가하고, 두 번째 Linear는 이를 다시 $d_{model}$ 차원의 representation으로 조합한다.

실제 model feature 하나가 사람이 해석할 수 있는 특정 개념 하나와 정확히 대응한다고 가정해서는 안 되지만,

> **Attention으로 얻은 contextual information을 token 내부에서 더 복잡하게 가공한다**

는 intuition은 유용하다.

---

## 33. Shared FFN Across Tokens

FFN은 token마다 독립적으로 적용되지만, 각 token마다 서로 다른 FFN parameter가 존재하는 것은 아니다.

모든 token이 같은

$$
W_1,\quad W_2
$$

를 공유한다.

예를 들어 PyTorch에서

```python
ffn = nn.Linear(512, 2048)
```

에

```text
[B,T,512]
```

tensor를 넣으면 `nn.Linear`는 마지막 dimension에 적용된다.

즉 개념적으로,

```text
x[:,0,:] → same W1
x[:,1,:] → same W1
x[:,2,:] → same W1
...
```

이다.

따라서 sequence 내 모든 위치에서 동일한 FFN transformation을 사용한다.

---

## 34. Second Residual Connection

FFN을 거친 뒤에도 Residual Connection을 사용한다.

첫 번째 Add & Norm output을

$$
Y
$$

라고 하면,

$$
FFN(Y)
$$

의 shape도

$$
[B,T,512]
$$

이다.

따라서

$$
Y+FFN(Y)
$$

를 계산할 수 있다.

이 Residual Connection도 첫 번째 Residual과 같은 역할을 한다.

- 기존 representation을 직접 보존
- 새롭게 계산된 FFN update를 추가
- gradient 전달과 optimization 안정화

---

## 35. Second Layer Normalization

두 번째 Residual 이후 다시 LayerNorm을 적용한다.

원래 Transformer의 Post-Norm 형태라면,

$$
Z =
LayerNorm
\left(
Y+FFN(Y)
\right)
$$

이다.

shape은

$$
Z:[B,T,512]
$$

이다.

이 $Z$가 **Transformer Encoder Block 하나의 최종 output**이다.

---

## 36. Complete Transformer Encoder Block

이제 Encoder Block 전체를 처음부터 끝까지 연결할 수 있다.

```text
X
[B,T,512]
│
│
├─────────────────────────────┐
│                             │
▼                             │
Multi-Head Self-Attention     │
│                             │
▼                             │
MHA(X)                        │
[B,T,512]                     │
│                             │
└────────────── + ◀───────────┘
                │
                ▼
           LayerNorm
                │
                ▼
                Y
          [B,T,512]
                │
                │
├─────────────────────────────┐
│                             │
▼                             │
FFN                           │
512 → 2048 → 512              │
│                             │
▼                             │
FFN(Y)                        │
[B,T,512]                     │
│                             │
└────────────── + ◀───────────┘
                │
                ▼
           LayerNorm
                │
                ▼
                Z
          [B,T,512]
```

수식으로는,

$$
Y =
LayerNorm
\left(
X+MHA(X)
\right)
$$

$$
Z =
LayerNorm
\left(
Y+FFN(Y)
\right)
$$

이다.

---

## 37. Complete Shape Flow

기준:

$$
d_{model}=512,\quad
h=8,\quad
d_{head}=64,\quad
d_{ff}=2048
$$

전체 흐름:

```text
Input
[B,T,512]

↓ Multi-Head Self-Attention

[B,T,512]

↓ Residual Add

[B,T,512]

↓ LayerNorm

[B,T,512]

↓ FFN Linear 1

[B,T,2048]

↓ Activation

[B,T,2048]

↓ FFN Linear 2

[B,T,512]

↓ Residual Add

[B,T,512]

↓ LayerNorm

Encoder Block Output
[B,T,512]
```

---

## 38. Why Does the Shape Keep Returning to $[B,T,d_{model}]$?

Transformer Encoder Block의 입력은

$$
[B,T,d_{model}]
$$

이고 output도

$$
[B,T,d_{model}]
$$

이다.

이 구조에는 두 가지 중요한 장점이 있다.

### 1. Residual Addition

Residual Connection에서

$$
x+F(x)
$$

를 계산하려면 두 tensor의 shape이 같아야 한다.

따라서 MHA와 FFN 모두 최종적으로 $d_{model}$ dimension으로 돌아온다.

### 2. Stack Multiple Encoder Blocks

한 Encoder Block의 output이 다음 Encoder Block의 input과 같은 shape이므로,

```text
Encoder Block 1
↓
Encoder Block 2
↓
Encoder Block 3
↓
...
```

처럼 쉽게 여러 층을 쌓을 수 있다.

---

## 39. Why Stack Multiple Encoder Blocks?

한 Encoder Block을 지나면 token들은 이미 다른 token의 정보를 참고한 contextual representation이 된다.

그런데 다음 Encoder Block에서는 **이미 contextualized된 representation끼리 다시 Self-Attention**을 수행한다.

즉 첫 번째 block에서 얻은 관계를 기반으로 다음 block에서 더 복잡한 관계를 만들 수 있다.

개념적으로,

```text
Initial token embeddings
↓
Encoder Block 1
basic contextual relationships
↓
Encoder Block 2
relationships between contextualized representations
↓
Encoder Block 3
more abstract / higher-level representation
↓
...
```

와 같이 이해할 수 있다.

각 layer에서 shape은 같지만 representation의 **내용과 의미는 계속 변한다.**

---

## 40. The Core Roles Inside an Encoder Block

Transformer Encoder Block의 구성 요소를 역할별로 압축하면 다음과 같다.

### Multi-Head Self-Attention

> 다른 token들을 참고하여 contextual information을 가져온다.

즉 **token mixing**을 담당한다.

### Residual Connection

> 기존 representation을 직접 보존하면서 새로운 update를 더하고, 깊은 network에서 gradient 전달과 optimization을 안정화한다.

### LayerNorm

> 각 token 내부의 feature distribution을 normalize하여 representation의 중심과 scale을 안정화한다.

### FFN

> Attention으로 문맥 정보가 반영된 각 token representation을 독립적으로 비선형 변환한다.

즉 **feature processing / feature mixing**을 담당한다.

---

## 41. One-Sentence View of the Encoder Block

한 Encoder Block을 가장 압축해서 보면,

> **Self-Attention으로 token들 사이의 정보를 섞고, FFN으로 각 token 내부 feature를 가공하며, Residual Connection과 LayerNorm으로 이 과정을 안정적으로 반복할 수 있게 한다.**

라고 정리할 수 있다.

---

## 42. Post-Norm vs Pre-Norm

지금까지는 원래 2017 Transformer의 흐름을 기준으로

$$
LayerNorm(x+Sublayer(x))
$$

형태를 사용했다.

이를 **Post-Norm**이라고 한다.

예:

$$
Y =
LayerNorm
\left(
X+MHA(X)
\right)
$$

하지만 현대 Transformer architecture에서는 LayerNorm을 sublayer 앞에 두는 **Pre-Norm**도 매우 흔하다.

개념적으로,

$$
Y =
X
+
MHA(LayerNorm(X))
$$

같은 형태이다.

FFN도 마찬가지로,

$$
Z =
Y
+
FFN(LayerNorm(Y))
$$

와 같이 구성할 수 있다.

현재 단계에서는 Post-Norm과 Pre-Norm의 세부적인 optimization 차이를 깊게 다룰 필요는 없다.

핵심은,

> **Residual Connection과 LayerNorm의 순서는 architecture에 따라 달라질 수 있다.**

는 점을 알고 있는 것이다.

이 노트에서는 Transformer의 기본 구조를 이해하기 위해 원래 논문의 Post-Norm 흐름을 기준으로 설명했다.

---

## 43. Important Distinctions and Common Confusions

### Residual does not mean "use only the identity path"

Residual Connection에서는

$$
1
$$

경로와

$$
F'(x)
$$

경로가 모두 사용된다.

identity path가 transformation path를 대체하는 것이 아니다.

### "Gradient flow" is not information physically moving

gradient flow는

> 계산 그래프를 역방향으로 따라 chain rule을 계산하는 과정

을 의미하는 비유적 표현이다.

### A complex $F(x)$ is not problematic because differentiation is difficult

문제는 사람이 $F(x)$를 미분하기 어렵다는 것이 아니라, 깊은 network에서 derivative/Jacobian이 반복적으로 곱해진다는 것이다.

### LayerNorm does not normalize across tokens

각 token의 $d_{model}$ feature를 normalize한다.

```text
[B,T,d_model]
      ↑
normalized dimension
```

### LayerNorm does not simply flatten all values

feature의 상대적 pattern을 없애는 것이 아니라 중심과 scale을 정돈한다.

### FFN does not exchange information between tokens

token 간 정보 교환은 Attention의 역할이다.

FFN은 각 token을 독립적으로 처리한다.

### Expanding to $d_{ff}$ alone is not enough

FFN의 핵심은 단순히

$$
512\rightarrow2048\rightarrow512
$$

가 아니라,

$$
Linear
\rightarrow
Nonlinearity
\rightarrow
Linear
$$

구조이다.

activation이 없다면 두 Linear layer는 하나로 합칠 수 있다.

---

## 44. Final Conceptual Summary

Transformer Encoder Block을 이해하기 위한 전체 흐름은 다음과 같다.

```text
Input X
[B,T,d_model]

↓
Multi-Head Self-Attention
token-to-token information exchange

↓
Residual Connection
old representation + contextual update

↓
LayerNorm
feature scale stabilization

↓
Feed Forward Network
token-wise nonlinear feature transformation

↓
Residual Connection
old representation + FFN update

↓
LayerNorm

↓
Encoder Block Output
[B,T,d_model]
```

이를 역할 중심으로 다시 압축하면,

```text
Attention
= token mixing

FFN
= feature processing

Residual
= preserve + update + easier optimization

LayerNorm
= stabilize representation scale
```

이다.

---

## 45. From One Encoder Block to the Full Encoder

지금까지는 Transformer **Encoder Block 하나**의 내부 구조를 이해했다.

하지만 실제 Transformer Encoder는 보통 이 block 하나로 끝나지 않는다.

전체 Encoder를 이해하려면 block 앞의 입력 준비 과정과 block을 여러 층 쌓는 구조까지 연결해야 한다.

전체 흐름은 다음과 같다.

```text
Token IDs
↓
Token Embedding
↓
+ Positional Encoding
↓
X^(0) [B,T,d_model]
↓
Encoder Block 1
↓
X^(1) [B,T,d_model]
↓
Encoder Block 2
↓
X^(2) [B,T,d_model]
↓
...
↓
Encoder Block N
↓
Encoder Output X^(N)
[B,T,d_model]
```

---

## 46. Token Embedding Before the Encoder

Transformer에 text token을 그대로 넣을 수는 없다.

먼저 tokenizer가 각 token에 **token ID**를 부여한다.

예를 들어 개념적으로,

```text
"I"    → 17
"love" → 203
"cats" → 891
```

처럼 표현할 수 있다.

여기서 `17`, `203`, `891`이라는 숫자 자체에 의미가 있는 것은 아니다.  
이 값들은 vocabulary 안에서 해당 token을 찾기 위한 **번호표 / index**에 가깝다.

모델에는 학습 가능한 embedding matrix가 있다.

$$
E:[V,d_{model}]
$$

여기서 $V$는 vocabulary size이다.

각 token ID는 embedding matrix에서 하나의 $d_{model}$차원 vector를 가져온다.

```text
Token
↓
Token ID
↓
Embedding lookup
↓
Embedding vector [d_model]
```

sequence 전체와 batch까지 고려하면,

$$
X_{embed}:[B,T,d_{model}]
$$

이다.

Embedding vector는 사람이 미리 각 의미를 직접 넣어주는 것이 아니라, 모델 학습 과정에서 다른 parameter들과 함께 backpropagation으로 학습된다.

현재 Transformer 구조를 이해하는 단계에서는

> **token ID는 index이고, Token Embedding은 그 token을 $d_{model}$차원의 학습 가능한 representation으로 바꾸는 과정**

정도로 이해하면 충분하다.

---

## 47. Why Positional Encoding Is Needed

RNN은 sequence를 시간 순서대로 처리한다.

```text
x1 → h1 → x2 → h2 → x3 → h3 → ...
```

따라서 recurrent computation 구조 자체에 순서가 포함되어 있다.

반면 Self-Attention은 모든 token의 Q/K/V를 병렬적으로 만들고 token 간 관계를 한꺼번에 계산할 수 있다.

즉 Self-Attention 연산 자체에는

> "이 token이 먼저 나왔는가, 나중에 나왔는가"

라는 순서 정보가 자동으로 들어 있지 않다.

그래서 token representation에 별도의 **position signal**을 넣어준다.

---

## 48. Adding Token and Position Information

Token Embedding을

$$
X_{embed}:[B,T,d_{model}]
$$

이라고 하자.

각 sequence position에 대응하는 Positional Encoding은 개념적으로

$$
P:[T,d_{model}]
$$

이다.

두 값을 element-wise로 더한다.

$$
X^{(0)} = X_{embed}+P
$$

broadcasting을 이용하면 같은 positional encoding을 각 batch에 적용할 수 있으므로 결과 shape은 그대로

$$
X^{(0)}:[B,T,d_{model}]
$$

이다.

예를 들어 같은 `cat` token이라도 위치가 다르면,

$$
e_{cat}+p_1
$$

과

$$
e_{cat}+p_5
$$

처럼 Transformer가 받는 input representation 자체가 달라진다.

따라서 모델은 token의 내용뿐 아니라 position에 따른 차이도 attention 계산에 사용할 수 있다.

---

## 49. Does Adding Position Destroy Token Information?

Token Embedding과 Positional Encoding을 더하면 두 정보가 한 vector 안에 섞인다.

하지만 Transformer가 이 둘을 다시 사람처럼

```text
이 부분 = token meaning
이 부분 = position
```

으로 명시적으로 분리할 필요는 없다.

Self-Attention의 projection을 보면,

$$
Q=XW_Q,
\quad
K=XW_K,
\quad
V=XW_V
$$

이고,

$$
X=E+P
$$

이므로 예를 들어

$$
Q=(E+P)W_Q
$$

가 된다.

즉 Q/K/V projection부터 이미 **token information과 positional information이 결합된 representation**을 입력으로 받는다.

모델의 learned weights는 학습을 통해 이 결합된 정보를 task에 유용한 방식으로 활용한다.

중요한 intuition은,

> 모델이 token과 position을 다시 완벽하게 분리하는 법을 배우는 것이 아니라, **"이 token이 이 위치에 있다"는 결합된 representation을 유용하게 사용하는 법을 학습한다**

는 것이다.

---

## 50. Sinusoidal Positional Encoding

원래 Transformer에서는 sin/cos 함수를 이용한 positional encoding을 사용했다.

$$
PE(pos,2i)=
\sin\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

$$
PE(pos,2i+1)=
\cos\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

이 식 자체를 암기하는 것이 현재 학습의 목표는 아니다.

핵심은 각 dimension이 서로 다른 주기의 sin/cos pattern을 사용하기 때문에 position이 변할 때 positional vector도 **규칙적인 방식으로 변한다**는 것이다.

완전히 random한 vector를 각 position에 붙여도 position 자체를 서로 구별하는 것은 가능하다.

하지만 random vector에는 위치 사이의 관계에 특별한 구조가 없다.

반면 sinusoidal encoding에서는 위치가 이동할 때 vector가 규칙적으로 변하므로 모델이

- 상대적인 위치
- 위치 사이의 이동
- 거리와 관련된 pattern

을 활용하기 좋은 구조를 제공한다.

이것이 모델에게 `3칸 차이`라는 정수를 직접 제공한다는 뜻은 아니다.  
정확히는 **위치 사이의 상대적 관계를 학습에 활용할 수 있는 structured signal을 제공한다**는 의미이다.

---

## 51. Where Positional Encoding Enters

Positional Encoding은 Multi-Head Attention이 끝난 뒤에 넣는 것이 아니다.

**첫 Encoder Block에 들어가기 전에** Token Embedding에 더한다.

```text
Token IDs
↓
Token Embedding
[B,T,d_model]

        +

Positional Encoding
[T,d_model]

        ↓

X^(0)
[B,T,d_model]

        ↓

Encoder Block 1
```

그래야 첫 Self-Attention에서 Q/K/V를 만들 때부터 position information을 사용할 수 있다.

---

## 52. Multi-Head vs Multiple Encoder Layers

한 Encoder Block 안의 Multi-Head Attention도 이미 여러 관계를 병렬적으로 학습할 수 있다.

그렇다면 왜 Encoder Block 자체를 다시 여러 층 쌓는지가 중요한 질문이다.

둘의 역할은 다음처럼 구분하면 좋다.

> **Multi-Head:** 같은 layer의 같은 input representation을 여러 learned projection space에서 병렬적으로 바라본다.

> **Multiple Encoder Layers:** 한 layer에서 이미 contextualized된 representation을 다음 layer의 새로운 input으로 사용하여 다시 Attention과 FFN을 수행한다.

첫 block에서는

$$
X^{(0)}
\rightarrow
Encoder_1
\rightarrow
X^{(1)}
$$

이다.

두 번째 block은 원래 embedding을 다시 보는 것이 아니라,

$$
X^{(1)}
\rightarrow
Encoder_2
\rightarrow
X^{(2)}
$$

처럼 **이미 한 번 문맥화된 representation들 사이의 관계를 다시 계산**한다.

따라서 여러 layer를 쌓는다는 것은 단순히 같은 관계를 더 많이 보는 것만을 의미하지 않는다.

- 이미 만들어진 관계를 다시 해석할 수 있고
- 여러 관계를 조합할 수 있으며
- representation을 반복적으로 refinement할 수 있다.

단, 특정 layer가 반드시 `문법`, 다음 layer가 반드시 `의미`처럼 고정된 역할을 가진다고 이해하면 안 된다.

---

## 53. Why the Shape Stays the Same Across Encoder Blocks

Encoder Block 하나의 input과 output은 모두

$$
[B,T,d_{model}]
$$

이다.

Multi-Head Attention은 head들을 concatenate한 뒤 $W_O$를 통해 다시 $d_{model}$로 돌아온다.

FFN도 내부적으로

$$
d_{model}
\rightarrow
d_{ff}
\rightarrow
d_{model}
$$

로 확장했다가 다시 돌아온다.

따라서,

```text
X^(0) [B,T,d_model]
↓ Encoder 1
X^(1) [B,T,d_model]
↓ Encoder 2
X^(2) [B,T,d_model]
↓ Encoder 3
X^(3) [B,T,d_model]
```

처럼 block을 연속해서 연결할 수 있다.

> **shape은 유지되지만 representation의 값과 의미는 계속 변한다.**

---

## 54. Do Encoder Blocks Share Parameters?

일반적인 Transformer에서는 서로 다른 Encoder Block이 같은 parameter를 반복해서 재사용하지 않는다.

예를 들어 Encoder Block 1에는 자신만의

$$
W_Q^{(1)},
W_K^{(1)},
W_V^{(1)},
W_O^{(1)}
$$

와 FFN / LayerNorm parameter가 있다.

Encoder Block 2에는 별도의

$$
W_Q^{(2)},
W_K^{(2)},
W_V^{(2)},
W_O^{(2)}
$$

와 FFN / LayerNorm parameter가 있다.

즉,

> **구조는 같지만 각 layer의 학습 parameter는 보통 서로 다르다.**

그래서 각 layer는 자신에게 들어온 단계의 representation을 서로 다른 transformation으로 가공할 수 있다.

---

## 55. Padding Mask

실제 batch에서는 문장마다 sequence length가 다를 수 있다.

예를 들어,

```text
I love cats <PAD> <PAD>
I really love this movie
```

처럼 짧은 sequence 뒤에 `<PAD>` token을 추가해 길이를 맞출 수 있다.

하지만 `<PAD>`는 실제 문장 내용이 아니므로 Self-Attention이 이 위치를 참고해서는 안 된다.

그래서 **Padding Mask**를 사용한다.

Attention score가 예를 들어,

```text
real real real PAD PAD
2.1  1.4  0.8  0.3 0.5
```

라면 softmax 전에 PAD 위치를 사실상 $-\infty$로 만든다.

```text
[2.1, 1.4, 0.8, -∞, -∞]
```

softmax 이후에는 PAD 위치의 attention weight가 0이 된다.

```text
[0.58, 0.29, 0.13, 0, 0]
```

따라서 Value weighted sum에 PAD 정보가 들어오지 않는다.

핵심은,

> **Padding Mask = "이 위치는 실제 sequence 정보가 아니므로 attention하지 마"라고 알려주는 장치**

이다.

이 mask는 이후 Decoder에서 배우게 될 **Causal / Look-Ahead Mask**와 목적이 다르다.

---

## 56. Complete Transformer Encoder Flow

지금까지의 내용을 하나로 연결하면 다음과 같다.

```text
Input text
↓
Tokenization
↓
Token IDs
↓
Token Embedding
[B,T,d_model]
↓
+ Positional Encoding
↓
X^(0)
[B,T,d_model]
↓

Encoder Block 1
  Multi-Head Self-Attention
  + Padding Mask when needed
  Residual + LayerNorm
  FFN
  Residual + LayerNorm
↓
X^(1)
[B,T,d_model]
↓

Encoder Block 2
↓
X^(2)
[B,T,d_model]
↓
...
↓
Encoder Block N
↓

Final Encoder Output X^(N)
[B,T,d_model]
```

---

## 57. What Does the Encoder Finally Output?

Transformer Encoder는 입력 sequence 전체를 하나의 vector로 압축해서 출력하는 것이 아니다.

최종 output은

$$
X^{(N)}:[B,T,d_{model}]
$$

이다.

즉 **각 input token마다 하나의 최종 contextual representation이 남아 있다.**

예를 들어,

```text
The   cat   sat   on   mat
 ↓     ↓     ↓     ↓    ↓
z1    z2    z3    z4   z5
```

각각

$$
z_i\in\mathbb{R}^{d_{model}}
$$

이다.

하지만 이 $z_i$는 처음의 단순 Token Embedding과 다르다.

예를 들어 `cat`의 최종 $z_{cat}$은 단순히 사전적인 `cat`의 representation이 아니라, 여러 Encoder layer를 거치며 다른 token들과 상호작용한

> **"이 문장 안에서의 cat"에 대한 contextualized representation**

이라고 볼 수 있다.

---

## 58. Connection to the Decoder

초기 Encoder-Decoder Attention에서 배웠던 구조는

$$
Q=\text{Decoder state}
$$

$$
K,V=\text{Encoder hidden representations}
$$

였다.

Transformer에서도 이 intuition이 그대로 이어진다.

Encoder의 최종 output

$$
X^{(N)}:[B,T_{src},d_{model}]
$$

전체가 이후 Decoder의 **Cross-Attention에서 K/V의 source**가 된다.

즉 Transformer Encoder는 각 source token의 contextual representation을 모두 유지해 두고, Decoder가 필요한 시점마다 이를 참고할 수 있게 한다.

---

## 59. Final Encoder Summary

Transformer Encoder 전체를 가장 압축해서 정리하면 다음과 같다.

> **Token을 embedding vector로 바꾸고 positional information을 더한 뒤, 여러 Encoder Block에서 Self-Attention과 FFN을 반복하여 각 token representation을 점점 문맥화한다. Padding 위치는 mask로 attention에서 제외한다. 최종적으로 모든 input token의 contextual representation `[B,T,d_model]`을 출력하며, 이 representation들은 이후 Decoder Cross-Attention에서 사용된다.**

현재까지 완료한 흐름:

```text
RNN / LSTM / GRU
↓
Encoder-Decoder Attention
↓
Q / K / V
↓
Self-Attention
↓
Scaled Dot-Product Attention
↓
Multi-Head Attention
↓
Residual Connection + LayerNorm
↓
FFN
↓
One Encoder Block
↓
Token Embedding + Positional Encoding
↓
Stacked Encoder Blocks
↓
Padding Mask
↓
Complete Transformer Encoder ✅
```

다음 학습 단계는 **Transformer Decoder**이다.

```text
Decoder input / overall structure
↓
Masked (Causal) Self-Attention
↓
Encoder-Decoder Cross-Attention
↓
Decoder FFN / Residual / LayerNorm
↓
Stacked Decoder Blocks
↓
Complete Transformer
```
