# Attention: From Encoder-Decoder Attention to Multi-Head Self-Attention

이 문서는 RNN 계열 학습 이후 Attention을 처음 접했을 때부터  
**Encoder-Decoder Attention → Q/K/V → Self-Attention → Single-Head → Multi-Head Attention**으로 이어진 학습 흐름을 정리한 노트이다.

단순히 공식만 외우는 것이 아니라,

- Attention이 왜 필요했는지
- 각 계산이 어떤 의미를 가지는지
- Query / Key / Value가 왜 등장하는지
- Self-Attention에서는 무엇이 달라지는지
- Multi-Head Attention은 왜 필요한지
- 실제 tensor shape은 어떻게 변하는지

를 단계적으로 이해하는 것을 목표로 한다.

---

## 1. Why Attention?

Attention을 이해하려면 먼저 초기 **Seq2Seq Encoder-Decoder RNN** 구조의 한계를 볼 필요가 있다.

기본적인 Seq2Seq에서는 Encoder가 입력 sequence를 timestep 순서대로 처리한다.

```text
x1 → x2 → x3 → ... → xT
                  ↓
             final hidden state
```

Encoder의 마지막 hidden state 하나가 입력 sequence 전체의 정보를 담은 **fixed context vector** 역할을 하고,  
Decoder는 이 하나의 vector를 받아 출력 sequence를 생성한다.

즉,

```text
Input sequence
      ↓
   Encoder
      ↓
one fixed context vector
      ↓
   Decoder
      ↓
Output sequence
```

와 같은 구조이다.

### Problem: Fixed Context Bottleneck

짧은 sequence에서는 마지막 hidden state 하나에 정보를 압축하는 방식이 어느 정도 가능하지만,  
sequence가 길어지면 입력의 모든 정보를 하나의 vector에 담아야 한다.

따라서 앞부분의 세부 정보가 손실될 수 있고, Decoder가 특정 출력 token을 생성할 때  
입력의 어느 부분이 중요한지를 직접 다시 참고할 수 없다.

Attention의 핵심 아이디어는 이 bottleneck을 완화하는 것이다.

> Decoder에게 Encoder의 마지막 hidden state 하나만 주지 말고,  
> **Encoder가 만든 모든 timestep의 hidden state를 다시 볼 수 있게 하자.**

---

## 2. Encoder-Decoder Attention

Encoder가 입력 sequence로부터 다음 hidden states를 만들었다고 하자.

$$
h_1, h_2, \dots, h_T
$$

Attention을 사용하면 Decoder의 각 timestep마다 이 전체 hidden states를 다시 참고한다.

예를 들어 Decoder가 현재 timestep $t$에서 state $s_t$를 가지고 있다면,

```text
Encoder hidden states
h1   h2   h3   ...   hT
 \    |   /           /
  \   |  /           /
   attention scores
         ↑
   Decoder state st
```

Decoder는 현재 자신이 필요한 정보와 각각의 Encoder hidden state가 얼마나 관련 있는지 계산한다.

중요한 점은 **Decoder timestep마다 attention이 다시 계산된다는 것**이다.

따라서 첫 번째 출력 token을 만들 때 중요했던 Encoder 위치와,  
다섯 번째 출력 token을 만들 때 중요했던 Encoder 위치가 서로 다를 수 있다.

---

## 3. Basic Attention Calculation

Attention의 기본 계산은 크게 네 단계로 볼 수 있다.

### Step 1. Score

현재 상태와 각 후보 정보 사이의 관련성을 계산한다.

$$
e_i = score(query, key_i)
$$

여기서 `score`는 반드시 dot product일 필요는 없다.

Attention의 핵심은 **관련성을 나타내는 scalar score를 만드는 것**이고,  
그 방법으로 dot product, additive attention 등 여러 방식이 존재할 수 있다.

---

### Step 2. Softmax

score들을 softmax에 넣어 합이 1인 attention weight로 바꾼다.

$$
\alpha_i =
\frac{\exp(e_i)}
{\sum_j \exp(e_j)}
$$

이때 $\alpha_i$가 각 정보가 얼마나 중요하게 사용될지를 나타낸다.

```text
scores
[2.1, 0.5, -0.3]

        ↓ softmax

attention weights
[0.76, 0.15, 0.09]
```

---

### Step 3. Weighted Sum of Values

각 Value에 attention weight를 곱하고 합한다.

$$
c =
\sum_i \alpha_i v_i
$$

이렇게 얻은 $c$가 **context vector / attention output**이다.

즉 Attention은 후보 정보를 하나 선택하는 것이 아니라,  
중요도에 따라 여러 Value를 가중합하여 새로운 representation을 만든다.

---

### Step 4. Use the Context

Encoder-Decoder Attention에서는 이 context vector를 Decoder의 현재 state와 함께 사용해  
다음 출력 token을 예측하는 데 활용할 수 있다.

핵심 흐름은 다음과 같다.

```text
score
  ↓
softmax
  ↓
attention weights
  ↓
weighted sum of Values
  ↓
context vector
```

---

## 4. Attention Weight Is Not a Learned Parameter

Attention weight $\alpha$ 자체는 모델이 직접 저장하고 학습하는 parameter가 아니다.

$$
\alpha =
softmax(score(Q,K))
$$

처럼 **현재 Q와 K로부터 매 forward pass마다 동적으로 계산되는 값**이다.

학습되는 것은 Q/K/V를 만들거나 score를 계산하는 데 사용되는 weight이다.

예를 들어 Transformer 방식이라면,

$$
W_Q,\quad W_K,\quad W_V
$$

가 loss와 backpropagation을 통해 학습된다.

그리고 학습된 projection으로 만들어진 Q와 K가 달라지면서  
결과적으로 attention weight도 task에 맞게 달라진다.

---

## 5. Query, Key, Value

Attention을 일반적인 형태로 표현하기 위해 Q/K/V 개념을 사용한다.

### Query

> 지금 어떤 정보를 찾고 있는가?

현재 token 또는 현재 state가 다른 정보들과 관련성을 비교하기 위해 사용하는 representation이다.

---

### Key

> 내가 가진 정보가 어떤 정보인지 비교하기 위한 representation

Query와 비교되어 attention score를 만드는 역할을 한다.

---

### Value

> 실제로 가져올 정보

Attention weight가 계산된 뒤 weighted sum에 사용되는 representation이다.

---

이를 한 문장으로 정리하면,

> **Query와 Key를 비교해서 어디를 볼지 정하고, 그 결과로 Value를 가져온다.**

---

## 6. Encoder-Decoder Attention에서 Q/K/V

초기 Encoder-Decoder Attention을 Q/K/V 관점으로 다시 보면 이해가 쉬워진다.

- Query: 현재 Decoder state
- Key: Encoder hidden states
- Value: Encoder hidden states에서 가져올 정보

즉 Decoder는

> "현재 출력 token을 만들기 위해 Encoder의 어느 부분이 필요한가?"

라는 Query를 가지고 Encoder 전체를 검색한다고 볼 수 있다.

개념적으로,

$$
Q = \text{Decoder state}
$$

$$
K,V = \text{Encoder representations}
$$

이다.

### Why is the Decoder State the Query?

Decoder가 지금 생성하려는 출력에 따라 필요한 입력 정보가 달라지기 때문이다.

따라서 Decoder의 현재 state가 **찾는 쪽**이 되고,  
Encoder hidden states가 **찾아볼 대상**이 된다.

이 구조에서 Attention은 입력 sequence 전체를 다시 참고할 수 있게 해준다.

---

## 7. From Attention to Self-Attention

Encoder-Decoder Attention에서는 Q와 K/V의 출처가 달랐다.

```text
Query  ← Decoder
Key    ← Encoder
Value  ← Encoder
```

Self-Attention에서는 한 sequence 안의 token들이 **서로를 참고한다.**

즉 Q/K/V가 모두 같은 입력 $X$로부터 만들어진다.

```text
                X
          ┌─────┼─────┐
          ↓     ↓     ↓
         Q      K      V
```

하지만 Q, K, V가 같은 값이라는 뜻은 아니다.

각각 서로 다른 learned projection을 사용한다.

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

따라서 같은 token $x_i$라도

$$
q_i = x_iW_Q,\quad
k_i = x_iW_K,\quad
v_i = x_iW_V
$$

라는 서로 다른 역할의 representation으로 변환된다.

---

## 8. Why Does a Token Attend to Other Tokens?

Self-Attention에서는 각 token이 sequence의 다른 token들과 관계를 계산한다.

예를 들어 token $i$의 Query $q_i$와 모든 Key를 비교한다.

$$
q_i k_1^T,\quad
q_i k_2^T,\quad
\dots,\quad
q_i k_T^T
$$

이를 softmax하면

$$
\alpha_{i1}, \alpha_{i2}, \dots, \alpha_{iT}
$$

가 되고,

$$
z_i =
\sum_{j=1}^{T}
\alpha_{ij}v_j
$$

로 token $i$의 새로운 representation을 만든다.

즉 $z_i$는 더 이상 token $i$ 자신의 정보만 담고 있는 것이 아니다.

> token $i$가 sequence 전체를 보고, 자신과 관련 있는 다른 token들의 정보를 반영한  
> **contextual representation**이다.

---

## 9. Single-Head Scaled Dot-Product Attention

Transformer에서는 대표적으로 Scaled Dot-Product Attention을 사용한다.

$$
Attention(Q,K,V) =
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

각 부분의 의미는 다음과 같다.

### $QK^T$

각 Query와 모든 Key 사이의 dot product를 계산한다.

즉 token 간 attention score matrix를 만든다.

---

### $\sqrt{d_k}$로 나누기

차원이 커질수록 dot product의 값도 커질 수 있다.

score가 지나치게 커지면 softmax가 매우 뾰족해지고 gradient가 작아질 수 있으므로  
$\sqrt{d_k}$로 scale을 조정한다.

---

### Softmax

각 Query가 모든 Key에 주는 attention weight를 만든다.

각 Query 기준으로 weight의 합은 1이다.

---

### Multiply by $V$

attention weight를 이용해 Value들을 weighted sum한다.

결과적으로 각 token마다 context를 반영한 새로운 representation이 생성된다.

---

## 10. Single-Head Shape Flow

입력을 다음과 같이 두자.

$$
X:[B,T,d_{model}]
$$

Self-Attention에서 projection 후

$$
Q,K,V:[B,T,d_k]
$$

라고 하면,

$$
QK^T:
[B,T,d_k]
@
[B,d_k,T]
$$

이므로

$$
Attention\ Score:[B,T,T]
$$

이다.

Softmax 이후에도 shape은 동일하다.

$$
Attention\ Weight:[B,T,T]
$$

그리고

$$
[B,T,T]
@
[B,T,d_v]
$$

를 계산하면

$$
Z:[B,T,d_v]
$$

를 얻는다.

마지막 두 개의 $T$가 의미하는 것은,

> 각 Query token $T$개가 모든 Key token $T$개와 관계를 계산한다

는 것이다.

---

## 11. Limitation of a Single Attention Head

Single-head attention에서도 한 token은 sequence 전체를 참고할 수 있다.

따라서 single head가 여러 정보를 **전혀 표현하지 못하는 것**은 아니다.

하지만 각 Query token에 대해 만들어지는 attention distribution은 하나이다.

예를 들어,

```text
token i attention
→ [0.05, 0.70, 0.10, 0.10, 0.05]
```

처럼 sequence를 바라보는 하나의 attention pattern이 만들어진다.

그런데 한 token은 동시에 여러 종류의 관계를 가질 수 있다.

- 문법적 관계
- 의미적 관계
- 가까운 문맥
- 멀리 떨어진 dependency
- 특정 표현이나 entity와의 관계

하나의 attention distribution만 사용하는 것보다  
여러 개의 서로 다른 projection space에서 attention을 병렬로 계산할 수 있다면  
다양한 관계를 별도로 학습할 수 있다.

이 아이디어가 **Multi-Head Attention**이다.

---

## 12. Multi-Head Attention

Multi-Head Attention은 Self-Attention을 여러 개 복사해서 단순 반복하는 것이라기보다,

> 같은 입력 전체를 **서로 다른 learned projection space**로 투영한 뒤  
> 각 space에서 독립적인 attention을 계산하는 구조

로 이해하는 것이 중요하다.

예를 들어,

$$
d_{model}=512,\quad
h=8
$$

이라면

$$
d_{head} =
\frac{d_{model}}{h} =
64
$$

이다.

각 head $i$는 개념적으로 자신의 projection matrix를 가진다.

$$
W_Q^{(i)}:[512,64]
$$

$$
W_K^{(i)}:[512,64]
$$

$$
W_V^{(i)}:[512,64]
$$

따라서

$$
Q_i=XW_Q^{(i)}
$$

$$
K_i=XW_K^{(i)}
$$

$$
V_i=XW_V^{(i)}
$$

이다.

### Important

각 head가 입력 512차원을 64차원씩 잘라서 보는 것이 아니다.

**모든 head가 원래 입력의 512차원 전체를 본다.**

다만 서로 다른 learned weight를 사용해서 각자의 64차원 projection space로 변환한다.

즉,

> input feature를 head별로 나누는 것이 아니라,  
> **projection output space를 head별로 나눈다.**

---

## 13. One Head in Multi-Head Attention

각 head에서는 동일한 Scaled Dot-Product Attention을 계산한다.

$$
head_i =
softmax
\left(
\frac{Q_iK_i^T}{\sqrt{d_{head}}}
\right)V_i
$$

각 head가 서로 다른 Q/K/V projection을 사용하기 때문에  
서로 다른 attention distribution을 학습할 수 있다.

예를 들어 같은 Query token이라도,

```text
Head 1
→ token 2에 큰 attention

Head 2
→ token 5에 큰 attention

Head 3
→ token 1과 token 4에 분산된 attention
```

처럼 서로 다른 관계를 볼 수 있다.

따라서 같은 token에 대해서도 각 head는 서로 다른 output

$$
z_i^{(1)},\;
z_i^{(2)},\;
\dots,\;
z_i^{(h)}
$$

를 만든다.

---

## 14. Conceptual Projection vs Actual Implementation

개념적으로는 head마다 다음 weight들이 존재한다고 볼 수 있다.

$$
W_Q^{(1)},\dots,W_Q^{(8)}
$$

각각

$$
[512,64]
$$

이다.

하지만 실제 구현에서 이 작은 matrix를 head마다 따로 계산하면 비효율적이다.

따라서 보통 이를 큰 하나의 matrix로 묶는다.

$$
W_Q:[512,512]
$$

이는 개념적으로

$$
W_Q =
[
W_Q^{(1)}
|
W_Q^{(2)}
|
\dots
|
W_Q^{(8)}
]
$$

처럼 출력 차원 방향으로 붙인 것으로 볼 수 있다.

따라서 한 번에

$$
Q=XW_Q
$$

를 계산하면

$$
Q:[B,T,512]
$$

가 나오고,

이를

$$
[B,T,8,64]
$$

로 reshape하여 head별 결과를 다시 구분한다.

K와 V도 같은 방식이다.

즉,

> **개념적으로는 head별 projection을 따로 계산하지만,  
> 구현에서는 큰 matrix multiplication 한 번으로 묶어서 계산한 뒤 reshape한다.**

---

## 15. Multi-Head Tensor Shape Flow

기준:

$$
d_{model}=512,\quad
h=8,\quad
d_{head}=64
$$

입력:

$$
X:[B,T,512]
$$

### 1. Q/K/V Projection

$$
Q,K,V:[B,T,512]
$$

### 2. Split Heads

$$
[B,T,512]
\rightarrow
[B,T,8,64]
$$

즉,

$$
[B,T,h,d_{head}]
$$

### 3. Move the Head Dimension Forward

보통 계산하기 편하게

$$
[B,T,h,d_{head}]
\rightarrow
[B,h,T,d_{head}]
$$

로 차원 순서를 바꾼다.

따라서

$$
Q,K,V:[B,8,T,64]
$$

이다.

---

## 16. Why Move the Head Dimension Forward?

여기서 두 과정을 구분해야 한다.

### Split Heads

$$
[B,T,512]
\rightarrow
[B,T,8,64]
$$

이 과정은 **Multi-Head Attention의 개념적 구조**이다.

512차원의 projection output을 8개의 head output으로 구분한다.

### Move Head Dimension

$$
[B,T,8,64]
\rightarrow
[B,8,T,64]
$$

이 과정은 주로 **효율적인 계산을 위한 tensor arrangement**이다.

이렇게 두면 각 `(batch, head)`마다 하나의

$$
[T,64]
$$

Q/K/V matrix가 있다고 볼 수 있다.

PyTorch의 batched matrix multiplication을 사용하면  
모든 batch와 모든 head의 attention을 한꺼번에 계산하기 쉽다.

즉 head 차원을 반드시 두 번째에 두어야 Multi-Head Attention이 되는 것은 아니다.

> **head로 나누는 것은 개념적인 구조이고,  
> head 차원을 앞으로 옮기는 것은 효율적인 병렬 계산을 위한 구현상의 배열이다.**

---

## 17. Attention Score Shape in Multi-Head Attention

Attention 계산 직전,

$$
Q:[B,8,T,64]
$$

$$
K:[B,8,T,64]
$$

이다.

$QK^T$를 계산할 때는 **K의 마지막 두 차원**을 transpose한다.

$$
K^T:[B,8,64,T]
$$

따라서

$$
[B,8,T,64]
@
[B,8,64,T]
$$

의 결과는

$$
Attention\ Score:[B,8,T,T]
$$

이다.

여기서 8은 head 개수이므로  
각 head마다 독립적인 $T\times T$ attention score matrix가 존재한다.

Softmax 이후에도

$$
Attention\ Weight:[B,8,T,T]
$$

이다.

---

## 18. Weighted Sum of Values

Value는

$$
V:[B,8,T,64]
$$

이므로,

$$
[B,8,T,T]
@
[B,8,T,64]
$$

를 계산하면

$$
Head\ Outputs:[B,8,T,64]
$$

가 된다.

이 tensor 안에는 모든 batch, 모든 head, 모든 token의 output이 들어 있다.

특정 token $i$, 특정 head $r$의 output을

$$
z_i^{(r)}
$$

라고 생각할 수 있고,

$$
z_i^{(r)}\in\mathbb{R}^{64}
$$

이다.

---

## 19. Concatenate the Heads

모든 head의 output은 현재

$$
[B,8,T,64]
$$

이다.

token 기준으로 다시 정렬하면,

$$
[B,T,8,64]
$$

이고 마지막 두 차원을 합치면

$$
[B,T,512]
$$

가 된다.

token $i$ 하나만 보면,

$$
Concat
(
z_i^{(1)},
z_i^{(2)},
\dots,
z_i^{(8)}
)
\in
\mathbb{R}^{512}
$$

이다.

즉 각 head가 서로 다른 관점에서 가져온 정보를 다시 하나의 token representation으로 합친다.

---

## 20. Output Projection $W_O$

Multi-Head Attention의 완성된 식은 다음과 같다.

$$
MultiHead(Q,K,V) =
Concat(head_1,\dots,head_h)W_O
$$

Concat 직후 shape은

$$
[B,T,512]
$$

이고,

$$
W_O:[512,512]
$$

를 적용하면 최종 output도

$$
[B,T,512]
$$

가 된다.

### Why $W_O$?

Concat만 한 상태는 각 head에서 얻은 feature들을 단순히 옆에 붙여놓은 상태이다.

```text
head 1 information | head 2 information | ... | head 8 information
```

$W_O$는 이 전체 feature를 다시 linear projection하여  
여러 head에서 얻은 정보를 서로 섞고 통합하는 역할을 한다.

---

## 21. Complete Multi-Head Attention Formula

각 head:

$$
Q_i=XW_Q^{(i)},\quad
K_i=XW_K^{(i)},\quad
V_i=XW_V^{(i)}
$$

$$
head_i =
softmax
\left(
\frac{Q_iK_i^T}{\sqrt{d_{head}}}
\right)V_i
$$

전체 Multi-Head Attention:

$$
MHA(X) =
Concat(head_1,\dots,head_h)W_O
$$

---

## 22. Complete Shape Summary

$$
d_{model}=512,\quad
h=8,\quad
d_{head}=64
$$

```text
X
[B, T, 512]

↓ W_Q, W_K, W_V

Q, K, V
[B, T, 512]

↓ split heads

[B, T, 8, 64]

↓ move head dimension

[B, 8, T, 64]

↓ Q @ K^T

Attention Scores
[B, 8, T, T]

↓ scale + softmax

Attention Weights
[B, 8, T, T]

↓ Attention Weights @ V

Head Outputs
[B, 8, T, 64]

↓ move dimensions back

[B, T, 8, 64]

↓ concatenate heads

[B, T, 512]

↓ W_O

Multi-Head Attention Output
[B, T, 512]
```

---

## 23. Important Distinctions and Common Confusions

### Attention score is not necessarily a dot product

Dot product is one scoring method.  
Attention의 본질은 Query와 Key 사이의 attention score를 계산하는 것이다.

---

### Attention weights are not direct model parameters

$$
\alpha
$$

는 저장된 parameter가 아니라 Q/K를 통해 매번 계산된다.

학습되는 것은 Q/K/V projection 등에 사용되는 weight이다.

---

### Self-Attention에서 Q/K/V의 source는 같지만 값은 다르다

모두 $X$에서 나오지만,

$$
W_Q,\quad W_K,\quad W_V
$$

가 다르므로 서로 다른 representation이다.

---

### A head does not see only part of the original input features

각 head는 원래 $d_{model}$ 전체를 입력으로 보고  
자신의 $d_{head}$ projection space로 투영한다.

---

### Splitting heads and moving the head dimension are different operations

- split: Multi-Head의 개념적 구조
- dimension rearrangement: 병렬 matrix multiplication을 위한 구현상의 편의

---

### $QK^T$에서 transpose하는 것은 K이다

$$
Q:[B,h,T,d_{head}]
$$

$$
K^T:[B,h,d_{head},T]
$$

이다.

앞서 `[B,T,h,d_head] → [B,h,T,d_head]`로 바꾸는 transpose/permute와  
$QK^T$에서의 $K^T$는 목적이 다른 연산이다.

---

### Single-head도 여러 정보를 표현할 수 있다

Multi-head의 장점은 single-head가 아무것도 못한다는 것이 아니다.

Multi-head는 서로 다른 learned projection에서 **여러 attention distribution을 명시적으로 병렬 계산**할 수 있어  
다양한 관계를 표현하기 더 좋은 구조를 제공한다.

---

## 24. Conceptual Summary

Attention을 배우면서 가장 중요한 흐름은 다음과 같다.

### Encoder-Decoder Attention

> 하나의 fixed context vector에 모든 정보를 압축하지 않고,  
> Decoder가 필요한 순간마다 Encoder hidden states 전체를 다시 참고한다.

### Q/K/V

> Query와 Key를 비교해서 어디를 볼지 정하고,  
> 그 attention weight를 이용해 Value의 정보를 가져온다.

### Self-Attention

> 같은 sequence의 token들이 서로를 참고하면서  
> 각 token을 문맥이 반영된 contextual representation으로 바꾼다.

### Multi-Head Attention

> 같은 입력을 서로 다른 learned projection space에서 여러 번 바라보면서  
> 서로 다른 attention pattern과 관계를 병렬로 학습한다.

따라서 전체적인 발전 흐름은,

```text
RNN Seq2Seq
↓
Fixed context bottleneck
↓
Encoder-Decoder Attention
↓
Query / Key / Value
↓
Self-Attention
↓
Scaled Dot-Product Attention
↓
Multi-Head Attention
```

으로 정리할 수 있다.

---

## 25. Where This Note Stops

이 문서는 **Multi-Head Self-Attention의 output을 만드는 과정까지**를 다룬다.

다음 단계에서는 이 Multi-Head Attention output이 Transformer Encoder Block 안에서 어떻게 사용되는지 살펴본다.

```text
Multi-Head Self-Attention
↓
Residual Connection + LayerNorm
↓
Feed Forward Network
↓
Residual Connection + LayerNorm
↓
Transformer Encoder Block
```

이후의 내용은 별도의 Transformer Encoder 노트에서 이어서 정리한다.
