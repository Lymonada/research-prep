# Transformer Decoder

이 문서는 Transformer Encoder를 이해한 이후, **Transformer Decoder가 target sequence를 어떻게 처리하고 실제 next-token prediction까지 이어지는지**를 정리한 노트이다.

특히 Decoder를 공부하면서 헷갈리기 쉬웠던 다음 질문들을 중심으로 정리한다.

- 왜 Decoder에도 target sequence가 input으로 들어가는가?
- target sequence는 label과 어떤 관계인가?
- Masked Self-Attention은 무엇을 출력하는가?
- 왜 Decoder input도 Self-Attention을 해야 하는가?
- Causal Mask는 정확히 무엇을 막는가?
- 왜 Decoder가 Query이고 Encoder가 Key / Value인가?
- Cross-Attention은 모든 target token에서 일어나는가?
- Cross-Attention output은 prediction인가, representation인가?
- Decoder Block 안에서 Residual / LayerNorm / FFN은 어떻게 연결되는가?
- Decoder Block을 여러 층 쌓으면 무엇이 달라지는가?
- 실제 단어 prediction은 언제 처음 등장하는가?
- Training과 Inference는 왜 다르게 동작하는가?
- Autoregressive Generation은 무엇인가?

이 노트는 원래 2017 Transformer의 **Encoder–Decoder 구조**와 **Post-Norm** 흐름을 기준으로 설명한다.

---

## 1. Starting Point: What Does the Decoder Do?

Encoder의 역할은 source sequence를 contextual representation으로 바꾸는 것이다.

예를 들어 번역 문제에서,

```text
Source
I love cats
```

가 Encoder로 들어가면 최종적으로

$$
E:[B,T_{src},d_{model}]
$$

형태의 Encoder output을 얻는다.

각 source token은 전체 source 문맥을 반영한 representation이 된다.

하지만 Encoder는 source를 **이해하고 표현하는 것**까지 담당할 뿐,

```text
나는 고양이를 좋아한다
```

라는 target sequence를 직접 생성하지는 않는다.

이 target sequence를 한 token씩 생성하는 역할이 **Decoder**이다.

따라서 전체적인 역할은 다음과 같이 볼 수 있다.

```text
Source sequence
      ↓
   Encoder
      ↓
Encoder representations
      ↓
   Decoder
      ↓
Target sequence generation
```

---

## 2. Does the Decoder Receive the Correct Target Sentence?

Training에서는 **그렇다.**

예를 들어 학습 데이터가

```text
Source:
I love cats

Target:
나는 고양이를 좋아한다
```

라고 하자.

Encoder에는 source가 들어가고, Decoder에도 target sequence를 이용한 input이 들어간다.

하지만 target을 그대로 넣는 것이 아니라 **한 칸 오른쪽으로 shift한 sequence**를 사용한다.

```text
Decoder input:
[BOS]   나는   고양이를   좋아한다

Label:
나는    고양이를   좋아한다   [EOS]
```

즉 target sequence는 동시에

- Decoder input을 만드는 재료
- next-token prediction의 정답 label

로 사용된다.

---

## 3. Relation to Labels in MLP / CNN

MNIST 같은 classification에서는,

```text
Input  = image
Label  = class index
```

처럼 input과 label이 완전히 분리되어 있었다.

Transformer Decoder에서는 target sequence가 정답 역할을 하면서도, **이전 정답 token들이 Decoder input으로도 사용된다.**

예를 들어,

```text
Decoder input             Label
[BOS]                  →  나는
[BOS] 나는             →  고양이를
[BOS] 나는 고양이를    →  좋아한다
[BOS] 나는 고양이를 좋아한다 → [EOS]
```

라고 볼 수 있다.

Training에서 이전 정답 token을 Decoder에 제공하는 방식을 흔히 **Teacher Forcing**이라고 부른다.

---

## 4. Why Does the Decoder Need Self-Attention?

Decoder가 생성해야 할 문장을 이미 input으로 받는다고 생각하면,

> "어차피 생성해야 할 target인데 왜 다시 Self-Attention을 하는가?"

라는 의문이 생길 수 있다.

핵심은 **다음 token은 지금까지 생성된 target 문맥에 의존한다**는 것이다.

예를 들어,

```text
나는 고양이를 ???
```

에서 다음 token을 예측하려면 마지막 token인 `고양이를`만 보는 것이 아니라,

- 앞에 `나는`이 나왔다는 것
- `고양이를`이 목적어처럼 사용되고 있다는 것
- 지금까지 어떤 target-side 문맥이 형성되어 있는지

를 함께 이해해야 한다.

따라서 Decoder의 Self-Attention은

> **지금까지 생성된 target sequence의 문맥을 이해하는 역할**

을 한다.

---

## 5. Encoder Self-Attention vs Decoder Self-Attention

Encoder와 Decoder의 Self-Attention은 본질적으로 같은 연산이지만, 볼 수 있는 범위가 다르다.

### Encoder

```text
I ↔ love ↔ cats
```

source sequence 전체가 이미 주어져 있으므로 모든 token이 서로를 볼 수 있다.

### Decoder

```text
[BOS] → 나는 → 고양이를 → 좋아한다
```

현재 위치보다 미래에 있는 target token을 미리 보면 안 된다.

따라서 Decoder Self-Attention에는 **Causal Mask**가 필요하다.

---

## 6. Decoder Input Shape

Decoder input token들은 먼저 Token Embedding과 Positional Encoding을 거쳐

$$
X_{dec}:[B,T_{tgt},d_{model}]
$$

이 된다.

예를 들어,

$$
B=32,\quad T_{tgt}=20,\quad d_{model}=512
$$

이면,

$$
X_{dec}:[32,20,512]
$$

이다.

이 representation에서 Masked Multi-Head Self-Attention용 Q / K / V를 만든다.

$$
Q=X_{dec}W_Q
$$

$$
K=X_{dec}W_K
$$

$$
V=X_{dec}W_V
$$

즉 Q / K / V가 모두 Decoder 쪽에서 나오므로 **Self-Attention**이다.

---

## 7. What Does Masked Self-Attention Output?

중요한 구분은 다음과 같다.

> **Masked Self-Attention output은 next-token prediction이 아니다.**

Encoder Self-Attention과 마찬가지로 각 token을 **contextual representation**으로 바꾸는 단계이다.

예를 들어 Decoder input이

```text
[BOS]   나는   고양이를
```

라고 하자.

Masked Self-Attention output을

$$
z_1,z_2,z_3
$$

라고 하면,

```text
z1 = [BOS]만 반영한 representation

z2 = [BOS], 나는
     를 반영한 "나는" 위치 representation

z3 = [BOS], 나는, 고양이를
     를 반영한 "고양이를" 위치 representation
```

이라고 볼 수 있다.

즉 $z_3$는 단순히 `고양이를`이라는 token embedding이 아니라,

> **"지금까지 [BOS] 나는 고양이를까지 생성된 상태"**

를 담은 contextual representation이다.

---

## 8. Why Is the Attention "Masked"?

Training에서는 target sequence 전체를 한 번에 Decoder에 넣는다.

```text
[BOS]   나는   고양이를   좋아한다
```

그런데 mask 없이 Self-Attention을 하면 각 위치가 오른쪽의 미래 token까지 볼 수 있다.

```text
                 Key
             BOS   나는   고양이를   좋아한다
Query BOS      O      O       O          O
      나는     O      O       O          O
      고양이를 O      O       O          O
      좋아한다 O      O       O          O
```

예를 들어 `나는` 위치의 정답은 `고양이를`인데, Self-Attention에서 이미 오른쪽의 `고양이를`를 볼 수 있다면 정답을 미리 보는 셈이다.

따라서 미래 방향의 연결을 막는다.

```text
                 Key
             BOS   나는   고양이를   좋아한다
Query BOS      O      X       X          X
      나는     O      O       X          X
      고양이를 O      O       O          X
      좋아한다 O      O       O          O
```

이것이 **Causal Mask**이다.

---

## 9. Where Is the Causal Mask Applied?

Scaled Dot-Product Attention에서 먼저 score를 계산한다.

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

sequence 길이가 4라면 score matrix는 개념적으로

$$
S=
\begin{bmatrix}
s_{11}&s_{12}&s_{13}&s_{14}\\
s_{21}&s_{22}&s_{23}&s_{24}\\
s_{31}&s_{32}&s_{33}&s_{34}\\
s_{41}&s_{42}&s_{43}&s_{44}
\end{bmatrix}
$$

형태이다.

Softmax를 적용하기 전에 미래 위치를 $-\infty$로 바꾼다.

$$
S_{masked}=
\begin{bmatrix}
s_{11}&-\infty&-\infty&-\infty\\
s_{21}&s_{22}&-\infty&-\infty\\
s_{31}&s_{32}&s_{33}&-\infty\\
s_{41}&s_{42}&s_{43}&s_{44}
\end{bmatrix}
$$

이후 Softmax를 적용하면

$$
e^{-\infty}=0
$$

이므로 미래 위치의 attention weight가 0이 된다.

---

## 10. Why Can a Token Attend to Itself?

예를 들어 `고양이를` 위치는

```text
[BOS], 나는, 고양이를
```

까지 볼 수 있다.

자기 자신인 `고양이를`를 보는 것이 cheating처럼 느껴질 수 있지만, 이 위치의 정답은 `고양이를`이 아니다.

shifted target 때문에,

```text
Decoder input        Label
[BOS]             →  나는
나는              →  고양이를
고양이를          →  좋아한다
좋아한다          →  [EOS]
```

이다.

따라서 `고양이를` 위치가 자기 자신을 보는 것은 문제가 없고, **오른쪽의 미래 token만 보면 안 된다.**

---

## 11. Why Is It Called "Causal"?

Autoregressive generation에서는 정보가 시간 순서상

```text
past → present → future
```

방향으로만 사용되어야 한다.

미래 token이 과거 representation에 영향을 주면 실제 생성 시점에는 존재하지 않는 정보를 사용하게 된다.

따라서 Causal Mask는

> **미래 → 과거 방향의 정보 유출을 막는 mask**

라고 이해할 수 있다.

---

## 12. Causal Mask vs Padding Mask

두 mask는 목적이 다르다.

### Causal Mask

> 미래 token을 보지 못하게 한다.

### Padding Mask

> 의미 없는 `[PAD]` token을 보지 못하게 한다.

예를 들어,

```text
[BOS] 나는 고양이를 [PAD] [PAD]
```

같은 sequence에서는 두 종류의 mask가 동시에 필요할 수 있다.

---

## 13. Why Is Causal Mask Especially Important During Training?

Inference에서는 미래 token 자체가 아직 존재하지 않는다.

예를 들어 현재 생성 상태가

```text
[BOS] 나는 고양이를
```

이라면 오른쪽의 `좋아한다`는 아직 생성되지 않았다.

반면 Training에서는 효율적인 병렬 계산을 위해

```text
[BOS] 나는 고양이를 좋아한다
```

전체를 한 번에 넣는다.

따라서 Causal Mask는

> **계산은 병렬로 하되, 각 위치가 실제 inference 시점에 볼 수 있는 정보만 보도록 강제한다.**

라고 이해하는 것이 가장 중요하다.

---

## 14. From Masked Self-Attention to Cross-Attention

Masked Self-Attention 이후 각 target 위치는 지금까지의 target-side 문맥을 담은 representation을 가진다.

예를 들어 `고양이를` 위치를

$$
z_3
$$

라고 하면,

> `나는 고양이를`까지 생성된 상태

를 나타내는 representation이라고 볼 수 있다.

하지만 아직 source sentence인

```text
I love cats
```

의 정보는 직접 참고하지 않았다.

이제 Decoder가 Encoder output을 참고하는 단계가 **Cross-Attention**이다.

---

## 15. Why Is the Decoder the Query?

Cross-Attention에서 Decoder representation은 현재 target-side 상태를 나타낸다.

예를 들어 $z_3$는

> "나는 지금까지 `나는 고양이를`이라고 생성했다. 다음 token을 만들기 위해 source의 어느 정보를 참고해야 하는가?"

라는 상태를 표현한다.

따라서 **찾는 쪽**인 Decoder가 Query가 된다.

$$
Q=Z W_Q
$$

---

## 16. Why Are Encoder Representations the Key and Value?

Encoder final output을

$$
E:[B,T_{src},d_{model}]
$$

이라고 하자.

각 source token representation에서 Key와 Value를 만든다.

$$
K=EW_K
$$

$$
V=EW_V
$$

### Key

Query가 source의 어떤 위치와 관련 있는지 비교하는 데 사용한다.

### Value

Attention weight가 결정된 뒤 실제로 가져올 source information이다.

따라서 Cross-Attention은

$$
Q=Decoder
$$

$$
K,V=Encoder
$$

구조가 된다.

---

## 17. Does Cross-Attention Happen at Every Target Position?

그렇다.

`고양이를` 위치에 와서 처음 source를 보는 것이 아니다.

Decoder input이

```text
[BOS]   나는   고양이를
```

라면 모든 위치에서 Cross-Attention이 수행된다.

```text
[BOS]      → 첫 token 생성을 위해 source의 무엇을 볼까?
나는       → 다음 token 생성을 위해 source의 무엇을 볼까?
고양이를   → 다음 token 생성을 위해 source의 무엇을 볼까?
```

Tensor shape으로 보면 더 명확하다.

Decoder Query:

$$
Q:[B,h,T_{tgt},d_{head}]
$$

Encoder Key:

$$
K:[B,h,T_{src},d_{head}]
$$

따라서

$$
QK^T:[B,h,T_{tgt},T_{src}]
$$

이다.

즉 **모든 target 위치가 모든 source 위치를 얼마나 참고할지 계산한다.**

---

## 18. Cross-Attention Output

한 target 위치 $i$에서 attention weight를

$$
\alpha_{ij}
$$

라고 하면 Cross-Attention output은

$$
c_i=\sum_j \alpha_{ij}v_j
$$

이다.

예를 들어 `고양이를` 위치의 attention weight가

```text
I       0.1
love    0.7
cats    0.2
```

라면,

$$
c_3
=
0.1V_{I}
+
0.7V_{love}
+
0.2V_{cats}
$$

가 된다.

$c_3$는

> **현재 target 문맥을 기준으로 source에서 지금 필요한 정보를 가져온 representation**

이라고 볼 수 있다.

중요하게도 이것 역시 아직 `좋아한다`라는 token prediction이 아니다.

---

## 19. $z_i$ vs $c_i$

Decoder를 이해할 때 다음 구분이 유용하다.

### $z_i$: Masked Self-Attention 쪽

> 지금까지 target에서 무엇이 생성되었는가?

### $c_i$: Cross-Attention 쪽

> 그 target 상태에서 source의 어떤 정보가 필요한가?

둘 다 **representation**이지 vocabulary token 자체가 아니다.

---

## 20. Decoder Block: First Sublayer

이제 실제 Transformer Decoder Block 구조로 연결한다.

Decoder Block의 첫 번째 sublayer는

```text
Masked Multi-Head Self-Attention
↓
Residual Connection
↓
LayerNorm
```

이다.

입력을

$$
X:[B,T_{tgt},d_{model}]
$$

이라고 하고 Masked Self-Attention output을

$$
A=MaskedMHA(X)
$$

라고 하면 원래 Transformer의 Post-Norm 기준으로,

$$
H^{(1)}
=
LayerNorm(X+A)
$$

이다.

즉 정확한 Transformer Block 관점에서는 단순 Attention output $A$만 다음 단계로 보내지 않고, **Residual + LayerNorm까지 끝난 $H^{(1)}$**가 다음 Cross-Attention의 Decoder-side representation이 된다.

---

## 21. Decoder Block: Cross-Attention Sublayer

Cross-Attention에서는

$$
Q=H^{(1)}W_Q
$$

이고 Encoder final output $E$에서

$$
K=EW_K
$$

$$
V=EW_V
$$

를 만든다.

Cross-Attention output을

$$
C=CrossAttention(H^{(1)},E,E)
$$

라고 하면,

$$
H^{(2)}
=
LayerNorm(H^{(1)}+C)
$$

가 된다.

즉 이 시점의 representation에는

```text
target-side context
+
source-side information
```

이 함께 반영되어 있다.

---

## 22. Decoder Block: FFN Sublayer

Cross-Attention 이후에는 Encoder와 마찬가지로 Position-wise FFN을 사용한다.

$$
F=FFN(H^{(2)})
$$

원래 Transformer의 FFN은

$$
FFN(x)=ReLU(xW_1+b_1)W_2+b_2
$$

형태이다.

예를 들어,

$$
d_{model}=512,\quad d_{ff}=2048
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

형태이다.

FFN은 다른 token을 새롭게 참고하지 않고, 각 token representation 내부 feature를 비선형적으로 가공한다.

---

## 23. Decoder Block: Third Add & Norm

FFN 이후에도 Residual Connection과 LayerNorm을 적용한다.

$$
H^{(3)}
=
LayerNorm(H^{(2)}+F)
$$

이 $H^{(3)}$가 **Decoder Block 하나의 최종 output**이다.

shape은 계속

$$
[B,T_{tgt},d_{model}]
$$

로 유지된다.

---

## 24. Complete Decoder Block

원래 Transformer의 Post-Norm 기준 전체 Decoder Block은 다음과 같다.

```text
Decoder Input X
[B,T_tgt,d_model]
        │
        ▼
Masked Multi-Head Self-Attention
        │
        ▼
Residual Add + LayerNorm
        │
        ▼
H^(1)
        │
        │  Q from Decoder
        ▼
Cross-Attention  ◀──── Encoder Final Output E
        │               K,V from Encoder
        ▼
Residual Add + LayerNorm
        │
        ▼
H^(2)
        │
        ▼
FFN
        │
        ▼
Residual Add + LayerNorm
        │
        ▼
Decoder Block Output H^(3)
[B,T_tgt,d_model]
```

Encoder Block과 비교하면 가장 큰 차이는 **Cross-Attention sublayer가 하나 더 존재한다는 것**이다.

---

## 25. Following One Token Through the Decoder Block

`고양이를` 위치 하나만 따라가면 다음과 같다.

### Initial representation

$$
x_3
$$

`고양이를` token embedding + positional information.

### Masked Self-Attention

```text
[BOS], 나는, 고양이를
```

까지만 참고해서 target-side contextual update를 만든다.

### First Residual + LayerNorm

$$
h_3^{(1)}
=
LayerNorm(x_3+a_3)
$$

지금까지의 target 문맥을 반영한 representation.

### Cross-Attention

$h_3^{(1)}$로 Query를 만들고 Encoder의 `I / love / cats` representations를 K / V로 사용한다.

### Second Residual + LayerNorm

$$
h_3^{(2)}
=
LayerNorm(h_3^{(1)}+c_3)
$$

이제 target-side 문맥과 source-side 정보가 함께 반영된다.

### FFN + Third Add & Norm

$$
h_3^{(3)}
=
LayerNorm
\left(
h_3^{(2)}+FFN(h_3^{(2)})
\right)
$$

이것이 Decoder Block 하나를 지난 `고양이를` 위치의 output이다.

---

## 26. Stacking Multiple Decoder Blocks

실제 Transformer Decoder는 Decoder Block 하나만 사용하는 것이 아니라 여러 층을 쌓는다.

```text
Decoder input
↓
Decoder Block 1
↓
Decoder Block 2
↓
Decoder Block 3
↓
...
↓
Final Decoder Output
```

첫 번째 Decoder Block의 output을

$$
X^{(1)}
$$

이라고 하면 두 번째 Decoder Block은 이 $X^{(1)}$을 input으로 받는다.

즉 block이 깊어질수록 이미 contextualized된 Decoder representations를 다시 Masked Self-Attention / Cross-Attention / FFN으로 가공한다.

---

## 27. Does Every Decoder Block Use the Same Encoder Output?

그렇다.

Encoder의 최종 output을

$$
E
$$

라고 하면, 각 Decoder Block의 Cross-Attention은 모두 이 Encoder final output을 직접 참고한다.

```text
              Encoder Final Output E
                   │
          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼
Decoder 1      Decoder 2  Decoder 3
Cross-Attn     Cross-Attn Cross-Attn
```

하지만 각 Decoder layer가 사용하는 projection parameter는 서로 다르다.

---

## 28. Are the Attention Weights Shared Across Decoder Layers?

일반적인 Transformer에서는 공유하지 않는다.

Block 1의 Masked Self-Attention에는

$$
W_{Q,self}^{(1)},
W_{K,self}^{(1)},
W_{V,self}^{(1)}
$$

가 있고,

같은 Block 1의 Cross-Attention에는 별도의

$$
W_{Q,cross}^{(1)},
W_{K,cross}^{(1)},
W_{V,cross}^{(1)}
$$

가 있다.

Block 2에서도 다시 별도의 parameter를 사용한다.

$$
W_{Q,self}^{(2)}
\neq
W_{Q,self}^{(1)}
$$

$$
W_{Q,cross}^{(2)}
\neq
W_{Q,cross}^{(1)}
$$

그리고 같은 block 안에서도

$$
W_{Q,self}^{(1)}
\neq
W_{Q,cross}^{(1)}
$$

이다.

즉,

> **layer가 다르면 parameter도 다르고, 같은 layer 안에서도 attention 종류가 다르면 parameter가 다르다.**

Multi-Head Attention에서는 head마다 서로 다른 projection space를 학습한다고 이해할 수 있으며, 실제 구현에서는 여러 head의 projection을 큰 matrix에 묶어 계산하는 경우가 일반적이다.

---

## 29. When Does a Representation Become an Actual Token Prediction?

Masked Self-Attention도 prediction이 아니고,
Cross-Attention도 prediction이 아니다.

여러 Decoder Block을 모두 지난 뒤에도 output은 여전히

$$
H_{dec}:[B,T_{tgt},d_{model}]
$$

형태의 representation이다.

예를 들어 `고양이를` 위치의 최종 representation을

$$
h_3
$$

라고 하면,

$$
h_3\in\mathbb{R}^{d_{model}}
$$

이다.

이 representation을 실제 vocabulary token score로 바꾸는 것이 마지막 **Linear projection**이다.

---

## 30. Vocabulary Projection

Vocabulary size를

$$
V
$$

라고 하자.

마지막 Linear layer는

$$
W_{vocab}:[d_{model},V]
$$

를 사용한다.

따라서,

$$
[B,T_{tgt},d_{model}]
\rightarrow
[B,T_{tgt},V]
$$

가 된다.

예를 들어,

$$
d_{model}=512,
\quad
V=30000
$$

이면 각 target 위치에서

$$
[512]\rightarrow[30000]
$$

으로 바뀐다.

즉 vocabulary의 30,000개 token 각각에 대한 점수를 만든다.

---

## 31. What Are Logits?

Linear layer output은 확률이 아니라 **logits**이다.

예를 들어 `고양이를` 위치의 logits가 개념적으로

```text
나는        0.7
고양이를    1.1
좋아한다    6.8
먹는다      2.3
본다        1.9
...
```

처럼 나올 수 있다.

각 값은 vocabulary token에 대한 raw score이다.

Softmax를 적용하면 확률 분포처럼 해석할 수 있다.

$$
P(y_{t+1}=j)
=
\frac{e^{logit_j}}
{\sum_k e^{logit_k}}
$$

---

## 32. Prediction at Every Target Position

Training에서는 모든 target 위치에서 vocabulary logits를 동시에 계산한다.

```text
Decoder input:
[BOS]   나는   고양이를   좋아한다

Prediction target:
나는    고양이를   좋아한다   [EOS]
```

따라서 각 final Decoder representation은 다음 token을 예측한다.

```text
h1 → "나는"
h2 → "고양이를"
h3 → "좋아한다"
h4 → [EOS]
```

전체 logits shape은

$$
[B,T_{tgt},V]
$$

이다.

---

## 33. Cross Entropy Loss Across the Sequence

각 target 위치마다 Cross Entropy Loss가 생긴다고 생각할 수 있다.

$$
L_1,L_2,\dots,L_{T_{tgt}}
$$

예를 들어,

```text
[BOS] 위치       → 정답 "나는"
나는 위치        → 정답 "고양이를"
고양이를 위치    → 정답 "좋아한다"
좋아한다 위치    → 정답 [EOS]
```

각 위치에서 model logits와 정답 token ID를 비교한다.

보통 유효한 token들의 loss를 평균하거나 합쳐 최종적으로 하나의 scalar loss를 만든다.

$$
L
=
\frac{1}{T}
\sum_{t=1}^{T}L_t
$$

Batch가 있다면 여러 sample과 여러 token 위치를 함께 reduction해서 결국 하나의 scalar loss를 얻는다.

`[PAD]` 위치는 보통 loss 계산에서 제외한다.

---

## 34. Backpropagation Through the Whole Transformer

최종 scalar loss에 대해

```python
loss.backward()
```

를 호출하면 전체 computation graph를 따라 gradient가 계산된다.

개념적으로,

```text
Loss
↑
Vocabulary Linear
↑
Final Decoder Layer
↑
...
↑
Masked Self-Attention / Cross-Attention / FFN
↑
Earlier Decoder Layers
↑
Cross-Attention connection
↑
Encoder Layers
```

방향으로 gradient contribution이 계산된다.

따라서 next-token prediction loss는

- Vocabulary Linear
- Decoder Self-Attention
- Decoder Cross-Attention
- Decoder FFN
- Encoder layers
- Embeddings

등 전체 Transformer의 학습 가능한 parameter에 영향을 준다.

정확히는 `loss.backward()`가 parameter를 직접 수정하는 것은 아니다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

에서

- `backward()` = gradient 계산
- `optimizer.step()` = 실제 parameter update

이다.

---

## 35. Why Is Training Parallelizable Across Target Positions?

Training에서는 정답 target 전체를 알고 있다.

따라서

```text
[BOS] 나는 고양이를 좋아한다
```

를 한 번에 넣고 Causal Mask를 이용해 각 위치의 미래 정보만 차단한다.

그러면 개념적으로 다음 문제들을 동시에 계산할 수 있다.

```text
[BOS]                  → 나는
[BOS] 나는             → 고양이를
[BOS] 나는 고양이를    → 좋아한다
[BOS] 나는 고양이를 좋아한다 → [EOS]
```

즉 **sequence 위치 방향으로 병렬 계산**할 수 있다.

이것이 Transformer training의 중요한 장점 중 하나이다.

---

## 36. Inference: There Is No Correct Target Sequence

Inference에서는 정답 target이 없다.

source가

```text
I love cats
```

만 주어졌다고 하자.

Encoder는 source를 한 번 처리해 Encoder output을 만든다.

Decoder는 처음에

```text
[BOS]
```

만 입력으로 받는다.

Decoder를 거쳐 vocabulary logits가 나오고 첫 token을 선택한다.

예를 들어,

```text
[BOS]
  ↓
나는
```

가 생성되었다고 하자.

그러면 다음 Decoder input은

```text
[BOS] 나는
```

가 된다.

다시 Decoder를 실행해 다음 token을 생성한다.

```text
[BOS] 나는
      ↓
    고양이를
```

그리고 다시,

```text
[BOS] 나는 고양이를
           ↓
        좋아한다
```

이 과정을 `[EOS]`가 생성될 때까지 반복한다.

---

## 37. What Is Autoregressive Generation?

**Autoregressive Generation**은

> **모델이 이전에 생성한 output을 다시 조건으로 사용해서 다음 output을 생성하는 방식**

이다.

수식으로는 전체 target sequence의 conditional probability를

$$
P(y_1,\dots,y_T\mid x)
$$

한 번에 독립적으로 예측하는 것이 아니라,

$$
P(y_1,\dots,y_T\mid x)
=
\prod_{t=1}^{T}
P(y_t\mid y_{<t},x)
$$

처럼 분해한다.

여기서

$$
y_{<t}
$$

는 현재보다 앞에서 생성된 모든 target token이다.

예를 들어,

$$
P(\text{나는}\mid x)
$$

$$
P(\text{고양이를}\mid \text{나는},x)
$$

$$
P(\text{좋아한다}\mid \text{나는, 고양이를},x)
$$

처럼 생성한다.

---

## 38. Training vs Inference

둘의 차이를 압축하면 다음과 같다.

### Training

정답 target을 알고 있다.

```text
[BOS] 나는 고양이를 좋아한다
```

전체를 한 번에 넣고 Causal Mask를 사용한다.

각 위치에서 다음 token을 병렬로 예측한다.

이전 token은 **정답 target token**이다.

### Inference

정답 target을 모른다.

```text
[BOS]
↓
prediction
↓
[BOS] + prediction
↓
next prediction
↓
...
```

이전 token은 **모델 자신이 방금 생성한 token**이다.

---

## 39. Is Inference Slower Than Training for One Sequence?

비교 기준에 따라 다르다.

### Token-direction parallelism

Training은 target 전체를 알고 있으므로 여러 target 위치를 병렬 계산할 수 있다.

Inference는 다음 token이 생성되어야 그 다음 step을 진행할 수 있으므로 **순차적 dependency**가 있다.

출력 길이가 $T$라면 기본적으로 $T$개의 generation step이 필요하다.

따라서 autoregressive inference는 latency 측면에서 불리하다.

### But Training Also Has Backpropagation

Training은

```text
Forward
+
Loss
+
Backward
+
Optimizer Step
```

이 필요하다.

Inference는 backward가 없다.

따라서 단순히 "한 sequence의 전체 계산시간은 항상 inference가 더 길다"고 말하기보다는,

> **Training은 token 위치를 병렬 처리할 수 있지만 backward가 필요하고, Inference는 backward가 없지만 output token을 순차적으로 생성해야 한다.**

라고 이해하는 것이 정확하다.

---

## 40. Causal Mask and Autoregressive Generation Are Connected

Inference에서는 실제로 미래 token을 모른다.

```text
나는 고양이를 ???
```

상태에서 다음 token을 예측한다.

Training에서는 정답 target 전체를 이미 알고 있지만, Causal Mask를 사용해서 각 위치가

> **Inference 때와 마찬가지로 과거와 현재 정보만 아는 상태**

가 되도록 만든다.

따라서 다음 세 개념은 하나의 흐름으로 연결된다.

```text
Target Shift
↓
Causal Mask
↓
Autoregressive Next-Token Generation
```

---

## 41. Why the Original Encoder–Decoder Transformer Is a Useful Starting Point

현재 배우는 구조는 특정 번역기만을 위한 특수한 예외가 아니다.

2017년 원래 Transformer는 **Encoder–Decoder sequence-to-sequence 구조**로 제안되었고, 기계번역이 대표적인 적용 문제였다.

따라서

```text
Source sequence
→ Encoder
→ Decoder
→ Target sequence
```

구조를 먼저 배우는 것은 Transformer 전체 구조를 이해하는 정석적인 출발점이다.

이후 다른 Transformer 계열을 보면,

- Encoder-only 계열: Encoder 구조 중심
- Decoder-only 계열: causal / autoregressive Decoder 구조 중심
- Encoder–Decoder 계열: source understanding + target generation

처럼 연결해서 볼 수 있다.

---

## 42. Important Distinctions and Common Confusions

### Decoder receives the target during training, but not during inference

Training에서는 shifted target을 Decoder input으로 사용한다.

Inference에서는 정답 target이 없으므로 `[BOS]`에서 시작해 자기 prediction을 다시 input에 붙인다.

### Decoder input is not identical to the label

한 칸 shift되어 있다.

```text
Input : [BOS] 나는 고양이를 좋아한다
Label : 나는  고양이를 좋아한다 [EOS]
```

### Masked Self-Attention does not predict the next token

각 target 위치의 **causal contextual representation**을 만든다.

### Cross-Attention does not directly predict the next token either

현재 Decoder 상태를 기준으로 Encoder Value에서 필요한 source information을 가져와 새로운 representation을 만든다.

### Cross-Attention happens at every target position

특정 마지막 token에서만 source를 보는 것이 아니다.

### Decoder is Q, Encoder is K/V in Cross-Attention

Decoder는 현재 필요한 source 정보를 찾는 쪽이고, Encoder는 검색 대상과 가져올 정보를 제공하는 쪽이다.

### Causal Mask blocks attention connections, not tokens themselves

미래 token을 sequence에서 삭제하는 것이 아니라 attention score에서 미래 위치를 볼 수 없도록 막는다.

### A token can attend to itself

현재 위치의 input token은 다음 token을 예측하기 위한 context이므로 자기 자신까지 볼 수 있다.

### Encoder final output is reused by every Decoder layer

각 Decoder layer의 Cross-Attention은 같은 Encoder final representations를 참고하지만, layer별 projection parameter는 서로 다르다.

### Different attention modules have different parameters

같은 Decoder Block 안에서도 Masked Self-Attention과 Cross-Attention은 서로 다른 $W_Q,W_K,W_V$를 사용한다.

다른 Decoder Block끼리도 parameter를 공유하지 않는 것이 일반적이다.

### Prediction first appears after the final Decoder representation

Attention / FFN은 representation을 만들고 가공한다.

실제 vocabulary logits는 마지막 Linear projection에서 처음 등장한다.

### `loss.backward()` does not directly update the weights

`backward()`는 gradient를 계산하고, 실제 parameter update는 `optimizer.step()`에서 이루어진다.

---

## 43. Final Conceptual Summary

Transformer Decoder의 전체 흐름을 역할 중심으로 압축하면 다음과 같다.

```text
Shifted Target Input
[B,T_tgt]

↓
Embedding + Positional Encoding
[B,T_tgt,d_model]

↓
Masked Self-Attention
"지금까지 target에서 무엇이 생성되었는가?"

↓
Residual + LayerNorm

↓
Cross-Attention
Q = Decoder
K,V = Encoder Final Output
"현재 target 상태에서 source의 무엇을 참고해야 하는가?"

↓
Residual + LayerNorm

↓
FFN
각 token representation 내부 feature processing

↓
Residual + LayerNorm

↓
Decoder Block Output
[B,T_tgt,d_model]

↓
Decoder Blocks × N

↓
Final Decoder Representations
[B,T_tgt,d_model]

↓
Vocabulary Linear

↓
Logits
[B,T_tgt,V]

↓
각 위치에서 next-token prediction

↓
Cross Entropy Loss during training
```

한 문장으로 정리하면,

> **Decoder는 지금까지 생성된 target 문맥을 Masked Self-Attention으로 이해하고, 그 상태에서 필요한 source 정보를 Cross-Attention으로 가져온 뒤, 여러 layer를 통해 representation을 가공하여 최종적으로 vocabulary logits를 만들어 다음 token을 예측한다.**

---

## 44. Decoder Training Flow in One Example

예시:

```text
Source:
I love cats

Target:
나는 고양이를 좋아한다 [EOS]
```

Training에서는,

```text
Encoder input:
I love cats

Decoder input:
[BOS] 나는 고양이를 좋아한다

Label:
나는 고양이를 좋아한다 [EOS]
```

이다.

Decoder input 전체가 병렬로 들어가지만 Causal Mask 때문에 각 위치는 실제 생성 시점에 알 수 있는 정보만 본다.

```text
[BOS]                    → 나는
[BOS] 나는               → 고양이를
[BOS] 나는 고양이를      → 좋아한다
[BOS] 나는 고양이를 좋아한다 → [EOS]
```

각 위치의 logits에 대해 Cross Entropy Loss를 계산하고, 이를 하나의 scalar loss로 reduction한 뒤 전체 Encoder–Decoder Transformer에 backpropagation한다.

---

## 45. Decoder Inference Flow in One Example

Inference에서는 정답 target이 없다.

```text
Source:
I love cats
```

Encoder output을 먼저 만든 뒤,

```text
[BOS]
  ↓
나는

[BOS] 나는
  ↓
고양이를

[BOS] 나는 고양이를
  ↓
좋아한다

[BOS] 나는 고양이를 좋아한다
  ↓
[EOS]
```

처럼 모델이 이전에 생성한 token을 다시 Decoder input에 추가하며 한 token씩 생성한다.

이것이 **Autoregressive Generation**이다.

---

## 46. Decoder Checkpoint

Decoder를 이해했다면 다음 질문에 답할 수 있어야 한다.

1. 왜 target sequence를 Decoder input으로도 사용하는가?
2. Decoder input과 target label은 왜 한 칸 shift되어 있는가?
3. Masked Self-Attention output은 왜 prediction이 아니라 contextual representation인가?
4. Causal Mask는 attention score의 어느 부분을 왜 막는가?
5. 왜 현재 token은 자기 자신까지 볼 수 있는가?
6. Cross-Attention에서 왜 Decoder가 Q이고 Encoder가 K/V인가?
7. 왜 Cross-Attention score shape이 $[B,h,T_{tgt},T_{src}]$인가?
8. Cross-Attention은 모든 target 위치에서 수행되는가?
9. Decoder Block 하나에서 세 번의 Add & Norm은 각각 어디에 위치하는가?
10. 여러 Decoder layer가 같은 Encoder output을 보면서도 서로 다른 계산을 할 수 있는 이유는 무엇인가?
11. 실제 vocabulary prediction은 Decoder의 어느 지점에서 처음 만들어지는가?
12. Training에서 token별 loss가 어떻게 하나의 scalar loss가 되는가?
13. `loss.backward()`와 `optimizer.step()`의 역할 차이는 무엇인가?
14. Training은 왜 target 위치를 병렬 계산할 수 있는가?
15. Inference는 왜 한 token씩 순차적으로 생성해야 하는가?
16. Autoregressive Generation은 무엇인가?
17. Causal Mask와 Autoregressive Generation은 어떤 관계인가?

이 질문들을 자연스럽게 설명할 수 있다면 Transformer Decoder의 핵심 구조와 학습 / 생성 흐름을 이해한 상태라고 볼 수 있다.

---

## 47. Next Step

이제 Encoder와 Decoder를 각각 따로 이해했으므로 다음 단계에서는

```text
Source Token IDs
↓
Encoder
↓
Encoder Final Output
↓
Decoder
↓
Final Decoder Representations
↓
Vocabulary Linear
↓
Logits / Loss
```

를 **Transformer 전체 forward pass 하나의 흐름**으로 연결해서 보는 것이 자연스럽다.

그 이후에는 이 전체 구조를 PyTorch tensor shape과 작은 `.py` 구현으로 확인하면 된다.
