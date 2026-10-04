# 직접 구현한 Transformer — 구조, Tensor Flow, 학습과 추론

이 문서는 내가 PyTorch로 bottom-up 구현한 **작은 Post-LN Encoder–Decoder Transformer**를 나중에 다시 빠르게 복원하기 위한 최종 학습 노트다.

Embedding부터 Attention, Encoder, Decoder, Mask, Training, Autoregressive Inference까지 하나씩 직접 구현했기 때문에, 단순히 각 부품의 정의를 나열하기보다 **실제 tensor가 `src`와 `tgt`에서 시작해서 어떤 순서로 이동하고, 각 단계의 출력이 무엇을 의미하는지**를 중심으로 정리한다.

구현 과정에서 사용한 세부 테스트는 Colab notebook에 남겨 두었고, `12_Transformer.py`에는 최종 모델 구현과 Copy Task training/final test를 정리해 두었다.

**복습 순서:**  
전체 구조가 헷갈릴 때는 **2절 End-to-End Forward Flow**를 먼저 읽고, 특정 부품을 복습하고 싶을 때 3~7절로 내려간다. 아주 빠르게 shape만 복원하고 싶다면 마지막 **Final Shape & Forward Flow Cheat Sheet**를 본다.

---

# 1. 구현 범위와 표기

구현은 다음 순서로 진행했다.

```text
Embedding
→ Positional Encoding
→ attention()
→ Multi-Head Attention
→ FFN
→ Residual + LayerNorm
→ EncoderBlock
→ Encoder
→ DecoderBlock
→ Decoder
→ Masks
→ 전체 Transformer
→ Output Linear / Loss
→ Copy Task Training
→ Autoregressive Inference
→ Variable-length batch
```

`nn.Transformer`로 전체 모델을 가져다 쓰지 않고 핵심 구조를 직접 연결했다.

다만 모든 연산을 밑바닥부터 만든 것은 아니다.

- `nn.Embedding`
- `nn.Linear`
- `nn.LayerNorm`
- `F.softmax`
- `CrossEntropyLoss`
- `Adam`
- `pad_sequence`

등의 PyTorch 기본 모듈과 연산은 사용했다.

## 1.1 Shape 표기

| 기호 | 의미 |
| --- | --- |
| `B` | Batch size |
| `S` | Source sequence length. Padding batch에서는 `S_max` |
| `T` | Decoder input sequence length. Padding batch에서는 `T_max` |
| `D` | `d_model` |
| `h` | `num_heads` |
| `d` | `d_head = D // h` |
| `d_ff` | FFN의 hidden feature size |
| `T_q` | Query의 sequence length |
| `T_k` | Key/Value의 sequence length |
| `d_k` | Query/Key의 feature dimension |
| `d_v` | Value의 feature dimension |
| `V_src` | Source vocabulary size |
| `V_tgt` | Target vocabulary size |
| `N` | Encoder/Decoder block의 수 |

Attention의 `V` tensor와 vocabulary size의 `V`는 서로 다른 의미다. 필요할 때 vocabulary size는 `V_tgt`처럼 따로 표기한다.

Token ID는 정수 인덱스다.

```text
src [B,S]
tgt [B,T]
```

Embedding을 통과한 뒤부터는 각 token을 나타내는 연속값 representation이 된다.

```text
[B,S,D]
[B,T,D]
```

Attention은 token ID 숫자 자체를 비교하는 연산이 아니다.

## 1.2 마지막 Copy Task 설정

최종적으로 사용한 작은 Copy Task의 기본 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| Vocabulary size | 20 |
| PAD / BOS / EOS | 0 / 1 / 2 |
| 실제 data token | 3~19 |
| `d_model` | 32 |
| `num_heads` | 4 |
| `d_head` | 8 |
| `d_ff` | 64 |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Learning rate | `1e-3` |
| Training steps | 1000 |
| Loss | `CrossEntropyLoss(ignore_index=pad_idx)` |

Copy Task에서는 target sequence가 source를 그대로 복사하기 때문에

```text
y = src
```

이지만, Transformer 자체가 source와 target의 길이나 내용을 같게 요구하는 것은 아니다.

---

# 2. 코드 구조와 End-to-End Forward Flow

먼저 세부 구현을 떠나 전체를 한 번에 보면 다음과 같다.

```text
src token ids [B,S]
        │
        ▼
TokenEmbedding
[B,S,D]
        │
        ▼
PositionalEncoding
[B,S,D]
        │
        ▼
Encoder × N
[B,S,D]
        │
        │ encoder_output
        │
        └───────────────────────────┐
                                    │
                                    ▼
tgt token ids [B,T]          Decoder Cross-Attention
        │                           ▲
        ▼                           │
TokenEmbedding                     │
[B,T,D]                             │
        │                           │
        ▼                           │
PositionalEncoding                  │
[B,T,D]                             │
        │                           │
        ▼                           │
Decoder × N  ◀──────────────────────┘
[B,T,D]
        │
        ▼
Output Linear
D → V_tgt
        │
        ▼
vocab_logits
[B,T,V_tgt]
```

Training에서는 이 logits를 `tgt_label`과 비교해서 loss를 계산한다.

```text
vocab_logits [B,T,V_tgt]
        +
tgt_label [B,T]
        │
        ▼
CrossEntropyLoss
        │
        ▼
scalar loss
```

핵심 흐름을 한 문장으로 줄이면 다음과 같다.

> Encoder는 source를 문맥화된 memory인 `encoder_output [B,S,D]`로 바꾸고, Decoder는 target prefix와 그 memory를 함께 이용해 `decoder_output [B,T,D]`를 만든다. 마지막 Linear가 이를 vocabulary prediction인 `[B,T,V_tgt]` logits로 바꾼다.

---

## 2.1 시작점: `src`, `tgt`, `tgt_label`

Transformer의 입력은 처음부터 vector가 아니라 token ID다.

```text
src [B,S]
tgt [B,T]
```

Training에서 `tgt`는 정확히 말하면 `tgt_input`이다.

Copy Task에서 source가

```text
[7, 4, 9, 6]
```

이라면

```text
src       = [7, 4, 9, 6]

tgt_input = [BOS, 7, 4, 9, 6]
tgt_label = [7, 4, 9, 6, EOS]
```

가 된다.

여기서 역할을 구분해야 한다.

```text
src
tgt_input
    ↓
Transformer.forward(src, tgt)에 들어감

tgt_label
    ↓
모델 forward에는 들어가지 않고 loss 계산에 사용
```

---

## 2.2 Source: Token Embedding

시작:

```text
src [B,S]
```

`TokenEmbedding`의 핵심은

```python
nn.Embedding(V_src, D)
```

이다.

각 token ID가 embedding table의 한 row를 선택한다.

```text
[B,S]
→
[B,S,D]
```

그리고 원래 Transformer 방식에 맞춰

```text
Embedding × √D
```

scaling을 적용한다.

결과:

```text
src_embedding [B,S,D]
```

`src_embedding[b,s,:]`는

> batch `b`에서 source의 `s`번째 token을 나타내는 D차원 vector

다.

아직 다른 token의 문맥은 포함하지 않는다.

---

## 2.3 Source: Positional Encoding

Self-Attention 자체에는 token의 순서를 알 수 있는 구조가 없기 때문에 sinusoidal positional encoding을 embedding에 더한다.

```text
src_embedding [B,S,D]
        +
PE [1,S,D]
        ↓
src_x [B,S,D]
```

Shape은 그대로다.

이제 각 위치의 representation에는

```text
token 정보 + position 정보
```

가 함께 들어 있다.

현재 구현에서는 미리

```text
PE [1,max_len,D]
```

를 만들어 `register_buffer`로 저장하고, forward에서는 필요한 sequence length만큼 자른다.

```python
self.pe[:, :x.size(1), :]
```

현재 구현은 sinusoidal table 구성상 `D`가 짝수이고, 실제 입력 sequence length가 `max_len` 이하라는 전제를 가진다.

---

## 2.4 Source Padding Mask

Encoder에 들어가기 전에 source token ID를 이용해 padding mask를 만든다.

```text
src [B,S]
↓
create_padding_mask(src, pad_idx)
↓
src_mask [B,1,1,S]
```

예를 들어

```text
[8, 5, 3, PAD, PAD]
```

라면 key 방향으로 대략

```text
[True, True, True, False, False]
```

가 된다.

중요한 점은 mask를 embedding 이후 representation으로 만드는 것이 아니라

```text
token ID == pad_idx ?
```

를 직접 확인해서 만든다는 것이다.

이 `src_mask`는 두 군데에서 사용한다.

```text
Encoder Self-Attention
Decoder Cross-Attention
```

---

## 2.5 Encoder로 진입

Encoder 입력:

```text
src_x [B,S,D]
```

우리 `Encoder`는

```text
EncoderBlock × N
```

구조다.

```text
[B,S,D]
   │
   ▼
EncoderBlock 1
[B,S,D]
   │
   ▼
EncoderBlock 2
[B,S,D]
   │
  ...
   ▼
EncoderBlock N
[B,S,D]
```

각 layer를 지나도 shape은 변하지 않는다.

변하는 것은 representation의 내용이다.

초기에는

```text
token + position
```

정도의 정보였다면, 각 EncoderBlock을 지나면서 source의 다른 token들을 참고한 contextual representation으로 바뀐다.

---

## 2.6 EncoderBlock: Self-Attention

Encoder Self-Attention에서는 projection 전 입력 기준으로

```text
Q = x
K = x
V = x
```

이다.

입력:

```text
x [B,S,D]
```

Q/K/V projection 후에도 각각

```text
[B,S,D]
```

이고 head를 나누면

```text
Q [B,h,S,d]
K [B,h,S,d]
V [B,h,S,d]
```

가 된다.

Attention score:

```text
Q @ Kᵀ

[B,h,S,d]
@
[B,h,d,S]

→ [B,h,S,S]
```

여기서 마지막 두 축은

```text
S_query × S_key
```

다.

즉 각 source 위치가 다른 source 위치들을 얼마나 참고할지 나타낸다.

`src_mask [B,1,1,S]`는 이 score에 broadcast되어 PAD key를 차단한다.

Softmax 이후:

```text
encoder self-attention weights
[B,h,S,S]
```

이 나온다.

그리고

```text
weights @ V
```

를 통해 각 query가 참고한 Value의 가중합을 얻는다.

각 head의 출력:

```text
[B,h,S,d]
```

head들을 다시 합치고 `W_O`를 통과하면:

```text
[B,S,D]
```

가 된다.

### 이 출력이 의미하는 것

이 시점의 각 source token representation은 더 이상 자기 token 정보만 갖고 있지 않다.

> 자신을 query로 삼아 source의 다른 token들을 참고해 만든 contextual representation

이다.

---

## 2.7 EncoderBlock: Add & Norm → FFN → Add & Norm

우리 구현은 Post-LN 구조다.

Self-Attention 이후:

```text
h1 = LayerNorm(x + self_attn_output)
```

```text
[B,S,D]
```

FFN은 각 token 위치에 독립적으로 적용한다.

```text
D
→ d_ff
→ ReLU
→ D
```

전체 tensor shape으로 보면

```text
[B,S,D]
→ [B,S,d_ff]
→ [B,S,D]
```

이다.

그 후:

```text
output = LayerNorm(h1 + FFN(h1))
```

결과:

```text
[B,S,D]
```

Attention이 **token 간 정보 교환**을 담당한다면, FFN은 각 token의 **feature 자체를 비선형적으로 변환**한다.

---

## 2.8 Encoder 최종 출력

N개의 EncoderBlock을 모두 지나면

```text
encoder_output [B,S,D]
```

가 나온다.

이 tensor는 문장 전체를 하나의 vector로 압축한 결과가 아니다.

```text
S개의 source position
×
각각 D차원 contextual representation
```

이다.

이걸 다음처럼 기억하면 편하다.

> `encoder_output`은 Decoder가 source를 읽을 수 있도록 만들어 놓은 memory다.

모든 DecoderBlock의 Cross-Attention은 이 최종 `encoder_output`을 K/V로 사용한다.

---

## 2.9 Target: Embedding + Positional Encoding

Target 입력:

```text
tgt [B,T]
```

Training에서는 이것이 `tgt_input`이고, inference에서는 현재까지 생성한 `generated` sequence다.

Source와 같은 과정을 거친다.

```text
tgt [B,T]
↓
TokenEmbedding
[B,T,D]
↓
PositionalEncoding
[B,T,D]
```

Decoder 진입 직전:

```text
tgt_x [B,T,D]
```

이다.

Source와 target embedding은 서로 독립된 `nn.Embedding` parameter를 가진다.

---

## 2.10 Target Mask

Decoder Self-Attention에는 두 종류의 제한이 필요하다.

1. 미래 target token을 볼 수 없어야 한다.
2. Target의 PAD key를 볼 수 없어야 한다.

그래서

```text
causal mask
[1,1,T,T]

AND

target padding mask
[B,1,1,T]

↓

tgt_mask
[B,1,T,T]
```

를 만든다.

이 mask가 Decoder Self-Attention score

```text
[B,h,T,T]
```

에 broadcast된다.

---

## 2.11 DecoderBlock: Masked Self-Attention

Decoder Self-Attention에서도 projection 전 기준으로

```text
Q = K = V = x
```

이다.

```text
x [B,T,D]
→
Q/K/V [B,h,T,d]
```

Score:

```text
[B,h,T,T]
```

여기에 causal + padding mask를 적용한다.

따라서 target position `t`는

```text
0 ... t
```

까지만 볼 수 있고, 미래 target은 볼 수 없다.

Self-Attention 결과는 다시

```text
[B,T,D]
```

가 된다.

첫 번째 Add & Norm:

```text
h1 = LayerNorm(x + self_attn_output)
```

```text
h1 [B,T,D]
```

### 이 시점의 representation

여기까지는 아직 source를 읽지 않았다.

따라서 `h1`은

> 지금까지 허용된 target prefix의 문맥을 반영한 representation

이라고 보면 된다.

---

## 2.12 DecoderBlock: Cross-Attention

이제 Decoder가 Encoder output을 읽는다.

Cross-Attention에서는 Q/K/V의 출처가 다르다.

```text
Q = Decoder h1
K = encoder_output
V = encoder_output
```

Projection 전 shape:

```text
Q : [B,T,D]
K : [B,S,D]
V : [B,S,D]
```

Head를 나누면:

```text
Q : [B,h,T,d]
K : [B,h,S,d]
V : [B,h,S,d]
```

따라서 score는

```text
Q @ Kᵀ

[B,h,T,d]
@
[B,h,d,S]

→ [B,h,T,S]
```

가 된다.

이 shape이 Cross-Attention의 핵심이다.

```text
T = target query 위치
S = source key 위치
```

즉

> 각 target 위치가 source의 어느 위치를 얼마나 참고할 것인가

를 나타낸다.

Cross-Attention에는 `src_mask [B,1,1,S]`를 사용해서 source PAD key를 막는다.

결과:

```text
cross_attention_output [B,T,D]
cross_attention_weights [B,h,T,S]
```

그리고:

```text
h2 = LayerNorm(h1 + cross_attention_output)
```

```text
h2 [B,T,D]
```

### 이 시점의 representation

이제 Decoder representation에는

```text
target prefix 정보
+
source 정보
```

가 모두 들어 있다.

---

## 2.13 DecoderBlock: FFN

Cross-Attention 이후 다시 token별 FFN을 적용한다.

```text
[B,T,D]
→ [B,T,d_ff]
→ [B,T,D]
```

그리고 마지막 Add & Norm:

```text
h3 = LayerNorm(h2 + FFN(h2))
```

결과:

```text
[B,T,D]
```

이 값이 다음 DecoderBlock으로 넘어간다.

---

## 2.14 Decoder 최종 출력

N개의 DecoderBlock을 모두 지나면

```text
decoder_output [B,T,D]
```

을 얻는다.

이것은 아직 token prediction이 아니다.

예를 들어

```text
decoder_output[b,t,:]
```

는

> batch `b`의 target position `t`에서, 현재까지의 target prefix와 source 문맥을 모두 반영한 D차원 hidden representation

이다.

즉 다음 세 가지를 구분해야 한다.

```text
Hidden representation
[B,T,D]

≠

Vocabulary logits
[B,T,V_tgt]

≠

Predicted token ID
[B,T] 또는 [B]
```

---

## 2.15 Output Linear와 Vocabulary Logits

마지막으로

```python
nn.Linear(D, V_tgt)
```

를 적용한다.

```text
decoder_output
[B,T,D]

↓

output_linear

↓

vocab_logits
[B,T,V_tgt]
```

예를 들어

```text
vocab_logits[b,t,:]
```

는 target position `t`에서 vocabulary의 모든 token에 대한 raw score다.

```text
token 0 score
token 1 score
token 2 score
...
token V_tgt-1 score
```

아직 probability가 아니며, training에서는 `CrossEntropyLoss`가 내부적으로 필요한 계산을 처리하므로 softmax를 먼저 하지 않는다.

---

## 2.16 Transformer.forward() 전체를 코드 순서로 압축

실제 `Transformer.forward(src, tgt)`의 사고 순서는 다음과 같다.

```text
src [B,S]
│
├─ src_token_embedding
│    └─ [B,S,D]
│
├─ src_positional_encoding
│    └─ [B,S,D]
│
├─ create_padding_mask(src)
│    └─ src_mask [B,1,1,S]
│
└─ Encoder × N
     ├─ layer별 self-attn weights [B,h,S,S]
     └─ encoder_output [B,S,D]


tgt [B,T]
│
├─ tgt_token_embedding
│    └─ [B,T,D]
│
├─ tgt_positional_encoding
│    └─ [B,T,D]
│
├─ create_decoder_mask(tgt)
│    └─ tgt_mask [B,1,T,T]
│
└─ Decoder × N
     ├─ self-attn weights [B,h,T,T]
     ├─ cross-attn weights [B,h,T,S]
     └─ decoder_output [B,T,D]


decoder_output
│
└─ output_linear D → V_tgt
     ↓
vocab_logits [B,T,V_tgt]
```

---

## 2.17 Forward 반환값

현재 `Transformer.forward()`는 네 가지를 반환한다.

| 반환값 | Shape |
| --- | --- |
| `vocab_logits` | `[B,T,V_tgt]` |
| `encoder_attn_weights` | N개 list, 각 `[B,h,S,S]` |
| `decoder_self_attn_weights` | N개 list, 각 `[B,h,T,T]` |
| `decoder_cross_attn_weights` | N개 list, 각 `[B,h,T,S]` |

Attention weights list는 다음 layer로 전달되는 주 representation이 아니다.

```text
encoder_output / decoder_output
= 실제 forward 흐름의 메인 representation

attention weights
= 각 layer에서 attention 관계를 확인하기 위해 함께 반환한 값
```

---

# 3. Attention과 Multi-Head의 Shape

## 3.1 Scaled Dot-Product Attention

직접 구현한 `attention()`의 핵심은 다음과 같다.

```python
d_k = Q.shape[-1]
att_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

if mask is not None:
    att_scores = att_scores.masked_fill(~mask, float("-inf"))

att_weights = F.softmax(att_scores, dim=-1)
output = att_weights @ V
```

일반적인 입력:

| Tensor | Shape |
| --- | --- |
| Q | `[B,h,T_q,d_k]` |
| K | `[B,h,T_k,d_k]` |
| V | `[B,h,T_k,d_v]` |

Score 계산:

```text
Q
[B,h,T_q,d_k]

@

Kᵀ
[B,h,d_k,T_k]

=

att_scores
[B,h,T_q,T_k]
```

`att_scores[b,head,q,k]`는 query 위치 `q`와 key 위치 `k`의 compatibility score다.

Scaling:

```text
QKᵀ / √d_k
```

를 적용하고, mask가 있다면 허용하지 않는 위치를 `-inf`로 만든다.

Softmax:

```python
F.softmax(att_scores, dim=-1)
```

여기서 마지막 축은 `T_k`이므로

> 하나의 query가 여러 key 중 어디에 얼마나 attention을 줄 것인가

를 정규화한다.

결과:

```text
att_weights [B,h,T_q,T_k]
```

각 query row의 허용된 weight 합은 1이다.

마지막으로:

```text
att_weights
[B,h,T_q,T_k]

@

V
[B,h,T_k,d_v]

=

output
[B,h,T_q,d_v]
```

따라서 Attention output은

- sequence length는 Query 쪽 `T_q`
- feature size는 Value 쪽 `d_v`

를 따른다.

### 기억할 조건

- Q/K의 feature dimension은 같아야 한다.
- K/V의 sequence length는 같아야 한다.
- `T_q`와 `T_k`는 달라도 된다.
- Score와 attention weights의 shape은 같지만 의미는 다르다.
  - Score = raw compatibility
  - Weight = softmax 이후 실제 비중

---

## 3.2 `Q=K=V=x`의 정확한 의미

Self-Attention에서

```python
self.mha(x, x, x)
```

라고 쓰지만 실제 projected Q/K/V tensor가 같은 것은 아니다.

각각

```text
Q = W_Q(x)
K = W_K(x)
V = W_V(x)
```

이고 `W_Q`, `W_K`, `W_V`는 서로 다른 `nn.Linear`다.

즉

> Q/K/V를 만드는 원본 representation이 같다.

는 의미다.

---

## 3.3 Multi-Head Attention

MHA 입력:

```text
query [B,T_q,D]
key   [B,T_k,D]
value [B,T_k,D]
```

Linear projection:

```text
[B,T,D]
→
[B,T,D]
```

Head split:

```text
[B,T_q,D]
→ unflatten
[B,T_q,h,d]
→ transpose
[B,h,T_q,d]
```

Attention 이후:

```text
[B,h,T_q,d]
```

Head들을 다시 합친다.

```text
[B,h,T_q,d]
→ transpose
[B,T_q,h,d]
→ flatten
[B,T_q,D]
```

그리고:

```text
W_O : D → D
```

를 통과한다.

최종 MHA output:

```text
[B,T_q,D]
```

Head 결과는 평균내는 것이 아니라 **concat 후 W_O로 다시 섞는다.**

---

# 4. 세 Attention과 Block 내부 흐름

## 4.1 세 Attention 비교

아래 Q/K/V 출처는 projection 이전 representation 기준이다.

| Attention | Q | K / V | Score |
| --- | --- | --- | --- |
| Encoder Self | Encoder x | Encoder x | `[B,h,S,S]` |
| Decoder Masked Self | Decoder x | Decoder x | `[B,h,T,T]` |
| Decoder Cross | Decoder self-attn 이후 h1 | Encoder output | `[B,h,T,S]` |

Shape을 관계로 기억하면 쉽다.

```text
Encoder Self
source → source
[S,S]

Decoder Self
target → target
[T,T]

Decoder Cross
target → source
[T,S]
```

---

## 4.2 EncoderBlock

```text
x [B,S,D]
│
├─ Self-Attention
│    ├─ weights [B,h,S,S]
│    └─ output [B,S,D]
│
├─ Add + Norm
│    └─ [B,S,D]
│
├─ FFN
│    └─ [B,S,D]
│
└─ Add + Norm
     └─ [B,S,D]
```

식으로 쓰면:

```text
attn_output = SelfAttention(x)
h1 = LayerNorm(x + attn_output)

ffn_output = FFN(h1)
output = LayerNorm(h1 + ffn_output)
```

---

## 4.3 DecoderBlock

```text
x [B,T,D]
│
├─ Masked Self-Attention
│    ├─ weights [B,h,T,T]
│    └─ output [B,T,D]
│
├─ Add + Norm
│    └─ h1 [B,T,D]
│
├─ Cross-Attention
│    ├─ Q = h1
│    ├─ K/V = encoder_output [B,S,D]
│    ├─ weights [B,h,T,S]
│    └─ output [B,T,D]
│
├─ Add + Norm
│    └─ h2 [B,T,D]
│
├─ FFN
│    └─ [B,T,D]
│
└─ Add + Norm
     └─ h3 [B,T,D]
```

모든 DecoderBlock은 같은 최종 `encoder_output`을 받지만, 각 block의 Cross-Attention은 서로 독립된 projection parameter를 가진다.

Decoder Self-Attention과 Cross-Attention도 서로 다른 `MultiHeadAttention` 모듈이다.

---

## 4.4 FFN, Residual, LayerNorm

FFN:

```text
Linear(D,d_ff)
→ ReLU
→ Linear(d_ff,D)
```

각 token position에 동일한 network를 독립적으로 적용한다.

```text
Attention
= token들 사이에서 정보를 가져옴

FFN
= 각 token의 feature를 변환
```

Residual connection은 각 sublayer마다

```text
input + sublayer_output
```

을 계산한다.

우리 구현은:

```text
LayerNorm(x + Sublayer(x))
```

인 Post-LN 구조다.

LayerNorm은 각 token의 마지막 `D` feature 축을 정규화한다.

---

# 5. Mask 최종 정리

현재 mask 규약은:

```text
True  = attention 허용
False = attention 차단
```

이다.

## 5.1 Mask 종류

| Mask | Shape | 사용처 |
| --- | --- | --- |
| Source padding | `[B,1,1,S]` | Encoder Self / Decoder Cross |
| Causal | `[1,1,T,T]` | Decoder Self |
| Target padding | `[B,1,1,T]` | Decoder Self |
| Combined decoder | `[B,1,T,T]` | Decoder Self에 실제 전달 |

---

## 5.2 Source Padding Mask

```text
src [B,S]
↓
src != pad_idx
↓
[B,S]
↓
unsqueeze
[B,1,1,S]
```

Attention score:

```text
[B,h,T_q,S]
```

에 broadcast되어 모든 head와 모든 query가 동일한 PAD key를 보지 못하게 한다.

---

## 5.3 Causal Mask

Decoder Self-Attention에서는 미래 token을 볼 수 없어야 한다.

```text
T × T lower triangular matrix
```

를 만든다.

예:

```text
T T F F
T T T F
T T T T
T T T T
```

정확히는 대각선을 포함한 lower triangle이다.

현재 token이 자기 자신을 볼 수 있는 이유는 target shift 때문이다.

예를 들어:

```text
tgt_input
[BOS, a, b, c]

tgt_label
[a, b, c, EOS]
```

position 2의 입력 token은 `b`이고, 이 위치에서 예측해야 하는 것은 `c`다.

따라서

```text
BOS, a, b
```

까지 보는 것이 맞고, 미래의 `c`를 보는 것만 막아야 한다.

---

## 5.4 Decoder Combined Mask

```text
causal
[1,1,T,T]

AND

target padding
[B,1,1,T]

=

combined
[B,1,T,T]
```

이 mask가

```text
Decoder Self-Attention score
[B,h,T,T]
```

에 적용된다.

---

## 5.5 Broadcasting에서 차원 1의 의미

Mask가 attention score와 물리적으로 같은 shape일 필요는 없다.

```text
score
[B,h,T_q,T_k]
```

에 broadcast 가능하면 된다.

예:

```text
[B,1,1,S]
```

은

- `h`축의 모든 head
- `T_q`축의 모든 query

에 같은 source padding rule을 적용한다는 뜻이다.

차원 `1`이 “head 1개만 적용한다”는 뜻은 아니다.

---

## 5.6 Attention PAD Mask와 Loss의 `ignore_index`

둘은 역할이 다르다.

| 기능 | 역할 |
| --- | --- |
| Padding attention mask | PAD를 Key/Value로 참고하지 못하게 함 |
| `ignore_index=pad_idx` | Label이 PAD인 위치의 loss를 계산하지 않음 |

Padding mask가 PAD query의 output 자체를 0으로 만드는 것은 아니다.

또한 현재 `nn.Embedding`에서 `padding_idx`를 지정하지 않았지만, 이것과 attention padding mask는 별개의 문제다.

---

## 5.7 현재 구현에서 알아둘 전제

현재 Copy Task와 mask 구현은 다음 조건에서 사용했다.

- Source에는 적어도 하나 이상의 유효 token이 있다.
- Target은 BOS로 시작한다.
- Padding은 오른쪽에 붙인다.
- Source/target이 같은 `pad_idx`를 사용한다.
- 한 query의 모든 key가 mask되는 경우는 따로 처리하지 않는다.

모든 score가 `-inf`가 되는 row가 만들어지면 softmax에서 NaN이 생길 수 있기 때문에, 현재 구현은 All-PAD source나 특수한 left-padding 상황까지 일반적으로 처리하는 구현은 아니다.

---

# 6. Training

## 6.1 Target Shift

이 문서에서 `y`는 EOS가 붙기 전 실제 정답 token sequence를 의미한다.

```text
y = [a,b,c]
```

그러면:

```text
tgt_input = [BOS,a,b,c]
tgt_label = [a,b,c,EOS]
```

이다.

Copy Task에서는

```text
y = src
```

이므로:

```text
tgt_input = [BOS] + src
tgt_label = src + [EOS]
```

가 된다.

위치별 대응:

| Position | Decoder Input Prefix | Label |
| --- | --- | --- |
| 0 | BOS | a |
| 1 | BOS, a | b |
| 2 | BOS, a, b | c |
| 3 | BOS, a, b, c | EOS |

---

## 6.2 Teacher Forcing

Training에서는 target prefix를 모델이 직접 생성하지 않고 정답으로 제공한다.

그래서 Decoder input 전체가 한 번에

```text
[B,T]
```

tensor로 들어간다.

하지만 causal mask 때문에 position `t`는 미래 target 위치를 볼 수 없다.

따라서 한 번의 forward에서 T개 위치의 next-token prediction을 병렬로 학습할 수 있다.

Training에서 모델이 배우는 형태는:

```text
p(y_t | source, y_<t)
```

이다.

Autoregressive inference에서는 같은 조건 구조를 사용하지만 `y_<t`에 정답이 아니라 모델이 이전에 생성한 token들이 들어간다.

---

## 6.3 Logits와 Loss

Transformer 출력:

```text
vocab_logits [B,T,V_tgt]
```

Label:

```text
tgt_label [B,T]
```

`CrossEntropyLoss`에 넣기 위해:

```text
vocab_logits
[B,T,V_tgt]

→ reshape

[B*T,V_tgt]
```

Label도:

```text
[B,T]
→
[B*T]
```

로 펼친다.

```text
CrossEntropyLoss(
    logits [B*T,V_tgt],
    labels [B*T]
)
```

결과:

```text
scalar loss
```

현재는:

```python
CrossEntropyLoss(ignore_index=pad_idx)
```

를 사용한다.

따라서 `tgt_label`이 PAD인 위치는 loss 계산에서 제외된다.

CE에는 raw logits를 넣기 때문에 model output에 softmax를 먼저 적용하지 않는다.

---

## 6.4 Training Loop

현재 `.py`에 남겨둔 training loop의 핵심은 다음과 같다.

```text
generate batch
↓
optimizer.zero_grad()
↓
model(src, tgt_input)
↓
reshape logits / labels
↓
CrossEntropyLoss
↓
loss.backward()
↓
optimizer.step()
```

`argmax`는 training loss 경로에 들어가지 않는다.

---

## 6.5 Variable-Length Batch

가변 길이 Copy Task에서는 sample마다 먼저 실제 sequence를 만든다.

예를 들어 lengths:

```text
[2,5,3,4]
```

라면:

```text
B = 4
S_max = 5
T_max = 6
```

이 된다.

첫 번째 sample이 실제로 `[a,b]`라면:

```text
src
[a,b,PAD,PAD,PAD]

tgt_input
[BOS,a,b,PAD,PAD,PAD]

tgt_label
[a,b,EOS,PAD,PAD,PAD]
```

각 sequence에 BOS/EOS를 먼저 붙인 뒤 `pad_sequence()`를 사용해 오른쪽 padding한다.

따라서 EOS는 각 sample의 실제 정답 마지막 token 바로 뒤에 놓인다.

---

# 7. Autoregressive Inference

Training과 inference에서 Transformer 내부 구조 자체가 바뀌는 것은 아니다.

가장 큰 차이는 Decoder에 들어가는 target prefix다.

```text
Training
정답 prefix 사용

Inference
모델 자신이 생성한 prefix 사용
```

---

## 7.1 시작

`greedy_decode()`는:

```text
generated = [BOS]
```

에서 시작한다.

Batch라면:

```text
generated [B,1]
```

이다.

---

## 7.2 한 step

현재 prefix를 모델에 넣는다.

```text
model(src, generated)
```

현재 generated length가 `T_current`라면:

```text
vocab_logits
[B,T_current,V_tgt]
```

가 나온다.

새로운 token을 만들기 위해 필요한 것은 마지막 position의 logits뿐이다.

```python
next_token_logits = vocab_logits[:, -1, :]
```

Shape:

```text
[B,V_tgt]
```

Greedy decoding에서는:

```python
next_token = next_token_logits.argmax(dim=-1)
```

으로 가장 score가 큰 token을 선택한다.

```text
next_token [B]
```

이후:

```text
[B]
→ unsqueeze
[B,1]
```

하고

```text
generated + next_token
```

을 concat한다.

```text
[B,T_current]
+
[B,1]

→
[B,T_current+1]
```

---

## 7.3 반복

전체 흐름:

```text
[BOS]
↓
forward
↓
next token
↓
[BOS, token1]
↓
forward
↓
next token
↓
[BOS, token1, token2]
↓
...
```

EOS가 나오거나 `max_new_tokens`에 도달할 때까지 반복한다.

---

## 7.4 Batch별 EOS 처리

Batch 안의 sequence마다 EOS가 나오는 시점이 다를 수 있다.

그래서:

```text
finished [B]
```

bool tensor로 각 sample의 종료 여부를 추적한다.

순서:

1. 현재 next token을 예측한다.
2. 이미 `finished=True`였던 sample은 PAD로 바꾼다.
3. 이번 step에서 새로 EOS가 나온 sample을 `finished=True`로 기록한다.
4. Next token을 generated에 붙인다.
5. `finished.all()`이면 종료한다.

따라서 먼저 끝난 sample은 이후 PAD만 붙고, 다른 sample은 계속 생성할 수 있다.

---

## 7.5 현재 구현의 계산 방식

현재 `greedy_decode()`는 매 step:

```python
model(src, generated)
```

전체를 다시 호출한다.

따라서

- Encoder도 다시 계산하고
- Decoder도 현재 prefix 전체를 다시 계산한다.

이것은 **동작 자체는 올바르지만 효율 최적화를 적용하지 않은 구현**이다.

이번 구현에서는 Transformer의 구조와 autoregressive flow를 직접 연결하는 데 집중했기 때문에 Encoder output 재사용이나 KV cache는 구현하지 않았다.

---

# 8. 직접 구현하면서 확인한 동작

구현 과정에서는 각 부품을 만든 뒤 작은 tensor로 shape과 동작을 직접 확인했다.

## 8.1 단계별 테스트

확인한 내용은 크게 다음과 같다.

| 단계 | 확인한 내용 |
| --- | --- |
| Embedding / PE | `[B,T] → [B,T,D]`, PE 이후 shape 유지 |
| `attention()` | `T_q != T_k`, `d_k != d_v`에서도 올바른 output shape |
| MHA | Head split / concat 후 `[B,T_q,D]` 복원 |
| EncoderBlock | Input/output `[B,S,D]`, attention `[B,h,S,S]` |
| Encoder | N개 layer를 지나며 representation shape 유지 |
| DecoderBlock | Self `[B,h,T,T]`, Cross `[B,h,T,S]` |
| Decoder | N개 layer별 self/cross weights 저장 |
| Masks | 미래 token과 PAD key 차단 |
| Transformer | 최종 logits `[B,T,V_tgt]` |
| Loss | Logits/label flatten 후 scalar CE |
| Copy Task | Training loss 감소 및 teacher-forced prediction 확인 |
| Greedy inference | BOS부터 모델 자신의 prediction을 다시 input으로 사용 |
| B>1 inference | Sample별 EOS 종료 추적 |
| Variable-length | Padding / EOS / loss ignore가 함께 동작 |

---

## 8.2 Fixed-Length에서 Variable-Length까지

처음에는 고정 길이 Copy Task로 학습했다.

이 단계에서는 전체 Transformer가 실제로 학습되고, teacher-forced prediction과 autoregressive generation까지 연결되는 것을 확인할 수 있었다.

하지만 고정 길이 데이터만 학습한 모델을 다른 길이에 적용하면 실제 sequence 끝에서 EOS를 안정적으로 생성하지 못하는 경우가 있었다.

이 경험을 통해:

> Mask가 variable length tensor를 처리할 수 있다는 것과, 모델이 sequence length에 따른 종료 패턴을 학습했다는 것은 서로 다른 문제다.

라는 점을 확인했다.

이후 가변 길이 sequence들을 섞어 학습하고, 각 sample의 실제 끝에 EOS를 두도록 data generation을 바꿨다.

최종 테스트에서는 variable-length batch에 대해

- Teacher-forcing prediction
- B>1 autoregressive generation
- Sample별 EOS 종료
- EOS 이후 PAD 처리

를 함께 확인했다.

이 테스트는 Transformer가 모든 일반적인 sequence task에 대해 검증되었다는 의미가 아니라, **내가 구현한 구조의 forward / training / autoregressive generation flow가 작은 Copy Task에서 처음부터 끝까지 연결된다는 것을 확인하기 위한 테스트**였다.

---

# 9. 표준 Transformer와 비교한 내 구현의 범위

## 9.1 핵심적으로 구현한 구조

내 구현은 원래 Encoder–Decoder Transformer의 핵심 흐름을 그대로 포함한다.

- Learned token embedding
- `√D` embedding scaling
- Sinusoidal positional encoding
- Scaled dot-product attention
- Multi-head projection
- Encoder Self-Attention
- Decoder Masked Self-Attention
- Encoder–Decoder Cross-Attention
- FFN
- Residual connection
- Post-LayerNorm
- Encoder / Decoder layer stacking
- Causal mask
- Padding mask
- Shifted target training
- Vocabulary output projection
- Autoregressive generation

즉 작은 Copy Task용 모델이지만, 내가 공부하려고 했던 Transformer의 핵심 데이터 흐름은 직접 연결했다.

---

## 9.2 학습 목적상 단순화한 부분

| 항목 | 내 구현 |
| --- | --- |
| 규모 | `N=2`, `D=32`, `h=4`, `d_ff=64` |
| 데이터 | 작은 vocabulary의 Copy Task |
| Dropout | 없음 |
| Label smoothing | 없음 |
| Optimizer schedule | Adam + 고정 learning rate |
| Decode | Greedy |
| Embedding / output weight sharing | 하지 않음 |
| Source / target embedding | 독립 parameter |
| Attention Linear bias | PyTorch `nn.Linear` 기본 bias 사용 |

원 논문의 Base Transformer보다 훨씬 작은 설정이고 training recipe도 단순화했다.

목적은 논문의 번역 성능을 재현하는 것이 아니라:

> Transformer 내부의 tensor flow와 training / inference 구조를 직접 구현해 이해하는 것

이었다.

---

## 9.3 이번 구현에서 다루지 않은 부분

다음 요소들은 현재 `12_Transformer.py`의 범위에는 들어 있지 않다.

| 요소 | 현재 구현에서의 상태 |
| --- | --- |
| Dropout | 사용하지 않음 |
| Label smoothing | 사용하지 않음 |
| LR warmup / Transformer schedule | 사용하지 않음 |
| Beam search | Greedy decoding만 구현 |
| KV cache | Prefix 전체를 매 step 다시 계산 |
| Encoder output caching | Autoregressive step마다 Encoder도 다시 계산 |
| SDPA / FlashAttention | 직접 score → mask → softmax → V 계산 |
| Mixed precision | 별도로 사용하지 않음 |
| Weight tying | Embedding과 output projection이 독립 |
| Large-scale evaluation | 작은 Copy Task 기능 검증 중심 |
| Checkpoint / resume system | 별도로 구현하지 않음 |

이 항목들은 “앞으로 반드시 구현해야 하는 TODO”가 아니라, **현재 코드가 어디까지를 구현한 것인지 경계를 명확히 하기 위한 기록**이다.

Transformer 직접 구현 학습은 여기까지를 하나의 완성된 단위로 본다.

---

# 참고 자료

1. Vaswani et al., **Attention Is All You Need** — 원래 Encoder–Decoder Transformer 구조와 scaled dot-product / multi-head attention.
2. PyTorch `CrossEntropyLoss` documentation — Raw logits, `ignore_index`, reduction.
3. PyTorch `scaled_dot_product_attention`, `MultiheadAttention` documentation — Attention API와 bool mask 규약 비교.

---

# Final Shape & Forward Flow Cheat Sheet

## A. 전체 Forward

```text
src IDs
[B,S]
↓
Embedding × √D
[B,S,D]
↓
Positional Encoding
[B,S,D]
↓
Encoder × N
[B,S,D]
= encoder_output


tgt IDs
[B,T]
↓
Embedding × √D
[B,T,D]
↓
Positional Encoding
[B,T,D]
↓
Decoder × N
  └─ encoder_output 참고
[B,T,D]
= decoder_output

↓
Linear(D → V_tgt)

vocab_logits
[B,T,V_tgt]
```

---

## B. EncoderBlock

```text
x [B,S,D]
↓
Self-Attention
weights [B,h,S,S]
output [B,S,D]
↓
Add + Norm
[B,S,D]
↓
FFN
[B,S,D]
↓
Add + Norm
[B,S,D]
```

---

## C. DecoderBlock

```text
x [B,T,D]
↓
Masked Self-Attention
weights [B,h,T,T]
↓
Add + Norm
[B,T,D]
↓
Cross-Attention
Q: Decoder [B,T,D]
K/V: Encoder [B,S,D]
weights [B,h,T,S]
↓
Add + Norm
[B,T,D]
↓
FFN
[B,T,D]
↓
Add + Norm
[B,T,D]
```

---

## D. Attention

```text
Q [B,h,T_q,d_k]
K [B,h,T_k,d_k]
V [B,h,T_k,d_v]

Q @ Kᵀ
→ [B,h,T_q,T_k]

÷ √d_k
→ score

mask
→ forbidden score = -inf

softmax(dim=-1)
→ attention weights
[B,h,T_q,T_k]

weights @ V
→ output
[B,h,T_q,d_v]
```

---

## E. Multi-Head

```text
[B,T,D]
↓ W_Q / W_K / W_V
[B,T,D]
↓ split
[B,T,h,d]
↓ transpose
[B,h,T,d]
↓ attention
[B,h,T,d]
↓ transpose
[B,T,h,d]
↓ concat
[B,T,D]
↓ W_O
[B,T,D]
```

---

## F. 세 Attention

| Attention | 관계 | Weights |
| --- | --- | --- |
| Encoder Self | Source → Source | `[B,h,S,S]` |
| Decoder Self | Target → Target | `[B,h,T,T]` |
| Decoder Cross | Target → Source | `[B,h,T,S]` |

---

## G. Masks

```text
Encoder Self
src padding mask
[B,1,1,S]


Decoder Self
causal
[1,1,T,T]

AND

target padding
[B,1,1,T]

=

combined
[B,1,T,T]


Decoder Cross
src padding mask
[B,1,1,S]
```

**True = 허용 / False = 차단**

차원이 1이면 해당 축 전체에 같은 mask가 broadcasting된다.

---

## H. Training

정답:

```text
y = [a,b,c]
```

Shift:

```text
tgt_input
[BOS,a,b,c]

tgt_label
[a,b,c,EOS]
```

Forward:

```text
src + tgt_input
↓
Transformer
↓
logits [B,T,V]
```

Loss:

```text
logits
[B,T,V]
→ [B*T,V]

labels
[B,T]
→ [B*T]

CrossEntropyLoss
→ scalar
```

PAD label은 `ignore_index=pad_idx`로 제외한다.

---

## I. Autoregressive Inference

```text
generated = [BOS]

↓ model(src, generated)

logits [B,current_T,V]

↓ logits[:,-1,:]

[B,V]

↓ argmax

next_token [B]

↓ concat

generated [B,current_T+1]

↓ 반복
```

EOS가 생성되면 해당 sample을 `finished=True`로 기록하고, 다음 step부터 PAD를 붙인다.

---

## J. 마지막으로 구분할 세 가지

```text
decoder_output
[B,T,D]
=
source + target prefix를 반영한 hidden representation


vocab_logits
[B,T,V]
=
각 위치에서 vocabulary token별 raw score


argmax(logits)
[B,T] 또는 [B]
=
실제로 선택한 token ID
```

그리고 전체 모델을 한 줄로 줄이면:

```text
Source
[B,S]
→ [B,S,D]
→ Encoder
→ memory [B,S,D]
                ↘
                  Cross-Attention
                ↗
Target
[B,T]
→ [B,T,D]
→ Decoder
→ [B,T,D]
→ Linear
→ [B,T,V]
```
