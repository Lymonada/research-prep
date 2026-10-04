# 직접 구현한 Transformer — 구조, Tensor Flow, 학습과 추론

이 문서는 내가 bottom-up으로 구현한 **작은 Post-LN Encoder–Decoder Transformer**를 복원하기 위한 학습 기록이다. 코드에서 실제로 수행하는 연산과 tensor의 의미를 중심으로, 기존 학습 노트의 중복 설명을 통합했다. 코드 수정 사항을 적용했다고 가정하지 않으며, 원본 파일의 미해결 정리 항목은 10절에 따로 기록한다.

**복습 순서:** 전체 연결은 2절 → attention과 mask는 3~7절. 시간이 짧으면 마지막 **Final Shape & Forward Flow Cheat Sheet**부터 읽는다.

## 1. 구현 범위와 표기

구현 순서는 다음과 같았다.

Embedding → Positional Encoding → attention → Multi-Head Attention → FFN → Residual/LayerNorm → Encoder → Decoder → Masks → Transformer 조립 → Logits/Loss → Copy Task 학습 → Greedy autoregressive inference → 가변 길이 batch.

핵심 모델은 직접 조립했고, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` 등은 PyTorch 모듈을 사용했다. `nn.Transformer` 하나를 호출해 모델 전체를 대체하지 않았다.

### 1.1 Shape 표기

| 기호 | 의미 |
| --- | --- |
| `B` | Batch size |
| `S` | 현재 source tensor의 sequence length. Padding batch에서는 `S_max` |
| `T` | 현재 decoder input tensor의 sequence length. 학습에서는 padded target input 길이, 추론에서는 현재 prefix 길이 |
| `D` | `d_model`: token representation의 feature 수 |
| `h` | `num_heads` |
| `d` | `d_head = D // h`. 현재 MHA에서는 `d_k = d_v = d` |
| `T_q` | Query 위치의 개수: attention output의 sequence length |
| `T_k` | Key/Value 쌍의 개수: 각 query가 참고할 위치의 개수 |
| `d_k`, `d_v` | 일반 attention 함수의 Q/K feature 수와 Value feature 수 |
| `V_src`, `V_tgt` | Source/target vocabulary 크기. Logits에서 쓰는 `V`는 `V_tgt` |
| `N` | Encoder와 Decoder 각각의 block 수 |

**Attention의 `V` tensor와 vocabulary 크기 `V`는 다른 의미다.** 이 문서에서는 필요할 때 vocabulary 크기를 `V_tgt`로 쓴다.

Token IDs는 정수 인덱스이고, embedding 이후 tensor는 연속값 representation이다. Attention은 token ID 숫자 자체를 비교하는 연산이 아니다.

### 1.2 마지막 Copy Task 실험 설정

| 항목 | 값 |
| --- | --- |
| Vocabulary / special tokens | 20 / PAD=0, BOS=1, EOS=2 |
| 실제 데이터 token IDs | 3~19 |
| D / h / d | 32 / 4 / 8 |
| FFN hidden size / layers | 64 / Encoder 2층, Decoder 2층 |
| Positional table | 기본 `max_len=5000` |
| Optimizer / learning rate | Adam / 고정 `1e-3` |
| Training steps | 1000 |
| 최종 training batch | 33개 sample, 길이 2~9가 섞인 고정 lengths 목록; token 값은 step마다 새로 생성 |
| Loss | `CrossEntropyLoss(ignore_index=0)` |

따라서 마지막 학습 설정에서는 `src [33,9]`, `tgt_input/tgt_label [33,10]`, logits `[33,10,20]`이 된다. 이것은 특정 실험의 크기이며 모델 자체는 `S=T`나 `T=S+1`을 요구하지 않는다. `T=S+1`은 이 Copy Task의 데이터 구성에서 나온다.

## 2. 코드 구조와 End-to-End Forward Flow

### 2.1 코드에서 각 부품이 맡는 역할

| 코드 | 역할 |
| --- | --- |
| `TokenEmbedding` | Token ID를 D차원 학습 가능한 vector로 바꾸고 √D scaling |
| `PositionalEncoding` | Sinusoidal 위치 정보를 embedding에 더함 |
| `attention` | Projected Q/K/V로 scaled dot-product attention 계산 |
| `MultiHeadAttention` | Q/K/V projection, head split, attention, concat, W_O |
| `FFN` | 각 token의 feature 변환: D → d_ff → D |
| `ResidualLayerNorm` | 각 sublayer에서 `LayerNorm(x + sublayer_output)` |
| `EncoderBlock`, `Encoder` | Source self-attention + FFN을 N층 수행 |
| `DecoderBlock`, `Decoder` | Masked self-attention + cross-attention + FFN을 N층 수행 |
| Mask 함수 3개 | Causal mask, padding mask, decoder combined mask 생성 |
| `Transformer` | Source/target 경로 연결 후 vocabulary logits 반환 |
| Copy batch 생성 함수 2개 | 정답 prefix와 next-token label 생성, 필요하면 padding |
| Training loop | Forward → CE loss → backward → parameter update |
| `greedy_decode` | BOS에서 시작해 자신의 예측을 prefix에 추가 |

### 2.2 Source side

`src token ids [B,S]`  
→ `src_token_embedding`: lookup × √D → `[B,S,D]`  
→ `src_positional_encoding`: PE를 더함 → `[B,S,D]`  
→ `encoder`: EncoderBlock × N → `encoder_output [B,S,D]`  

별도 mask 경로:

`src token ids [B,S]` → `create_padding_mask(src, pad_idx)` → `src_mask [B,1,1,S]`

Mask는 embedding 값에서 추정하지 않는다. **원래 token ID를 PAD ID와 비교**해서 만든다. 이 `src_mask`가 Encoder self-attention과 Decoder cross-attention 양쪽에 전달된다.

최종 Encoder output은 문장 하나를 압축한 단일 vector가 아니다. **Source의 각 위치에 대응하는 contextual representation S개**이며, 모든 Decoder block이 이 최종 출력을 참고한다.

**Embedding과 PE에서 복원할 세부 연산:** embedding weight는 `[V_src,D]`이고 학습된다. Token ID는 그 행을 선택하는 인덱스다. PE는 `position [max_len,1]`과 `div_term [D/2]`를 곱해 `[max_len,D/2]`를 만들고, 짝수 feature에 sin, 홀수 feature에 cos를 채운다. 같은 feature pair는 같은 주파수를 사용한다.

`PE(pos,2i)=sin(pos / 10000^(2i/D))`, `PE(pos,2i+1)=cos(pos / 10000^(2i/D))`.

완성된 PE는 `[1,max_len,D]`로 `register_buffer`에 등록한다. 학습 parameter는 아니지만 모델의 device 이동과 저장에 포함된다. Forward에서는 현재 길이 L만큼 잘라 `[1,L,D]`를 `[B,L,D]` embedding에 더한다. Token 정보와 position 정보가 결합된 값을 이후 layer가 학습해 활용한다. Target 쪽도 같은 방식이며 source/target embedding parameter는 서로 독립이다.

### 2.3 Target side와 output

`tgt token ids [B,T]`  
→ `tgt_token_embedding`: lookup × √D → `[B,T,D]`  
→ `tgt_positional_encoding` → `[B,T,D]`  
→ `decoder(tgt_x, encoder_output, tgt_mask, src_mask)` → `[B,T,D]`  
→ `output_linear: Linear(D,V_tgt)` → `vocab_logits [B,T,V_tgt]`  

별도 mask 경로:

`tgt token ids [B,T]` → `create_decoder_mask(tgt, pad_idx)` → `tgt_mask [B,1,T,T]`

여기서 `tgt`의 값은 실행 목적에 따라 달라진다.

| 실행 상황 | `Transformer.forward(src, tgt)`에 전달하는 tgt |
| --- | --- |
| 학습 / teacher-forced prediction | 정답으로 구성한 `tgt_input` |
| Autoregressive generation | BOS와 지금까지 모델이 생성한 token으로 구성한 `generated` |

**`Transformer.forward`가 target shift나 token 생성을 수행하는 것은 아니다.** Shift는 데이터 생성 함수가, 생성 반복은 `greedy_decode`가 담당한다.

### 2.4 Forward 반환값

| 순서 | 값 | Shape |
| --- | --- | --- |
| 1 | `vocab_logits` | `[B,T,V_tgt]` |
| 2 | `encoder_attn_weights` | 길이 N의 list, 각 `[B,h,S,S]` |
| 3 | `decoder_self_attn_weights` | 길이 N의 list, 각 `[B,h,T,T]` |
| 4 | `decoder_cross_attn_weights` | 길이 N의 list, 각 `[B,h,T,S]` |

List는 layer별 결과를 담는다. Batch나 head를 list로 나눈 것이 아니다. 현재 첫 반환값은 Decoder hidden state가 아니라 **output projection까지 지난 logits**다.

## 3. Attention과 Multi-Head의 Shape

### 3.1 Scaled Dot-Product Attention

현재 함수의 핵심 연산은 다음과 같다.

```python
d_k = Q.shape[-1]
att_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
if mask is not None:
    att_scores = att_scores.masked_fill(~mask, float('-inf'))
att_weights = F.softmax(att_scores, dim=-1)
output = att_weights @ V
```

| 단계 | Shape | 의미 |
| --- | --- | --- |
| Q | `[B,h,T_q,d_k]` | 정보를 요청하는 위치별 query |
| K | `[B,h,T_k,d_k]` | Query와 비교할 key |
| V | `[B,h,T_k,d_v]` | 실제로 모아 올 정보 |
| Kᵀ | `[B,h,d_k,T_k]` | 마지막 두 축을 교환 |
| QKᵀ / √d_k | `[B,h,T_q,T_k]` | 각 query–key 쌍의 score |
| Mask 적용 | `[B,h,T_q,T_k]` | 금지된 연결의 score를 −∞로 만듦 |
| Softmax | `[B,h,T_q,T_k]` | 각 query의 key별 attention weights |
| Weights @ V | `[B,h,T_q,d_v]` | 각 query가 모아 온 Value의 가중합 |

기억할 제약:

- Q와 K의 feature 수는 같아야 한다: `d_k`.
- K와 V의 sequence length는 같아야 한다: `T_k`.
- Q의 길이 `T_q`와 K/V의 길이 `T_k`는 달라도 된다.
- Output의 sequence length는 **Query 쪽 `T_q`**, feature 수는 **Value 쪽 `d_v`**다.
- `softmax(dim=-1)`은 key 축을 정규화한다. 허용 key가 하나 이상 있으면 각 query row의 weights 합은 1이다.

Mask는 score에 적용하고, softmax 후 얻은 weights로 V를 가중합한다. Score와 weights는 shape은 같지만 각각 raw compatibility와 정규화된 비중이라는 차이가 있다.

독립 함수 테스트에서는 `d_k=8`, `d_v=6`으로 다르게 설정해도 output `[B,h,T_q,6]`이 나왔다. MHA에서는 구현을 단순하게 하기 위해 `d_k=d_v=D/h`를 사용한다.

### 3.2 Q/K/V의 입력과 projection 결과 구분

`self.mha(x,x,x)`는 Q/K/V를 만드는 **원본 representation이 같다**는 뜻이다. Projection 이후 실제 tensor 값까지 같다는 뜻이 아니다.

`Q = W_Q(query)`, `K = W_K(key)`, `V = W_V(value)`이며 세 Linear는 독립 parameter를 가진다. 코드의 `nn.Linear`는 bias도 포함한다.

### 3.3 Head split과 concat

| 단계 | Q / query 쪽 shape |
| --- | --- |
| MHA 입력 | `[B,T_q,D]` |
| W_Q projection | `[B,T_q,D]` |
| `unflatten(-1, (h,d))` | `[B,T_q,h,d]` |
| `transpose(1,2)` | `[B,h,T_q,d]` |
| Attention output | `[B,h,T_q,d]` |
| `transpose(1,2)` | `[B,T_q,h,d]` |
| `flatten(-2)` | `[B,T_q,D]` |
| W_O projection | `[B,T_q,D]` |

K/V도 같은 방식으로 분리하되 길이는 `T_k`다. `D % h == 0` 조건을 확인한다.

큰 `Linear(D,D)`의 출력 feature를 head별로 나누면 각 head는 서로 다른 가중치 묶음을 사용한다. 따라서 head마다 Linear를 따로 작성하지 않아도 서로 다른 projection을 표현할 수 있다. 먼저 원본 feature를 잘라 독립 attention에 넣는 것과는 다르다.

Head output은 **평균이 아니라 concat**하고, W_O로 결합한다. W_O는 head 결과를 D차원 representation으로 섞는 projection이다. 마지막 vocabulary projection `output_linear`와 역할이 다르다.

`transpose → flatten`은 필요한 경우 내부 복사가 일어날 수 있지만 올바른 결합이다. 코드 모양을 `contiguous().view(...)`로 바꾸어야만 정답인 것은 아니다.

## 4. 세 Attention과 Block 내부 흐름

### 4.1 Attention 비교

아래 Q/K/V 출처는 **projection 이전**을 뜻한다.

| 종류 | Q의 출처 | K/V의 출처 | `[T_q,T_k]` | Score/weights | Output |
| --- | --- | --- | --- | --- | --- |
| Encoder Self | 현재 Encoder block 입력 x | 같은 x | `[S,S]`: source → source | `[B,h,S,S]` | `[B,S,D]` |
| Decoder Masked Self | 현재 Decoder block 입력 x | 같은 x | `[T,T]`: target → target | `[B,h,T,T]` | `[B,T,D]` |
| Decoder Cross | Self-Attention 후 Add & Norm 결과 | 최종 Encoder output | `[T,S]`: target → source | `[B,h,T,S]` | `[B,T,D]` |

Encoder는 source의 모든 유효 위치를 참고한다. Decoder self-attention은 현재 위치까지의 유효 target 위치만 참고한다. Cross-attention은 각 target 위치가 source의 모든 유효 위치를 참고한다.

Cross-attention에서 **Query의 개수가 output 위치의 개수를 결정**하므로 S개의 source를 참고해도 output 길이는 T다. 모든 Decoder 위치에서 수행되며 마지막 위치에서만 수행되는 것이 아니다.

### 4.2 EncoderBlock — 두 번의 Add & Norm

입력 `x [B,S,D]`에 대해:

1. `attn_output = MHA(x,x,x, src_mask)` → `[B,S,D]`
2. `h1 = LayerNorm(x + attn_output)` → `[B,S,D]`
3. `ffn_output = FFN(h1)` → `[B,S,D]`
4. `h2 = LayerNorm(h1 + ffn_output)` → `[B,S,D]`

Encoder는 이 block을 N번 순차 적용한다. 앞 layer의 contextual representation이 다음 layer의 입력이 된다. 같은 구조를 반복하지만 각 block의 parameter는 별도다.

### 4.3 DecoderBlock — 세 번의 Add & Norm

입력 `x [B,T,D]`, Encoder memory `E [B,S,D]`에 대해:

1. `self_output = self_attn(x,x,x, self_mask)` → `[B,T,D]`
2. `h1 = LayerNorm(x + self_output)` → `[B,T,D]`
3. `cross_output = cross_attn(h1,E,E, cross_mask)` → `[B,T,D]`
4. `h2 = LayerNorm(h1 + cross_output)` → `[B,T,D]`
5. `ffn_output = FFN(h2)` → `[B,T,D]`
6. `h3 = LayerNorm(h2 + ffn_output)` → `[B,T,D]`

코드의 Cross-Attention query 입력은 `residual_ln1`, 즉 h1이다. 각 Decoder block은 같은 최종 E를 받지만 서로 다른 cross-attention projection을 학습한다. Self-attention과 cross-attention도 서로 별개의 MHA다.

### 4.4 FFN, Residual, LayerNorm의 역할

`FFN = Linear(D,d_ff) → ReLU → Linear(d_ff,D)`.

각 token에 동일한 FFN을 적용한다. FFN은 feature를 변환하며 token 위치들을 직접 섞지 않는다. Token 간 정보 교환은 attention에서 이루어진다.

Residual은 **각 sublayer의 입력과 출력**을 더한다. Encoder/Decoder block 전체를 한 번 감싸는 덧셈이 아니다. LayerNorm은 각 token의 마지막 D축을 정규화하고 학습 가능한 scale/shift를 적용한다.

현재 구조는 `LN(x + Sublayer(x))`인 **Post-LN**이다. 원 논문 계열의 이 흐름에서는 stack 끝에 Pre-LN 모델처럼 별도 final norm을 반드시 추가해야 하는 것은 아니다.

Multi-Head는 같은 layer 안의 서로 다른 projection, Multi-Layer는 이미 갱신된 representation에 대한 다음 단계의 변환이다. 특정 head나 layer가 특정 문법 기능을 반드시 맡는다고 가정하지 않는다.

## 5. Mask 최종 정리

### 5.1 네 가지 mask와 실제 사용처

현재 규약은 **True=허용, False=차단**이다.

| Mask | 생성 기준 | 저장 shape | 사용처 |
| --- | --- | --- | --- |
| Source padding | `src != pad_idx`, 두 번 unsqueeze | `[B,1,1,S]` | Encoder self, Decoder cross |
| Causal | 대각선을 포함하는 lower triangle | `[1,1,T,T]` | Decoder self의 미래 차단 |
| Target padding | `tgt != pad_idx`, 두 번 unsqueeze | `[B,1,1,T]` | Decoder self의 PAD key 차단 |
| Decoder combined | `causal & target_padding` | `[B,1,T,T]` | Decoder self에 실제 전달 |

Cross-attention에서는 source padding mask를 재사용한다. Target/source의 위치 번호를 비교해 삼각형으로 막지 않는다. Encoder가 source 전체를 보는 것은 조건부 생성에서 허용되는 정보다.

### 5.2 Broadcasting

Attention score의 일반 shape은 `[B,h,T_q,T_k]`다. Mask는 이 크기로 **broadcast 가능하면** 되고, 물리적으로 똑같은 크기의 tensor일 필요는 없다.

| Mask shape | Broadcast 의미 |
| --- | --- |
| `[B,1,1,S]` | 각 batch의 source key 차단 규칙을 모든 head와 query에 동일 적용 |
| `[1,1,T,T]` | 같은 causal 규칙을 모든 batch와 head에 동일 적용 |
| `[B,1,T,T]` | Batch·query·key마다 정해진 허용 규칙을 모든 head에 동일 적용 |

예를 들어 Decoder combined mask는 `[1,1,T,T] & [B,1,1,T] → [B,1,T,T]`이고, `[B,h,T,T]` score에 적용된다. 차원 1은 “특정 head/query 하나만 처리”가 아니라 **그 축 전체에서 같은 값을 재사용**한다는 뜻이다.

### 5.3 Causal mask와 자기 자신

Decoder input이 `[BOS,a,b,c]`라면 query 위치 2의 input은 b다. 이 위치는 `[BOS,a,b]`를 보고 c를 예측한다. 따라서 대각선을 포함해 현재 input token까지 보는 것이 맞다. 미래 input인 c를 참고하면 그 위치의 label을 미리 보게 된다.

Shift와 causal mask는 함께 작동한다. Shift만 하고 mask를 빼면 미래 정답을 볼 수 있고, input/label을 잘못 정렬한 채 mask만 넣어도 올바른 next-token 학습이 되지 않는다.

### 5.4 PAD key 차단과 PAD label 무시는 다르다

| 장치 | 막는 것 | 막지 않는 것 |
| --- | --- | --- |
| Padding attention mask | PAD 위치를 K/V로 참고하는 연결 | PAD query의 output 계산 자체 |
| `ignore_index=pad_idx` | Label이 PAD인 위치의 loss | PAD vocabulary class의 존재·생성 가능성 |

PAD query row의 representation이나 prediction은 0일 필요가 없다. 다음 attention에서도 PAD key가 차단되고, FFN/LayerNorm은 token별 연산이며, PAD label의 loss를 제외하므로 현재 경로에서 유효 token의 학습을 방해하지 않는다.

`nn.Embedding`에 `padding_idx`를 지정하지 않았다고 attention padding이 잘못된 것은 아니다. 그 옵션으로 attention mask를 대신할 수도 없다.

### 5.5 지원 범위와 주의점

- 현재 데이터는 **유효 source가 있고, BOS로 시작하는 target을 오른쪽 padding**한다.
- 한 query의 key가 모두 차단되면 score가 전부 −∞가 되어 현재 softmax에서 NaN이 생길 수 있다. All-PAD source나 왼쪽 PAD target의 앞 query 등이 해당한다.
- PAD query도 없애겠다며 row 전체를 무조건 −∞로 만드는 것은 안전한 해결이 아니다.
- Bool mask 규약은 API마다 다르다. 현재 코드와 PyTorch SDPA는 True=허용이지만, `nn.MultiheadAttention`의 bool mask는 True=차단이다. 내장 API로 옮길 때 확인한다.

## 6. Training: Shift, Teacher Forcing, Logits, Loss

### 6.1 Target 표기를 먼저 고정한다

이 문서에서 **y는 EOS 없는 정답 token sequence**다.

`tgt_input = [BOS] + y`, `tgt_label = y + [EOS]`.

Copy Task에서는 `y=src`이므로 구현이 `[BOS]+src`, `src+[EOS]`다. 일반 번역에서는 y가 source와 다르고 길이도 다를 수 있다.

| 위치 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| `tgt_input` | BOS | a | b | c |
| `tgt_label` | a | b | c | EOS |
| 허용되는 target prefix | BOS | BOS,a | BOS,a,b | BOS,a,b,c |

EOS를 포함한 정답 전체를 `labels`라고 부른다면 같은 관계를 `tgt_input=[BOS]+labels[:-1]`, `tgt_label=labels`로 쓸 수 있다. **`BOS+target[:-1]`와 `target+EOS`를 같은 target 정의로 섞지 않는다.**

### 6.2 Teacher forcing과 병렬 학습

Teacher forcing은 각 위치에 **정답 prefix**를 주는 학습 방식이다. 정답 기반 target 전체가 tensor에 들어 있어도 causal mask가 각 위치의 미래 접근을 막는다.

그래서 하나의 forward에서 T개 위치의 next-token prediction을 병렬로 계산할 수 있다. 학습 중 이전 위치의 argmax 결과를 다음 입력으로 직접 넣어야 하는 것은 아니다.

모델이 학습하는 조건부 예측은 `p(y_t | source, y_<t)`다. 생성 때는 같은 형태의 조건에 정답 대신 자신의 이전 예측이 들어간다. 미래 정보 제한은 같지만, **주어진 prefix의 값은 달라질 수 있다.** 따라서 teacher-forced prediction 성공이 자유 생성의 성공을 자동 보장하지 않는다.

### 6.3 Vocabulary logits와 loss

Decoder hidden state `[B,T,D]`는 source와 target prefix를 반영한 contextual representation이다. 다음 token 그 자체의 embedding이나 token ID가 아니다.

`output_linear`가 이를 `[B,T,V_tgt]`로 바꾸면 각 위치마다 모든 vocabulary token에 대한 raw score가 나온다. 예를 들어 `vocab_logits[b,t,5]`는 해당 위치에서 **다음 token이 ID 5일 score**다.

학습 경로:

`vocab_logits [B,T,V_tgt]`
→ `reshape(-1,V_tgt)` → `[B*T,V_tgt]`
→ `CrossEntropyLoss` with `tgt_label.reshape(-1) [B*T]`
→ scalar loss.

Logits와 label은 같은 batch/position 순서로 펴진다. CE에는 softmax를 먼저 적용하지 않고 raw logits를 전달한다. 기본 mean reduction에서는 PAD가 아닌 label 위치들의 loss를 평균하므로, 각 문장을 동일 비중으로 평균하는 것과는 다를 수 있다. 즉, 모든 문장을 1/B씩 똑같은 가중치로 반영하는 게 아니라 유효한 토큰이 많이 들어있는 긴 문장이 자연스럽게 이번 스텝의 파라미터 업데이트에 더 큰 영향력을 행사하게 된다.

학습 loop의 역할:

| 코드 | 의미 |
| --- | --- |
| `optimizer.zero_grad()` | 이전 gradient 초기화 |
| `model(train_src, train_tgt_input)` | 조건부 next-token logits 계산 |
| `loss_function(logits, label)` | 정답과 비교 |
| `loss.backward()` | Parameter별 gradient 계산 |
| `optimizer.step()` | Gradient에 따라 parameter 갱신 |

`argmax`는 이 loss 경로에 들어가지 않는다. 예측을 읽거나 generation할 때 사용한다.

### 6.4 Variable-length batch

`generate_variable_copy_batch()`는 각 sample마다 BOS/EOS를 먼저 붙이고, 세 목록을 각각 `pad_sequence(..., batch_first=True)`로 오른쪽 padding한다. 따라서 EOS가 실제 정답 끝 바로 뒤에 놓인다.

길이가 `[2,5,3,4]`이면 `B=4`, `S_max=5`, `T_max=6`이다.

| Tensor | Shape | 첫 sample 예시 |
| --- | --- | --- |
| src | `[4,5]` | `[a,b,PAD,PAD,PAD]` |
| tgt_input | `[4,6]` | `[BOS,a,b,PAD,PAD,PAD]` |
| tgt_label | `[4,6]` | `[a,b,EOS,PAD,PAD,PAD]` |

이 sample은 input의 b 위치에서 EOS를 예측한다. 그 뒤 PAD label의 예측값은 loss에서 제외한다. Encoder source에 EOS를 붙이지 않은 것은 현재 데이터 표현의 선택이며 구조상 필수 누락이 아니다.

## 7. Inference: BOS부터 Greedy Generation까지

### 7.1 현재 prefix로 다음 token 하나 선택

`greedy_decode()`는 처음 `generated [B,1]`을 BOS로 채운다. 이후 다음 과정을 반복한다.

| 단계 | Shape / 역할 |
| --- | --- |
| `model(src, generated)` | `[B,current_length,V_tgt]` |
| `vocab_logits[:, -1, :]` | `[B,V_tgt]`: 현재 prefix 다음 token용 logits |
| `argmax(dim=-1)` | `[B]`: 각 sample의 greedy next-token ID |
| `unsqueeze(-1)` | `[B,1]` |
| `cat([generated,next_token], dim=1)` | `[B,current_length+1]` |

이전 위치의 logits도 계산되지만 이미 입력한 prefix의 중간 token들을 예측하는 값이다. **현재 새로 붙일 token에는 마지막 위치의 logits만 필요하다.** Softmax 없이 argmax해도 선택되는 index는 같다.

Greedy는 매 step 현재 score가 가장 높은 token 하나를 선택한다. 전체 sequence의 확률이 가장 높은 결과를 반드시 찾아주는 방법은 아니다.

### 7.2 EOS와 batch별 종료

`finished [B]`는 sample별 EOS 생성 여부다. 코드의 처리 순서는 다음과 같다.

1. 현재 logits에서 next token을 선택한다.
2. **이미 종료된** row만 `torch.where`로 PAD로 바꾼다.
3. 이번에 EOS를 새로 선택한 row도 finished에 기록한다.
4. Next token을 generated에 추가한다.
5. `finished.all()`이면 전체 loop를 종료한다.

이번 step의 새 EOS는 그대로 기록되고, 그다음 step부터 PAD를 붙인다. 먼저 끝난 sample을 batch에서 제거하지 않아도 아직 진행 중인 sample과 함께 처리할 수 있다.

길이 `[2,5,3,4]`를 정확하게 복사하면 다음과 같은 형태가 된다.

| Source 실제 내용 | 생성 결과 |
| --- | --- |
| a,b | BOS,a,b,EOS,PAD,PAD,PAD |
| c,d,e,f,g | BOS,c,d,e,f,g,EOS |
| h,i,j | BOS,h,i,j,EOS,PAD,PAD |
| k,l,m,n | BOS,k,l,m,n,EOS,PAD |

학습 때 EOS 뒤의 PAD label은 loss에서 제외했다. **생성 결과의 EOS 뒤 PAD는 모델이 정답으로 학습해서 출력한 것이 아니라 종료 처리 코드가 강제한 값**이다.

### 7.3 Length, mode, 계산 비용

- `max_new_tokens`는 BOS를 제외하고 새로 생성할 최대 개수다. EOS도 생성 token 수에 포함된다.
- Copy Task에서는 최대 source 길이 + 1이면 실제 token들과 EOS를 생성할 공간을 준다. 일반 task의 정답 길이는 source 길이로 정해지지 않는다.
- 실제 G step을 수행했다면 반환 shape은 `[B,1+G]`이고 BOS가 포함된다. `G ≤ max_new_tokens`다.
- 길이 제한에 도달했는데 EOS를 못 만들면 그대로 끝난다. 제한값이 EOS 생성을 보장하지 않는다.
- `model.eval()`은 평가 모드 설정, `torch.no_grad()`는 gradient graph 생성 방지다. 두 기능은 다르며 `eval()` 자체가 gradient를 끄지는 않는다.
- 현재는 source/model/새 tensor가 같은 device라는 전제로 동작한다.

현재 매 step `model(src, generated)` 전체를 호출하므로 Encoder와 Decoder prefix를 반복 계산한다. 이는 올바르지만 느린 구현이다. 나중에는 **Encoder output 한 번 계산 → layer별 KV cache 사용**으로 연결할 수 있다. Cache를 도입하면 현재처럼 PE를 처음부터 prefix 전체에 더하는 방식도 새 token의 실제 position에 맞게 조정해야 한다.

## 8. 구현 과정에서 실제로 확인한 것

| 검증 단계 | 노트북에서 확인한 내용 |
| --- | --- |
| Attention 단독 | T_q≠T_k, d_k≠d_v 상황의 shape와 weights 합 |
| MHA / Encoder / Decoder | Head 분리·결합 및 layer별 output/weights shape |
| Mask | 미래 key와 PAD key의 attention weight가 0 |
| Output / Loss | `[2,5,30] → [10,30]`, label `[2,5] → [10]`, scalar CE |
| Fixed-length Copy Task | Loss 감소, teacher-forced prediction과 정답 일치 사례 |
| Greedy inference | B=1과 B>1에서 BOS부터 복사 및 EOS 생성 |
| Variable-length training | 길이가 다른 4개 sample에서 각 실제 끝에 EOS, 이후 PAD |

중요했던 실패와 해석도 함께 기억한다.

**길이 5로만 학습 → 가변 길이 추론:** 짧은 source에서도 실제 끝에 EOS를 내지 않고 일반 token을 더 생성하는 결과가 있었다. Mask가 가변 길이를 처리한다는 것과, 학습된 모델이 길이에 따른 종료 규칙을 잘 사용한다는 것은 별개다.

**가변 길이로 학습 → 가변 길이 추론:** 마지막 실험에서는 각 row의 실제 token부터 EOS까지 올바르게 생성했다. Teacher-forced prediction의 PAD label 구간에 EOS 등이 반복되는 것은 평가 오류가 아니다.

저장된 성공 사례는 구조와 학습·생성의 기능 검증이다. 많은 새 sample에서의 정확도, 학습 길이 밖의 일반화, 모든 입력 경계 조건까지 증명한 것은 아니다. 마지막 실험의 마지막 출력 loss는 약 0.324였으며 전체 분포에서 완벽히 수렴했다고 해석하지 않는다.

추가 검증이 필요할 때는 규모를 키우기보다 두 성질을 먼저 확인할 수 있다: 미래 target을 바꿔도 이전 logits가 유지되는가, 오른쪽 source PAD를 더 붙여도 유효 target logits가 유지되는가. 이는 후속 검증 후보이며 이번 문서 작성 중 새로 실행한 테스트는 아니다.

## 9. 구현 검토: 일치, 단순화, 생략, 다음 연결

### 9.1 표준 구조와 일치하는 핵심

학습 가능한 scaled embedding, sinusoidal PE, scaled dot-product attention, multi-head projection/concat, ReLU FFN, Post-LN, 독립 layer stacking, masked Decoder self-attention, Encoder–Decoder cross-attention, shifted next-token 학습의 연결이 원 논문 Encoder–Decoder 구조에 충실하다.[1]

일반 함수·tensor 연산으로 나누어 작성했거나 batch-first shape을 사용했다는 것은 구조적 오류가 아니다. 정상 입력 범위에서 검토한 핵심 연산·shape 연결의 오류는 발견되지 않았다.

### 9.2 학습 목적의 단순화와 다른 설계 선택

| 항목 | 현재 구현 | 분류 / 의미 |
| --- | --- | --- |
| 규모 | N=2, D=32, h=4, d_ff=64 | 작은 실험을 위한 축소 |
| 데이터 | 작은 vocabulary의 Copy Task | 정보 흐름과 학습·생성 검증 목적 |
| Dropout | 생략 | 정규화 단순화 |
| Weight tying | Source/target embedding과 output weight가 독립 | 실제 parameter 구조가 다른 설계 |
| Attention projection bias | `nn.Linear` 기본 bias 사용 | 논문의 행렬곱 표기와 구분할 세부 선택 |
| Label smoothing | 없음 | Loss 설정 단순화 |
| Optimizer / schedule | 기본 Adam, 고정 LR | 논문의 별도 Adam 설정·warmup 생략 |
| Decode strategy | Greedy | 논문의 beam search 대신 단순한 선택 |

원 논문 Base는 6층, D=512, h=8, d_ff=2048이며, embedding/output weight 공유, dropout, label smoothing, warmup 등을 사용했다.[1] 이 차이 때문에 논문 실험을 그대로 재현한 것은 아니지만, 핵심 원리 구현의 정확성과는 별개다.

### 9.3 현재 학습 범위에서 생략한 실제 구현 요소

이 항목들은 이번 구현에서 생략된 것으로 확인된 것들이다. 모든 항목을 사전에 명시적으로 비교·선택했다는 뜻은 아니다.

| 생략한 요소 | 추가할 때 얻는 것 / 현재 판단 |
| --- | --- |
| Encoder 결과 재사용, KV cache | 반복 계산 감소. 현재 정확성의 필수 조건은 아님 |
| SDPA / FlashAttention | Attention kernel의 속도·메모리 최적화. 직접 score를 보는 현재 방식은 학습에 유리 |
| 모든 attention weights 반환의 선택화 | 큰 입력에서 불필요한 저장 감소 |
| Mixed precision | 학습·추론 효율 향상. 현재 FP32 중심 코드의 dtype 점검 필요 |
| BOS/PAD 생성 금지 등 제약 | 특수 token의 부적절한 생성 제한. 현재 argmax는 vocabulary 전체가 후보 |
| 독립 평가 loop와 checkpoint 관리 | 재현·지속 학습·여러 sample 평가. 현재 기능 확인 수준을 넘어설 때 추가 |

### 9.4 발전된 Transformer로 연결할 개념

| 현재 이해를 출발점으로 | 다음에 연결할 개념 | 무엇이 달라지는가 |
| --- | --- | --- |
| `LN(x + f(x))` | Pre-LN, RMSNorm | Norm 위치 또는 정규화 연산 |
| ReLU FFN | Gated FFN / SwiGLU | Token별 feature 변환 |
| Embedding에 PE를 더함 | RoPE | Q/K에 위치 정보를 반영하는 방식 |
| Prefix 전체 재계산 | KV cache | 과거 layer별 K/V 재사용 |
| 수동 score·softmax·V 곱 | SDPA / FlashAttention | Attention의 효율적인 계산 |
| Encoder–Decoder | Decoder-only | 전체 모델 구성과 조건 입력 방식 |

Pre-Norm/RMSNorm, SwiGLU, RoPE 등은 LLaMA 계열의 대표적인 변화다.[4] 원 논문 Transformer에 빠진 부품을 복구하는 작업이 아니라 **다른 설계를 공부하는 단계**다. 모든 현대 모델이 같은 선택을 한다고 일반화하지 않는다.

## 10. 원본 코드·노트에서 정리할 사항과 지원 전제

이 문서는 아래 문제를 설명에 반영했지만 원본 `.py`나 notebook을 수정하지 않았다.

| 항목 | 검토 결과 |
| --- | --- |
| `.py` 함수 배치 | `generate_variable_copy_batch()` 정의가 training loop보다 뒤에 있어 처음부터 실행하면 호출 시점에 이름이 없음. 마지막 notebook 셀에는 앞에 배치되어 있음 |
| `.py` 테스트 설정 | Fixed-length prediction test에서 `seq_len`을 사용하기 전에 정의하지 않음 |
| 옛 Transformer Test | 첫 반환값을 `decoder_output`이라 부르지만 현재는 logits. 해당 설정의 shape은 `[2,5,30]` |
| Target mask 주석 | 실제 코드는 tgt를 사용하지만 주석에 src라고 남은 부분이 있음 |
| Target shift 표기 | 이 문서는 EOS 없는 y 기준 `BOS+y`, `y+EOS`로 통일 |
| Q=K=V 표기 | Projection 전 입력 출처가 같다는 뜻으로 한정 |
| PositionalEncoding | 현재 구현은 짝수 D, 실제 forward 입력 길이 ≤ max_len 전제 |
| Padding | All-masked row 미처리. 유효 source 및 BOS로 시작하는 오른쪽 padding target 전제 |
| PAD ID | Transformer에 하나의 pad_idx만 있어 source/target이 같은 PAD ID를 사용한다는 전제 |
| Data generator | `randint(3,vocab_size)`는 0/1/2를 특수 token으로 예약한 현재 규약 전제 |
| 학습 모드 | 새 모델은 기본 train mode. Decode 후 재학습할 때는 `model.train()` 호출 필요 |

검토는 원본 코드 분석과 notebook에 저장된 결과를 근거로 했다. 검토 환경에 PyTorch가 없어 새로 재학습하거나 경계 조건을 실행 검증하지 않았다. `.py`와 마지막 notebook 셀의 클래스·함수 17개는 주석을 제외한 구문 구조가 같았지만, 최상위 실행 순서까지 같았던 것은 아니다.

## 참고 자료

외부 자료는 표준 구조 및 API 의미를 확인하는 용도다. 이 문서의 구현·실험 설명은 첨부한 코드와 notebook을 기준으로 한다.

1. [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 원 논문 구조·학습·평가 설정.
2. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — Raw logits, ignore_index, reduction.
3. [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — Bool mask 규약과 attention API.
4. [Touvron et al., LLaMA](https://arxiv.org/abs/2302.13971) — 대표적인 현대 구조 변경.
5. [Hugging Face: How caching works](https://huggingface.co/docs/transformers/cache_explanation) — Autoregressive KV cache.

## Final Shape & Forward Flow Cheat Sheet

**표기:** B=batch, S=source tensor 길이, T=현재 decoder input 길이, D=d_model, h=head 수, d=D/h, V=target vocab 크기. Padding된 S/T와 sample별 실제 길이를 구분한다.

### A. 전체 forward

| 경로 | 연결 |
| --- | --- |
| Source | IDs `[B,S]` → Embedding×√D → PE → `[B,S,D]` → Encoder×N → **E `[B,S,D]`** |
| Target | IDs `[B,T]` → Embedding×√D → PE → `[B,T,D]` → Decoder×N, E 참고 → **H `[B,T,D]`** |
| Output | H → Linear(D,V) → **logits `[B,T,V]`** |
| Training | Logits `[B*T,V]` + labels `[B*T]` → CE, PAD label 제외 → scalar loss → backward → step |
| Generation | BOS `[B,1]` → forward → last logits `[B,V]` → argmax `[B]` → concat → EOS 또는 길이 제한까지 반복 |

### B. Block과 attention

| 항목 | 핵심 |
| --- | --- |
| EncoderBlock | Self → Add & Norm → FFN → Add & Norm |
| DecoderBlock | Masked Self → Add & Norm → Cross → Add & Norm → FFN → Add & Norm |
| Post-LN | 각 sublayer마다 `LN(x + sublayer_output)` |
| FFN | `[B,L,D] → [B,L,d_ff] → [B,L,D]` |
| Head split | `[B,L,D] → [B,L,h,d] → [B,h,L,d]` |
| Attention | Q `[B,h,T_q,d_k]` × Kᵀ `[B,h,d_k,T_k]` → score `[B,h,T_q,T_k]` → mask → softmax(key축) → × V `[B,h,T_k,d_v]` → `[B,h,T_q,d_v]` |
| Head concat | `[B,h,T_q,d] → [B,T_q,h,d] → [B,T_q,D] → W_O` |

### C. 세 attention과 mask

| Attention | Q 입력 / K,V 입력 | Score | 사용 mask |
| --- | --- | --- | --- |
| Encoder Self | Encoder x / 같은 x | `[B,h,S,S]` | Source PAD `[B,1,1,S]` |
| Decoder Self | Decoder x / 같은 x | `[B,h,T,T]` | Causal `[1,1,T,T]` & Target PAD `[B,1,1,T]` = `[B,1,T,T]` |
| Cross | Decoder self 후 Add & Norm / 최종 E | `[B,h,T,S]` | Source PAD `[B,1,1,S]` |

**True=허용.** Mask의 1축은 broadcast한다. PAD key는 차단하지만 PAD query output을 0으로 만들지는 않는다. Q/K/V는 원본 입력에 서로 다른 projection을 적용한 결과다.

### D. 학습과 생성의 정렬

| 구분 | 기억할 것 |
| --- | --- |
| 정답 y=`[a,b,c]` | Input=`[BOS,a,b,c]`, label=`[a,b,c,EOS]` |
| Teacher forcing | 정답 prefix 사용. Causal mask로 미래 차단. T개 위치를 병렬 계산 |
| Logits | 다음 token의 raw score. CE 전 softmax 불필요 |
| Greedy | 생성한 prefix 사용. `logits[:,-1,:].argmax(-1)` |
| EOS batch 처리 | 이미 finished인 row만 PAD 강제 → 새 EOS 기록 → concat → 모두 finished면 종료 |
| 반환 shape | BOS 포함 `[B,1+G]`, G는 실제 생성 step 수 |
| 핵심 구분 | **Hidden state는 representation, logits는 vocabulary score, argmax 결과는 token ID** |
