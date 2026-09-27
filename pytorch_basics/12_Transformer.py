import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model
            )

    def forward(self, x):
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        return x

# 2. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # [max_len, d_model] 크기의 0으로 채워진 빈 테이블 생성. 
        # 나중에 position을 알려주는 값으로 채우고 embedding된 token이랑 합쳐질 예정.
        pe = torch.zeros(max_len, d_model)
        # 각 row가 몇 번째 position인지 나타냄. 
        # sin/cos 계산할거라서 float로 만들기. 0부터 max_len-1까지 만들고 unsqueeze로 제일 안쪽에 차원 추가
        # shape은 [max_len, 1]
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(-1)
        # 각 embedding dimension pair마다 서로 다른 sin/cos 주기를 만들어주는 scale 값들을 생성. shape은 [d_model / 2]  (d_model이 짝수라고 가정)
        # 각 dimension pair라는게 sin/cos을 의미. dim 0과 1 → 같은 frequency 사용, 즉 dim 0 → sin(frequency 0), dim 1 → cos(frequency 0) 이어서 dim 2,3도 같은 frquency 사용..
        # div_term은 사실 각 포지션이 다른 속도의 sin/cos값을 갖게 하기위한 scale임.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        # position*div_term으로 나온 [max_len, d_model/2] 모양의 텐서가 sin/cos에 들어가서 sin은 짝수, cos은 홀수 column에 들어가게 함. 여기서 column은 d_model. row는 포지션.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        # 미리 만든 self.pe는 shape이 [1, max_len, d_model]이라서 그걸 바로 
        # x(embedding이 끝난 입력값, shape은 [B, T, d_model])에 더하면 안되서 x.size(1), 즉 토큰개수만큼 잘라줌.
        x = x + self.pe[:, :x.size(1), :]
        return x






# 3. Attention
def attention(Q, K, V, mask=None): 
    # Expected shapes: Q/K/V : [B, h, T, d_head], mask: attention score [B,h,T_q,T_k]에 broadcast 가능한 bool tensor
    # ex) Causal mask: [1, 1, T_q, T_k] , Padding mask: [B, 1, 1, T_k]
    d_k = Q.shape[-1]
    K_trans = torch.transpose(K, -2, -1)
    att_scores = Q @ K_trans / math.sqrt(d_k)
    # 만약 mask가 있다면, pytorch의 masked_fill은 mask값이 true인 위치를 바꾸기 때문에,
    # ~mask으로 false, true를 뒤집고 기존에 false였던 부분을 -inf로 채우기(현재 코드에선 true가 허용, false가 차단 인 방식으로 mask를 만들었기 때문)
    # 중요한 부분은, 이 mask가 causal인지 padding인지 알 필요없이 false인 위치를 막음. 그래서 구현이 간단해짐.
    if mask is not None: att_scores = att_scores.masked_fill(~mask, float("-inf")) 
    att_weights = F.softmax(att_scores, dim = -1)
    output = att_weights @ V
    return output, att_weights # output: [B, h, T_q, d_v], att_weights: [B, h, T_q, T_k]

# 4. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None): # Expected query, key, value shape: [B, T, d_model]
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Q/K/V를 head split 하고 head차원을 앞으로 옮기기
        # ex) Q: [B, T_q, d_model] -> [B, T_q, h, d_head] -> [B, h, T_q, d_head]
        Q = Q.unflatten(-1, (self.num_heads, self.d_head))
        Q = torch.transpose(Q, 1, 2)

        K = K.unflatten(-1, (self.num_heads, self.d_head))
        K = torch.transpose(K, 1, 2)

        V = V.unflatten(-1, (self.num_heads, self.d_head))
        V = torch.transpose(V, 1, 2)

        output, att_weights = attention(Q, K, V, mask)
        # attention에서 나온 output을 다시 원래 shape으로 돌려놓기 head_output [B, h, T_q, d_head] -> [B, T_q, d_model]
        output = torch.transpose(output, 1, 2)
        output = output.flatten(-2) # 숫자하나만 적으면 맨끝에 -1이 default값으로 들어가있어서 (-2,-1) 이렇게 처리됨. 즉 그 두개를 합침.

        output = self.W_O(output)

        return output, att_weights




# 5. Feed-Forwad Network
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x): # Expected x shape: [B, T, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        output = self.linear2(x)
        return output

# 6. Residual Layer Norm
class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        x = x + sublayer_output # Residual connection
        output = self.layernorm(x) # Layernormalization
        return output





# 7. Encoder Block
## 하나의 encoder block이 어떻게 작동해야하는지 
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = ResidualLayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.add_norm2 = ResidualLayerNorm(d_model)
    def forward(self, x, src_mask=None):
        attn_output, attn_weights = self.mha(x,x,x, mask=src_mask) # multihead attention에서 새로운 representation과 attetion weight matrix 받기
        residual_ln1 = self.add_norm1(x, attn_output) # 첫번쨰 add+norm 통과
        ffn_output = self.ffn(residual_ln1) # FFN층 통과
        residual_ln2 = self.add_norm2(residual_ln1, ffn_output) # 두번째 add+norm 통과
        return residual_ln2, attn_weights

# 8. Encoder
## 실제로 encoder block들을 조립해서 작동하게 하기
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderBlock(d_model, num_heads, d_ff) for i in range(num_layers)
            )
    
    def forward(self, x, src_mask=None): # x(encoder input) shape: [B, S, d_model]
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask=src_mask)
            all_attn_weights.append(attn_weights)
        return x, all_attn_weights
        




# 9. Decoder Block 
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = ResidualLayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = ResidualLayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff)
        self.add_norm3 = ResidualLayerNorm(d_model)

    def forward(self, x, encoder_output, self_mask=None ,cross_mask=None): # x(decoder input) shape: [B, T, d_model] , encoder_output shape: [B, S, d_model]
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, mask=self_mask) # q/k/v 모두 decoder input을 가져와서 projection
        residual_ln1 = self.add_norm1(x, self_attn_output)

        cross_attn_output, cross_attn_weights = self.cross_attn(residual_ln1, encoder_output, encoder_output, mask=cross_mask) # cross attention은 k/v를 encoder outputd에서 가져와서 projection
        residual_ln2 = self.add_norm2(residual_ln1, cross_attn_output)
        
        ffn_output = self.ffn(residual_ln2)
        residual_ln3 = self.add_norm3(residual_ln2, ffn_output)
        # self_attn_weights: [B,h, T, T] , cross_atnn_weights: [B, H, T, S]
        return residual_ln3, self_attn_weights, cross_attn_weights

# 10. Decoder
## 실제로 decoder block들을 조립해서 작동하게 하기
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderBlock(d_model, num_heads, d_ff) for i in range(num_layers)
            )
    
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None): # x(decoder input) shape: [B, T, d_model]
        all_self_attn_weights =[]
        all_cross_attn_weights =[]
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, encoder_output, self_mask=self_mask, cross_mask=cross_mask)
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)
        return x, all_self_attn_weights, all_cross_attn_weights





        
# 11. Masks
def create_causal_mask(seq_len, device): # Decoder self-attention에서 미래 target token을 attend하지 못하게 함.
    # seq_len = target sequence length T.

    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # Decoder self-attention에서는 T_q = T_k = T이므로 [T_q, T_k] = [T, T] lower-triangular mask를 만듦.

    mask = mask.unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0).bool() 
    # mask: [1, 1, T, T]
    # 앞 쪽에 두 차원을 추가해서 나중에 [B, h, T_q, T_k] shape을 가진 attention score와 계산될 예정.
    return mask

def create_padding_mask(seq, pad_idx): # 현재 K/V를 제공하는 sequence의 PAD 위치를 아무 Query도 참고하지 못하게 함.

    # seq 텐서와 PAD token의 정수 id를 받음. seq: [B, T_k]
    mask = seq != pad_idx 
    mask = mask.unsqueeze(1).unsqueeze(2) # mask: [B, 1, 1, T_k] -> # head dimension과 query dimension에 broadcasting되도록 함.
    # 왜냐면 모든 head의 모든 query에서 PAD key를 참고하지 못하게 해야해서.

    return mask # 따라서 각 batch에서 모든 head와 모든 query가 PAD key를 보지 못함.

def create_decoder_mask(target_seq, pad_idx):# target_seq: [B, T]
    # Decoder self-attention용 causal mask + target padding mask 생성.
    seq_len = target_seq.shape[1] 
    # target_seq: [B, T] 에서 T 가져오기
    current_device = target_seq.device
    causal_mask = create_causal_mask(seq_len, current_device) 
    # causal mask: [1, 1, T_q, T_k] = [1, 1, T, T]

    padding_mask = create_padding_mask(target_seq, pad_idx) 
    # paddin mask: [B, 1, 1, T_k]

    combined_mask = causal_mask & padding_mask # combined mask: [B, 1, T, T_k]
    return combined_mask 


# 12. Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, pad_idx, max_len=5000):
        super().__init__()
        self.src_token_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_token_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        self.src_positional_encoding = PositionalEncoding(d_model, max_len)
        self.tgt_positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)

        self.pad_idx = pad_idx

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt): # source sequence: [B, S] , target sequence: [B, T]
        src_x = self.src_token_embedding(src)
        src_x = self.src_positional_encoding(src_x)

        src_mask = create_padding_mask(seq=src, pad_idx=self.pad_idx) # src_mask는 padding mask만
        # padding mask를 만들때는 embedding하기전의 src sequence를 가져와야지 token id를 보고 pad인지 확인할 수 있음.
        encoder_output, encoder_attn_weights = self.encoder(src_x, src_mask)
        # encoder_output: [B,S,d_model], encoder_attn_weights: list of num_layers, each element: [B,h,S,S]
        tgt_x = self.tgt_token_embedding(tgt)
        tgt_x = self.tgt_positional_encoding(tgt_x)

        tgt_mask = create_decoder_mask(target_seq=tgt, pad_idx=self.pad_idx) # tgt_mask는 causal mask+padding mask
        # padding mask를 만들때는 embedding하기전의 src sequence를 가져와야지 token id를 보고 pad인지 확인할 수 있음.

        decoder_output, decoder_self_attn_weights, decoder_cross_attn_weights = self.decoder(tgt_x, encoder_output, tgt_mask, src_mask)
        # decoder_output: [B,T,d_model], decoder_self_attn_weights: list of num_layers, each [B,h,T,T], decoder_cross_attn_weights: list of num_layers, each [B,h,T,S]
        vocab_logits = self.output_linear(decoder_output)
        # vocab_logits: [B,T,tgt_vocab_size]

        return (
                vocab_logits,
                encoder_attn_weights,
                decoder_self_attn_weights,
                decoder_cross_attn_weights
                )

# 13. Copy Task
def generate_copy_batch(batch_size, seq_len, vocab_size, bos_idx, eos_idx, device): # 
    src = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    tgt_input = F.pad(src, (1,0), value=bos_idx) # (1,0)을 하면 앞에 하나 추가
    tgt_label = F.pad(src, (0,1), value=eos_idx) # (0,1)을 하면 뒤에 하나 추가
    return src, tgt_input, tgt_label # src: [B, S], tgt_input: [B, S+1], tgt_label: [B, S+1]


# 14. Training Loop
vocab_size = 20
pad_idx = 0
bos_idx = 1
eos_idx = 2

seq_len = 5
batch_size = 32

d_model = 32
num_heads = 4
d_ff = 64
num_layers = 2

learning_rate = 1e-3
num_steps = 1000
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Transformer(vocab_size, vocab_size, d_model, num_heads, d_ff, num_layers, pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()


for step in range(num_steps):

    train_src, train_tgt_input, train_tgt_label = generate_copy_batch(batch_size, seq_len, vocab_size, bos_idx, eos_idx, device)

    optimizer.zero_grad()

    vocab_logits,encoder_attn_weights,decoder_self_attn_weights,decoder_cross_attn_weights = model(train_src, train_tgt_input)


    logits = vocab_logits.reshape(-1, vocab_logits.size(-1))
    label = train_tgt_label.reshape(-1)

    loss = loss_function(logits, label)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Loss: ")
        print(loss)


# 15. Training-time prediction test
test_batch_size = 1
model.eval()
# 위에서 학습시킨 모델을 그대로 가져와서 batch_size만 1로 바꾸고 추론.
with torch.no_grad():
    src, tgt_input, tgt_label = generate_copy_batch(test_batch_size, seq_len, vocab_size, bos_idx, eos_idx, device)
    vocab_logits,encoder_attn_weights,decoder_self_attn_weights,decoder_cross_attn_weights = model(src, tgt_input)

    prediction = vocab_logits.argmax(dim=-1)

print(src)
print(tgt_input)
print(tgt_label)
print(prediction)
print(torch.equal(prediction, tgt_label))


#16. Autoregressive Inference
@torch.no_grad()
def greedy_decode(model, src, bos_idx, eos_idx, max_new_tokens, device): # src: [B,S]
    model.eval()
    batch_size = src.shape[0]

    # [B, 1] 크기로 bos_idx 채우기
    generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    # model에 넣어줄 target sequence로 [BOS]만 있는 텐서 넣기

    # 최대 max_new_tokens개의 새로운 token을 autoregressive하게 생성
    for tokens in range(max_new_tokens):
        vocab_logits,encoder_attn_weights,decoder_self_attn_weights,decoder_cross_attn_weights = model(src, generated)
        next_token_logits = vocab_logits[:,-1,:] # next_token_logits: [B, vocab_size]
        # vocab_logits에는 모든 timestep의 다음 토큰 예상값이 logits으로 들어있기 때문에
        # 마지막 시점의, 즉 마지막 position의 예상값을 가져오기 위해 T차원을 -1로 가져옴.

        next_token = next_token_logits.argmax(dim=-1) # next_token_logits에서 가장 큰 값의 인덱스는 모델이 예상한 다음 token id. next_token: [B]
        next_token = next_token.unsqueeze(-1) # 제일 안쪽 차원을 하나 추가. next_token: [B, 1]
        generated = torch.cat([generated, next_token], dim=1) # generated와 next_token을 붙임. generated: [B, current_length+1]

        if next_token.item() == eos_idx: # [EOS]를 생성했으면 문장이 끝났기 때문에 break. next_token이 값을 딱 하나만 가지기 때문에 .item()으로 꺼낼 수 있음
            break # 만약 batch_size가 1이 아니라서 next_token에 여러 값이 있다면 작동X
    
    return generated # generated: [B,T]


max_new_tokens = src.shape[1] + 1
result = greedy_decode(model, src, bos_idx, eos_idx, max_new_tokens, device)





# 17. Teacher-forcing vs Autoregressive Inference Test

# 새로운 테스트용 copy sample 1개 생성
test_src, test_tgt_input, test_tgt_label = generate_copy_batch(
    batch_size=1,
    seq_len=seq_len,
    vocab_size=vocab_size,
    bos_idx=bos_idx,
    eos_idx=eos_idx,
    device=device
)

# 1) Teacher-forcing prediction
model.eval()

with torch.no_grad():
    vocab_logits, _, _, _ = model(test_src, test_tgt_input)
    teacher_prediction = vocab_logits.argmax(dim=-1)

# 2) Autoregressive prediction
max_new_tokens = src.shape[1] + 1

generated = greedy_decode(
    model,
    test_src,
    bos_idx,
    eos_idx,
    max_new_tokens,
    device
)

print("src:                 ", test_src)
print("tgt_input:           ", test_tgt_input)
print("tgt_label:           ", test_tgt_label)
print("teacher prediction:  ", teacher_prediction)
print("autoregressive:      ", generated)






############################
# generate_copy_batch() Test
############################

batch_size = 2
seq_len = 5
vocab_size = 20

bos_idx = 1
eos_idx = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

src, tgt_input, tgt_label = generate_copy_batch(
    batch_size=batch_size,
    seq_len=seq_len,
    vocab_size=vocab_size,
    bos_idx=bos_idx,
    eos_idx=eos_idx,
    device=device
)

print("src:")
print(src)
print("shape:", src.shape)

print("\ntgt_input:")
print(tgt_input)
print("shape:", tgt_input.shape)

print("\ntgt_label:")
print(tgt_label)
print("shape:", tgt_label.shape)

# shape 확인
assert src.shape == (batch_size, seq_len)
assert tgt_input.shape == (batch_size, seq_len + 1)
assert tgt_label.shape == (batch_size, seq_len + 1)

# BOS / EOS 위치 확인
assert torch.all(tgt_input[:, 0] == bos_idx)
assert torch.all(tgt_label[:, -1] == eos_idx)

# 실제 copy 관계 확인
assert torch.equal(tgt_input[:, 1:], src)
assert torch.equal(tgt_label[:, :-1], src)

print("\nAll tests passed!")














###########################
# Transformer Logits Test #
###########################

B = 2
S = 7
T = 5

src_vocab_size = 20
tgt_vocab_size = 30

d_model = 8
num_heads = 2
d_ff = 32
num_layers = 3
max_len = 20
pad_idx = 0

src = torch.tensor([
    [1, 4, 7, 3, 0, 0, 0],
    [2, 5, 8, 9, 3, 4, 0]
])

tgt = torch.tensor([
    [1, 6, 4, 0, 0],
    [1, 3, 7, 8, 0]
])

dummy = torch.tensor([
    [4, 7, 9, 0, 0],
    [3, 8, 2, 5, 0]
])

my_model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, pad_idx, max_len)

vocab_logits,encoder_attn_weights,decoder_self_attn_weights,decoder_cross_attn_weights = my_model(src, tgt)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

logits = vocab_logits.reshape(-1, vocab_logits.size(-1))
flattened_dummy = dummy.reshape(-1)

loss = criterion(logits, flattened_dummy)

print("Logits Shape: ")
print(vocab_logits.shape)

print("Labels Shape:")
print(dummy.shape)

print("Flattened Logits Shape: ")
print(logits.shape)

print("Flattened Labels Shape:")
print(flattened_dummy.shape)

print("Loss: ")
print(loss)

####################
# Transformer Test #
####################

B = 2
S = 7
T = 5

src_vocab_size = 20
tgt_vocab_size = 30

d_model = 8
num_heads = 2
d_ff = 32
num_layers = 3
max_len = 20
pad_idx = 0

src = torch.tensor([
    [1, 4, 7, 3, 0, 0, 0],
    [2, 5, 8, 9, 3, 4, 0]
])

tgt = torch.tensor([
    [1, 6, 4, 0, 0],
    [1, 3, 7, 8, 0]
])

my_model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, pad_idx, max_len)

decoder_output,encoder_attn_weights,decoder_self_attn_weights,decoder_cross_attn_weights = my_model(src, tgt)

print("Decoder output shape:")
print(decoder_output.shape)

print(len(encoder_attn_weights))
for i, weights in enumerate(encoder_attn_weights):
    print(f"Layer {i}: {weights.shape}")

print(len(decoder_self_attn_weights))
for i, weights in enumerate(decoder_self_attn_weights):
    print(f"Layer {i}: {weights.shape}")

print(len(decoder_cross_attn_weights))
for i, weights in enumerate(decoder_cross_attn_weights):
    print(f"Layer {i}: {weights.shape}")



#####################
# Decoder Mask Test #
#####################

B = 2
T = 5
S = 6
d_model = 8
num_heads = 2
d_ff = 32
num_layers = 3
pad_idx = 0

# decoder input representation
x = torch.randn(B, T, d_model)

# encoder output
encoder_output = torch.randn(B, S, d_model)

# target token ids
target_seq = torch.tensor([
    [1, 5, 8, 9, 2],     # PAD 없음
    [1, 7, 3, 0, 0]      # 마지막 두 개가 PAD
])

# source token ids
src_seq = torch.tensor([
    [4, 6, 8, 2, 9, 3],  # PAD 없음
    [5, 7, 2, 1, 0, 0]   # 마지막 두 개가 PAD
])

# Decoder Self-Attention용:
# causal + target padding
self_mask = create_decoder_mask(target_seq, pad_idx)

# Decoder Cross-Attention용:
# source padding
cross_mask = create_padding_mask(src_seq, pad_idx)

decoder = Decoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)

output, all_self_attn_weights, all_cross_attn_weights = decoder(
    x,
    encoder_output,
    self_mask=self_mask,
    cross_mask=cross_mask
)

print("Decoder output shape:")
print(output.shape)

print("\nNumber of decoder layers:")
print(len(all_self_attn_weights))
print(len(all_cross_attn_weights))

for i in range(num_layers):
    print(f"\nLayer {i}")
    print("Self-attention shape:",
          all_self_attn_weights[i].shape)
    print("Cross-attention shape:",
          all_cross_attn_weights[i].shape)

print("\nLast layer - sequence 1 - head 0 - SELF attention:")
print(all_self_attn_weights[-1][1, 0])

print("\nLast layer - sequence 1 - head 0 - CROSS attention:")
print(all_cross_attn_weights[-1][1, 0])










#####################
# Encoder Mask Test #
#####################

B = 2
S = 5
d_model = 8
num_heads = 2
d_ff = 32
num_layers = 3

x = torch.randn(B, S, d_model)

src_seq = torch.tensor([
    [5, 8, 2, 9, 4],
    [7, 3, 6, 0, 0]
])

src_mask = create_padding_mask(src_seq, pad_idx=0)

encoder = Encoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)

output, all_attn_weights = encoder(
    x,
    src_mask=src_mask
)

print(output.shape)
print(len(all_attn_weights))

for i, weights in enumerate(all_attn_weights):
    print(f"Layer {i}: {weights.shape}")

print("\nLast layer, sequence 1, head 0:")
print(all_attn_weights[-1][1, 0])





# Encoder Self:
# source padding mask 
# [B, 1, 1, S]

# Decoder Self:
# target padding mask + causal mask -> combined mask
# [B, 1, T, T]

# Decoder Cross:
# source padding mask
# [B, 1, 1, S]

# Encoder Self: 모든 query가 source의 PAD key를 못 봄
# Decoder Self: 모든 query가 target의 PAD key를 못 보고, 동시에 미래 key도 못 봄
# Decoder Cross: decoder query들이 encoder/source의 PAD key를 못 봄




#####################
# Mask Test #
#####################

target_seq = torch.tensor([
    [1, 5, 8, 9, 2],  # PAD 없음
    [1, 7, 3, 0, 0]   # 마지막 두 개가 PAD
])

mask = create_decoder_mask(target_seq, pad_idx=0)

print(mask.shape)
print("Sequence 0:")
print(mask[0, 0])

print("\nSequence 1:")
print(mask[1, 0])


################
# Decoder test #
################

B = 2
T = 5          # target sequence length
S = 7          # source sequence length
d_model = 8
num_heads = 2
d_ff = 32
num_layers = 3

x = torch.randn(B, T, d_model)
encoder_output = torch.randn(B, S, d_model)

decoder = Decoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)

output, all_self_attn_weights, all_cross_attn_weights = decoder(
    x,
    encoder_output
)

print("input shape           :", x.shape)
print("encoder output shape  :", encoder_output.shape)
print("decoder output shape  :", output.shape)

print("num decoder layers    :", len(all_self_attn_weights))

for i in range(num_layers):
    print(f"\nLayer {i+1}")
    print("self attention shape  :", all_self_attn_weights[i].shape)
    print("cross attention shape :", all_cross_attn_weights[i].shape)

    print(
        "self attn sum         :",
        all_self_attn_weights[i].sum(dim=-1)
    )

    print(
        "cross attn sum        :",
        all_cross_attn_weights[i].sum(dim=-1)
    )


#####################
# DecoderBlock test #
#####################

B = 2
T = 5          # target length
S = 7          # source length
d_model = 8
num_heads = 2
d_ff = 32

decoder_block = DecoderBlock(d_model, num_heads, d_ff)

x = torch.randn(B, T, d_model)
encoder_output = torch.randn(B, S, d_model)

output, self_attn_weights, cross_attn_weights = decoder_block(
    x,
    encoder_output
)

print("output:", output.shape)
print("self attention:", self_attn_weights.shape)
print("cross attention:", cross_attn_weights.shape)

print("self attn sum:", self_attn_weights.sum(dim=-1))
print("cross attn sum:", cross_attn_weights.sum(dim=-1))






################
# Encoder test #
################

B = 2
S = 5          # source sequence length
d_model = 8
num_heads = 2
d_ff = 32
num_layers = 3
0
# [B, T, d_model]
x = torch.randn(B, S, d_model)

encoder = Encoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)

output, all_attn_weights = encoder(x)

print("input shape :", x.shape)
print("output shape:", output.shape)

print("number of attention matrices:", len(all_attn_weights))

for i, attn_weights in enumerate(all_attn_weights):
    print(f"layer {i+1} attention shape:", attn_weights.shape)
    print(f"layer {i+1} attention sum:")
    print(attn_weights.sum(dim=-1))


#####################
# Encoder Block test #
#####################

B = 2
S = 5          # source sequence length
d_model = 8
num_heads = 2
d_ff = 32

x = torch.randn(B, S, d_model)

encoder_block = EncoderBlock(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff
)

output, attn_weights = encoder_block(x)

print("input shape       :", x.shape)
print("output shape      :", output.shape)
print("attention shape   :", attn_weights.shape)

# 각 query가 key 방향으로 만든 attention weight의 합
print("attention sum     :", attn_weights.sum(dim=-1))





##############################
# MultiHeadAttention Test
##############################

if __name__ == "__main__":
    B = 2
    T_q = 5
    T_k = 7
    d_model = 32
    num_heads = 4

    query = torch.randn(B, T_q, d_model)
    key = torch.randn(B, T_k, d_model)
    value = torch.randn(B, T_k, d_model)

    mha = MultiHeadAttention(d_model, num_heads)

    output, att_weights = mha(query, key, value)

    print(output.shape)
    print(att_weights.shape)
    print(att_weights.sum(dim=-1))




####################
# attention() Test #
#####################

if __name__ == "__main__":
    B=2
    h=4
    T_q=5
    T_k=7
    d_k=8
    d_v=6
    Q = torch.randn(B, h, T_q, d_k)
    K = torch.randn(B, h, T_k, d_k)
    V = torch.randn(B, h, T_k, d_v)

    output , att_weights = attention(Q, K, V)
    print(output.shape)
    print(att_weights.shape)
    print(att_weights.sum(dim=-1))





##########################################
#   Embedding + Positional Encoding Test #
##########################################

if __name__ == "__main__":
    # Dummy settings
    vocab_size = 20
    d_model = 8
    max_len = 10

    # Batch = 2, Sequence Length = 5
    x = torch.tensor([
        [3, 8, 15, 2, 1],
        [4, 7, 3, 9, 1]
    ])

    # Modules
    token_embedding = TokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model
    )

    positional_encoding = PositionalEncoding(
        d_model=d_model,
        max_len=max_len
    )

    # 1. Token IDs
    print("Token IDs shape:")
    print(x.shape)

    # 2. Embedding
    embedded = token_embedding(x)

    print("\nEmbedding shape:")
    print(embedded.shape)

    # 3. Positional Encoding
    encoded = positional_encoding(embedded)

    print("\nAfter Positional Encoding:")
    print(encoded.shape)

    # Additional Test
    print("\nEmbedding first token:")
    print(embedded[0, 0])

    print("\nAfter PE first token:")
    print(encoded[0, 0])

    # Shape check
    assert x.shape == (2, 5)
    assert embedded.shape == (2, 5, 8)
    assert encoded.shape == (2, 5, 8)

    print("\nAll shape checks passed!")
