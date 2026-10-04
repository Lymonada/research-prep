import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# 1. Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
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

        # 각 embedding dimension pair마다 서로 다른 sin/cos 주기를 만들어주는 scale 값들을 생성.
        # shape은 [d_model / 2]  (d_model이 짝수라고 가정)
        # 각 dimension pair라는게 sin/cos을 의미.
        # dim 0과 1 → 같은 frequency 사용, 즉 dim 0 → sin(frequency 0), dim 1 → cos(frequency 0)
        # 이어서 dim 2,3도 같은 frquency 사용..
        # div_term은 사실 각 포지션이 다른 속도의 sin/cos값을 갖게 하기위한 scale임.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # position*div_term으로 나온 [max_len, d_model/2] 모양의 텐서가 sin/cos에 들어가서
        # sin은 짝수, cos은 홀수 column에 들어가게 함.
        # 여기서 column은 d_model. row는 포지션.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 미리 만든 self.pe는 shape이 [1, max_len, d_model]이라서 그걸 바로
        # x(embedding이 끝난 입력값, shape은 [B, T, d_model])에 더하면 안되서
        # x.size(1), 즉 토큰개수만큼 잘라줌.
        x = x + self.pe[:, :x.size(1), :]
        return x


# 3. Attention
def attention(Q, K, V, mask=None):
    # Expected shapes:
    # Q/K/V : [B, h, T, d_head]
    # mask: attention score [B,h,T_q,T_k]에 broadcast 가능한 bool tensor
    # ex) Causal mask: [1, 1, T_q, T_k], Padding mask: [B, 1, 1, T_k]

    d_k = Q.shape[-1]
    K_trans = torch.transpose(K, -2, -1)
    att_scores = Q @ K_trans / math.sqrt(d_k)

    # 만약 mask가 있다면, pytorch의 masked_fill은 mask값이 true인 위치를 바꾸기 때문에,
    # ~mask으로 false, true를 뒤집고 기존에 false였던 부분을 -inf로 채우기
    # (현재 코드에선 true가 허용, false가 차단 인 방식으로 mask를 만들었기 때문)
    # 중요한 부분은, 이 mask가 causal인지 padding인지 알 필요없이 false인 위치를 막음.
    # 그래서 구현이 간단해짐.
    if mask is not None:
        att_scores = att_scores.masked_fill(~mask, float("-inf"))

    att_weights = F.softmax(att_scores, dim=-1)
    output = att_weights @ V

    # output: [B, h, T_q, d_v]
    # att_weights: [B, h, T_q, T_k]
    return output, att_weights


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

    def forward(self, query, key, value, mask=None):
        # Expected query, key, value shape: [B, T, d_model]

        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Q/K/V를 head split 하고 head차원을 앞으로 옮기기
        # ex) Q: [B, T_q, d_model]
        # -> [B, T_q, h, d_head]
        # -> [B, h, T_q, d_head]
        Q = Q.unflatten(-1, (self.num_heads, self.d_head))
        Q = torch.transpose(Q, 1, 2)

        K = K.unflatten(-1, (self.num_heads, self.d_head))
        K = torch.transpose(K, 1, 2)

        V = V.unflatten(-1, (self.num_heads, self.d_head))
        V = torch.transpose(V, 1, 2)

        output, att_weights = attention(Q, K, V, mask)

        # attention에서 나온 output을 다시 원래 shape으로 돌려놓기
        # head_output [B, h, T_q, d_head] -> [B, T_q, d_model]
        output = torch.transpose(output, 1, 2)

        # 숫자하나만 적으면 맨끝에 -1이 default값으로 들어가있어서
        # (-2,-1) 이렇게 처리됨. 즉 그 두개를 합침.
        output = output.flatten(-2)

        output = self.W_O(output)

        return output, att_weights


# 5. Feed-Forward Network
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Expected x shape: [B, T, d_model]

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
        x = x + sublayer_output  # Residual connection
        output = self.layernorm(x)  # Layernormalization

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
        # multihead attention에서 새로운 representation과 attention weight matrix 받기
        attn_output, attn_weights = self.mha(x, x, x, mask=src_mask)

        residual_ln1 = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(residual_ln1)
        residual_ln2 = self.add_norm2(residual_ln1, ffn_output)

        return residual_ln2, attn_weights


# 8. Encoder
## 실제로 encoder block들을 조립해서 작동하게 하기
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        self.layers = nn.ModuleList(EncoderBlock(d_model, num_heads, d_ff) for i in range(num_layers))

    def forward(self, x, src_mask=None):
        # x(encoder input) shape: [B, S, d_model]

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

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # x(decoder input) shape: [B, T, d_model]
        # encoder_output shape: [B, S, d_model]

        # q/k/v 모두 decoder input을 가져와서 projection
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, mask=self_mask)
        residual_ln1 = self.add_norm1(x, self_attn_output)

        # cross attention은 k/v를 encoder output에서 가져와서 projection
        cross_attn_output, cross_attn_weights = self.cross_attn(residual_ln1, encoder_output, encoder_output, mask=cross_mask)
        residual_ln2 = self.add_norm2(residual_ln1, cross_attn_output)
        ffn_output = self.ffn(residual_ln2)
        residual_ln3 = self.add_norm3(residual_ln2, ffn_output)

        # self_attn_weights: [B, h, T, T]
        # cross_attn_weights: [B, h, T, S]
        return residual_ln3, self_attn_weights, cross_attn_weights


# 10. Decoder
## 실제로 decoder block들을 조립해서 작동하게 하기
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        self.layers = nn.ModuleList(DecoderBlock(d_model, num_heads, d_ff) for i in range(num_layers))

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # x(decoder input) shape: [B, T, d_model]

        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, encoder_output, self_mask=self_mask, cross_mask=cross_mask)

            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)

        return x, all_self_attn_weights, all_cross_attn_weights


# 11. Masks
def create_causal_mask(seq_len, device):
    # Decoder self-attention에서 미래 target token을 attend하지 못하게 함.
    # seq_len = target sequence length T.

    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # Decoder self-attention에서는 T_q = T_k = T이므로
    # [T_q, T_k] = [T, T] lower-triangular mask를 만듦.

    mask = mask.unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0).bool()

    # mask: [1, 1, T, T]
    # 앞 쪽에 두 차원을 추가해서 나중에
    # [B, h, T_q, T_k] shape을 가진 attention score와 계산될 예정.

    return mask


def create_padding_mask(seq, pad_idx):
    # 현재 K/V를 제공하는 sequence의 PAD 위치를 아무 Query도 참고하지 못하게 함.

    # seq 텐서와 PAD token의 정수 id를 받음.
    # seq: [B, T_k]
    mask = seq != pad_idx

    # pad_idx가 아닌 원소는 True,
    # pad_idx랑 같으면 False를 가지는 텐서를 만들어줌 = mask

    mask = mask.unsqueeze(1).unsqueeze(2)

    # mask: [B, 1, 1, T_k]
    # head dimension과 query dimension에 broadcasting되도록 함.
    # 왜냐면 모든 head의 모든 query에서 PAD key를 참고하지 못하게 해야해서.

    # 따라서 각 batch에서 모든 head와 모든 query가 PAD key를 보지 못함.
    return mask


def create_decoder_mask(target_seq, pad_idx):
    # target_seq: [B, T]
    # Decoder self-attention용 causal mask + target padding mask 생성.

    seq_len = target_seq.shape[1]
    # target_seq: [B, T] 에서 T 가져오기

    current_device = target_seq.device

    causal_mask = create_causal_mask(seq_len, current_device)

    # causal mask: [1, 1, T_q, T_k]
    # = [1, 1, T, T]

    padding_mask = create_padding_mask(target_seq, pad_idx)

    # padding mask: [B, 1, 1, T_k]

    combined_mask = causal_mask & padding_mask

    # combined mask: [B, 1, T, T_k]
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

    def forward(self, src, tgt):
        # source sequence: [B, S]
        # target sequence: [B, T]

        src_x = self.src_token_embedding(src)
        src_x = self.src_positional_encoding(src_x)

        # src_mask는 padding mask만
        src_mask = create_padding_mask(seq=src, pad_idx=self.pad_idx)

        # padding mask를 만들때는 embedding하기전의 src sequence를 가져와야지
        # token id를 보고 pad인지 확인할 수 있음.
        encoder_output, encoder_attn_weights = self.encoder(src_x, src_mask)

        # encoder_output: [B,S,d_model]
        # encoder_attn_weights:
        # list of num_layers, each element: [B,h,S,S]

        tgt_x = self.tgt_token_embedding(tgt)
        tgt_x = self.tgt_positional_encoding(tgt_x)

        # tgt_mask는 causal mask + padding mask
        tgt_mask = create_decoder_mask(target_seq=tgt, pad_idx=self.pad_idx)

        # padding mask를 만들때는 embedding하기전의 tgt sequence를 가져와야지
        # token id를 보고 pad인지 확인할 수 있음.

        decoder_output, decoder_self_attn_weights, decoder_cross_attn_weights = self.decoder(tgt_x, encoder_output, tgt_mask, src_mask)

        # decoder_output: [B,T,d_model]
        # decoder_self_attn_weights:
        # list of num_layers, each [B,h,T,T]
        # decoder_cross_attn_weights:
        # list of num_layers, each [B,h,T,S]

        vocab_logits = self.output_linear(decoder_output)

        # vocab_logits: [B,T,tgt_vocab_size]

        return vocab_logits, encoder_attn_weights, decoder_self_attn_weights, decoder_cross_attn_weights
        


# 13. Fixed-Length Copy Task Batch
def generate_copy_batch(batch_size, seq_len, vocab_size, bos_idx, eos_idx, device):
    src = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)

    tgt_input = F.pad(src, (1, 0), value=bos_idx) # (1,0)을 하면 앞에 하나 추가

    tgt_label = F.pad(src, (0, 1), value=eos_idx) # (0,1)을 하면 뒤에 하나 추가

    # src: [B, S]
    # tgt_input: [B, S+1]
    # tgt_label: [B, S+1]
    return src, tgt_input, tgt_label


# 14. Variable-Length Copy Task Batch
def generate_variable_copy_batch(lengths, vocab_size, bos_idx, eos_idx, pad_idx, device):
    src_list = []
    tgt_input_list = []
    tgt_label_list = []

    # lengths는 각 sequence의 길이를 알려주는 1차원 배열
    for length in lengths:
        src = torch.randint(3, vocab_size, (length,), device=device)

        tgt_input = F.pad(src, (1, 0), value=bos_idx)
        # (1,0)을 하면 앞에 하나 추가

        tgt_label = F.pad(src, (0, 1), value=eos_idx)
        # (0,1)을 하면 뒤에 하나 추가

        src_list.append(src)
        tgt_input_list.append(tgt_input)
        tgt_label_list.append(tgt_label)

    # src_list를 PAD해서 [B, S_max]
    # pad_sequence가 입력받은 리스트에서 각 텐서의 길이 중 가장 큰 값을 찾아냄.
    src_batch = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    # batch_first=True로 output을 [B, S_max] 형태로 만들고,
    # S_max는 src_list에서 가장 긴 sequence 길이(max(lengths)).
    # 짧은 sequence의 오른쪽 부족한 부분은 pad_idx로 채움.

    # tgt_input_list를 PAD해서 [B, T_max]
    tgt_input_batch = pad_sequence(tgt_input_list, batch_first=True, padding_value=pad_idx)

    # tgt_input의 각 sequence는 [BOS]가 추가되어 src보다 길이가 1 크므로,
    # padding 후 shape은 [B, T_max], T_max = max(lengths) + 1.

    # tgt_label_list를 PAD해서 [B, T_max]
    tgt_label_batch = pad_sequence(tgt_label_list, batch_first=True, padding_value=pad_idx)

    # tgt_label의 각 sequence는 [EOS]가 추가되어 src보다 길이가 1 크므로,
    # padding 후 shape은 [B, T_max], T_max = max(lengths) + 1.

    return src_batch, tgt_input_batch, tgt_label_batch


# 15. Autoregressive Inference
@torch.no_grad()
def greedy_decode(model, src, bos_idx, eos_idx, max_new_tokens, device, pad_idx):
    # src: [B,S]

    model.eval()
    batch_size = src.shape[0]

    # [B, 1] 크기로 bos_idx 채우기
    generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    # model에 넣어줄 target sequence로 [BOS]만 있는 텐서 넣기

    # 텐서 하나를 만들어서 batch의 각 sequence가
    # EOS를 생성해 종료되었는지 개별적으로 추적
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 최대 max_new_tokens개의 새로운 token을 autoregressive하게 생성
    for tokens in range(max_new_tokens):
        vocab_logits, _, _, _ = model(src, generated)

        # next_token_logits: [B, vocab_size]
        next_token_logits = vocab_logits[:, -1, :]

        # vocab_logits에는 모든 timestep의 다음 토큰 예상값이 logits으로 들어있기 때문에
        # 마지막 시점의, 즉 마지막 position의 예상값을 가져오기 위해
        # T차원을 -1로 가져옴.

        # next_token_logits에서 가장 큰 값의 인덱스는
        # 모델이 예상한 다음 token id.
        # next_token: [B]
        next_token = next_token_logits.argmax(dim=-1)

        # 이전 step에서 이미 EOS를 생성한 sequence는 이제 PAD만 추가
        next_token = torch.where(
            # finished 텐서의 값 중 true인 위치에는
            # torch.full_like(next_token, pad_idx)에서 채워넣고
            # false면 next_token에서 채워넣기.
            # 즉, 문장이 EOS를 마지막으로 생성해서 끝난상태면
            # pad token id를 next_token으로 하고,
            # 아니면 그대로 next_token에서 가져옴.
            # finished=True -> pad_idx
            # finished=False -> 모델이 예측한 next_token 유지
            finished,
            torch.full_like(next_token, pad_idx),
            next_token
        )

        # 이번 step에서 새롭게 EOS를 생성한 sequence까지 finished에 기록
        finished = finished | (next_token == eos_idx)

        # 제일 안쪽 차원을 하나 추가.
        # next_token: [B, 1]
        next_token = next_token.unsqueeze(-1)

        # generated와 next_token을 붙임.
        # generated: [B, current_length+1]
        generated = torch.cat([generated, next_token], dim=1)

        # finished 텐서안의 값이 모두 true면 break
        if finished.all():
            break

    # generated: [B,T]
    return generated

# 16. Training Loop + Final Test
if __name__ == "__main__":

    vocab_size = 20
    pad_idx = 0
    bos_idx = 1
    eos_idx = 2

    # 각 sample의 실제 source sequence length
    lengths = [
        2, 5, 3, 4, 8, 6, 9, 4, 5, 3,
        2, 5, 3, 4, 8, 6, 9, 4, 5, 3,
        2, 5, 3, 4, 8, 6, 9, 4, 5, 3,
        5, 3, 7
    ]

    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2

    learning_rate = 1e-3
    num_steps = 1000

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, pad_idx=pad_idx).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = nn.CrossEntropyLoss(ignore_index=pad_idx)


    #################
    # Training Loop #
    #################

    model.train()

    for step in range(num_steps):

        train_src, train_tgt_input, train_tgt_label = generate_variable_copy_batch(lengths=lengths, vocab_size=vocab_size, bos_idx=bos_idx, eos_idx=eos_idx, pad_idx=pad_idx, device=device)

        # train_src: [B, S_max]
        # train_tgt_input: [B, T_max]
        # train_tgt_label: [B, T_max]

        optimizer.zero_grad()

        vocab_logits, _, _, _ = model(train_src, train_tgt_input)

        # vocab_logits: [B, T_max, vocab_size]
        # CrossEntropyLoss에 넣기 위해
        # [B, T_max, vocab_size] -> [B*T_max, vocab_size]
        logits = vocab_logits.reshape(-1, vocab_logits.size(-1))

        # [B, T_max] -> [B*T_max]
        label = train_tgt_label.reshape(-1)

        loss = loss_function(logits, label)

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f}")


    ##############################################################
    # Final Variable-length Teacher-forcing vs Autoregressive Test
    ##############################################################

    # lengths 배열의 길이가 자연스럽게 batch_size가 됨.
    test_lengths = [2, 5, 3, 4]

    test_src, test_tgt_input, test_tgt_label = generate_variable_copy_batch(lengths=test_lengths, vocab_size=vocab_size, bos_idx=bos_idx, eos_idx=eos_idx, pad_idx=pad_idx, device=device)

    # test_src: [B, S_max]
    # test_tgt_input / test_tgt_label: [B, T_max]


    # 1) Teacher-forcing prediction
    model.eval()

    with torch.no_grad():
        vocab_logits, _, _, _ = model(test_src, test_tgt_input)
        teacher_prediction = vocab_logits.argmax(dim=-1)


    # 2) Autoregressive prediction
    max_new_tokens = max(test_lengths) + 1

    generated = greedy_decode(model=model, src=test_src, bos_idx=bos_idx, eos_idx=eos_idx, max_new_tokens=max_new_tokens, device=device, pad_idx=pad_idx)


    print("\n================ Final Test ================\n")

    print("src:")
    print(test_src)

    print("\ntgt_input:")
    print(test_tgt_input)

    print("\ntgt_label:")
    print(test_tgt_label)

    print("\nteacher prediction:")
    print(teacher_prediction)

    print("\nautoregressive:")
    print(generated)