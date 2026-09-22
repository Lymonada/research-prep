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
def attention(Q, K, V): # Expected Q/K/V shape: [B, h, T, d_head]
    d_k = Q.shape[-1]
    K_trans = torch.transpose(K, -2, -1)
    att_scores = Q @ K_trans / math.sqrt(d_k)
    att_weights = F.softmax(att_scores, dim = -1)
    output = att_weights @ V
    return output, att_weights # output: [B, h, T_q, d_k], att_weights: [B, h, T_q, T_k]

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
    
    def forward(self, query, key, value): # Expected query, key, value shape: [B, T, d_model]
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

        # attention에서 나온 output을 다시 원래 shape으로 돌려놓기 head_output [B,h,T_q,d_head] -> [B, T, d_model]
        output, att_weights = attention(Q,K,V)
        output = torch.transpose(output, 1, 2)
        output = output.flatten(-2) # 

        output = self.W_O(output)

        return output, att_weights

# 5. Feed-Forwad Network
class FeedForward(nn.Module):
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
        x = x + sublayer_output
        output = self.layernorm(x)
        return output


















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
