# Mini RAG Retrieval Pipeline

## 1. 목표

이번 단계의 목표는 RAG 전체를 한 번에 구현하는 것이 아니라, 그중에서도 가장 핵심적인 **Retrieval Pipeline**을 직접 구현하면서 내부 흐름을 이해하는 것이다.

최종적으로 구현한 흐름은 다음과 같다.

```text
Raw Document
    ↓
Chunking
    ↓
Chunks
    ↓
Embedding Model
    ↓
Chunk Embeddings

Query Text
    ↓
Embedding Model
    ↓
Query Embedding

Chunk Embeddings + Query Embedding
    ↓
Cosine Similarity
    ↓
Similarity Scores
    ↓
Top-k
    ↓
Relevant Chunks
```

즉,

> 긴 document를 여러 chunk로 나누고, 각 chunk를 embedding한 뒤, query와 가장 의미적으로 가까운 Top-k chunk를 찾는 것

이 이번 구현의 핵심이다.

---

# 2. RAG에서 Retrieval이 하는 일

RAG는 크게 보면 다음 두 부분으로 나눌 수 있다.

```text
Retrieval
+
Generation
```

이번 단계에서는 아직 LLM generation은 구현하지 않고 **Retrieval 부분만** 구현했다.

Retrieval의 역할은 사용자 query가 들어왔을 때 외부 문서에서 관련 있는 정보를 찾아내는 것이다.

예를 들어:

```text
Query:
"How does a RAG system find relevant information?"
```

이 들어오면 전체 문서를 그대로 LLM에 전달하는 것이 아니라,

```text
관련 있는 Chunk 1
관련 있는 Chunk 2
...
```

를 먼저 찾아낸다.

이후 실제 RAG에서는 이 chunk들을 LLM prompt의 context로 넣는다.

---

# 3. Document와 Chunk

처음에는 긴 document 하나가 있다고 가정했다.

```python
document = """
Transformers are neural network architectures that use attention mechanisms
to model relationships between tokens in a sequence.

Self-attention allows each token to examine other tokens in the sequence
and build a contextual representation.

Retrieval-Augmented Generation, or RAG, combines information retrieval
with language generation.

...
"""
```

자료형은 단순한 문자열이다.

```text
document: str
```

하지만 긴 document 전체를 하나의 embedding으로 만들면 여러 주제의 정보가 하나의 vector에 섞일 수 있다.

예를 들어 하나의 문서 안에:

```text
Transformer
RAG
CNN
Optimizer
```

가 모두 들어 있다면,

```text
"What is retrieval in RAG?"
```

라는 query에 필요한 RAG 관련 정보가 전체 document embedding 속에서 희석될 수 있다.

그래서 실제 retrieval에서는 document를 여러 개의 **chunk**로 나눈다.

```text
Document
   ↓
Chunk 0
Chunk 1
Chunk 2
...
Chunk N-1
```

즉,

> Document는 원본 정보의 저장 단위이고, Chunk는 실제 검색 단위가 될 수 있다.

---

# 4. Fixed-size Chunking

먼저 가장 단순한 fixed-size chunking을 구현했다.

예를 들어:

```text
A B C D E F G H I J
```

에서

```text
chunk_size = 4
```

라면:

```text
Chunk 0: A B C D
Chunk 1: E F G H
Chunk 2: I J
```

처럼 나눌 수 있다.

핵심 Python 문법은 다음과 같다.

```python
words = text.split()
```

문자열을 공백 기준으로 나눠 list로 만든다.

```python
words[start:start + chunk_size]
```

slicing으로 chunk를 만들고,

```python
" ".join(chunk)
```

으로 다시 문자열 형태로 합친다.

---

# 5. Overlap Chunking

Fixed-size chunking의 문제는 중요한 정보가 chunk 경계에서 끊길 수 있다는 것이다.

예를 들어:

```text
RAG first retrieves relevant documents before generation begins.
```

이 문장이 다음처럼 잘린다고 하자.

```text
Chunk 0:
RAG first retrieves relevant

Chunk 1:
documents before generation begins
```

그러면

```text
retrieves relevant documents
```

라는 의미 단위가 두 chunk로 갈라진다.

이를 완화하기 위해 이전 chunk의 일부 내용을 다음 chunk에도 포함시키는 **overlap**을 사용할 수 있다.

예를 들어:

```text
chunk_size = 4
overlap = 2
```

이면:

```text
Chunk 0: A B C D
Chunk 1: C D E F
Chunk 2: E F G H
Chunk 3: G H I J
Chunk 4: I J
```

가 된다.

핵심은 이동 간격이다.

```text
step = chunk_size - overlap
```

예를 들어:

```text
chunk_size = 4
overlap = 2

step = 2
```

이므로 시작 index는:

```text
0 → 2 → 4 → 6 → 8
```

로 이동한다.

---

# 6. `chunk_text_overlap()`

구현한 함수는 다음과 같다.

```python
def chunk_text_overlap(text, chunk_size, overlap):
    assert 0 <= overlap < chunk_size

    results = []
    words = text.split()

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        result = " ".join(chunk)
        results.append(result)

    return results
```

## 입력

```text
text: str
chunk_size: int
overlap: int
```

## 출력

```text
chunks: list[str]
```

예시:

```python
text = "A B C D E F G H I J"

chunks = chunk_text_overlap(
    text,
    chunk_size=4,
    overlap=2
)
```

출력:

```python
[
    "A B C D",
    "C D E F",
    "E F G H",
    "G H I J",
    "I J"
]
```

---

# 7. Text Embedding

처음 toy retrieval에서는 embedding vector를 직접 만들었다.

```python
query = torch.tensor([1.0, 2.0, 3.0])

doc1 = torch.tensor([1.0, 2.0, 3.0])
doc2 = torch.tensor([3.0, 2.0, 1.0])
doc3 = torch.tensor([-1.0, -2.0, -3.0])
```

하지만 실제 RAG에서는 사람이 embedding을 직접 만들지 않는다.

실제 text를 embedding model에 넣는다.

이번 구현에서는:

```python
from sentence_transformers import SentenceTransformer
```

를 사용했고,

```python
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

를 사용했다.

이 모델의 embedding dimension은:

```text
D = 384
```

이다.

---

# 8. Token Embedding과 Sentence Embedding의 차이

Transformer를 직접 구현할 때 사용했던:

```python
nn.Embedding(vocab_size, d_model)
```

은 token ID 하나를 vector 하나로 바꾸는 역할이었다.

```text
Token IDs
[B, T]
↓
nn.Embedding
[B, T, d_model]
```

즉 token마다 representation이 존재한다.

반면 RAG에서 사용하는 sentence/chunk embedding은:

```text
Raw Text
↓
Tokenization
↓
Transformer
↓
Pooling
↓
Sentence / Chunk Embedding
[D]
```

의 형태다.

즉 하나의 sentence 또는 chunk 전체가 하나의 semantic vector로 표현된다.

---

# 9. Chunk Embedding

Chunk가 `N`개 있다고 하자.

```text
chunks: list[str]
length = N
```

이를 embedding model에 넣으면:

```python
chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True
)
```

출력 shape은:

```text
[N, D]
```

이다.

이번 모델에서는:

```text
D = 384
```

이므로 chunk가 6개라면:

```text
chunk_embeddings
[6, 384]
```

가 된다.

각 row는 해당 chunk와 1:1로 대응된다.

```text
chunk_embeddings[0] ↔ chunks[0]
chunk_embeddings[1] ↔ chunks[1]
chunk_embeddings[2] ↔ chunks[2]
...
```

이 index 대응 관계는 retrieval 결과에서 원본 text를 다시 찾을 때 중요하다.

---

# 10. Query Embedding

Query도 같은 embedding model을 사용한다.

```python
query_text = "How does a RAG system find relevant information?"
```

이를 embedding하면:

```python
query_embedding = model.encode(
    query_text,
    convert_to_tensor=True
)
```

shape은:

```text
[D]
```

즉 이번 모델에서는:

```text
[384]
```

이다.

중요한 점은 chunk와 query가 **같은 embedding space**에 존재해야 한다는 것이다.

그래야 두 vector의 similarity를 비교할 수 있다.

---

# 11. Cosine Similarity

Query와 각 chunk의 semantic similarity를 계산하기 위해 cosine similarity를 사용했다.

공식:

```text
cosine_similarity(q, d)

= (q · d)
  -------
  ||q|| ||d||
```

의미는 두 vector가 얼마나 비슷한 방향을 가리키는지를 비교하는 것이다.

Cosine similarity가 크다고 해서:

```text
0.7 = 70% 확률
```

이라는 뜻은 아니다.

이는 단순히 embedding 공간에서의 similarity score이며, retrieval에서는 보통 여러 문서 또는 chunk 사이의 **상대적인 순위**가 중요하다.

---

# 12. Toy Cosine Similarity

처음에는 두 vector만 비교하는 함수를 만들었다.

```python
def cosine_similarity(a, b):
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)

    return dot / (norm_a * norm_b)
```

입력:

```text
a [D]
b [D]
```

출력:

```text
scalar []
```

예를 들어:

```python
query = torch.tensor([1.0, 2.0, 3.0])

doc1 = torch.tensor([1.0, 2.0, 3.0])
doc2 = torch.tensor([3.0, 2.0, 1.0])
doc3 = torch.tensor([-1.0, -2.0, -3.0])
```

결과:

```text
query ↔ doc1 = 1.0000
query ↔ doc2 = 0.7143
query ↔ doc3 = -1.0000
```

---

# 13. Batch Cosine Similarity

실제 retrieval에서는 query 하나와 여러 chunk를 비교해야 한다.

따라서:

```text
query
[D]

documents
[N, D]
```

형태에서 모든 similarity를 한 번에 계산하는 함수를 구현했다.

```python
def cosine_similarity_batch(query, documents):

    dot_products = documents @ query
    q_norm = torch.norm(query)
    doc_norm = torch.norm(documents, dim=1)

    scores = dot_products / (q_norm * doc_norm)

    return scores
```

---

# 14. Cosine Similarity Shape Flow

입력:

```text
query
[D]

documents
[N, D]
```

먼저:

```python
dot_products = documents @ query
```

shape:

```text
[N, D] @ [D]
→ [N]
```

즉 각 document 또는 chunk와 query의 dot product를 한 번에 계산한다.

---

Query norm:

```python
q_norm = torch.norm(query)
```

shape:

```text
[D]
→ []
```

scalar 하나가 나온다.

---

각 document/chunk의 norm:

```python
doc_norm = torch.norm(documents, dim=1)
```

shape:

```text
[N, D]
→ [N]
```

각 row마다 norm 하나를 계산한다.

---

최종:

```python
scores = dot_products / (q_norm * doc_norm)
```

shape:

```text
[N] / ([] * [N])
→ [N]
```

PyTorch broadcasting에 의해 scalar `q_norm`이 모든 document norm에 적용된다.

최종 결과:

```text
scores [N]
```

각 값은 다음과 대응된다.

```text
scores[0] ↔ query vs chunk 0
scores[1] ↔ query vs chunk 1
scores[2] ↔ query vs chunk 2
...
```

---

# 15. Retrieval

Similarity score를 계산한 뒤에는 가장 점수가 높은 chunk를 선택해야 한다.

이를 위해:

```python
torch.topk(scores, k)
```

를 사용한다.

예를 들어:

```text
scores
[N]
```

에서:

```python
top_scores, top_indices = torch.topk(scores, k)
```

를 실행하면:

```text
top_scores
[k]

top_indices
[k]
```

가 나온다.

예:

```text
top_scores
[0.6957, 0.5200]

top_indices
[2, 1]
```

이면:

```text
1위 → chunk 2
2위 → chunk 1
```

이라는 의미다.

---

# 16. `retrieve()`

전체 retrieval을 담당하는 함수:

```python
def retrieve(query, documents, texts, k=2):

    scores = cosine_similarity_batch(query, documents)

    top_scores, top_indices = torch.topk(scores, k)

    results = []

    for score, idx in zip(top_scores, top_indices):
        index = idx.item()
        results.append((texts[index], score.item()))

    return results
```

## 입력

```text
query
[D]

documents
[N, D]

texts
list[str], length N

k
int
```

## 출력

```text
list[(text, score)]
```

예:

```python
[
    ("Relevant chunk text ...", 0.6957),
    ("Another relevant chunk ...", 0.5200)
]
```

---

# 17. 왜 `texts[index]`가 필요한가

Embedding vector 자체에는 원래 text가 들어 있지 않다.

예를 들어:

```text
chunk_embeddings[2]
```

는 단순히:

```text
[384개의 숫자]
```

일 뿐이다.

하지만:

```text
chunk_embeddings[2] ↔ chunks[2]
```

라는 index 관계가 존재한다.

따라서 retrieval에서:

```python
top_indices
```

를 얻은 뒤:

```python
texts[index]
```

를 사용하면 해당 embedding에 대응하는 원본 chunk를 다시 가져올 수 있다.

핵심은:

> Retrieval의 세 번째 인자는 각 embedding row와 같은 index를 공유하는 원본 데이터여야 한다.

예를 들어 chunk embedding을 사용했다면:

```python
retrieve(
    query_embedding,
    chunk_embeddings,
    chunks
)
```

가 된다.

반대로 document 전체를 embedding했다면:

```python
retrieve(
    query_embedding,
    document_embeddings,
    documents_text
)
```

처럼 document text가 들어갈 수도 있다.

---

# 18. 전체 Retrieval Pipeline

이번에 최종적으로 만든 흐름은 다음과 같다.

```text
Raw Document
str
   ↓
chunk_text_overlap()
   ↓
Chunks
list[str], length N
   ↓
SentenceTransformer.encode()
   ↓
Chunk Embeddings
[N, D]


Query Text
str
   ↓
SentenceTransformer.encode()
   ↓
Query Embedding
[D]


Query Embedding [D]
        +
Chunk Embeddings [N, D]
        ↓
cosine_similarity_batch()
        ↓
Similarity Scores
[N]
        ↓
torch.topk(k)
        ↓
Top Scores [k]
Top Indices [k]
        ↓
Original Chunk Lookup
        ↓
Top-k Relevant Chunks
```

---

# 19. Shape Summary

```text
document
str

↓ chunk_text_overlap()

chunks
list[str], length = N

↓ model.encode()

chunk_embeddings
[N, D]

query_text
str

↓ model.encode()

query_embedding
[D]

↓ cosine_similarity_batch()

scores
[N]

↓ torch.topk(k)

top_scores
[k]

top_indices
[k]

↓ original text lookup

results
list[(chunk_text, score)]
```

이번에 사용한 embedding model에서는:

```text
D = 384
```

였다.

예를 들어 chunk가 6개라면:

```text
chunk_embeddings
[6, 384]

query_embedding
[384]

scores
[6]

k = 2

top_scores
[2]

top_indices
[2]
```

가 된다.

---

# 20. 실제 Retrieval 결과

사용한 query:

```text
How does a RAG system find relevant information?
```

Top-1:

```text
Score: 0.6957

of documents for relevant information. Documents in a RAG system
are usually split into smaller chunks. Each chunk is converted into
an embedding vector using an embedding model. These vectors can then
be searched using ...
```

Top-2:

```text
Score: 0.5200

Retrieval-Augmented Generation, or RAG, combines information retrieval
with language generation. Before answering a question, a RAG system
searches an external collection of documents for relevant information.
...
```

Query와 실제로 관련 있는 chunk들이 높은 similarity score로 검색되었다.

---

# 21. Transformer Attention과 Retrieval 비교

Transformer에서 attention을 구현할 때:

```text
Q @ K^T
```

를 사용해 token representation 사이의 관계를 계산했다.

Retrieval에서는:

```text
chunk_embeddings @ query_embedding
```

을 통해 query와 chunk representation 사이의 similarity를 계산한다.

둘은 같은 알고리즘은 아니지만,

```text
representation vector 사이의 관계를 계산한다
```

는 공통점이 있다.

차이는 이후 처리 방식이다.

Transformer Attention:

```text
Similarity Scores
↓
Softmax
↓
Weighted Sum of V
```

Retrieval:

```text
Similarity Scores
↓
Top-k
↓
Relevant Chunks
```

즉 Retrieval에서는 similarity를 이용해 중요한 chunk를 직접 선택한다.

---

# 22. 구현한 전체 코드

```python
import torch
from sentence_transformers import SentenceTransformer


def cosine_similarity_batch(query, documents):
    # query: [D]
    # documents: [N, D]

    dot_products = documents @ query
    # [N, D] @ [D]
    # → [N]

    q_norm = torch.norm(query)
    # scalar []

    doc_norm = torch.norm(documents, dim=1)
    # [N, D]
    # → [N]

    scores = dot_products / (q_norm * doc_norm)
    # [N]

    return scores


def retrieve(query, documents, texts, k=2):

    scores = cosine_similarity_batch(query, documents)
    # [N]

    top_scores, top_indices = torch.topk(scores, k)
    # top_scores: [k]
    # top_indices: [k]

    results = []

    for score, idx in zip(top_scores, top_indices):
        index = idx.item()
        results.append((texts[index], score.item()))

    return results


def chunk_text_overlap(text, chunk_size, overlap):

    assert 0 <= overlap < chunk_size

    results = []

    words = text.split()

    for i in range(
        0,
        len(words),
        chunk_size - overlap
    ):
        chunk = words[i:i + chunk_size]

        result = " ".join(chunk)

        results.append(result)

    return results


document = """
Transformers are neural network architectures that use attention mechanisms
to model relationships between tokens in a sequence.

Self-attention allows each token to examine other tokens in the sequence
and build a contextual representation.

Retrieval-Augmented Generation, or RAG, combines information retrieval
with language generation. Before answering a question, a RAG system searches
an external collection of documents for relevant information.

Documents in a RAG system are usually split into smaller chunks.
Each chunk is converted into an embedding vector using an embedding model.
These vectors can then be searched using similarity measures such as cosine similarity.

When a user submits a query, the query is also converted into an embedding.
The system compares the query embedding with the stored chunk embeddings
and retrieves the most relevant chunks.

The retrieved chunks are placed into the prompt as additional context.
A language model then uses both the user question and the retrieved context
to generate an answer.
"""


query_text = "How does a RAG system find relevant information?"


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


chunks = chunk_text_overlap(
    document,
    chunk_size=35,
    overlap=8
)


for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(chunk)
    print()


chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True
)


query_embedding = model.encode(
    query_text,
    convert_to_tensor=True
)


results = retrieve(
    query_embedding,
    chunk_embeddings,
    chunks,
    k=2
)


for text, score in results:
    print(f"Score: {score:.4f}")
    print(text)
    print()
```

---

# 23. Checkpoint

이번 단계가 끝났다면 다음 질문에 답할 수 있어야 한다.

1. 왜 긴 document 전체를 하나의 embedding으로 만들지 않고 chunk로 나누는가?
2. Chunk와 Document의 차이는 무엇인가?
3. Overlap은 왜 사용하는가?
4. `chunk_size - overlap`이 step이 되는 이유는 무엇인가?
5. `SentenceTransformer.encode()`는 text를 어떤 형태로 변환하는가?
6. Token embedding과 sentence/chunk embedding은 어떤 차이가 있는가?
7. Chunk가 `N`개이고 embedding dimension이 `D`라면 `chunk_embeddings`의 shape은 무엇인가?
8. Query 하나의 embedding shape은 무엇인가?
9. `documents @ query`가 `[N]` shape이 되는 이유는 무엇인가?
10. `torch.norm(documents, dim=1)`이 `[N]`을 반환하는 이유는 무엇인가?
11. Cosine similarity score는 확률인가?
12. `scores [N]`의 각 index는 무엇과 대응되는가?
13. `torch.topk(scores, k)`가 반환하는 두 값은 무엇인가?
14. 왜 `top_indices`를 사용해서 다시 `chunks[index]`를 가져와야 하는가?
15. `retrieve()`에서 embedding matrix와 text list가 같은 index를 공유해야 하는 이유는 무엇인가?
16. Transformer attention의 similarity 계산과 retrieval similarity 계산은 어떤 점에서 비슷하고 다른가?
17. 현재 구현한 pipeline은 RAG 전체 중 어느 부분까지 구현한 것인가?

---

# 24. 현재 Mini RAG 진행 상태

완료:

```text
RAG Architecture                  ✅
Embedding                         ✅
Cosine Similarity                 ✅
Toy Vector Retrieval              ✅
Semantic Retrieval                ✅
Fixed-size Chunking               ✅
Overlap Chunking                  ✅
Chunk Embedding                   ✅
Top-k Chunk Retrieval             ✅
Basic Retrieval Pipeline          ✅
```

현재 완성한 범위:

```text
Document
↓
Chunking
↓
Embedding
↓
Similarity Search
↓
Top-k Retrieval
```

즉,

> Document Ingestion + Dense Retrieval의 가장 기본적인 형태를 직접 구현한 상태

이다.

아직 진행하지 않은 부분:

```text
Real File Loading
PDF / Markdown Loading
Metadata
Vector Database
Prompt Construction
LLM Generation
End-to-End RAG
Retrieval Evaluation
Advanced Chunking
Hybrid Search
Reranking
```

---

# 25. 한 줄 정리

이번 단계에서 구현한 Retrieval Pipeline은:

> 긴 document를 overlapping chunk로 나누고, 각 chunk를 semantic embedding으로 변환한 뒤, query embedding과 cosine similarity를 계산하여 가장 관련 있는 Top-k chunk를 검색하는 구조이다.
