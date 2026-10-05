# Mini RAG Retrieval Pipeline

## 1. 목표

이번 단계의 목표는 RAG 전체를 한 번에 구현하는 것이 아니라, 그중에서도 가장 핵심적인 **Retrieval Pipeline**을 직접 구현하면서 내부 흐름을 이해하는 것이다.

처음에는 toy text와 직접 만든 vector로 retrieval의 원리를 확인했고, 이후 실제 Markdown 파일을 불러와 여러 문서를 chunking하고 metadata를 붙인 뒤 semantic embedding과 cosine similarity를 사용해 관련 chunk를 검색하는 구조까지 확장했다.

현재 최종적으로 구현한 흐름은 다음과 같다.

```text
Markdown Files
    ↓
File Loading
    ↓
Document + Metadata
    ↓
Overlap Chunking
    ↓
Chunks + Metadata
    ↓
Multiple Documents Merge
    ↓
All Chunks
    ↓
Text Extraction
    ↓
SentenceTransformer
    ↓
Chunk Embeddings [N, D]

Query Text
    ↓
SentenceTransformer
    ↓
Query Embedding [D]

Chunk Embeddings + Query Embedding
    ↓
Cosine Similarity
    ↓
Similarity Scores [N]
    ↓
Top-k
    ↓
Original Chunk Lookup
    ↓
Text + Score + Metadata
```

즉,

> 실제 여러 문서를 읽고, 검색 가능한 chunk 단위로 나눈 뒤, query와 의미적으로 가까운 Top-k chunk를 원본 출처 정보와 함께 찾아내는 것

이 이번 Retrieval Pipeline의 최종 목표이다.

---

# 2. RAG에서 Retrieval이 하는 일

RAG는 크게 보면 다음 흐름으로 볼 수 있다.

```text
Retrieval
    ↓
Augmentation
    ↓
Generation
```

이번 문서에서는 아직 LLM generation까지 연결하지 않고 **Retrieval 부분을 완성하는 것**에 집중했다.

Retrieval의 역할은 사용자 query가 들어왔을 때 외부 문서에서 관련 있는 정보를 찾아내는 것이다.

예를 들어:

```text
Query:
"How does attention make each token to contextual representation?"
```

이 들어오면 전체 문서를 그대로 LLM에 전달하는 것이 아니라,

```text
관련 있는 Chunk 1
관련 있는 Chunk 2
...
```

를 먼저 찾는다.

이후 실제 RAG에서는 이 retrieved chunks를 하나의 context로 묶고, query와 함께 LLM prompt에 넣게 된다.

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

> Document는 원본 정보의 저장 단위이고, Chunk는 실제 검색 단위이다.

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

현재 구현에서 `chunk_size`는 token 수나 문자 수가 아니라 **공백 기준 word 개수**이다.

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

이를 완화하기 위해 이전 chunk의 일부 내용을 다음 chunk에도 포함시키는 **overlap**을 사용한다.

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

입력:

```text
text: str
chunk_size: int
overlap: int
```

출력:

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

현재 구현에서는 `text.split()`을 사용하므로 Markdown의 줄바꿈도 공백처럼 처리된다. 지금 Mini RAG의 기본 원리를 익히는 단계에서는 이 단순한 word-based chunking을 유지한다.

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

model = SentenceTransformer("all-MiniLM-L6-v2")
```

를 사용했다.

이 모델의 embedding dimension은:

```text
D = 384
```

이다.

---

# 8. Token Embedding과 Sentence / Chunk Embedding의 차이

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
chunk_texts: list[str]
length = N
```

이를 embedding model에 넣으면:

```python
chunk_embeddings = model.encode(
    chunk_texts,
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

이다.

각 row는 해당 chunk와 1:1로 대응된다.

```text
chunk_embeddings[0] ↔ all_chunks[0]
chunk_embeddings[1] ↔ all_chunks[1]
chunk_embeddings[2] ↔ all_chunks[2]
...
```

이 index 대응 관계는 retrieval 결과에서 원본 text와 metadata를 다시 찾을 때 매우 중요하다.

---

# 10. Query Embedding

Query도 같은 embedding model을 사용한다.

```python
query_text = "How does attention make each token to contextual representation?"
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

Cosine similarity는 두 vector가 얼마나 비슷한 방향을 가리키는지를 비교한다.

Similarity가 크다고 해서:

```text
0.7 = 70% 확률
```

이라는 뜻은 아니다.

이는 embedding 공간에서의 similarity score이며, retrieval에서는 여러 chunk 사이의 **상대적인 순위**가 중요하다.

---

# 12. Batch Cosine Similarity

실제 retrieval에서는 query 하나와 여러 chunk를 비교해야 한다.

```text
query_embedding
[D]

chunk_embeddings
[N, D]
```

형태에서 모든 similarity를 한 번에 계산하는 함수를 구현했다.

```python
def cosine_similarity_batch(query, documents):
    # query: [D]
    # documents: [N, D]

    dot_products = documents @ query
    q_norm = torch.norm(query)
    doc_norm = torch.norm(documents, dim=1)

    scores = dot_products / (q_norm * doc_norm)

    return scores
```

Shape flow:

```text
documents @ query
[N, D] @ [D]
→ [N]

q_norm
[D] → []

doc_norm
[N, D] → [N]

scores
[N] / ([] * [N])
→ [N]
```

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

# 13. Top-k Retrieval

Similarity score를 계산한 뒤에는 가장 점수가 높은 chunk를 선택한다.

```python
top_scores, top_indices = torch.topk(scores, k)
```

Shape:

```text
top_scores  [k]
top_indices [k]
```

예를 들어:

```text
top_indices = [41, 40]
```

이라면 embedding matrix의 41번째, 40번째 row에 대응하는 원본 chunk를 다시 찾아가면 된다.

핵심은:

```text
embedding row index
        ↕
original chunk index
```

의 대응 관계이다.

---

# 14. 실제 파일 로딩

Toy string 대신 실제 GitHub repository의 Markdown 파일을 retrieval 대상으로 사용하도록 확장했다.

이번 구현에서는 `pathlib.Path`를 사용했다.

```python
from pathlib import Path
```

파일 하나를 읽는 함수:

```python
def load_document(file_path):
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    document_dict = {
        "text": text,
        "metadata": {
            "source": path.name,
            "path": str(path)
        }
    }

    return document_dict
```

예를 들어:

```python
document = load_document("notes/07_attention.md")
```

결과 구조:

```python
{
    "text": "# Attention: ...",
    "metadata": {
        "source": "07_attention.md",
        "path": "notes/07_attention.md"
    }
}
```

즉 실제 파일을 읽은 뒤 단순 문자열만 반환하는 것이 아니라 **문서 내용과 출처 정보를 함께 가진 dictionary**로 관리한다.

---

# 15. Metadata

Metadata는 chunk 내용 자체가 아니라 해당 정보가 **어디에서 왔는지 추적하기 위한 정보**이다.

현재 사용한 metadata:

```python
{
    "source": "07_attention.md",
    "path": "notes/07_attention.md"
}
```

이후 chunking 과정에서는 여기에:

```python
"chunk_index": 0
```

을 추가한다.

Metadata 자체를 embedding하는 것은 아니다.

```text
chunk["text"]
    ↓
Embedding Model
    ↓
Embedding Vector

chunk["metadata"]
    ↓
원본 위치 추적용으로 별도 보관
```

즉 semantic search에는 text를 사용하고, 검색된 결과의 출처를 복원할 때 metadata를 사용한다.

---

# 16. Document Metadata를 Chunk에 상속하기

실제 파일을 chunking할 때는 각 chunk가 원본 document의 metadata를 가져야 한다.

구현한 함수:

```python
def chunk_document(document, chunk_size, overlap):
    text_chunks = chunk_text_overlap(
        document["text"],
        chunk_size,
        overlap
    )

    chunks_with_metadata = []

    for i, chunk in enumerate(text_chunks):
        metadata = document["metadata"].copy()
        metadata["chunk_index"] = i

        chunk_dict = {
            "text": chunk,
            "metadata": metadata
        }

        chunks_with_metadata.append(chunk_dict)

    return chunks_with_metadata
```

결과:

```python
{
    "text": "...chunk text...",
    "metadata": {
        "source": "07_attention.md",
        "path": "notes/07_attention.md",
        "chunk_index": 0
    }
}
```

## 왜 `.copy()`를 사용하는가?

```python
metadata = document["metadata"]
```

처럼 그대로 대입하면 새로운 dictionary가 만들어지는 것이 아니라 같은 dictionary 객체를 가리킨다.

그 상태에서:

```python
metadata["chunk_index"] = i
```

를 하면 원본 document metadata까지 수정될 수 있다.

따라서:

```python
metadata = document["metadata"].copy()
```

로 각 chunk용 metadata를 따로 만든다.

---

# 17. `chunk_index`

`enumerate()`를 사용하면 각 chunk의 문서 내부 번호를 쉽게 만들 수 있다.

```python
for i, chunk in enumerate(text_chunks):
```

예:

```text
i = 0 → first chunk
i = 1 → second chunk
i = 2 → third chunk
```

이를 그대로:

```python
metadata["chunk_index"] = i
```

에 사용했다.

중요한 점은 `chunk_index`가 전체 corpus 기준 index가 아니라 **각 문서 내부 index**라는 것이다.

```text
07_attention.md
chunk 0
chunk 1
...

08_transformer_encoder.md
chunk 0
chunk 1
...
```

따라서 chunk를 식별할 때는 보통:

```text
(source, chunk_index)
```

조합을 사용하면 된다.

---

# 18. 여러 파일을 하나의 Corpus로 합치기

실제 RAG에서는 문서 하나가 아니라 여러 문서를 검색 대상으로 사용한다.

이번에는 다음 실제 Markdown 파일들을 사용했다.

```python
file_paths = [
    "notes/07_attention.md",
    "notes/08_transformer_encoder.md",
    "notes/09_transformer_decoder.md"
]
```

구현한 함수:

```python
def load_and_chunk_files(file_paths, chunk_size, overlap):
    all_chunks = []

    for file_path in file_paths:
        document = load_document(file_path)
        chunks = chunk_document(document, chunk_size, overlap)
        all_chunks.extend(chunks)

    return all_chunks
```

여기서 `extend()`를 사용하는 이유는 문서별 chunk list를 하나의 flat list로 합치기 위해서이다.

```python
a = [1, 2]
a.append([3, 4])
# [1, 2, [3, 4]]
```

반면:

```python
a = [1, 2]
a.extend([3, 4])
# [1, 2, 3, 4]
```

이다.

최종 구조:

```python
all_chunks = [
    {
        "text": "...",
        "metadata": {...}
    },
    {
        "text": "...",
        "metadata": {...}
    },
    ...
]
```

---

# 19. 실제 Multi-file Chunking 결과

다음 설정을 사용했다.

```python
all_chunks = load_and_chunk_files(
    file_paths,
    chunk_size=100,
    overlap=20
)
```

실제 결과:

```text
07_attention.md           → 42 chunks
08_transformer_encoder.md → 61 chunks
09_transformer_decoder.md → 50 chunks

Total                    → 153 chunks
```

따라서:

```text
len(all_chunks) = 153
```

이다.

각 chunk에는 text와 metadata가 같이 존재한다.

예:

```python
{
    "text": "...",
    "metadata": {
        "source": "09_transformer_decoder.md",
        "path": "notes/09_transformer_decoder.md",
        "chunk_index": 49
    }
}
```

---

# 20. Embedding용 Text 분리

`all_chunks`에는 text와 metadata가 같이 들어 있지만 SentenceTransformer에는 text만 넣는다.

```python
chunk_texts = [chunk["text"] for chunk in all_chunks]
```

즉:

```text
all_chunks
length = N

↓ text only

chunk_texts
list[str], length = N
```

그다음:

```python
chunk_embeddings = model.encode(
    chunk_texts,
    convert_to_tensor=True
)
```

를 수행한다.

실제 테스트에서는:

```text
N = 153
D = 384
```

이므로:

```text
chunk_embeddings
[153, 384]
```

가 나왔다.

Query는:

```text
query_embedding
[384]
```

였다.

---

# 21. 가장 중요한 Index Alignment

현재 구현에서 매우 중요한 invariant는 다음과 같다.

```text
chunk_embeddings[i]
        ↕
all_chunks[i]
```

예를 들어:

```text
chunk_embeddings[37]
```

이 검색되면 원본 데이터는:

```python
all_chunks[37]
```

에서 가져온다.

따라서 `all_chunks`를 새로 만들거나 file list / chunk_size / overlap을 변경했다면 반드시:

```python
chunk_texts = [chunk["text"] for chunk in all_chunks]
```

도 다시 만들고 embedding도 다시 생성해야 한다.

이 대응 관계가 깨지면 similarity search는 정상적으로 계산되더라도 **잘못된 text나 metadata가 검색 결과에 붙을 수 있다.**

---

# 22. Metadata-aware `retrieve()`

처음 구현한 `retrieve()`는 다음과 같이 text와 score만 반환했다.

```text
(text, score)
```

하지만 실제 파일과 metadata를 사용하게 되면서 retrieval 결과도 원본 chunk dictionary와 연결하도록 확장했다.

최종 함수:

```python
def retrieve(query_embedding, chunk_embeddings, chunks, k=2):
    # chunks = all_chunks

    scores = cosine_similarity_batch(
        query_embedding,
        chunk_embeddings
    )

    top_scores, top_indices = torch.topk(scores, k)

    results = []

    for score, idx in zip(top_scores, top_indices):
        index = idx.item()

        chunk_dict = chunks[index]

        result_dict = {
            "text": chunk_dict["text"],
            "score": score.item(),
            "metadata": chunk_dict["metadata"]
        }

        results.append(result_dict)

    return results
```

출력 구조:

```python
[
    {
        "text": "...",
        "score": 0.5619,
        "metadata": {
            "source": "09_transformer_decoder.md",
            "path": "notes/09_transformer_decoder.md",
            "chunk_index": 41
        }
    },
    ...
]
```

`score.item()`을 사용하는 이유는 PyTorch scalar tensor를 일반 Python `float`로 바꾸기 위해서이다.

---

# 23. 실제 Metadata-aware Retrieval 결과

사용한 query:

```text
How does attention make each token to contextual representation?
```

실제 embedding shape:

```text
chunk_embeddings
[153, 384]

query_embedding
[384]
```

Top-k retrieval 결과 중 하나:

```text
Score ≈ 0.5619
Source: 09_transformer_decoder.md
Chunk index: 41
```

또 다른 결과:

```text
Score ≈ 0.5554
Source: 07_attention.md
Chunk index: 40
```

두 번째 결과에는 다음과 같이 query와 직접적으로 관련된 내용이 포함되어 있었다.

```text
Query로 다른 모든 token의 Key와 비교하고,
해당 Value들을 가중합하면서 각 token을 문맥이 반영된
contextual representation으로 바꾼다.
```

즉 실제 repository의 여러 Markdown 파일을 대상으로 semantic retrieval이 정상적으로 동작했고, 검색된 chunk의 출처까지 추적할 수 있게 되었다.

---

# 24. Retrieval 결과를 사람이 읽기 좋게 출력하기

`results` 전체를 그대로 `print()`하면 dictionary가 길게 이어져 보이기 어렵다.

예를 들어 다음처럼 출력할 수 있다.

```python
def print_results(results):
    for i, result in enumerate(results):
        print(f"\n=== Result {i + 1} ===")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Chunk index: {result['metadata']['chunk_index']}")
        print(f"Text: {result['text']}")
```

출력 예:

```text
=== Result 1 ===
Score: 0.5619
Source: 09_transformer_decoder.md
Chunk index: 41
Text: ...

=== Result 2 ===
Score: 0.5554
Source: 07_attention.md
Chunk index: 40
Text: ...
```

Metadata를 추가한 목적이 여기서 명확해진다.

---

# 25. Transformer Attention과 Retrieval 비교

Transformer에서 attention을 구현할 때:

```text
Q @ K^T
```

를 사용해 token representation 사이의 관계를 계산했다.

Retrieval에서는:

```text
chunk_embeddings @ query_embedding
```

을 통해 query와 chunk representation 사이의 similarity 계산에 필요한 dot product를 구한다.

둘은 같은 알고리즘은 아니지만:

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

# 26. 구현한 Retrieval Pipeline 전체 코드

```python
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch


def cosine_similarity_batch(query, documents):
    # query: [D]
    # documents: [N, D]

    dot_products = documents @ query
    # [N, D] @ [D] → [N]

    q_norm = torch.norm(query)
    # scalar []

    doc_norm = torch.norm(documents, dim=1)
    # [N, D] → [N]

    scores = dot_products / (q_norm * doc_norm)
    # [N]

    return scores


def retrieve(query_embedding, chunk_embeddings, chunks, k=2):
    scores = cosine_similarity_batch(
        query_embedding,
        chunk_embeddings
    )

    top_scores, top_indices = torch.topk(scores, k)

    results = []

    for score, idx in zip(top_scores, top_indices):
        index = idx.item()
        chunk_dict = chunks[index]

        result_dict = {
            "text": chunk_dict["text"],
            "score": score.item(),
            "metadata": chunk_dict["metadata"]
        }

        results.append(result_dict)

    return results


def load_document(file_path):
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    document_dict = {
        "text": text,
        "metadata": {
            "source": path.name,
            "path": str(path)
        }
    }

    return document_dict


def chunk_text_overlap(text, chunk_size, overlap):
    assert 0 <= overlap < chunk_size

    results = []
    words = text.split()

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        result = " ".join(chunk)
        results.append(result)

    return results


def chunk_document(document, chunk_size, overlap):
    text_chunks = chunk_text_overlap(
        document["text"],
        chunk_size,
        overlap
    )

    chunks_with_metadata = []

    for i, chunk in enumerate(text_chunks):
        metadata = document["metadata"].copy()
        metadata["chunk_index"] = i

        chunk_dict = {
            "text": chunk,
            "metadata": metadata
        }

        chunks_with_metadata.append(chunk_dict)

    return chunks_with_metadata


def load_and_chunk_files(file_paths, chunk_size, overlap):
    all_chunks = []

    for file_path in file_paths:
        document = load_document(file_path)
        chunks = chunk_document(
            document,
            chunk_size,
            overlap
        )

        all_chunks.extend(chunks)

    return all_chunks
```

실제 사용:

```python
file_paths = [
    "notes/07_attention.md",
    "notes/08_transformer_encoder.md",
    "notes/09_transformer_decoder.md"
]

all_chunks = load_and_chunk_files(
    file_paths,
    chunk_size=100,
    overlap=20
)

chunk_texts = [chunk["text"] for chunk in all_chunks]

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(
    chunk_texts,
    convert_to_tensor=True
)

query_text = "How does attention make each token to contextual representation?"

query_embedding = model.encode(
    query_text,
    convert_to_tensor=True
)

results = retrieve(
    query_embedding,
    chunk_embeddings,
    all_chunks,
    k=2
)
```

---

# 27. 전체 Retrieval Flow와 Shape

```text
Markdown Files
list[path]

↓ load_document()

Document
{
    text,
    metadata: {source, path}
}

↓ chunk_document()

Chunks
list[dict]
{
    text,
    metadata: {source, path, chunk_index}
}

↓ load_and_chunk_files()

all_chunks
length = N

↓ text extraction

chunk_texts
list[str], length = N

↓ SentenceTransformer.encode()

chunk_embeddings
[N, D]


query_text
str

↓ SentenceTransformer.encode()

query_embedding
[D]


query_embedding [D]
        +
chunk_embeddings [N, D]
        ↓
cosine_similarity_batch()
        ↓
scores
[N]
        ↓
torch.topk(k)
        ↓
top_scores  [k]
top_indices [k]
        ↓
all_chunks[index]
        ↓
results
list[dict]
{
    text,
    score,
    metadata
}
```

실제 테스트에서는:

```text
N = 153
D = 384
k = 2

chunk_embeddings
[153, 384]

query_embedding
[384]

scores
[153]

top_scores
[2]

top_indices
[2]
```

였다.

---

# 28. 현재 구조에서 Metadata와 Embedding의 역할

둘의 역할을 명확히 구분해야 한다.

## Embedding

```text
chunk text
↓
semantic vector
↓
query와 similarity 계산
```

## Metadata

```text
retrieved index
↓
원본 chunk lookup
↓
source / path / chunk_index 확인
```

즉:

> Embedding은 **무엇이 관련 있는지 찾는 역할**이고, metadata는 **찾은 정보가 어디에서 왔는지 추적하는 역할**이다.

---

# 29. 현재 단순 구현의 한계

현재 구현은 retrieval의 핵심 원리를 직접 이해하기 위한 최소 구조이다.

아직 다음과 같은 개선 여지가 있다.

### 1. Word-based Chunking

현재:

```python
text.split()
```

을 사용한다.

따라서 문장, 문단, Markdown heading 같은 구조는 고려하지 않는다.

향후에는:

```text
Sentence-aware Chunking
Paragraph-aware Chunking
Token-based Chunking
Semantic Chunking
```

등을 고려할 수 있다.

### 2. In-memory Embedding Matrix

현재 모든 embedding을 Python / PyTorch memory에 직접 들고 있다.

문서 수가 매우 많아지면 Vector Database나 ANN search가 필요할 수 있다.

### 3. Dense Retrieval Only

현재 semantic embedding + cosine similarity만 사용한다.

향후에는 keyword/BM25와 결합한 Hybrid Search나 reranking 등을 사용할 수 있다.

이 기능들은 기본 Retrieval Pipeline을 이해한 다음 확장할 주제이다.

---

# 30. Checkpoint

이번 Retrieval 파트를 이해했다면 다음 질문에 답할 수 있어야 한다.

1. 왜 긴 document 전체를 하나의 embedding으로 만들지 않고 chunk로 나누는가?
2. Chunk와 Document의 차이는 무엇인가?
3. Overlap은 왜 사용하는가?
4. `chunk_size - overlap`이 step이 되는 이유는 무엇인가?
5. 현재 `chunk_size`는 문자, token, word 중 무엇을 기준으로 하는가?
6. `SentenceTransformer.encode()`는 text를 어떤 형태로 변환하는가?
7. Token embedding과 sentence/chunk embedding은 어떤 차이가 있는가?
8. Chunk가 `N`개이고 embedding dimension이 `D`라면 `chunk_embeddings`의 shape은 무엇인가?
9. Query 하나의 embedding shape은 무엇인가?
10. `documents @ query`가 `[N]` shape이 되는 이유는 무엇인가?
11. `torch.norm(documents, dim=1)`이 `[N]`을 반환하는 이유는 무엇인가?
12. Cosine similarity score는 확률인가?
13. `torch.topk(scores, k)`가 반환하는 두 값은 무엇인가?
14. 왜 `top_indices`를 사용해서 다시 원본 chunk를 가져와야 하는가?
15. `load_document()`는 text 외에 왜 metadata를 함께 반환하는가?
16. `document["metadata"].copy()`가 필요한 이유는 무엇인가?
17. `chunk_index`는 전체 corpus 기준인가, 문서 내부 기준인가?
18. 여러 파일의 chunk를 합칠 때 `append()` 대신 `extend()`를 사용하는 이유는 무엇인가?
19. 왜 SentenceTransformer에는 metadata가 아니라 `chunk["text"]`만 넣는가?
20. `chunk_embeddings[i] ↔ all_chunks[i]` 관계가 왜 중요한가?
21. `all_chunks`를 바꿨는데 기존 `chunk_texts`나 embedding을 그대로 사용하면 어떤 문제가 생길 수 있는가?
22. Metadata-aware retrieval 결과에는 어떤 정보들이 들어가는가?
23. Transformer attention의 similarity 계산과 retrieval similarity 계산은 어떤 점에서 비슷하고 다른가?
24. 현재 구현한 pipeline은 RAG 전체 중 어디까지 구현한 것인가?

---

# 31. 현재 Mini RAG 진행 상태

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
Real Markdown File Loading        ✅
Document Metadata                 ✅
Chunk Metadata                    ✅
Multi-file Loading                ✅
Multi-file Chunk Corpus           ✅
Metadata-aware Retrieval          ✅
Real Repository Retrieval Test    ✅
```

현재 완성한 범위:

```text
Real Markdown Files
↓
File Loading
↓
Metadata
↓
Chunking
↓
Multi-file Corpus
↓
Embedding
↓
Similarity Search
↓
Top-k Retrieval
↓
Text + Score + Source Metadata
```

즉,

> 실제 Markdown 여러 개를 대상으로 동작하는 기본 Dense Retrieval Pipeline을 직접 구현한 상태

이다.

아직 진행하지 않은 부분:

```text
Context Construction
Prompt Construction
LLM Generation
End-to-End RAG
Retrieval Evaluation
Advanced Chunking
Vector Database
Hybrid Search
Reranking
PDF / Other File Loaders
```

다음 단계는 retrieved chunks를 하나의 context 문자열로 조립하는 것이다.

```text
Retrieved Results
↓
build_context(results)
↓
Context String
↓
Context + User Query
↓
LLM Prompt
↓
Generation
```

이 단계부터 Retrieval → Augmentation → Generation이 연결된다.

---

# 32. 한 줄 정리

이번 단계에서 완성한 Retrieval Pipeline은:

> 실제 여러 Markdown 파일을 읽고 metadata와 함께 overlapping chunk로 나눈 뒤, 각 chunk를 semantic embedding으로 변환하고 query embedding과 cosine similarity를 계산하여 가장 관련 있는 Top-k chunk를 원본 source와 chunk 위치 정보까지 함께 검색하는 구조이다.
