# Mini RAG — Augmentation and Generation

이 문서는 Mini RAG 구현에서 **Retrieval 이후의 흐름**을 정리한다.

이전 단계에서는 다음 과정을 구현했다.

```text
Documents
↓
Load
↓
Chunk
↓
Embedding
↓
Query Embedding
↓
Cosine Similarity
↓
Top-k Retrieval
↓
Retrieved Chunks
```

이번 단계에서는 retrieval 결과를 실제 LLM이 사용할 수 있도록 변환하는 **Augmentation**을 구현하고, 이후 **Generation**을 연결하기 위한 구조를 정리한다.

현재 구현 상태는 다음과 같다.

```text
Retrieval      ✅
Augmentation   ✅
Generation     ⏭
```

---

# 1. RAG의 전체 구조

RAG는 크게 세 부분으로 나눌 수 있다.

```text
Retrieval
↓
Augmentation
↓
Generation
```

각 단계의 역할은 다음과 같다.

### Retrieval

사용자의 질문과 관련된 문서 chunk를 검색한다.

```text
Question
↓
Query Embedding
↓
Similarity Search
↓
Top-k Chunks
```

결과는 다음과 같은 형태의 Python object이다.

```python
retrieved_chunks = [
    {
        "text": "...",
        "score": ...,
        "metadata": {
            "source": "...",
            "path": "...",
            "chunk_index": ...
        }
    },
    ...
]
```

---

### Augmentation

Retrieval 결과를 LLM이 읽을 수 있는 **context 문자열**로 바꾸고, 원래 질문과 결합하여 최종 prompt를 만든다.

```text
retrieved_chunks
↓
format_context()
↓
context

context + question
↓
build_prompt()
↓
prompt
```

---

### Generation

최종 prompt를 LLM에게 전달하여 답변을 생성한다.

```text
prompt
↓
LLM
↓
answer
```

Generation은 아직 구현하지 않았으며 이후 OpenRouter API를 이용하여 연결할 예정이다.

---

# 2. Retrieval 결과에서 LLM Context로

Retrieval 단계의 output은 `list[dict]` 형태이다.

예를 들어:

```python
[
    {
        "text": "Query로 다른 모든 token의 Key와 비교하고 ...",
        "score": 0.81,
        "metadata": {
            "source": "07_attention.md",
            "path": "notes/07_attention.md",
            "chunk_index": 40
        }
    },
    ...
]
```

하지만 LLM에게 이 Python 자료구조 자체를 그대로 전달할 필요는 없다.

LLM이 읽기 편한 하나의 문자열로 변환한다.

```text
[1]
Source: 07_attention.md
Chunk: 40
Text: Query로 다른 모든 token의 Key와 비교하고 ...

[2]
Source: 08_transformer_encoder.md
Chunk: 35
Text: token들은 이미 다른 token의 정보를 참고한 ...

...
```

즉 자료형의 변화는 다음과 같다.

```text
list[dict]
↓
format_context()
↓
str
```

---

# 3. `format_context()`

Retrieval 결과를 하나의 context 문자열로 변환하기 위해 다음 함수를 구현했다.

```python
def format_context(retrieved_chunks):
    formatted_list = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk["text"]
        metadata = chunk["metadata"]

        source = metadata["source"]
        chunk_index = metadata["chunk_index"]

        formatted_text = (
            f"[{i}]\n"
            f"Source: {source}\n"
            f"Chunk: {chunk_index}\n"
            f"Text: {text}"
        )

        formatted_list.append(formatted_text)

    context = "\n\n".join(formatted_list)

    return context
```

입력:

```text
retrieved_chunks: list[dict]
```

출력:

```text
context: str
```

---

# 4. 왜 Metadata를 Context에 포함하는가?

각 chunk에는 다음과 같은 metadata를 저장했다.

```python
metadata = {
    "source": ...,
    "path": ...,
    "chunk_index": ...
}
```

현재 context에서는 이 중

```text
source
chunk_index
```

를 사용한다.

예:

```text
[1]
Source: 07_attention.md
Chunk: 40
Text: ...
```

이 정보는 단순히 보기 좋게 만들기 위한 것이 아니다.

향후 LLM이 답변을 생성할 때 어느 문서의 어느 chunk를 참고했는지 추적할 수 있고, 다음과 같은 citation 형태로 확장할 수도 있다.

```text
Self-Attention은 각 token이 다른 token의 정보를 참고하여
contextual representation을 만든다. [1]
```

따라서 metadata는 다음 목적에 사용할 수 있다.

```text
Source tracking
Citation
Debugging
Retrieval evaluation
```

---

# 5. Similarity Score는 왜 Context에 넣지 않았는가?

Retrieval 결과에는 각 chunk의 similarity score가 존재한다.

예:

```text
Chunk A → 0.83
Chunk B → 0.76
Chunk C → 0.68
```

이 score는 어떤 chunk를 top-k 결과로 선택할지 결정하는 데 중요하다.

하지만 LLM에게

```text
Similarity Score: 0.83
```

을 알려줄 필요는 일반적으로 없다.

각 정보의 역할을 구분하면 다음과 같다.

```text
text
→ LLM에게 전달할 실제 knowledge

metadata
→ source / citation / debugging

score
→ retrieval ranking / evaluation / debugging
```

따라서 현재 Mini RAG에서는 similarity score를 prompt의 context에 포함하지 않는다.

---

# 6. Context가 길어지는 것은 정상이다

현재 chunking 설정은 예를 들어 다음과 같다.

```python
chunk_size = 100
overlap = 20
```

그리고 top-k가 3이라면 최대 약 300 words 정도의 text가 context에 들어갈 수 있다.

따라서 `print(context)`를 했을 때 상당히 긴 문자열이 출력되는 것은 정상이다.

```text
Top-1 chunk ~100 words
Top-2 chunk ~100 words
Top-3 chunk ~100 words
```

사람이 터미널에서 보기에는 다소 길어 보일 수 있지만 LLM에게 context를 전달하는 목적에서는 문제가 없다.

출력을 깔끔하게 만들기 위해 chunk size를 지나치게 작게 줄일 필요는 없다.

---

# 7. Question + Context → Prompt

Context를 만든 다음 원래 사용자 질문과 결합하여 최종 prompt를 만든다.

이를 담당하는 함수가 `build_prompt()`이다.

```python
def build_prompt(question, context):
    prompt = (
        "Answer the question using the provided context.\n\n"
        "If the context does not contain enough information, "
        "say that the information is not available in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    return prompt
```

입력:

```text
question: str
context: str
```

출력:

```text
prompt: str
```

전체 변화는 다음과 같다.

```text
retrieved_chunks
↓
format_context()
↓
context: str
          \
           → build_prompt()
          /
question: str
↓
prompt: str
```

---

# 8. 최종 Prompt의 형태

현재 prompt는 대략 다음 구조를 가진다.

```text
Answer the question using the provided context.

If the context does not contain enough information,
say that the information is not available in the context.

Context:
[1]
Source: ...
Chunk: ...
Text: ...

[2]
Source: ...
Chunk: ...
Text: ...

Question:
...

Answer:
```

여기서 중요한 부분은 다음 두 instruction이다.

```text
Answer the question using the provided context.
```

그리고

```text
If the context does not contain enough information,
say that the information is not available in the context.
```

첫 번째 instruction은 모델이 retrieval 결과를 사용하도록 유도한다.

두 번째 instruction은 context에 없는 정보를 모델이 임의로 만들어내는 것을 줄이기 위한 기본적인 grounding instruction이다.

---

# 9. Triple-Quoted String과 Prompt Formatting

처음에는 다음과 같이 triple-quoted string을 사용할 수도 있다.

```python
prompt = f"""
    Answer the question ...

    Context:
    {context}
"""
```

하지만 Python에서는 코드의 indentation이 실제 문자열에 포함될 수 있다.

따라서 prompt formatting을 명확하게 통제하기 위해 다음과 같이 명시적으로 newline을 사용하는 방식을 사용했다.

```python
prompt = (
    "...\n\n"
    f"Context:\n{context}\n\n"
    f"Question:\n{question}\n\n"
    "Answer:"
)
```

LLM이 indentation이 있는 prompt를 읽지 못하는 것은 아니지만, prompt 자체를 확인하고 디버깅하기에는 이 방식이 더 명확하다.

---

# 10. Augmentation End-to-End Test

다음 세 개의 Transformer 학습 노트를 사용하여 테스트했다.

```python
file_paths = [
    "notes/07_attention.md",
    "notes/08_transformer_encoder.md",
    "notes/09_transformer_decoder.md"
]
```

문서를 load하고 chunking했다.

```python
all_chunks = load_and_chunk_files(
    file_paths,
    chunk_size=100,
    overlap=20
)
```

Embedding model:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

질문:

```python
query_text = (
    "How does attention turn each token "
    "into a contextual representation?"
)
```

Chunk text만 추출했다.

```python
chunk_texts = [
    chunk["text"]
    for chunk in all_chunks
]
```

Embedding:

```python
chunk_embeddings = model.encode(
    chunk_texts,
    convert_to_tensor=True
)

query_embedding = model.encode(
    query_text,
    convert_to_tensor=True
)
```

실제 shape:

```text
chunk_embeddings: [153, 384]
query_embedding:   [384]
```

Retrieval:

```python
retrieved = retrieve(
    query_embedding,
    chunk_embeddings,
    all_chunks,
    k=3
)
```

Augmentation:

```python
context = format_context(retrieved)

prompt = build_prompt(
    query_text,
    context
)
```

최종적으로 LLM에게 바로 전달 가능한 하나의 `prompt: str`을 만드는 데 성공했다.

---

# 11. Query 표현이 Retrieval 결과에 영향을 준다

처음 테스트에서는 질문을 다음과 같이 작성했다.

```text
How does attention make each token to contextual representation?
```

이 경우 가장 직접적인 Attention 문서가 아닌 Decoder 관련 chunk가 top-1으로 검색되었다.

질문을 다음처럼 자연스럽게 바꾸었다.

```text
How does attention turn each token into a contextual representation?
```

그러자 `07_attention.md`의 직접적으로 관련된 chunk가 top-1 결과로 올라왔다.

이를 통해 다음 흐름을 확인할 수 있었다.

```text
Query wording
↓
Query Embedding
↓
Similarity Scores
↓
Retrieval Ranking
```

Embedding 기반 retrieval에서는 사람이 같은 의미라고 느끼는 두 질문도 embedding representation이 달라질 수 있기 때문에 검색 결과가 달라질 수 있다.

또한 embedding retrieval의 ranking은 항상 사람이 판단하는 완벽한 relevance ordering과 일치하지 않는다.

---

# 12. Augmentation의 핵심

Augmentation은 복잡한 학습 과정이 아니다.

핵심은:

```text
Retrieved Knowledge
+
Original Question
```

을 하나의 LLM input으로 만드는 것이다.

즉 일반적인 LLM 호출이

```text
Question
↓
LLM
↓
Answer
```

이라면 RAG에서는 다음과 같이 바뀐다.

```text
Question
↓
Retrieve Relevant Knowledge
↓
Question + Retrieved Knowledge
↓
LLM
↓
Answer
```

LLM 자체를 다시 학습시키거나 parameter를 변경하는 것이 아니다.

외부에서 검색한 정보를 **LLM의 input context에 추가**한다.

---

# 13. LLM은 Retrieval 과정 자체를 알지 못한다

LLM에게 최종적으로 전달되는 것은 단순한 text prompt이다.

LLM은 다음 사실을 알 필요가 없다.

```text
SentenceTransformer를 사용했다.
Cosine similarity를 계산했다.
torch.topk를 사용했다.
Markdown 문서를 chunking했다.
```

LLM이 받는 것은 결국 다음뿐이다.

```text
Instruction
+
Context
+
Question
```

즉 RAG의 Retrieval / Augmentation logic과 LLM Generation은 서로 분리되어 있다.

```text
Retrieval System
    ↓
Relevant Context

        +

User Question
    ↓
Prompt
    ↓
LLM
```

이 분리는 이후 다른 embedding model이나 다른 LLM을 사용하더라도 전체 pipeline을 쉽게 교체할 수 있게 해준다.

---

# 14. 쉬운 질문만으로는 RAG를 검증하기 어렵다

현재 테스트 질문은 다음과 같다.

```text
How does attention turn each token
into a contextual representation?
```

하지만 일반적인 LLM은 retrieval context가 없어도 이 질문에 대답할 수 있다.

따라서 좋은 답변이 생성되었다고 해서 반드시

```text
LLM이 retrieved context를 사용했다.
```

라고 결론 내릴 수는 없다.

가능성은 두 가지이다.

```text
A. Retrieved context를 읽고 답변

B. Pretrained knowledge만 사용해서 답변
```

최종 답변만 보고 두 경우를 완전히 구분하기 어렵다.

---

# 15. Pipeline Test와 RAG Test를 구분한다

Generation을 구현한 뒤 테스트를 두 단계로 나눌 예정이다.

## Test 1 — Pipeline Test

현재 Attention 질문처럼 정답을 쉽게 확인할 수 있는 질문을 사용한다.

목적은 다음 전체 흐름이 오류 없이 실행되는지 확인하는 것이다.

```text
Question
↓
Embedding
↓
Retrieval
↓
Augmentation
↓
LLM API
↓
Generation
↓
Answer
```

이 테스트는 RAG의 grounding 자체를 증명하기 위한 것은 아니다.

---

## Test 2 — Grounding Test

LLM이 원래 알고 있을 수 없는, 문서 내부에만 존재하는 구체적인 정보를 질문한다.

예를 들어 직접 구현한 Transformer의 dummy test에 관한 질문:

```text
According to the notes,
what were the tensor shapes of the
self-attention and cross-attention weights
in the Decoder dummy test?
```

일반적인 Transformer shape인

```text
[B, h, T, T]
[B, h, T, S]
```

는 LLM이 알고 있을 수 있다.

하지만 실제 구현에서 사용한

```text
[2, 2, 5, 5]
[2, 2, 5, 7]
```

같은 구체적인 값은 개인 노트의 context를 읽어야 알 수 있다.

더 강한 테스트를 위해 문서 안에 임의의 사실을 넣는 방법도 있다.

```text
The internal experiment codename is Pineapple-47.
```

그리고 다음과 같이 질문한다.

```text
What is the internal experiment codename?
```

정상적인 결과는 다음과 같다.

```text
RAG ON
→ Pineapple-47

RAG OFF
→ 알 수 없음 / 추측
```

---

# 16. RAG ON vs RAG OFF

Grounding을 확인하기 위해 같은 질문을 두 가지 방식으로 LLM에 전달할 수 있다.

### RAG ON

```text
Retrieved Context
+
Question
↓
LLM
```

### RAG OFF

```text
Question Only
↓
LLM
```

그리고 두 답변을 비교한다.

```text
                    RAG ON      RAG OFF

Document-specific
information          O             X

General knowledge    O             O
```

이 비교를 통해 retrieval context가 실제 답변에 어떤 영향을 주는지 직관적으로 확인할 수 있다.

---

# 17. Citation만으로 Grounding이 증명되지는 않는다

Context에 `[1]`, `[2]`, `[3]`처럼 번호를 붙이면 나중에 LLM에게 source citation을 생성하도록 할 수 있다.

예:

```text
Attention combines information from other tokens
using a weighted sum of their Values. [1]
```

하지만 citation이 붙어 있다는 사실만으로 LLM이 실제로 그 source를 기반으로 reasoning했다고 완전히 증명할 수는 없다.

LLM이 이미 알고 있는 내용을 답하면서 단순히 `[1]`을 붙일 수도 있기 때문이다.

따라서 현재 Mini RAG에서는 citation 자체보다

```text
Document-specific question
+
RAG ON / OFF comparison
```

을 grounding 확인 방법으로 사용한다.

---

# 18. Generation Backend 선택

Generation을 위해 외부 LLM API를 사용해야 한다.

처음에는 OpenAI API를 고려했지만 Mini RAG의 현재 목적은 production system 구축이 아니라

```text
RAG 구조 이해
+
API 호출 경험
+
간단한 Generation 테스트
```

이다.

따라서 별도의 API 비용을 사용하는 대신 **OpenRouter의 무료 모델**을 이용하기로 결정했다.

현재 계획은 다음과 같다.

```text
Embedding
SentenceTransformer
all-MiniLM-L6-v2
        ↓
Retrieval
        ↓
Augmentation
        ↓
OpenRouter API
        ↓
Free LLM
        ↓
Answer
```

---

# 19. Generation Model은 고정한다

OpenRouter에는 사용 가능한 무료 모델 중 하나를 자동으로 선택하는 방식도 있지만, Mini RAG 실험에서는 특정 무료 모델 하나를 고정하여 사용할 예정이다.

이유는 테스트 재현성 때문이다.

모델이 매번 바뀌면:

```text
Run 1 → Model A
Run 2 → Model B
Run 3 → Model C
```

답변 차이가 발생했을 때 원인을 구분하기 어렵다.

```text
Retrieval 차이인가?
Prompt 차이인가?
Generation model 차이인가?
```

따라서 Mini RAG 테스트에서는 가능한 한 다음 조건을 고정한다.

```text
Embedding model
Retrieval logic
Chunking settings
Prompt
Generation model
```

그리고 필요한 요소 하나씩만 변경하여 결과를 비교한다.

---

# 20. Generation에서 구현할 함수

다음 단계에서는 다음 함수를 구현한다.

```python
def generate_answer(prompt):
    ...
```

입력:

```text
prompt: str
```

출력:

```text
answer: str
```

개념적인 내부 흐름은 다음과 같다.

```text
prompt
↓
API Request
↓
LLM
↓
Response Object
↓
Generated Text
↓
answer
```

---

# 21. OpenRouter API에서 배울 요소

다음 구현 단계에서는 OpenRouter를 기준으로 다음 요소를 먼저 이해한다.

```text
API key
↓
client
↓
messages
↓
model
↓
response
```

### API key

API 요청을 보내는 사용자를 인증하기 위한 값.

### Client

Python 프로그램에서 OpenRouter API와 통신하기 위한 객체.

### Messages

LLM에게 전달할 대화 입력.

현재 만들어둔 `prompt`가 실제 API request에 들어가는 부분이다.

### Model

Generation에 사용할 LLM을 지정한다.

Mini RAG 테스트에서는 특정 무료 모델을 하나 고정할 예정이다.

### Response

LLM API 호출 결과를 담고 있는 object.

이 response 안에서 실제 생성된 answer text를 추출한다.

이 구조를 이해한 뒤 `generate_answer(prompt)`를 직접 구현할 예정이다.

---

# 22. 현재 Mini RAG Pipeline

현재까지 구현된 전체 흐름은 다음과 같다.

```text
Markdown Files
      ↓
load_document()
      ↓
chunk_document()
      ↓
Chunks + Metadata
      ↓
SentenceTransformer
      ↓
Chunk Embeddings [N, 384]

Question
      ↓
Query Embedding [384]
      ↓
Cosine Similarity
      ↓
Top-k
      ↓

────────────────────────
       RETRIEVAL
────────────────────────

Retrieved Chunks
      ↓
format_context()
      ↓
Context String

Context + Question
      ↓
build_prompt()
      ↓
Prompt String

────────────────────────
      AUGMENTATION
────────────────────────

Prompt
      ↓
OpenRouter API
      ↓
LLM
      ↓
Answer

────────────────────────
       GENERATION
      (Next Step)
────────────────────────
```

---

# 23. Current Status

현재 완료:

```text
Document Loading                  ✅
Chunking                          ✅
Metadata                          ✅
Embedding                         ✅
Cosine Similarity                 ✅
Top-k Retrieval                   ✅
Metadata-aware Retrieval          ✅

format_context()                  ✅
build_prompt()                    ✅
Final Prompt Construction         ✅

Generation Backend Selection      ✅
OpenRouter API Understanding      ⏭
generate_answer()                 ⏭
End-to-End Generation Test        ⏭
RAG ON / OFF Grounding Test       ⏭
```

---

# 24. Next Step

다음 단계에서는 OpenRouter API를 연결한다.

먼저 다음 구조를 이해한다.

```text
API key
→ client
→ messages
→ model
→ response
```

그 후 직접 다음 함수를 구현한다.

```python
def generate_answer(prompt):
    ...
```

최종적으로:

```python
retrieved = retrieve(...)

context = format_context(retrieved)

prompt = build_prompt(
    question,
    context
)

answer = generate_answer(prompt)

print(answer)
```

까지 연결하면 최초의 end-to-end Mini RAG pipeline이 완성된다.

그 후:

```text
Pipeline Test
↓
Document-specific Question
↓
RAG ON vs RAG OFF
```

순서로 실제 retrieval context가 Generation에 미치는 영향을 확인할 예정이다.

---

# 25. Core Takeaway

현재 단계에서 가장 중요한 흐름은 다음과 같다.

```text
Retrieval은
"무슨 정보를 가져올 것인가?"

Augmentation은
"가져온 정보를 LLM에게 어떻게 줄 것인가?"

Generation은
"그 정보를 바탕으로 어떤 답변을 만들 것인가?"
```

이를 코드 기준으로 압축하면:

```text
retrieve()
↓
format_context()
↓
build_prompt()
↓
generate_answer()
```

현재는 앞의 세 단계까지 구현되었고, 다음 단계에서 `generate_answer()`를 연결하면 Mini RAG의 기본 구조가 완성된다.
