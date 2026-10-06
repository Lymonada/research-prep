# Mini RAG — Augmentation and Generation

이 문서는 Mini RAG에서 Retrieval 이후의 **Augmentation → Generation → End-to-End RAG** 흐름을 정리한다.

Retrieval 자체의 구현은 `11_Mini RAG Retrieval Pipeline.md`에서 다루었고, 여기서는 검색된 chunk를 LLM에게 전달하여 실제 답변을 생성하는 과정에 집중한다.

최종적으로 구현한 흐름은 다음과 같다.

```text
Question
↓
Retrieval
↓
Retrieved Chunks
↓
format_context()
↓
Context
↓
build_prompt()
↓
Prompt
↓
generate_answer()
↓
LLM
↓
Answer
```

최종 상태:

```text
Retrieval      ✅
Augmentation   ✅
Generation     ✅
End-to-End     ✅
Grounding Test ✅
```

---

# 1. Augmentation — Retrieved Chunks를 LLM Input으로 바꾸기

Retrieval 결과는 다음과 같은 `list[dict]` 형태이다.

```text
retrieved_chunks
[
    {
        text,
        score,
        metadata
    },
    ...
]
```

하지만 LLM에게 Python object 자체를 전달할 필요는 없다.

`format_context()`를 이용하여 사람이 읽을 수 있는 하나의 문자열로 변환했다.

```text
[1]
Source: 07_attention.md
Chunk: 40
Text: ...

[2]
Source: 08_transformer_encoder.md
Chunk: 35
Text: ...
```

즉 자료형의 변화는:

```text
retrieved_chunks: list[dict]
↓
format_context()
↓
context: str
```

이다.

현재 context에는 다음 정보를 넣는다.

```text
text
source
chunk_index
```

반면 similarity `score`는 context에 넣지 않는다.

```text
text
→ LLM이 실제로 사용할 knowledge

metadata
→ source tracking / debugging / citation

score
→ retrieval ranking / evaluation
```

---

# 2. Context + Question → Prompt

검색된 context와 원래 사용자 질문을 `build_prompt()`에서 결합한다.

현재 prompt의 기본 형태는 다음과 같다.

```text
Answer the question using the provided context.

If the context does not contain enough information,
say that the information is not available in the context.

Context:
...

Question:
...

Answer:
```

따라서 Augmentation 전체는:

```text
Retrieved Chunks
↓
format_context()
↓
Context

Context + Question
↓
build_prompt()
↓
Prompt
```

로 정리할 수 있다.

여기서 중요한 점은 **Augmentation이 별도의 model training 과정이 아니라는 것**이다.

```text
Retrieved Knowledge
+
Original Question
↓
LLM Input
```

즉 pretrained LLM의 parameter를 변경하지 않고 외부에서 검색한 정보를 input context에 추가한다.

---

# 3. Generation — OpenRouter로 LLM 연결하기

Generation에서는 OpenRouter API를 사용했다.

이번 Mini RAG의 목적은 production service를 만드는 것보다:

```text
RAG 구조 이해
+
외부 LLM API 호출 경험
+
End-to-End 동작 검증
```

이었기 때문에 OpenRouter의 무료 LLM endpoint를 사용했다.

API 호출 구조는 다음과 같이 이해했다.

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
↓
answer
```

### API key

OpenRouter API 사용자를 인증하는 secret이다.

Colab에서는 코드에 직접 작성하지 않고 Secrets에 저장했다.

```python
api_key = userdata.get("OPENROUTER_API_KEY")
```

### Client

Python에서 API server와 통신하는 객체이다.

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
```

여기서 `OpenAI()`를 사용한다고 해서 **OpenAI model을 사용한다는 뜻은 아니다.**

```text
OpenAI Python SDK
→ OpenAI-compatible 요청을 보내는 client

base_url
→ 실제 요청을 받을 server = OpenRouter

model
→ 실제 답변을 생성할 LLM
```

즉:

```text
Python
↓
OpenAI SDK
↓
OpenRouter
↓
Selected LLM
```

의 구조이다.

### Messages

현재 Mini RAG에서는 이미 `build_prompt()`가 instruction + context + question을 하나로 만들기 때문에 하나의 user message만 사용했다.

```python
messages = [
    {
        "role": "user",
        "content": prompt
    }
]
```

`role="user"`는 API 사용자의 신원을 뜻하는 것이 아니라 **이 message가 대화에서 user 발화 역할이라는 뜻**이다.

---

# 4. `generate_answer(prompt)`

Generation 단계에서 최종적으로 필요한 핵심은 다음 흐름이다.

```text
prompt
↓
messages
↓
client.chat.completions.create()
↓
response
↓
response.choices[0].message.content
↓
answer
```

실제 구현에서는 최종 answer뿐 아니라 어떤 model이 사용되었는지도 함께 반환했다.

```python
answer = response.choices[0].message.content
model_used = response.model
```

최종 Generation model은:

```text
nvidia/nemotron-3-ultra-550b-a55b:free
```

를 사용했다.

---

# 5. OpenRouter에서 겪은 시행착오

Generation 연결 과정에서 API 자체와 model routing을 구분해서 볼 필요가 있다는 것을 배웠다.

처음 Gemma와 Qwen의 무료 endpoint를 사용했을 때는:

```text
429
```

에러가 발생했다.

이는 RAG 코드의 문제가 아니라 무료 provider의 shared inference pool이 일시적으로 rate limit에 걸린 경우였다.

이후:

```python
model="openrouter/free"
```

를 사용해 보았다.

API 호출은 성공했지만 실제로 선택된 model은:

```text
nvidia/nemotron-3.5-content-safety:free
```

였고 결과는:

```text
User Safety: safe
```

였다.

즉:

```text
API 호출 성공
≠
목적에 맞는 model이 호출됨
```

이라는 점을 확인했다.

그래서 Mini RAG에서는 일반 text generation model인 Nemotron 3 Ultra를 명시적으로 고정했다.

무료 endpoint에서는 간헐적으로 정상적인 `choices`가 만들어지지 않는 경우도 있었다.

실제 service라면:

```text
retry
error handling
empty response check
provider fallback
```

등이 필요하지만, 이번 Mini RAG에서는 API robustness 자체보다 RAG pipeline 이해에 집중했다.

---

# 6. 첫 End-to-End Generation

질문:

```text
How does attention turn each token into a contextual representation?
```

Retrieval 결과의 top chunk에는 다음 내용이 있었다.

```text
Query로 다른 모든 token의 Key와 비교하고,
해당 Value들을 가중합하면서 각 token을
문맥이 반영된 contextual representation으로 바꾼다.
```

이 context를 LLM에게 전달한 결과:

```text
Attention turns each token into a contextual representation
by having the Query of a token compare with the Keys of all
other tokens, and then taking a weighted sum of the
corresponding Values...
```

라는 정상적인 답변이 생성되었다.

따라서 처음으로:

```text
Question
↓
Retrieval
↓
Augmentation
↓
Generation
↓
Answer
```

전체 pipeline이 실제로 연결되었다.

---

# 7. 정말 Retrieved Context를 사용한 것인가?

하지만 Attention에 대한 질문은 일반 LLM도 원래 알고 있을 가능성이 높다.

따라서 좋은 답변이 나왔다고 해서:

```text
LLM이 retrieved context를 사용했다.
```

라고 바로 결론 내릴 수는 없다.

가능성은:

```text
A. Retrieved context를 읽고 답함

B. Pretrained knowledge만으로 답함
```

두 가지이다.

그래서 LLM이 사전에 알 수 없는 **synthetic fact**를 이용하여 별도의 grounding test를 수행했다.

---

# 8. Grounding Test — `Purple Mango 731`

테스트 문서 `rag_test.md`에 다음과 같은 임의의 사실을 추가했다.

```text
The code name for the attention retrieval experiment is
"Purple Mango 731".
```

그리고 질문했다.

```text
What is the code name for the attention retrieval experiment?
```

## RAG OFF

Retrieved context를 전달하지 않고 질문만 model에게 보냈다.

결과:

```text
Purple Mango 731을 알지 못함
→ 일반적인 가능성을 추측
→ 추가 context를 요청
```

즉 pretrained knowledge만으로는 이 값을 알 수 없었다.

## RAG ON

같은 질문을 정상적인 RAG pipeline으로 처리했다.

Retrieval 결과의 top-1:

```text
Source: rag_test.md
Chunk: 0

... the code name ...
"Purple Mango 731"
```

Generation 결과:

```text
The code name for the attention retrieval experiment is
"Purple Mango 731".
```

따라서:

```text
RAG OFF
→ 모름

RAG ON
→ Purple Mango 731
```

이라는 차이를 확인했다.

이 실험을 통해 **retrieved context가 실제 Generation output에 영향을 준다는 것을 직접 검증했다.**

---

# 9. Citation과 Grounding은 다르다

Attention test에서는 LLM이 별도로 요청하지 않았는데도:

```text
(Source: 07_attention.md, Chunk 40)
```

같은 citation을 답변에 붙이는 경우가 있었다.

이는 `format_context()`가 source와 chunk metadata를 제공했기 때문에 LLM이 스스로 생성한 것이다.

하지만 citation을 붙였다는 사실만으로 실제 grounding이 증명되는 것은 아니다.

모델이 이미 알고 있던 내용을 답한 뒤 source 이름만 붙였을 수도 있기 때문이다.

따라서 이번 Mini RAG에서 더 중요한 검증은:

```text
Synthetic document-specific fact
+
RAG OFF / RAG ON comparison
```

이었다.

---

# 10. Indexing과 Query-time을 분리하기

최종 구현에서는 **문서 준비 단계**와 **질문 처리 단계**를 분리했다.

문서가 변하지 않는다면 매 질문마다 모든 chunk embedding을 다시 계산할 필요가 없다.

### Indexing / Preparation — 한 번 실행

```text
Files
↓
Load
↓
Chunk
↓
Chunk Embeddings
↓
준비 완료
```

현재 Mini RAG에서는:

```text
all_chunks
chunk_embeddings
```

을 memory에 유지한다.

### Query-time — 질문할 때마다 실행

```text
Question
↓
Query Embedding
↓
Retrieval
↓
Context
↓
Prompt
↓
Generation
↓
Answer
```

즉 실제 RAG의:

```text
Offline Indexing
vs
Online Query
```

구조를 작은 형태로 구현한 것이다.

---

# 11. `rag_answer(question)`과 최종 Pipeline

질문 이후의 모든 과정을 마지막으로 하나의 함수에 묶었다.

```text
rag_answer(question)
    ↓
Query Embedding
    ↓
retrieve()
    ↓
format_context()
    ↓
build_prompt()
    ↓
generate_answer()
    ↓
Answer
```

최종 사용은 다음처럼 단순해진다.

```python
question = (
    "How does attention turn each token "
    "into a contextual representation?"
)

answer, model_used = rag_answer(question, k=3)
```

최종 Mini RAG 전체 구조:

```text
────────────────────────────
     INDEXING / PREPARATION
────────────────────────────

Markdown Files
↓
Load
↓
Chunk + Metadata
↓
SentenceTransformer
↓
Chunk Embeddings [N, 384]


────────────────────────────
          QUERY-TIME
────────────────────────────

Question
↓
Query Embedding [384]
↓
Cosine Similarity
↓
Top-k Retrieval

        RETRIEVAL
            ↓
Retrieved Chunks
            ↓
format_context()
            ↓
Context
            ↓
build_prompt()

       AUGMENTATION
            ↓
Prompt
            ↓
OpenRouter API
            ↓
LLM

        GENERATION
            ↓
Answer
```

핵심 object 흐름만 압축하면:

```text
all_chunks: list[dict]
↓
chunk_embeddings: [N, 384]

question: str
↓
query_embedding: [384]
↓
scores: [N]
↓
retrieved: list[dict]
↓
context: str
↓
prompt: str
↓
messages: list[dict]
↓
response
↓
answer: str
```

현재 구현은 학습용 Mini RAG이므로:

```text
Word-based chunking
In-memory embeddings
Brute-force cosine similarity
No vector database
No reranker
No hybrid search
Minimal API error handling
```

등의 단순화가 있다.

하지만 핵심 구조는 그대로이다.

```text
Retrieve
→ Augment
→ Generate
```

그리고 `Purple Mango 731` 실험을 통해 retrieved knowledge가 실제 Generation에 사용된다는 것까지 확인했다.

이것으로 Mini RAG의 기본 End-to-End 구현을 완료했다.
