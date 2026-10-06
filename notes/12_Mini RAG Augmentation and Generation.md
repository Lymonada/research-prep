# Mini RAG — Augmentation and Generation

이 문서는 Mini RAG 구현에서 **Retrieval 이후의 Augmentation과 Generation**, 그리고 최종 **End-to-End RAG pipeline**까지 직접 구현한 과정을 정리한다.

이전 단계에서 구현한 Retrieval pipeline은 다음과 같다.

```text
Markdown Files
↓
Load
↓
Chunk
↓
Metadata
↓
Chunk Embeddings
↓
Query Embedding
↓
Cosine Similarity
↓
Top-k Retrieval
↓
Retrieved Chunks
```

이번 단계에서는 retrieval 결과를 LLM이 사용할 수 있는 prompt로 바꾸고, OpenRouter를 통해 실제 LLM에 전달하여 답변을 생성했다.

최종 구현 상태:

```text
Retrieval      ✅
Augmentation   ✅
Generation     ✅
End-to-End     ✅
Grounding Test ✅
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

사용자의 질문과 관련된 document chunk를 검색한다.

```text
Question
↓
Query Embedding
↓
Similarity Search
↓
Top-k Chunks
```

Retrieval 결과는 다음과 같은 Python object이다.

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

### Augmentation

검색된 chunk들을 LLM이 읽기 좋은 하나의 context 문자열로 만들고, 원래 질문과 결합하여 최종 prompt를 만든다.

```text
retrieved_chunks
↓
format_context()
↓
context: str

context + question
↓
build_prompt()
↓
prompt: str
```

### Generation

최종 prompt를 LLM API에 전달하고 실제 자연어 답변을 생성한다.

```text
prompt
↓
messages
↓
OpenRouter API
↓
LLM
↓
response
↓
answer
```

---

# 2. Retrieval 결과에서 LLM Context로

Retrieval의 output은 `list[dict]` 형태이다.

예:

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

LLM에게 이 Python 자료구조 자체를 그대로 넘기기보다 사람이 읽을 수 있는 하나의 문자열로 변환했다.

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

자료형의 변화는 다음과 같다.

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

`enumerate(..., start=1)`을 사용하여 사람이 읽기 쉬운 `[1]`, `[2]`, `[3]` 번호를 붙였다.

---

# 4. 왜 Metadata를 Context에 포함하는가?

각 chunk에는 다음 metadata를 저장했다.

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

이 정보는 다음 용도로 사용할 수 있다.

```text
Source tracking
Citation
Debugging
Retrieval evaluation
```

실제 Generation 테스트에서 LLM이 별도의 지시 없이도 다음과 같은 문구를 답변에 붙이는 경우가 있었다.

```text
(Source: 07_attention.md, Chunk 40)
```

하지만 이 citation은 코드가 강제로 붙인 것이 아니라 **LLM이 context의 metadata를 보고 스스로 생성한 문자열**이다.

따라서 안정적인 citation이 필요하다면 LLM이 생성한 citation을 그대로 신뢰하기보다, 코드가 가진 metadata를 이용해 citation을 직접 구성하거나 출력 형식을 명시적으로 강제하는 편이 더 안전하다.

---

# 5. Similarity Score는 왜 Context에 넣지 않았는가?

Retrieval 결과에는 similarity score도 존재한다.

예:

```text
Chunk A → 0.83
Chunk B → 0.76
Chunk C → 0.68
```

이 score는 top-k ranking에는 중요하지만 LLM에게 반드시 보여줄 필요는 없다.

역할을 구분하면:

```text
text
→ LLM에게 전달할 실제 knowledge

metadata
→ source / citation / debugging

score
→ retrieval ranking / evaluation / debugging
```

따라서 현재 Mini RAG에서는 similarity score를 prompt context에 포함하지 않는다.

---

# 6. Question + Context → Prompt

Context를 만든 다음 원래 질문과 결합하여 최종 prompt를 만든다.

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

전체 흐름:

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

# 7. Prompt의 구조와 Grounding Instruction

최종 prompt는 대략 다음 구조이다.

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

두 핵심 instruction은:

```text
Answer the question using the provided context.
```

그리고:

```text
If the context does not contain enough information,
say that the information is not available in the context.
```

첫 번째는 모델이 retrieved knowledge를 사용하도록 유도하고, 두 번째는 context에 없는 정보를 임의로 생성하는 것을 줄이기 위한 기본적인 grounding instruction이다.

---

# 8. Augmentation End-to-End Test

Transformer 학습 노트 세 개를 사용했다.

```python
file_paths = [
    "notes/07_attention.md",
    "notes/08_transformer_encoder.md",
    "notes/09_transformer_decoder.md"
]
```

Chunking 설정:

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

실제 embedding shape:

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
prompt = build_prompt(query_text, context)
```

이 시점에서 LLM에게 바로 전달할 수 있는 하나의 `prompt: str`이 완성된다.

---

# 9. Query 표현은 Retrieval Ranking에 영향을 준다

처음에는 다음과 같은 표현을 사용했다.

```text
How does attention make each token to contextual representation?
```

이 경우 가장 직접적인 Attention chunk가 아닌 다른 chunk가 top-1으로 검색되었다.

질문을 자연스럽게 바꾸었다.

```text
How does attention turn each token into a contextual representation?
```

그러자 `07_attention.md`의 직접적으로 관련된 chunk가 top-1 결과로 올라왔다.

즉:

```text
Query wording
↓
Query Embedding
↓
Similarity Scores
↓
Retrieval Ranking
```

Embedding 기반 retrieval에서는 사람이 비슷하다고 느끼는 두 질문도 representation이 달라질 수 있고, 따라서 ranking 역시 달라질 수 있다.

---

# 10. Augmentation의 핵심

Augmentation은 별도의 model training 과정이 아니다.

핵심은:

```text
Retrieved Knowledge
+
Original Question
```

을 하나의 LLM input으로 만드는 것이다.

일반적인 LLM 호출:

```text
Question
↓
LLM
↓
Answer
```

RAG:

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

즉 LLM의 parameter를 다시 학습시키는 것이 아니라 **외부에서 검색한 정보를 input context에 추가**한다.

---

# 11. LLM은 Retrieval 과정을 알 필요가 없다

LLM에게 최종적으로 전달되는 것은 text prompt뿐이다.

LLM은 다음 사실을 알 필요가 없다.

```text
SentenceTransformer를 사용했다.
Cosine similarity를 직접 계산했다.
torch.topk를 사용했다.
Markdown 문서를 chunking했다.
```

LLM이 보는 것은 결국:

```text
Instruction
+
Retrieved Context
+
Question
```

뿐이다.

따라서 Retrieval/Augmentation과 Generation은 서로 독립적으로 교체할 수 있다.

```text
Embedding Model 변경 가능
Retrieval Logic 변경 가능
LLM 변경 가능
```

---

# 12. Generation Backend — OpenRouter

Generation에는 OpenRouter를 사용했다.

현재 Mini RAG의 목적은 production service 구축보다:

```text
RAG 구조 이해
+
API 호출 경험
+
End-to-End 검증
```

이므로 무료 model endpoint를 사용했다.

전체 Generation 흐름은:

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

---

# 13. OpenAI SDK와 OpenRouter의 관계

Python에서는 다음 SDK를 사용했다.

```python
from openai import OpenAI
```

여기서 `OpenAI` client를 사용한다고 해서 OpenAI model을 사용한다는 뜻은 아니다.

역할을 분리하면:

```text
OpenAI Python SDK
→ OpenAI-compatible API 형식으로 요청을 보내는 client

base_url
→ 실제 요청을 받을 API server

model
→ 실제로 답변을 생성할 LLM
```

OpenRouter가 OpenAI-compatible API 형식을 지원하므로 다음처럼 사용할 수 있다.

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
```

즉 실제 흐름은:

```text
Python
↓
OpenAI SDK
↓
OpenRouter
↓
Selected LLM
```

이다.

---

# 14. API Key와 Colab Secrets

OpenRouter API key는 코드에 직접 작성하지 않고 Colab Secrets에 저장했다.

```python
from google.colab import userdata

api_key = userdata.get("OPENROUTER_API_KEY")
```

그리고 client 생성 시 사용했다.

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
```

API key는 인증 정보이므로 GitHub repository에 직접 포함하지 않는다.

---

# 15. Messages의 의미

LLM Chat API는 message list를 입력으로 받는다.

현재 Mini RAG에서는 하나의 user message만 사용했다.

```python
messages = [
    {
        "role": "user",
        "content": prompt
    }
]
```

여기서:

```text
role="user"
```

는 API를 호출하는 사람의 신원을 뜻하는 것이 아니라, 해당 message가 대화에서 **user 발화 역할**임을 나타낸다.

기본적인 role은 다음과 같이 이해할 수 있다.

```text
system
→ 모델의 전반적인 행동 규칙

user
→ 사용자 질문 / 요청

assistant
→ 이전 모델 답변
```

현재는 `build_prompt()`가 instruction, context, question을 모두 하나의 prompt로 만들기 때문에 단순히 `role="user"` 하나만 사용했다.

---

# 16. Generation Model 선택 과정

처음에는 여러 free endpoint를 테스트했다.

Gemma와 Qwen free endpoint에서는 일시적인 upstream rate limit으로 `429`가 발생했다.

이후:

```python
model="openrouter/free"
```

도 테스트했다.

하지만 이 router가 일반 generation model 대신:

```text
nvidia/nemotron-3.5-content-safety:free
```

를 선택한 경우가 있었고, 결과는:

```text
User Safety: safe
```

였다.

즉 API 호출 자체는 성공했지만, **RAG answer generation에 적합한 model이 선택된 것은 아니었다.**

따라서 최종 Mini RAG에서는 일반 text generation model을 명시적으로 고정했다.

```python
model="nvidia/nemotron-3-ultra-550b-a55b:free"
```

이 경험을 통해:

```text
API 호출 성공
≠
목적에 맞는 model이 호출됨
```

이라는 점을 확인했다.

---

# 17. `generate_answer(prompt)`

최종 Generation 함수는 다음과 같다.

```python
def generate_answer(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=messages
    )

    answer = response.choices[0].message.content
    model_used = response.model

    return answer, model_used
```

입력:

```text
prompt: str
```

출력:

```text
answer: str
model_used: str
```

내부 흐름:

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

`response` 자체는 단순 문자열이 아니라 API 응답 object이고, 실제 생성 text는:

```python
response.choices[0].message.content
```

에서 꺼낸다.

---

# 18. Free Endpoint에서 발생한 일시적인 응답 실패

무료 inference endpoint를 여러 번 호출하는 과정에서 간헐적으로 다음과 같은 문제가 발생했다.

```text
TypeError: 'NoneType' object is not subscriptable
```

발생 위치:

```python
response.choices[0].message.content
```

일부 호출에서 정상적인 `choices`가 만들어지지 않아 발생한 것으로 보인다.

학습용 Mini RAG에서는 성공한 호출로 pipeline을 검증했지만, 실제 service 수준에서는 다음과 같은 처리가 필요하다.

```text
Retry
Error handling
Empty response check
Provider fallback
```

즉 외부 LLM API의 일시적인 failure는 RAG retrieval logic 자체의 오류와 분리해서 봐야 한다.

---

# 19. 첫 End-to-End Generation Test

질문:

```text
How does attention turn each token into a contextual representation?
```

Retrieval top result에는 다음 내용이 포함되어 있었다.

```text
Query로 다른 모든 token의 Key와 비교하고,
해당 Value들을 가중합하면서 각 token을
문맥이 반영된 contextual representation으로 바꾼다.
```

Generation 결과:

```text
Attention turns each token into a contextual representation
by having the Query of a token compare with the Keys of all
other tokens, and then taking a weighted sum of the
corresponding Values ...
```

즉:

```text
Question
↓
Relevant Chunk Retrieval
↓
Prompt Augmentation
↓
LLM Generation
↓
Natural Language Answer
```

까지 실제로 동작했다.

---

# 20. 쉬운 질문만으로는 RAG Grounding을 증명할 수 없다

Attention 질문은 일반적인 LLM이 pretrained knowledge만으로도 답할 수 있다.

따라서 좋은 답변이 생성되었다고 해서 반드시:

```text
LLM이 retrieved context를 사용했다.
```

라고 결론 내릴 수는 없다.

가능성은 두 가지이다.

```text
A. Retrieved context를 읽고 답변

B. Context를 무시하고 pretrained knowledge만으로 답변
```

이를 구분하기 위해 별도의 synthetic fact test를 수행했다.

---

# 21. Synthetic Fact Grounding Test

LLM이 사전에 알 수 없는 임의의 사실을 테스트 document에 추가했다.

`rag_test.md`:

```text
# Mini RAG Test

In this project, the code name for the attention retrieval experiment is
"Purple Mango 731".

The experiment was created only for testing whether the RAG system actually
uses retrieved context.
```

질문:

```text
What is the code name for the attention retrieval experiment?
```

이 정보는 임의로 만든 값이므로 model의 pretrained knowledge로 알 수 없다.

---

# 22. A/B Test — RAG OFF

먼저 retrieved context를 LLM에게 전달하지 않고 question만 직접 보냈다.

```python
messages = [
    {
        "role": "user",
        "content": "What is the code name for the attention retrieval experiment?"
    }
]
```

모델은 `Purple Mango 731`을 알지 못했고, 해당 표현에는 일반적으로 알려진 하나의 code name이 없다고 답하면서 여러 가능성을 추측했다.

즉:

```text
Question Only
↓
LLM
↓
Purple Mango 731을 알지 못함
```

---

# 23. A/B Test — RAG ON

이번에는 정상적인 RAG pipeline으로 같은 질문을 처리했다.

Retrieval top-1:

```text
Source: rag_test.md
Chunk: 0
Text: ... the code name ... is "Purple Mango 731" ...
```

이 chunk를 context에 포함시켜 LLM에게 전달했다.

최종 답변:

```text
The code name for the attention retrieval experiment is
"Purple Mango 731".
```

결과를 비교하면:

```text
RAG OFF
→ Purple Mango 731을 모름

RAG ON
→ Purple Mango 731이라고 정확히 답함
```

따라서 retrieved context가 실제 Generation output에 영향을 준다는 것을 직접 확인했다.

---

# 24. Citation만으로 Grounding이 증명되지는 않는다

Attention test에서 LLM은 별도 citation instruction 없이 다음과 같은 source 표기를 생성하기도 했다.

```text
(Source: 07_attention.md, Chunk 40)
```

하지만 이것만으로 grounding을 증명할 수는 없다.

LLM이 pretrained knowledge로 답한 뒤 context에서 source 이름만 가져와 붙였을 가능성도 있기 때문이다.

따라서 이번 Mini RAG에서는 다음 검증이 더 중요했다.

```text
Document-specific synthetic fact
+
RAG OFF / RAG ON comparison
```

---

# 25. Indexing 단계와 Query-time 단계를 분리한다

최종 코드 구조에서는 문서 준비와 질문 처리를 분리했다.

문서가 바뀌지 않는다면 매 질문마다 모든 문서를 다시 읽고 모든 chunk embedding을 다시 계산할 필요가 없다.

## Indexing / Preparation — 한 번 실행

```text
Files
↓
Load
↓
Chunk
↓
Chunk Texts
↓
Chunk Embeddings
```

실제 코드:

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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [
    chunk["text"]
    for chunk in all_chunks
]

chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_tensor=True
)
```

## Query-time — 질문마다 실행

```text
Question
↓
Query Embedding
↓
Retrieve
↓
Context
↓
Prompt
↓
Generation
↓
Answer
```

이 분리는 실제 RAG system의:

```text
Offline Indexing
vs
Online Query
```

구조의 작은 형태로 볼 수 있다.

현재 Mini RAG에서는 vector database 대신 `all_chunks`와 `chunk_embeddings`를 memory에 유지한다.

---

# 26. `rag_answer(question)` — End-to-End 함수

질문이 들어온 이후의 모든 과정을 하나의 함수로 묶었다.

```python
def rag_answer(question, k=3):

    # 1. Question embedding
    query_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    # 2. Retrieval
    retrieved = retrieve(
        query_embedding,
        chunk_embeddings,
        all_chunks,
        k=k
    )

    # 3. Retrieved chunks → context
    context = format_context(retrieved)

    # 4. Context + question → prompt
    prompt = build_prompt(question, context)

    # 5. LLM generation
    answer, model_used = generate_answer(prompt)

    return answer, model_used
```

최종 사용법:

```python
question = (
    "How does attention turn each token "
    "into a contextual representation?"
)

answer, model_used = rag_answer(question, 3)

print(answer)
print("Model used:", model_used)
```

즉 사용자는 더 이상 다음 단계를 직접 하나씩 실행할 필요가 없다.

```text
query embedding
retrieve
format_context
build_prompt
generate_answer
```

대신:

```python
rag_answer(question)
```

하나로 query-time pipeline 전체를 실행할 수 있다.

---

# 27. 최종 End-to-End Flow

최종 Mini RAG의 전체 흐름은 다음과 같다.

```text
────────────────────────────────
        INDEXING / PREPARATION
────────────────────────────────

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


────────────────────────────────
             QUERY
────────────────────────────────

Question
      ↓
Query Embedding [384]
      ↓
Cosine Similarity
      ↓
torch.topk
      ↓

────────────────────────────────
          RETRIEVAL
────────────────────────────────

Retrieved Chunks
      ↓
format_context()
      ↓
Context String
      ↓
Context + Question
      ↓
build_prompt()
      ↓

────────────────────────────────
         AUGMENTATION
────────────────────────────────

Prompt String
      ↓
messages
      ↓
OpenRouter API
      ↓
Nemotron 3 Ultra
      ↓
response
      ↓
response.choices[0].message.content
      ↓

────────────────────────────────
          GENERATION
────────────────────────────────

Answer
```

코드 기준으로 압축하면:

```text
load_and_chunk_files()
↓
SentenceTransformer.encode(chunks)
↓
[initialization complete]

rag_answer(question)
    ↓
embedding_model.encode(question)
    ↓
retrieve()
    ↓
format_context()
    ↓
build_prompt()
    ↓
generate_answer()
    ↓
answer
```

---

# 28. Tensor / Python Object Flow

이번 Mini RAG에서 중요한 object와 shape을 연결하면:

```text
Markdown files
↓
documents: dict
↓
all_chunks: list[dict]
↓
chunk_texts: list[str]
↓
chunk_embeddings: [N, 384]

question: str
↓
query_embedding: [384]

chunk_embeddings [N,384]
query_embedding [384]
↓
cosine_similarity_batch()
↓
scores [N]
↓
torch.topk()
↓
top scores / indices

↓
retrieved: list[dict]
↓
context: str
↓
prompt: str
↓
messages: list[dict]
↓
response object
↓
answer: str
```

Transformer 구현에서 tensor shape을 따라갔다면, Mini RAG에서는 **tensor와 Python object가 서로 어떻게 변환되는지**를 따라가는 것이 중요했다.

---

# 29. 최종 구현 상태

```text
Document Loading                  ✅
Word-overlap Chunking            ✅
Metadata                          ✅
SentenceTransformer Embedding     ✅
Batch Cosine Similarity           ✅
Top-k Retrieval                   ✅
Metadata-aware Retrieval          ✅

format_context()                  ✅
build_prompt()                    ✅
Final Prompt Construction         ✅

OpenRouter API Key                ✅
OpenAI-compatible Client          ✅
Messages                          ✅
Generation Model                  ✅
generate_answer()                 ✅
Response Parsing                  ✅

Pipeline Test                     ✅
Synthetic Fact Grounding Test     ✅
RAG OFF / RAG ON Comparison       ✅
Indexing / Query Separation       ✅
rag_answer() End-to-End Function  ✅
```

---

# 30. 무엇을 직접 구현했고 무엇을 Library에 맡겼는가?

직접 구현:

```text
Document loading structure
Word-overlap chunking
Metadata propagation
Batch cosine similarity
Top-k result mapping
Retrieval result structure
Context formatting
Prompt construction
Generation wrapper
End-to-End query pipeline
```

Library / API 사용:

```text
SentenceTransformer
→ text embedding

torch.topk
→ top-k index selection

OpenRouter + external LLM
→ final text generation
```

즉 embedding model과 generation model 자체를 학습한 것은 아니지만, **RAG를 구성하는 data flow와 glue logic은 직접 구현했다.**

---

# 31. 현재 Mini RAG의 단순화된 부분

현재 구현은 RAG의 핵심 구조를 이해하기 위한 Mini version이므로 다음과 같은 단순화가 있다.

```text
Word-based chunking
In-memory embeddings
Brute-force cosine similarity
No vector database
No reranker
No hybrid search
No persistent index
No conversation memory
Minimal prompt
Minimal API error handling
```

하지만 핵심 구조는 실제 RAG와 동일하다.

```text
Index documents
↓
Embed
↓
Retrieve relevant context
↓
Augment prompt
↓
Generate answer
```

따라서 이후 FAISS, vector DB, reranker, better chunking 등을 배우더라도 현재 구현의 각 부분을 더 발전된 component로 교체하는 방식으로 이해할 수 있다.

---

# 32. Core Takeaways

## Retrieval

```text
"무슨 정보를 가져올 것인가?"
```

Question을 embedding하고 document chunk와 비교하여 관련 정보를 찾는다.

## Augmentation

```text
"가져온 정보를 LLM에게 어떻게 줄 것인가?"
```

Retrieved chunks를 context 문자열로 만들고 original question과 함께 prompt를 구성한다.

## Generation

```text
"그 정보를 바탕으로 어떤 답변을 만들 것인가?"
```

Prompt를 외부 LLM API에 전달하고 response에서 answer text를 추출한다.

최종적으로:

```text
Question
↓
Retrieve
↓
Augment
↓
Generate
↓
Answer
```

를 직접 구현하고 실제 document-specific synthetic fact를 이용해 **retrieved context가 Generation에 실제로 영향을 준다는 것까지 확인했다.**

이것으로 Mini RAG의 기본 End-to-End 구현을 완료했다.
