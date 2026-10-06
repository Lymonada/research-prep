from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from openai import OpenAI
from google.colab import userdata

def cosine_similarity_batch(query, documents): # query: [D], documents: [N,D]

    dot_products = documents @ query # dot_products: [N]
    q_norm = torch.norm(query) # a scalar
    doc_norm = torch.norm(documents, dim=1) # doc_norm: [N]
    scores = dot_products / (q_norm * doc_norm) # element-wise division
    return scores


def load_document(file_path): # ex) data/transformer_decoder.md

  path = Path(file_path)
  text = path.read_text(encoding="utf-8")

  document_dict = {
  "text": text,
  "metadata": {
      "source": path.name,
      "path": str(path)
  }
}
  return document_dict # 문서 내용과 metadata를 둘다 가지는 dict를 리턴


def chunk_text_overlap(text, chunk_size, overlap):
  # text는 그냥 긴 str
  assert 0 <= overlap < chunk_size
  results = []
  words = text.split() # 공백기준으로 잘라서 list 반환
  for i in range(0, len(words), chunk_size-overlap):
    chunk = words[i:i + chunk_size]
    result = " ".join(chunk) # 공백 넣어서 붙이기
    results.append(result)
  return results # list of str



def chunk_document(document, chunk_size, overlap):
    # document는 text와 metadata를 같이 가지는 dictionary

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

    return chunks_with_metadata # chunk내용과 metadata를 둘다 가지는 dict들을 리스트에 넣어서 리턴


def load_and_chunk_files(file_paths, chunk_size, overlap):

    all_chunks = []

    for file_path in file_paths:

        # document 로드
        document = load_document(file_path) # 로드해서 text와 metadata를 가지는 dictionary 리턴

        # document를 chunks로 변환
        chunks = chunk_document(document, chunk_size, overlap) # chunk내용과 metadata를 둘다 가지는 dict들을 리스트로 리턴

        # all_chunks에 chunks를 합치기
        all_chunks.extend(chunks)
        # chunks는 list라서 그 안의 값들을 append()하면 []에 감싸진 형태로 들어가기 때문에,
        # extend()로 []를 벗겨서 all_chunks에 넣기.

    return all_chunks # chunks 리스트의 값들(chunk dict들)을 모두 가지는 list


def retrieve(query_embedding, chunk_embeddings, chunks, k=2):# 여기서 chunks는 all_chunks. 즉 chunk_text와 metadata를 가진 dictionary를 가진 리스트

    scores = cosine_similarity_batch(query_embedding, chunk_embeddings)

    top_scores, top_indices = torch.topk(scores, k)

    results = []

    for score, idx in zip(top_scores, top_indices):

        index = idx.item()

        # 1. chunks[index]로 원래 chunk dictionary 찾기
        chunk_dict = chunks[index]

        # 2. result dictionary 만들기

        result_dict = {
          "text": chunk_dict["text"],
          "score": score.item(),
          "metadata": chunk_dict["metadata"]
        }

        # 3. results에 append
        results.append(result_dict)

    return results



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



api_key = userdata.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


def generate_answer(prompt): # prompt는 format_context를 거쳐서 나온 retrieved를 다시 build_prompt로 다듬어서 나온 str

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



def rag_answer(question, k=3): # 질문이 들어올 때마다 필요한 부분만 실행하는 함수


    # 1. question embedding
    query_embedding = embedding_model.encode(question, convert_to_tensor=True)

    # 2. retrieval
    retrieved = retrieve(
        query_embedding,
        chunk_embeddings,
        all_chunks,
        k=k
    )

    # 3. retrieved chunks → context
    context = format_context(retrieved)

    # 4. context + question → prompt
    prompt = build_prompt(question, context)

    # 5. LLM generation
    answer, model_used = generate_answer(prompt)

    return answer, model_used




# 문서준비/인덱싱 단계 - 한번만 실행

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

chunk_texts = [chunk["text"] for chunk in all_chunks]

chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_tensor=True
)


# 질문 처리 단계 — 질문할 때마다 실행

question = "How does attention turn each token into a contextual representation?"

answer, model_used = rag_answer(question, 3) # 한번에 모든 과정을 묶어둬서 깔끔하게 실행 가능하게 함

print(answer)
print("Model used:", model_used)
