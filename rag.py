from openai import OpenAI
import numpy as np
import faiss
import pickle

from embedding import model  # SentenceTransformer 모델
from get_chunk_from_vector import load_chunks

def search_similar_chunks(query, index_path="faiss_index.idx", chunk_path="chunks.pkl", top_k=3):
    """
    쿼리 임베딩 후 FAISS 인덱스에서 유사한 청크 인덱스와 내용을 반환
    """
    # 쿼리 임베딩
    query_embedding = model.encode([query])

    # FAISS 인덱스 로드
    index = faiss.read_index(index_path)

    # 청크 리스트 로드
    chunks = load_chunks(chunk_path)
    if not chunks:
        print("chunk 리스트를 불러올 수 없습니다.")
        return []

    # 유사한 청크 검색
    D, I = index.search(np.array(query_embedding), k=top_k)
    matched_chunks = [chunks[idx] for idx in I[0]]
    return matched_chunks

def main():
    # 1. 유저 쿼리 입력
    query = input("질문을 입력하세요(이 회사 연차는 어떻게 신청해?): ")

    # 2. 유사한 청크 검색
    matched_chunks = search_similar_chunks(query)

    if not matched_chunks:
        print("유사한 청크를 찾지 못했습니다.")
        return

    # 3. 프롬프트 생성
    context = "\n".join(matched_chunks)
    prompt = f"""
다음 내용을 바탕으로 질문에 대답해줘:

{context}

Q: {query}
A:"""

    # 4. OpenAI GPT 호출
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n=== 답변 ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()