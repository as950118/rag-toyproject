import numpy as np
import faiss
import pickle

from embedding import model

def load_chunks(chunk_path="chunks.pkl"):
    try:
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)
        return chunks
    except Exception as e:
        print(f"chunk 파일 로드 실패: {e}")
        return []

if __name__ == "__main__":
    # 1. 쿼리 입력 및 임베딩
    query = "이 회사 연차는 어떻게 신청해?"
    query_embedding = model.encode([query])

    # 2. FAISS 인덱스 로드
    index = faiss.read_index("faiss_index.idx")

    # 3. chunk 리스트 로드
    chunks = load_chunks("chunks.pkl")
    if not chunks:
        print("chunk 리스트를 불러올 수 없습니다.")
        exit(1)

    # 4. 유사한 chunk 검색
    D, I = index.search(np.array(query_embedding), k=3)  # 상위 3개 검색

    print("==== 유사한 chunk top 3 ====")
    for rank, idx in enumerate(I[0], 1):
        print(f"[{rank}위] (index: {idx}, 거리: {D[0][rank-1]:.4f})")
        print(chunks[idx])
        print("-" * 40)