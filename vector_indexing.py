import faiss
import numpy as np
from chunking import chunk_text_from_file
from embedding import get_embeddings

def create_faiss_index(embeddings):
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    filename = "취업규칙.txt"
    # 1. 텍스트를 chunking
    chunks = chunk_text_from_file(filename, chunk_size=500, chunk_overlap=100)
    if not chunks:
        print("chunk 생성 실패 또는 파일 없음.")
        exit(1)

    print(f"총 {len(chunks)}개의 chunk 생성됨.")

    # 2. 임베딩 생성
    embeddings = get_embeddings(chunks)
    print(f"임베딩 shape: {embeddings.shape}")

    # 3. FAISS 인덱스 생성 및 벡터 추가
    index = create_faiss_index(embeddings)

    print(f"FAISS 인덱스에 {embeddings.shape[0]}개 벡터 추가 완료.")

    # (Optional) 인덱스 저장
    faiss.write_index(index, "faiss_index.idx")
    print("FAISS 인덱스가 faiss_index.idx 파일로 저장되었습니다.")

