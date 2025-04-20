from sentence_transformers import SentenceTransformer
from chunking import chunk_text_from_file

model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 한국어 지원

if __name__ == "__main__":
    filename = "취업규칙.txt"
    chunks = chunk_text_from_file(filename, chunk_size=500, chunk_overlap=100)

    if not chunks:
        print("chunk 생성 실패 또는 파일 없음.")
        exit(1)

    print(f"총 {len(chunks)}개의 chunk 생성됨.")

    embeddings = model.encode(chunks, show_progress_bar=True)

    print(f"임베딩 shape: {embeddings.shape}")
    print("첫 번째 chunk 임베딩 벡터:", embeddings[0])
