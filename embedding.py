from sentence_transformers import SentenceTransformer
from chunking import chunk_text_from_file

model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 한국어 지원

def get_embeddings(chunks, model_name='jhgan/ko-sroberta-multitask'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

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