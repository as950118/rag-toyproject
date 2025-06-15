import requests
import numpy as np
import faiss
from embedding import model
from get_chunk_from_vector import load_chunks

def get_ollama_embedding(text, model_name="nomic-embed-text"):
    """Ollama를 통해 임베딩 생성"""
    url = "http://localhost:11434/api/embeddings"

    data = {
        "model": model_name,
        "prompt": text
    }

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return np.array(result.get('embedding', []))
    except Exception as e:
        print(f"임베딩 생성 실패: {e}")
        return None

def search_similar_chunks_ollama(query, index_path="faiss_index.idx", chunk_path="chunks.pkl", top_k=3):
    """
    Ollama 임베딩을 사용해 유사한 청크 검색
    """
    # 쿼리 임베딩
    query_embedding = get_ollama_embedding(query)
    if query_embedding is None:
        return []

    # FAISS 인덱스 로드
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        print(f"FAISS 인덱스를 로드할 수 없습니다: {e}")
        return []

    # 청크 리스트 로드
    chunks = load_chunks(chunk_path)
    if not chunks:
        print("chunk 리스트를 불러올 수 없습니다.")
        return []

    try:
        # 유사한 청크 검색
        D, I = index.search(query_embedding.reshape(1, -1), k=min(top_k, len(chunks)))
        matched_chunks = [chunks[idx] for idx in I[0] if idx < len(chunks)]
        return matched_chunks
    except Exception as e:
        print(f"유사 검색 실패: {e}")
        return []

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


def call_ollama_llm(prompt, model_name="qwen2.5:7b"):
    """
    Ollama를 통해 로컬 한글 LLM 호출
    """
    url = "http://localhost:11434/api/generate"

    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000
        }
    }

    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '응답을 받을 수 없습니다.')
    except requests.exceptions.ConnectionError:
        return "Ollama 서버에 연결할 수 없습니다. 'ollama serve' 명령으로 서버를 시작해주세요."
    except requests.exceptions.Timeout:
        return "응답 시간이 초과되었습니다. 다시 시도해주세요."
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

def main():
    print("=== Ollama 전용 한글 LLM RAG 시스템 ===")
    print("사용 전 준비사항:")
    print("1. Ollama 설치: brew install ollama")
    print("2. 모델 다운로드:")
    print("   - ollama pull qwen2.5:7b        (LLM, 한중일 언어)")
    print("   - ollama pull nomic-embed-text  (임베딩)")
    print("3. Ollama 서버 시작: ollama serve")
    print("4. Python 패키지: pip install requests faiss-cpu numpy")
    print("=" * 60)

    # 연결 테스트
    print("Ollama 서버 연결을 테스트하고 있습니다...")
    test_response = call_ollama_llm("안녕하세요", "qwen2.5:7b")
    if "연결할 수 없습니다" in test_response:
        print("❌ Ollama 서버가 실행되지 않았습니다.")
        print("다른 터미널에서 'ollama serve' 명령을 실행해주세요.")
        return
    else:
        print("✅ Ollama 서버 연결 성공!")

    # 임베딩 모델 테스트
    print("임베딩 모델을 테스트하고 있습니다...")
    test_embedding = get_ollama_embedding("테스트")
    if test_embedding is None:
        print("⚠️ 임베딩 모델이 없습니다. 벡터 검색 없이 진행합니다.")
        use_embedding = False
    else:
        print("✅ 임베딩 모델 사용 가능!")
        use_embedding = True

    while True:
        # 1. 유저 쿼리 입력
        query = input("\n질문을 입력하세요 (종료하려면 'quit' 입력): ")

        if query.lower() in ['quit', 'exit', '종료', '나가기']:
            print("프로그램을 종료합니다.")
            break

        if not query.strip():
            continue

        # 2. 유사한 청크 검색 (임베딩 모델이 있는 경우에만)
        matched_chunks = []
        print("관련 문서를 검색 중...")
        if use_embedding:
            matched_chunks = search_similar_chunks_ollama(query)
        else:
            matched_chunks = search_similar_chunks(query)

        if not matched_chunks:
            if use_embedding:
                print("관련 문서를 찾지 못했습니다. 일반 답변을 제공합니다.")
            context = "관련 문서가 없습니다."
        else:
            context = "\n".join(matched_chunks)
            print(f"관련 문서 {len(matched_chunks)}개를 찾았습니다.")

        # 3. 한글 최적화 프롬프트 생성
        prompt = f"""다음은 회사 문서의 내용입니다:

{context}

위 내용을 바탕으로 아래 질문에 대해 정확하고 친절하게 한국어로 답변해주세요. 
만약 문서에서 답을 찾을 수 없다면, 그렇다고 명시하고 일반적인 조언을 제공해주세요.

질문: {query}

답변:"""

        # 4. 로컬 한글 LLM 호출
        print("답변을 생성 중...")
        response = call_ollama_llm(prompt)

        print("\n=== 답변 ===")
        print(response)
        print("=" * 50)

if __name__ == "__main__":
    main()