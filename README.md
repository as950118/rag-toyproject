# RAG toyproject

한국어 문서에 대해 RAG(Retrieval-Augmented Generation) 파이프라인을 구현한 토이 프로젝트입니다.  
텍스트 파일을 청크로 분할하고, 임베딩을 생성하여 FAISS 벡터 DB에 저장한 뒤, 쿼리와 유사한 청크를 검색하여 LLM으로 답변을 생성합니다.

---

## 주요 기술 스택

- Python 3.9.6
- [LangChain](https://github.com/langchain-ai/langchain) (텍스트 청크 분할)
- [sentence-transformers](https://www.sbert.net/) (`jhgan/ko-sroberta-multitask` 모델 사용)
- [FAISS](https://github.com/facebookresearch/faiss) (벡터 검색)
- LLM 옵션:
  - [OpenAI GPT](https://platform.openai.com/docs/guides/gpt) (클라우드 기반)
  - [Ollama](https://ollama.ai/) (로컬 기반, Qwen2.5 모델 사용)

---

## 프로젝트 구조

```
rag-toyproject/ 
├── chunking.py # 텍스트 파일을 청크로 분할하고 pickle로 저장 
├── embedding.py # 청크 임베딩 생성 
├── vector_indexing.py # 임베딩으로 FAISS 인덱스 생성 및 저장 
├── get_chunk_from_vector.py # 쿼리 임베딩 후 유사 청크 검색 
├── rag.py # OpenAI GPT 기반 RAG 구현
├── rag_with_ollama.py # Ollama 기반 로컬 RAG 구현
├── requirements.txt # 의존성 목록 
├── README.md # 프로젝트 설명 
└── (기타)
```

---

## 설치 방법

1. **Python 3.9.6** 환경을 준비하세요.
2. 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

필요 패키지 예시:

```
langchain
sentence-transformers
faiss-cpu
openai
requests
```

### Ollama 설치 (로컬 LLM 사용 시)

1. Ollama 설치:
```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh
```

2. 필요한 모델 다운로드:
```bash
# 한중일 언어 지원 LLM
ollama pull qwen2.5:7b

# 임베딩 모델
ollama pull nomic-embed-text
```

3. Ollama 서버 시작:
```bash
ollama serve
```

## 사용 방법

1. 텍스트 파일을 청크로 분할 및 저장
```
python chunking.py
```
기본적으로 취업규칙.txt 파일을 읽어 chunks.pkl로 저장합니다.

2. 임베딩 생성
```
python embedding.py
```
chunks.pkl을 불러와 임베딩을 생성합니다.

3. FAISS 인덱스 생성
```
python vector_indexing.py
```
임베딩을 이용해 faiss_index.idx 파일을 생성합니다.

4. RAG 실행 (두 가지 옵션)

### OpenAI GPT 기반 RAG
```
python rag.py
```

### Ollama 기반 로컬 RAG
```
python rag_with_ollama.py
```

두 버전 모두 쿼리를 입력하면, FAISS에서 유사한 청크를 찾아 LLM으로 답변을 생성합니다.

예시
```
취업규칙.txt 파일을 프로젝트 폴더에 준비하세요.
위의 순서대로 각 스크립트를 실행하세요.
rag.py 또는 rag_with_ollama.py 실행 후, 예시 쿼리:
이 회사 연차는 어떻게 신청해?
```
→ 관련 규정 청크가 출력되고 LLM이 답변을 생성합니다.

## 참고사항

### OpenAI GPT 사용 시
- 환경 변수에 OPENAI_API_KEY를 설정해야 합니다.
- 한국어 임베딩 모델로 jhgan/ko-sroberta-multitask를 사용합니다.

### Ollama 사용 시
- Ollama 서버가 실행 중이어야 합니다.
- Qwen2.5:7b 모델은 한중일 언어를 지원합니다.
- nomic-embed-text 모델을 사용하여 임베딩을 생성합니다.
- 인터넷 연결 없이 로컬에서 실행 가능합니다.