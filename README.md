# RAG toyproject

한국어 문서에 대해 RAG(Retrieval-Augmented Generation) 파이프라인을 구현한 토이 프로젝트입니다.  
텍스트 파일을 청크로 분할하고, 임베딩을 생성하여 FAISS 벡터 DB에 저장한 뒤, 쿼리와 유사한 청크를 검색하여 LLM(OpenAI GPT)으로 답변을 생성합니다.

---

## 주요 기술 스택

- Python 3.9.6
- [LangChain](https://github.com/langchain-ai/langchain) (텍스트 청크 분할)
- [sentence-transformers](https://www.sbert.net/) (`jhgan/ko-sroberta-multitask` 모델 사용)
- [FAISS](https://github.com/facebookresearch/faiss) (벡터 검색)
- [OpenAI GPT](https://platform.openai.com/docs/guides/gpt) (질문 응답 생성)

---

## 프로젝트 구조

```
rag-toyproject/ 
├── chunking.py # 텍스트 파일을 청크로 분할하고 pickle로 저장 
├── embedding.py # 청크 임베딩 생성 
├── vector_indexing.py # 임베딩으로 FAISS 인덱스 생성 및 저장 
├── get_chunk_from_vector.py # 쿼리 임베딩 후 유사 청크 검색 
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
```

사용 방법
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

4. 쿼리로 유사한 청크 검색 및 답변 생성
```
python get_chunk_from_vector.py
```
쿼리를 입력하면, FAISS에서 유사한 청크를 찾아 출력합니다.
(OpenAI API 키가 있다면) LLM을 호출해 답변을 생성할 수 있습니다.

예시
```
취업규칙.txt 파일을 프로젝트 폴더에 준비하세요.
위의 순서대로 각 스크립트를 실행하세요.
get_chunk_from_vector.py 실행 후, 예시 쿼리:
이 회사 연차는 어떻게 신청해?
```
→ 관련 규정 청크가 출력됩니다.

참고
```
OpenAI API를 사용하는 경우, 환경 변수에 OPENAI_API_KEY를 설정해야 합니다.
한국어 임베딩 모델로 jhgan/ko-sroberta-multitask를 사용합니다.
```