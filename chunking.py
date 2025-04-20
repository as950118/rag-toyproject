from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

if __name__ == "__main__":
    filename = "취업규칙.txt"  # 확장자 명시 추천
    try:
        with open(filename, "r", encoding="utf-8") as f:
            long_text = f.read()
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {filename}")
        exit(1)
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        exit(1)

    chunks = text_splitter.split_text(long_text)
    print(f"총 {len(chunks)}개의 chunk 생성됨.\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()
