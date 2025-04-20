from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def chunk_text_from_file(filename, chunk_size=500, chunk_overlap=100, save_path="chunks.pkl"):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import pickle

    try:
        with open(filename, "r", encoding="utf-8") as f:
            long_text = f.read()
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {filename}")
        return []
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(long_text)

    # 청크를 pickle 파일로 저장
    try:
        with open(save_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"청크가 {save_path} 파일로 저장되었습니다.")
    except Exception as e:
        print(f"청크 저장 실패: {e}")

    return chunks


if __name__ == "__main__":
    filename = "취업규칙.txt"
    chunks = chunk_text_from_file(filename)
    print(f"총 {len(chunks)}개의 chunk 생성됨.\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()
