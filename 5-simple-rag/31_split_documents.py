"""Exercise 31: split text documents into small, readable passages."""

from pathlib import Path

DOCS = Path(__file__).parent / "sample_docs"
PASSAGE_WORDS = 18


def split_into_passages(text: str, words_per_passage: int = PASSAGE_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[start : start + words_per_passage])
        for start in range(0, len(words), words_per_passage)
    ]


def load_passages(folder: Path = DOCS) -> list[dict[str, str]]:
    passages = []
    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for index, passage in enumerate(split_into_passages(text), start=1):
            passages.append({"source": path.name, "passage_id": str(index), "text": passage})
    return passages


if __name__ == "__main__":
    if not DOCS.exists():
        print("Run exercise 30 first: python 5-simple-rag/30_create_knowledge_base.py")
    else:
        for item in load_passages():
            print(f"\n[{item['source']} · passage {item['passage_id']}]")
            print(item["text"])
