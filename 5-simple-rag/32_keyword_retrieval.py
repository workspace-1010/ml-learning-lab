"""Exercise 32: retrieve passages using transparent keyword matching."""

import re
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

DOCS = Path(__file__).parent / "sample_docs"


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def split_into_passages(text: str, words_per_passage: int = 18) -> list[str]:
    tokens = text.split()
    return [" ".join(tokens[i:i + words_per_passage]) for i in range(0, len(tokens), words_per_passage)]


def retrieve(question: str, passages: list[dict[str, str]], limit: int = 3) -> list[dict[str, str]]:
    query_words = words(question)
    ranked = []
    for item in passages:
        score = len(query_words & words(item["text"]))
        if score:
            ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[:limit]]


def load_all_passages() -> list[dict[str, str]]:
    items = []
    for path in sorted(DOCS.glob("*.txt")):
        for number, text in enumerate(split_into_passages(path.read_text(encoding="utf-8")), 1):
            items.append({"source": path.name, "passage_id": str(number), "text": text})
    return items


if __name__ == "__main__":
    if not DOCS.exists():
        print("Run exercise 30 first.")
    else:
        question = input("Ask a question about the sample documents: ").strip()
        matches = retrieve(question, load_all_passages())
        if not matches:
            print("No keyword matches. Try using words found in the documents.")
        else:
            for item in matches:
                print(f"\nSource: {item['source']} (passage {item['passage_id']})")
                print(item["text"])
