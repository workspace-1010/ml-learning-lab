"""Exercise 36: compare keyword retrieval with embedding-based retrieval.

Run from the repository root after exercise 30:
    python 5-simple-rag/36_semantic_retrieval.py

Requires the Part 4 Google Gen AI SDK and an API key. This lesson sends
the question and local passages to the embedding API; it does not use a
local embedding model or vector database. It is an optional extension.
"""

import math
import os
import re
import sys
from pathlib import Path

DOCS = Path(__file__).parent / "sample_docs"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def load_passages() -> list[dict[str, str]]:
    passages = []
    for path in sorted(DOCS.glob("*.txt")):
        tokens = path.read_text(encoding="utf-8").split()
        for number, start in enumerate(range(0, len(tokens), 18), start=1):
            passages.append({
                "source": path.name,
                "passage_id": str(number),
                "text": " ".join(tokens[start:start + 18]),
            })
    return passages


def keyword_score(question: str, passage: str) -> int:
    return len(words(question) & words(passage))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def embed(client, text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("The embedding API returned no vector.")
    return list(response.embeddings[0].values)


def main() -> int:
    if not DOCS.exists():
        print("Run exercise 30 first: python 5-simple-rag/30_create_knowledge_base.py")
        return 1
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY first. See 4-model-apis/README.md.")
        return 1
    try:
        from google import genai
    except ImportError:
        print('Install the SDK: python -m pip install -e ".[api]"')
        return 1

    passages = load_passages()
    if not passages:
        print(f"No .txt documents found in {DOCS}.")
        return 1

    question = input("Ask a question about the sample documents: ").strip()
    if not question:
        print("Please enter a question.")
        return 1

    # Show the baseline first so the difference between retrieval methods is visible.
    baseline = sorted(
        passages,
        key=lambda item: keyword_score(question, item["text"]),
        reverse=True,
    )
    print("\nKeyword baseline (top 3):")
    for item in baseline[:3]:
        print(
            f"- score={keyword_score(question, item['text'])} "
            f"{item['source']} · passage {item['passage_id']}: {item['text']}"
        )

    try:
        client = genai.Client()
        print(f"\nCreating embeddings with {EMBEDDING_MODEL}...")
        question_vector = embed(client, question)
        ranked = []
        for item in passages:
            vector = embed(client, item["text"])
            ranked.append((cosine_similarity(question_vector, vector), item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
    except Exception as exc:
        print(
            f"Embedding request failed ({type(exc).__name__}). "
            "Check your key, quota, model availability, and connection."
        )
        return 1

    print("\nSemantic retrieval (cosine similarity, top 3):")
    for score, item in ranked[:3]:
        print(
            f"- score={score:.3f} {item['source']} · passage {item['passage_id']}: "
            f"{item['text']}"
        )
    print(
        "\nCompare the passages, not just the scores. A higher similarity score "
        "does not prove that a passage answers the question."
    )
    print(
        "This teaching example embeds passages on each run. A larger application "
        "would cache document embeddings and update only changed documents."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
