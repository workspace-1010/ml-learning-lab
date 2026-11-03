"""Exercise 30: create a tiny local knowledge base.

Run from the repository root: python 5-simple-rag/30_create_knowledge_base.py
This creates sample_docs/ with fictional, public practice content.
"""

from pathlib import Path

DOCS = Path(__file__).parent / "sample_docs"

SAMPLES = {
    "school_library.txt": (
        "The school library opens at 8:30 AM on school days. "
        "Students may borrow two books at a time for seven days."
    ),
    "science_club.txt": (
        "The science club meets on Friday after classes in Lab 2. "
        "Members should bring a notebook and pencil."
    ),
    "computer_lab.txt": (
        "The computer lab is open for supervised projects on Wednesday. "
        "Students must save their work in their own project folder."
    ),
}


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for name, text in SAMPLES.items():
        (DOCS / name).write_text(text + "\n", encoding="utf-8")
        print(f"Created {DOCS / name}")
    print("These are fictional examples. Do not place private documents in this folder.")


if __name__ == "__main__":
    main()
