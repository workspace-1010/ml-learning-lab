"""Exercise 35: test a tiny RAG system's limits.

This is a no-network evaluation checklist. Answer each question after trying
exercise 34 with the sample documents.
"""

CHECKS = [
    "Ask a question whose answer is directly stated in one passage.",
    "Ask the same question using different words; keyword retrieval may fail.",
    "Ask about something not mentioned in any file.",
    "Add a second document that contradicts the first and inspect the answer.",
    "Check whether the retrieved passages actually support the model's answer.",
    "Remove the API key or temporarily disconnect from the internet and observe the error.",
]

if __name__ == "__main__":
    print("RAG evaluation checklist\n")
    for index, item in enumerate(CHECKS, 1):
        print(f"{index}. [ ] {item}")
    print("\nRAG can improve access to source material, but it does not guarantee truth.")
    print("Record failures. They reveal which part of the pipeline needs improvement.")
