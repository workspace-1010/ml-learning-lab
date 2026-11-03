# Build a basic RAG pipeline

## Goal

Answer a question using passages retrieved from a small document collection.

## Data flow

1. Load source documents.
2. Split them into chunks and preserve source metadata.
3. Convert chunks into embedding vectors.
4. Embed the question and retrieve the most similar chunks.
5. Build a prompt separating retrieved context from the question.
6. Ask a language model to answer using that context.
7. Show source passages and inspect whether the answer is supported.

The retriever selects candidate evidence; the language model writes an answer from the prompt. Retrieval is not generation, and retrieval alone does not guarantee correctness.

## Experiments

- Compare small and large chunks.
- Change the number of retrieved passages (top-k).
- Ask a question not answered by the documents and decide how the app should respond.
- Display source names next to answers.

Treat retrieved text as untrusted input. Do not let document contents override system instructions or place API keys in source code. A complete implementation should use a small document set and explain each library function before adding abstractions.
