# Capstone: document Q&A

Build a small application that accepts a question about a known document set, retrieves relevant passages, sends them to a language model, and displays an answer with source references.

## Acceptance criteria

- A new learner can follow setup from a clean environment.
- Source metadata remains traceable through chunking.
- Retrieval results can be inspected before generation.
- The prompt separates retrieved evidence from the question.
- The app signals when evidence is insufficient rather than confidently inventing an answer.
- Test at least five questions, including one not answered by the documents.
- Explain dependencies, model/license requirements, resource use and limitations.

Build in stages: print loaded documents, print chunks, inspect retrieved passages, add generation, then record a small evaluation table. A command-line app is enough; a web UI is optional.
