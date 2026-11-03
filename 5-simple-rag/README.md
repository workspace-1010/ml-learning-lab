# Part 5 — Build a tiny RAG application

## The goal

Build a small program that answers using passages retrieved from a few local text files. The program will not run a language model locally: it will send the question and retrieved passages to the hosted model API introduced in Part 4.

RAG means **retrieval-augmented generation**. It has two main actions:
- **Retrieval:** find document passages relevant to the question.
- **Generation:** send those passages and the question to a language model to generate a response.

Retrieval gives the model additional information; it does not guarantee that the model will use it correctly or that the answer will be true.

## The flow

```text
Local text files
      |
      v
Split into small passages
      |
      v
Find passages related to the question
      |
      v
Combine question + retrieved passages into a prompt
      |
      v
Send prompt to the model API
      |
      v
Display the returned response
```

## Exercise path

| Exercise | Idea |
|---:|---|
| 30 | Create a tiny local knowledge base |
| 31 | Split documents into passages |
| 32 | Retrieve passages with simple keyword matching |
| 33 | Build a prompt containing the evidence and question |
| 34 | Send the RAG prompt through the Part 4 API helper |
| 35 | Compare results and test failure cases |

## Why start with keyword search?

Keyword search keeps the first RAG implementation small and understandable. The learner can see exactly why a passage was selected before adding embeddings or a vector database. Those can be optional extensions after the basic system works.

## Hardware and cost

The documents, retrieval code, and prompt construction run on the learner's PC. The provider runs the language model. A small keyword-based example should be lightweight, including on a 4 GB RAM computer, but API requests need internet access and may incur charges. Keep the API key in an environment variable, never in the repository.

## Important limitation

DistilGPT-2 from Part 2 is a base text-completion model, not an instruction-tuned question-answering assistant. This API-based RAG project uses a model offered by the API provider instead. The learner can still use Part 2 to understand what language models do, and Part 3 to understand the inference process behind a model call. The exact API model is chosen according to the provider's current availability, requirements, and cost.

## Finish line

The learner can describe and implement the whole path: read local documents, retrieve relevant text, construct a prompt, call a remote model API, and inspect whether the response is supported by the retrieved passages.
