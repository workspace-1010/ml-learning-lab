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

## Why Part 4 comes before this part

The dedicated API section is intentional. It teaches how a local Python program sends a request to a remote model, handles authentication, reads a response, and deals with errors. This RAG project reuses that knowledge instead of hiding the model call inside a magic helper. Your computer runs document processing and retrieval; the API provider runs generation.

## The project grows in five versions

The current Exercises 30–35 are the foundation, not the whole journey. Keep them and extend the same project step by step:

### Version 1 — Retrieval only

Load local documents, split them into passages, run keyword retrieval, and display the selected passages and source names. Do not call a model yet. Learn to answer: *Why did the retriever select this passage, and did it contain the needed evidence?*

### Version 2 — Basic RAG

Use the selected passages to build a prompt with the question, call the Part 4 API helper, and inspect the answer. Trace one question through the complete pipeline.

### Version 3 — Inspectable RAG

Make the retrieved passages and source names visible. Handle empty or weak retrieval explicitly, and examine whether answer claims are supported by the retrieved text. The goal is to make the system explainable and debuggable.

### Version 4 — Semantic RAG

Add embeddings and similarity-based retrieval, then compare it with keyword matching using the same questions and documents. A vector database is optional; first understand what embeddings and similarity scores contribute. Any new external library must be documented in the root README and setup guide, including its purpose and installation command, before the learner reaches the lesson that needs it.

### Version 5 — Usable application

Add a simple interface, document ingestion or management, validation, useful error messages, and a repeatable evaluation set. Keep the earlier versions as baselines so you can tell whether each improvement actually helps.

For every version, draw the component diagram, trace one question from input to answer, inspect the actual evidence, and deliberately investigate at least one failure. Examples include a question whose answer is absent, wording that keyword matching misses, irrelevant retrieved passages, unsupported claims, and an unavailable API.

## Exercise path

| Exercise | Idea |
|---:|---|
| 30 | Create a tiny local knowledge base |
| 31 | Split documents into passages |
| 32 | Retrieve passages with simple keyword matching |
| 33 | Build a prompt containing the evidence and question |
| 34 | Send the RAG prompt through the Part 4 API helper |
| 35 | Compare results and test failure cases |
| 36 | Compare keyword retrieval with embedding-based semantic retrieval (optional API-backed extension) |
| 37 | Use the RAG pipeline through an interactive command-line application |

## Run the exercises

Run commands from the repository root. First create fictional sample documents: `python 5-simple-rag/30_create_knowledge_base.py`. Inspect splitting with `python 5-simple-rag/31_split_documents.py`, then try `python 5-simple-rag/32_keyword_retrieval.py` and `python 5-simple-rag/33_build_rag_prompt.py`. After completing the [Part 4 API setup](../4-model-apis/README.md), run `python 5-simple-rag/34_ask_gemini_with_rag.py` and `python 5-simple-rag/35_test_rag_limits.py`. When you understand the baseline, continue to Exercises 36–37. The live examples require internet, a personal key, and available API quota. Do not enable paid billing for this course; continue with offline exercises if live calls are unavailable.

### Version 1 — Retrieval only (Exercises 30–32)

Create the local knowledge base, split it into passages, and inspect keyword matches. Keep the model out of the loop until you can explain why a passage was selected. Deliverable: a question, a visible list of retrieved passages and their source names, and a short note about one query that fails.

### Version 2 — Basic RAG (Exercises 33–34)

Build a prompt from the question and retrieved passages, then call the Part 4 API. Deliverable: trace one question from input through retrieval and prompt construction to the generated response.

### Version 3 — Inspectable RAG (Exercises 34–35)

Inspect the passages printed before generation, check source names, and deliberately ask questions that are not answered in the files. Deliverable: record whether the evidence supports the answer and what happens when retrieval returns no matches. A fluent response is not proof of a correct retrieval.

### Version 4 — Semantic RAG (Exercise 36, optional API-backed extension)

Run `python 5-simple-rag/36_semantic_retrieval.py`. It compares the keyword baseline with embedding-based similarity search using the existing Google Gen AI SDK. No new Python package or vector database is required. The embedding API requires internet, an API key, and available quota. This teaching script recomputes embeddings each run; caching and indexing are later improvements. Compare the retrieved passages rather than assuming the highest score must be right.

### Version 5 — Usable application (Exercise 37)

Run `python 5-simple-rag/37_rag_cli_app.py` for an interactive command-line app. Ask multiple questions in one session; use `/sources`, `/reload`, `/help`, and `/quit`. To update the knowledge base, add or edit fictional/public `.txt` files in `5-simple-rag/sample_docs/`, then use `/reload`. The app displays retrieved evidence and handles request errors without printing secrets.

For every version, draw the component flow, trace one question end to end, and investigate a failure deliberately. Record what you observed, your explanation, and the next change you would try.

## Why start with keyword search?

Keyword search keeps the first RAG implementation small and understandable. The learner can see exactly why a passage was selected before adding embeddings or a vector database. Those can be optional extensions after the basic system works.

## Hardware and cost

The documents, retrieval code, and prompt construction run on the learner's PC. The provider runs the language model. A small keyword-based example is lightweight, including on a 4 GB RAM computer, but API requests need internet and available free-tier quota. The Gemini free tier may use submitted content to improve Google's products. Use only fictional/public sample documents and keep the API key in an environment variable, never in the repository. Do not enable paid billing to complete the course.

## Important limitation

DistilGPT-2 from Part 2 is a base text-completion model, not an instruction-tuned question-answering assistant. This API-based RAG project uses a model offered by the API provider instead. The learner can still use Part 2 to understand what language models do, and Part 3 to understand the inference process behind a model call. The exact API model is chosen according to the provider's current availability, requirements, and cost.

## Finish line

The learner can describe and implement the whole path: read local documents, retrieve relevant text, construct a prompt, call a remote model API, and inspect whether the response is supported by the retrieved passages.
