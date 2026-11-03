# Course design

## Design principle

Use an exercise-first format: numbered topic folders, short numbered source files, one concrete idea per exercise, and a README that acts as a navigable index. The structure is inspired by beginner-friendly courses such as Codédex Python 101 and C++ 101; lesson text and code are original.

Learners should be able to predict output, run one small program, and explain what changed. The goal is confidence through understanding the complete path, not memorizing functions or being overwhelmed by tools.

## Finish line

The learner can:
- write and explain a basic supervised-learning program;
- load and inspect one pretrained transformer without using `pipeline()`;
- explain tokenization, weights, logits, and next-token generation;
- wrap generation in a small inference-engine interface;
- explain request validation, decoding, stopping, KV caching, and streaming;
- explain what an API is and distinguish a local client from a remote model server;
- make a model API request, read the response, handle basic errors, and protect an API key;
- build a small RAG application that retrieves passages from local text files and supplies them as context to a model API;
- test when retrieval or generation fails and avoid claiming that RAG guarantees correctness.

## Course shape

### Part 1 — First ML programs

Introduce data, features and targets, train/test splitting, model creation, fitting, prediction, evaluation, classification and overfitting using small CPU-friendly examples.

### Part 2 — Understand DistilGPT-2

Use the fixed model ID `distilbert/distilgpt2` to study tokenization, configuration, checkpoint files, weights, forward-pass logits, next-token selection, autoregressive generation and sampling. This is a base English text-completion model, not an instruction-tuned assistant.

### Part 3 — Assemble a tiny inference engine

Organize the Part 2 mechanisms into a reusable component. Introduce request validation, a generation interface, decoding controls, stopping/context budgets, KV cache, and streaming. This is a learning engine, not a production serving system.

### Part 4 — Use a model through an API

This is the bridge from model internals to application development. Explain API, endpoint, request, response, JSON, HTTP/HTTPS, authentication, API keys, environment variables, errors, and provider pricing/limits. Learners send a prompt to a hosted model and read the returned text.

The local computer runs the Python client and application code; the provider runs the model. This makes the next project feasible on a modest PC without loading another large model locally. Use one documented provider consistently in runnable examples, but do not commit API keys or imply that API usage is always free.

### Part 5 — Build a simple RAG application

Use local text files as a tiny knowledge base. Split text into passages, retrieve passages with transparent keyword matching, build a prompt that includes the question and evidence, call the Part 4 API helper, and inspect the result. The learner should compare answers with and without retrieved context and test irrelevant or missing evidence.

Start without embeddings or a vector database. These add concepts and dependencies that are not necessary to understand the basic RAG architecture. They can be optional extensions later.

## The RAG project grows in stages

The goal is to bridge the gap between completing guided exercises and independently building an application. Do not remove the existing parts or skip the API section: Part 4 teaches the client/server boundary and API handling that Part 5 needs.

Treat the RAG work as one project that becomes more capable in small, explainable versions:

1. **Retrieval-only baseline:** load local text files, split them into passages, rank/select keyword matches, and print the selected passages. Do not call a model yet. Make retrieval visible and understandable.
2. **Basic RAG:** construct a prompt from the question and retrieved passages, call the Part 4 API helper, and inspect the answer.
3. **Inspectable RAG:** expose the retrieved passages and source names, detect when retrieval returns no useful evidence, and inspect unsupported answers. Keep enough visibility to explain what the system did.
4. **Semantic retrieval:** compare keyword matching with embeddings and similarity search. Add a vector index only when its purpose is understood and the simpler baseline provides a meaningful comparison.
5. **Usable application:** add a modest user interface, document ingestion/management, input validation, helpful errors, and a repeatable evaluation procedure.

For every version, the learner should:
- draw or update a component diagram;
- trace one question from input through retrieval, prompt construction, API request, and answer;
- inspect the actual retrieved evidence rather than judging only the final answer;
- deliberately test a failure case, explain a hypothesis for it, and change one thing at a time;
- write down what is known, what is uncertain, and what the next improvement should be.

The current Exercises 30–35 remain the foundation for Versions 1 and 2. Follow-on exercises can extend the project incrementally; they should not replace or delete the current course. Before any new non-standard dependency is introduced, document its purpose, installation command, and the point in the course where it becomes necessary in the root README and setup guide.

## Modest hardware guidance

- Parts 1 and 4–5 should be lightweight locally; Part 4–5 need an internet connection and a provider API key.
- Parts 2–3 use CPU inference with DistilGPT-2. Short prompts and small token budgets are important on a 4 GB RAM computer; fit is not guaranteed with every background workload.
- Keep credentials out of source code and Git history. Never commit model weights or local model caches.
- Explain API pricing, quotas, and data-handling terms as provider-dependent, not universal.

## Exercise pattern

Each source file should be small and runnable, with one main idea. Learners should be able to answer:
1. What is the input?
2. What important function or request is being used?
3. What is returned or changed?
4. What happens if one small thing changes?
5. What are the limitations or failure cases?

## Why no pipeline in Part 2?

The high-level Transformers `pipeline()` API hides loading, tokenization and output processing. In Part 2 those steps are the lesson, so examples use direct tokenizer and model APIs. Part 3 wraps them in a small interface while keeping the mechanics visible.
