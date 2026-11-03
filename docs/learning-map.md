# Learning map

## Part 1 — Basic ML

Data → features and targets → train/test split → model object → fit → predict → metric → compare a model's training performance with its test performance.

This part uses small local datasets and scikit-learn. It should run on CPU without downloading model weights.

## Part 2 — One pretrained language model

The fixed checkpoint is `distilbert/distilgpt2`. Follow the exercises in `2-distilgpt2/README.md`:

Text → tokenizer → token IDs → architecture + config + checkpoint weights → forward pass → logits → next-token choice → repeated generation.

A learner should be able to identify the model's main components, explain what the raw output means, and write a short next-token loop without using `pipeline()`.

## Part 3 — A tiny inference engine

Follow `3-inference-engine/README.md`:

Request → validate settings → tokenize prompt → check context budget → model forward pass → decoding policy → stop condition → decoded response.

Then inspect KV cache and streaming. These exercises reuse the same model and dependencies from Part 2. They are teaching examples, not a production inference server.

## Part 4 — Use a model through an API

Follow `4-model-apis/README.md`:

Python application → API request → HTTPS endpoint + authentication → provider runs model inference → JSON response → application extracts generated text.

The learner's PC runs the client application. The provider's server runs the model. API keys stay in environment variables; costs, limits and supported request formats depend on the provider.

## Part 5 — Build a simple RAG application

Follow `5-simple-rag/README.md`:

Local documents → small passages → keyword retrieval → question + retrieved evidence → model API → generated response → inspect whether the response is supported.

This first implementation deliberately uses keyword matching instead of embeddings or a vector database. That keeps the retrieval step visible and the hardware requirements modest. RAG adds context; it does not guarantee factual answers.

## From the first RAG exercise to an independent application

Keep the five existing course parts. Part 4 is a deliberate bridge: it teaches the request/response boundary and remote inference so the RAG project can run without loading another model on a modest computer.

Build the same RAG project in successive versions:

**Version 1 — Retrieval only**

Local documents → passages → keyword matching → ranked/selected passages → display passages and source names.

No language model or new dependency is needed. The key question is: *Did the retriever find the evidence that answers the question?*

**Version 2 — Basic RAG**

Question → retrieve passages → build a prompt from question + evidence → Part 4 API request → generated answer.

Trace a single request end to end and compare the response with the actual retrieved evidence.

**Version 3 — Inspectable RAG**

Question → retrieve → display evidence + sources → check for weak/empty retrieval → generate → inspect evidence support.

Make it possible to see what the system retrieved and to identify when the answer goes beyond that evidence.

**Version 4 — Semantic RAG**

Documents → chunks → embeddings → similarity search → selected evidence → prompt → model API → answer.

Compare results against keyword retrieval. Introduce embedding and vector-index dependencies only when their role and installation are documented.

**Version 5 — Usable application**

Document input/management → retrieval pipeline → source-aware answer → user interface → useful errors → evaluation records.

This is a learning application, not a production promise. Improve one component at a time and keep the earlier versions available as baselines.

At each version, draw the architecture, follow one question through the components, and intentionally test a failure (for example, a missing answer, irrelevant passage, ambiguous question, API quota error, or unsupported claim). A good final answer alone does not prove the retrieval was good.

## Deliberately out of scope for the first RAG version

Vector databases, embedding models, fine-tuning, training a transformer from random initialization, and production-grade serving infrastructure. They can be optional extensions after the learner understands the end-to-end flow.
