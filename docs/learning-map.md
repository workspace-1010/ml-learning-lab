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

## Deliberately out of scope for the first RAG version

Vector databases, embedding models, fine-tuning, training a transformer from random initialization, and production-grade serving infrastructure. They can be optional extensions after the learner understands the end-to-end flow.
