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

Then inspect two practical inference-system ideas: KV cache (reuse past key/value states) and streaming (show each token as it arrives). These exercises reuse the same model and dependencies from Part 2; no new model download is required if the checkpoint is already cached.

## Deliberately out of scope

This version does not cover RAG, vector databases, fine-tuning, training a transformer from random initialization, or production serving infrastructure. Those are useful topics, but they would dilute the current goal of understanding the model and its basic inference path.
