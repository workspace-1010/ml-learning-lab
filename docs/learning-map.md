# Learning map

## Part 1 — Basic ML

Data → features and targets → train/test split → model object → fit → predict → metric → compare a model's training performance with its test performance.

This part uses small local datasets and scikit-learn. It should run on CPU without downloading model weights.

## Part 2 — One pretrained language model

The fixed checkpoint is `distilbert/distilgpt2`. Follow the exercises in `2-distilgpt2/README.md`:

Text → tokenizer → token IDs → architecture + config + checkpoint weights → forward pass → logits → next-token choice → repeated generation.

A learner should be able to identify the model's main components, explain what the raw output means, and write a short next-token loop without using `pipeline()`.

## Deliberately out of scope

This version does not cover RAG, vector databases, fine-tuning or training a transformer from random initialization. Those are useful topics, but they would dilute the current goal of understanding and using a pretrained model.
