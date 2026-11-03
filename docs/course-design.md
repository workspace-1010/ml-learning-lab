# Course design

## Design principle

Use the same broad format that makes beginner exercise repositories approachable: numbered topic folders, short numbered source files, one concrete idea per program, and a README that acts as a navigable exercise index. This repository is inspired by the *structure* of Codédex's Python 101 and C++ 101 exercises; it does not copy their lesson text or code.

The examples here are standalone solutions/reference implementations. Learners should first predict the output or attempt the task in a Git branch, then run the reference file and compare.

## Finish line

The learner can:
- write and explain a basic supervised-learning program;
- explain features, targets, train/test split, fitting, prediction, metrics and basic overfitting;
- load one specific pretrained transformer and its tokenizer without using `pipeline()`;
- identify the roles of configuration, tokenizer assets, architecture and checkpoint weights;
- inspect learned parameter names and shapes, inspect the forward-pass logits, and understand how next-token generation works;
- write a small token-by-token generation loop using the model's raw output.

This course intentionally excludes RAG, vector databases, fine-tuning and training a language model from scratch. Model assembly and inference alone are a substantial goal.

## Course shape

### Part 1 — First ML programs

Short scripts introduce data, features/targets, splitting data, creating a model object, fitting, prediction, measuring error, classification and overfitting. Most files can be run on a CPU with a small scikit-learn dependency. The same dataset is reused so learners can compare each step.

### Part 2 — Build a mental model of DistilGPT2

Use the fixed model ID `distilbert/distilgpt2`, which is a compact 82M-parameter GPT-2-style language model. Load it with `AutoTokenizer` and `AutoModelForCausalLM`, not `pipeline()`. The standard safetensors checkpoint is about 353 MB. It is a base English text-completion model, not an instruction-tuned assistant.

Exercises progress from tokenization and configuration to checkpoint files, model parameters, the forward pass, logits, choosing one next token, manual greedy generation and sampling. The last exercises build on concepts shown by earlier files.

## The model is specific on purpose

Do not casually change the model ID halfway through the course. A fixed model lets every lesson refer to the same configuration fields and tensor shapes. If the model ID changes later, update the examples and expected observations together.

## 4 GB RAM guidance

- Use CPU inference; do not require a GPU or CUDA.
- Start with short prompts and generate only a small number of tokens.
- Close memory-heavy applications during model exercises.
- Model weights are downloaded by the learner on first use and must not be committed to Git.
- The model file is only one part of total RAM use: Python, PyTorch, Transformers and intermediate tensors also consume memory. 4 GB is a constrained target, not a guarantee that every machine and background workload will fit.
- Use the model's Safetensors checkpoint. Avoid downloading every alternate TensorFlow, Flax, Core ML and LiteRT file in the upstream repository.

## Exercise pattern

Each source file is a small, runnable reference. Files are grouped by topic and numbered in learning order. Keep the opening explanation short and concrete. Learners should be able to answer:
1. What is the input to this program?
2. What important function is being called?
3. What is returned or changed?
4. What happens if one small thing changes?

## Why no pipeline?

The high-level Transformers `pipeline()` API hides loading, tokenization and output processing. Here those steps are the lesson, so examples use direct tokenizer and model APIs. A later example manually selects next tokens from logits so learners can see the autoregressive loop.
