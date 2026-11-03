# Load a pretrained model

## Goal

Use an existing model repository and a library to run inference, understanding the roles of its files.

- **Configuration:** architecture and behavior settings.
- **Tokenizer files:** map text to token IDs.
- **Weights/checkpoint:** learned parameter tensors.
- **Model library:** implements the architecture's computations.
- **Inference:** run a forward computation without updating weights.

Hugging Face Transformers commonly provides `AutoTokenizer.from_pretrained(model_id)` and `AutoModelForCausalLM.from_pretrained(model_id)` for compatible causal language models. Exact classes depend on the model type.

Before running, choose a small model whose license permits your use. Downloads can be large and CPU inference slow. Read the model card and check hardware requirements. Never commit downloaded weights.

Exercises: identify the model ID/license; inspect tokenizer and configuration; run a short prompt; explain tokenization, inference and decoding; find the local cache location.
