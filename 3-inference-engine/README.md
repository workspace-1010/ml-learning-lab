# Part 3 — Build a tiny inference engine

Part 2 taught the pieces inside one model call. Part 3 connects those pieces into a **small reusable inference engine**: the kind of component an application calls when it wants text from a model.

We keep the same model, `distilbert/distilgpt2`, and use direct PyTorch and Transformers APIs. No new library is required beyond Part 2. The first model run downloads files if they are not already cached; see the [setup guide](../docs/setup-guide.md).

This is a learning engine, not a production server. It deliberately omits HTTP APIs, queues, multi-GPU scheduling, quantization, and multi-user concurrency. But it introduces the same *ideas* used in larger systems, in small readable programs.

## The request lifecycle

```text
request
  → validate settings
  → tokenize prompt
  → check context budget
  → run model
  → choose next token
  → repeat until stop condition
  → decode token IDs
  → response
```

## Exercises

| Exercise | Program | Big-system idea, made small |
|---:|---|---|
| 19 | [Define a request](19_define_a_request.py) | A request has inputs, settings, and constraints |
| 20 | [Build a tiny engine](20_build_a_tiny_engine.py) | Encapsulate model loading and generation behind one interface |
| 21 | [Control decoding](21_control_decoding.py) | Greedy decoding, temperature, and top-k sampling are policies |
| 22 | [Stop safely](22_stop_safely.py) | End tokens and context limits prevent runaway generation |
| 23 | [Reuse computation](23_reuse_computation.py) | A key-value cache avoids recalculating old tokens |
| 24 | [Stream the response](24_stream_the_response.py) | Deliver output incrementally instead of waiting for the full answer |

Run one file at a time from the repository root, using the same activated `.venv` as Part 2:

```bash
python 3-inference-engine/19_define_a_request.py
```

For lessons 20–24, make sure you installed the `[pretrained]` dependencies. The model is downloaded automatically if needed. Use short prompts and small token limits on a 4 GB RAM computer.

## What you should be able to explain

- What does the engine receive, and what does it return?
- Which parts belong to the model, and which parts are the engine's responsibility?
- Why does generation need a loop and a stopping rule?
- Why is a KV cache useful?
- What does streaming change for the user, even if the model computes one token at a time?

The examples prioritize clarity over speed. Real inference engines add scheduling, batching, memory management, monitoring, safety controls, and serving interfaces. You do not need to build those yet to understand the core flow.
