# ML Learning Lab

An exercise-first path from **basic machine learning** to understanding a pretrained model, building a tiny inference engine, using a hosted model API, and building a simple RAG application.

The format is inspired by the structure of beginner-friendly programming courses such as Codédex: numbered topic folders, one small runnable program per exercise, and a single index that helps you follow the sequence. The code and explanations here are original; this is not a copy of another course.

## What you will learn

**Part 1 — Train a neural network:** train a small NumPy neural network on MNIST handwritten digits, inspect loss and accuracy, visualize predictions, then build a maze agent that learns from actions and rewards. The original 01–09 exercises remain as foundations while the project sequence is developed.

**Part 2 — Assemble and inspect DistilGPT2:** tokenize text, inspect configuration and files, load the checkpoint, examine learned parameters, run the forward pass, inspect next-token scores, and write a token-by-token generation loop.

**Part 3 — Build a tiny inference engine:** define a request, wrap model use behind an interface, control decoding, enforce stop/context limits, use a KV cache, and stream output token by token.

The target is to understand useful APIs and how the pieces fit together—not to memorize every function, build a production server, or train an LLM from scratch.

## Libraries used in this course

**Rule: every external library is named here before learners encounter it.** Python's built-in syntax and standard library are assumed; everything else is listed below with its purpose and when to install it.

| Course section | Library/tool | What it does | When it is needed |
|---|---|---|---|
| All sections | Python 3.10+ | Language used to write the exercises | Install Python first |
| Part 1 — current exercises | NumPy | Stores numerical data in arrays and supports numerical calculations | Basic setup |
| Part 1 — current exercises | scikit-learn | Provides starter ML models, datasets, train/test splitting and evaluation metrics | Basic setup |
| Part 1 — tests | pytest | Runs automated checks for course code | Optional development install via `.[dev]` |
| Part 1 — MNIST project | NumPy | Store image arrays and implement the neural network, backpropagation and gradient-descent updates | Basic setup |
| Part 1 — MNIST project | Matplotlib | Display test digits and model predictions | Optional: install with `python -m pip install -e ".[visual]"` |
| Part 1 — MNIST project | MNIST dataset | 28×28 handwritten digit images and labels; downloaded automatically from a public Google-hosted mirror | Exercise 01 downloads it on first run; internet required |
| Part 1 — maze project | Python standard library (`random`, `argparse`, `collections`, `dataclasses`) | Train a tabular Q-learning agent in a maze using actions and rewards | No extra install needed |
| Part 2 — pretrained model | PyTorch | Runs tensor operations and the neural network computations | Install `.[pretrained]` |
| Part 2 — pretrained model | Hugging Face Transformers | Loads the tokenizer, model architecture and pretrained checkpoint | Install `.[pretrained]` |
| Parts 2–3 — model files | Hugging Face Hub (installed as a dependency) | Downloads and caches model files | Installed with the model dependencies |
| Part 4 — model API | Google Gen AI SDK (`google-genai`) | Sends prompts to a remotely hosted Gemini model | Install `.[api]`; internet and an available API quota required |
| Part 5 — simple RAG | Python standard library | Reads local text files and builds the prompt | No extra install planned |
| Part 5 — simple RAG | Google Gen AI SDK (`google-genai`) | Sends retrieved passages and a question to the hosted model | Same optional `.[api]` install as Part 4 |

The MNIST project is available in [`1-first-ml/mnist/`](1-first-ml/mnist/README.md), and the maze reinforcement-learning project is available in [`1-first-ml/maze/`](1-first-ml/maze/README.md). The maze agent uses only Python's standard library. No library is required just because it is popular: we add dependencies only when a project needs them and explain their role.

## Start here: setup in VS Code

For the complete beginner walkthrough, including every required library installation and the model-file download steps, read **[the setup guide](docs/setup-guide.md)** first. This project assumes you already have Python and VS Code installed and have opened the repository folder in VS Code.

Open **Terminal → New Terminal** in VS Code. Create and activate a virtual environment:

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the libraries for the existing Part 1 foundations (NumPy, scikit-learn, and pytest):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the MNIST visualization, optionally install Matplotlib with `python -m pip install -e ".[visual]"`. Exercises 01–02 in the MNIST project need no additional library beyond NumPy.

When you reach Part 2, install the additional libraries (PyTorch and Transformers):

```bash
python -m pip install -e ".[pretrained]"
```

Part 3 uses those same libraries and the same DistilGPT2 model. **No additional library or model download is required** if Part 2 is already set up. If you skipped Part 2, follow its setup instructions first.

When you reach Part 4, install the optional Gemini SDK with `python -m pip install -e ".[api]"`. Follow the [Part 4 setup guide](4-model-apis/README.md) and do not enable paid billing for this course.

## Part 1 — Train a neural network on MNIST

The original exercises 01–09 remain as a foundation for datasets, features, fitting, prediction, evaluation, classification and overfitting. The first hands-on project trains a small neural network **from scratch with NumPy** to recognize handwritten digits. It includes automatic or manual dataset acquisition, training with backpropagation and gradient descent, and a visualization of test predictions. Start with the [MNIST project guide](1-first-ml/mnist/README.md). The second project trains a maze agent using reinforcement learning and Q-learning. Start with the [maze project guide](1-first-ml/maze/README.md).

| Exercise | Program | Idea |
|---:|---|---|
| 01 | [The dataset](1-first-ml/01_the_dataset.py) | Inspect examples before fitting |
| 02 | [Features and targets](1-first-ml/02_features_and_targets.py) | Separate input data from the value to predict |
| 03 | [Split the data](1-first-ml/03_split_the_data.py) | Hold back examples for testing |
| 04 | [Create a model](1-first-ml/04_create_a_model.py) | Model object vs. trained model |
| 05 | [Fit the model](1-first-ml/05_fit_the_model.py) | Learn parameters from data |
| 06 | [Make predictions](1-first-ml/06_make_predictions.py) | Predict values for new inputs |
| 07 | [Measure error](1-first-ml/07_measure_error.py) | Evaluate held-out predictions |
| 08 | [Classify flowers](1-first-ml/08_classify_flowers.py) | Predict categories |
| 09 | [Spot overfitting](1-first-ml/09_spot_overfitting.py) | Compare training and test performance |

## Part 2 — DistilGPT2, without a pipeline

We use one specific open model throughout: **[`distilbert/distilgpt2`](https://huggingface.co/distilbert/distilgpt2)**. Its model card lists 82 million parameters and the Apache 2.0 license. It is a small English text-completion model, not a chat assistant. Read [Part 2's guide](2-distilgpt2/README.md) before running these exercises.

| Exercise | Program | Idea |
|---:|---|---|
| 10 | [Tokenize text](2-distilgpt2/10_tokenize_text.py) | Text becomes token IDs |
| 11 | [Read the configuration](2-distilgpt2/11_read_the_configuration.py) | Inspect the model blueprint |
| 12 | [List model files](2-distilgpt2/12_list_the_model_files.py) | Distinguish weights, config and tokenizer files |
| 13 | [Load the weights](2-distilgpt2/13_load_the_weights.py) | Construct the architecture and load its checkpoint |
| 14 | [Inspect the weights](2-distilgpt2/14_inspect_the_weights.py) | Names, shapes, dtypes and parameter counts |
| 15 | [Run the forward pass](2-distilgpt2/15_run_the_forward_pass.py) | Inspect raw logits and their dimensions |
| 16 | [Choose the next token](2-distilgpt2/16_choose_the_next_token.py) | Read the top candidate scores |
| 17 | [Generate token by token](2-distilgpt2/17_generate_one_token_at_a_time.py) | Implement greedy autoregressive decoding |
| 18 | [Explore sampling](2-distilgpt2/18_explore_sampling.py) | Turn scores into a sampled next token |

## Part 3 — Build a tiny inference engine

Part 2 showed how the model predicts tokens. Part 3 wraps that process into a component an application can call. The lessons introduce request validation, a reusable engine interface, decoding policies, stopping/context limits, KV caching, and streaming.

| Exercise | Program | Idea |
|---:|---|---|
| 19 | [Define a request](3-inference-engine/19_define_a_request.py) | Make input and generation settings explicit |
| 20 | [Build a tiny engine](3-inference-engine/20_build_a_tiny_engine.py) | Encapsulate loading and generation behind one interface |
| 21 | [Control decoding](3-inference-engine/21_control_decoding.py) | Temperature and top-k sampling |
| 22 | [Stop safely](3-inference-engine/22_stop_safely.py) | End tokens and model context limits |
| 23 | [Reuse computation](3-inference-engine/23_reuse_computation.py) | Reuse past key/value states with a KV cache |
| 24 | [Stream the response](3-inference-engine/24_stream_the_response.py) | Show output incrementally as tokens arrive |

Run one exercise at a time from the repository root, for example:

```bash
python 3-inference-engine/19_define_a_request.py
```

For model exercises, use the same activated Python environment and keep prompts/generation lengths short on a 4 GB RAM computer. Read [the Part 3 guide](3-inference-engine/README.md) first.

Every model exercise uses direct Transformers APIs such as `AutoTokenizer` and `AutoModelForCausalLM`. **No `pipeline()` is used**, because loading, tokenization, logits and generation are what we want to learn.

## Part 4 — Use a model through an API

We use the Gemini Developer API free tier where available. Each learner creates their own key; do not enable paid billing to finish the course. See [Part 4's guide](4-model-apis/README.md) for setup, privacy, and quota notes.

| Exercise | Program | Idea |
|---:|---|---|
| 25 | [API basics](4-model-apis/25_api_basics.py) | Understand client, endpoint, request, response and authentication |
| 26 | [HTTP and JSON](4-model-apis/26_http_json_demo.py) | Inspect a sample JSON response offline |
| 27 | [Environment variables](4-model-apis/27_environment_variables.py) | Check configuration without printing the key |
| 28 | [First Gemini request](4-model-apis/28_first_gemini_request.py) | Send a prompt and display generated text |
| 29 | [Error handling](4-model-apis/29_api_errors.py) | Recognize common failure categories |

| Exercise | Program | Idea |
|---:|---|---|
| 25 | API concepts | Understand requests, responses, endpoints and authentication |
| 26 | HTTP and JSON | Inspect a simple API response |
| 27 | API settings | Read configuration from environment variables |
| 28 | Call a model API | Send a prompt and read generated text |
| 29 | Handle API errors | Deal with missing settings, failed requests and bad responses |

## Part 5 — Build a simple RAG application

Use fictional local text files as a knowledge base, retrieve passages with keyword matching, construct a prompt, and send it to the API. See [Part 5's guide](5-simple-rag/README.md).

| Exercise | Program | Idea |
|---:|---|---|
| 30 | [Create a knowledge base](5-simple-rag/30_create_knowledge_base.py) | Store fictional practice facts locally |
| 31 | [Split documents](5-simple-rag/31_split_documents.py) | Turn documents into passages |
| 32 | [Retrieve passages](5-simple-rag/32_keyword_retrieval.py) | Find useful text with keyword matching |
| 33 | [Build the prompt](5-simple-rag/33_build_rag_prompt.py) | Combine evidence and the question |
| 34 | [Generate with RAG](5-simple-rag/34_ask_gemini_with_rag.py) | Call the API using retrieved context |
| 35 | [Test RAG limits](5-simple-rag/35_test_rag_limits.py) | Inspect retrieval and unsupported-answer failures |

| Exercise | Program | Idea |
|---:|---|---|
| 30 | Create a knowledge base | Store a few facts locally |
| 31 | Split documents | Turn documents into passages |
| 32 | Retrieve passages | Find useful text with keyword matching |
| 33 | Build the context prompt | Combine evidence and the question |
| 34 | Generate with RAG | Call the model API using retrieved context |
| 35 | Test failures | Compare results and inspect unsupported answers |

Start with keyword retrieval so the learner can understand the complete RAG process before learning embeddings or vector databases.

## Designed for a modest computer

Start with CPU inference, short prompts, and small generation lengths. The checkpoint is about 353 MB, but the full program uses more RAM because Python, PyTorch and intermediate tensors also require memory. Four GB is a constrained target, so close memory-heavy programs. Do not download every alternate model export, and never commit weights or model caches to this repository.

## Course design and study method

- [Course design](docs/course-design.md)
- [How to study](docs/how-to-study.md)
- [Setup guide](docs/setup-guide.md)
- [Learning map](docs/learning-map.md)
- [Glossary](docs/glossary.md)
- [Contribution guidelines](CONTRIBUTING.md)

RAG is introduced only after the learner has studied the model, the inference path, and remote model APIs. Vector databases and embeddings are optional extensions, not prerequisites for the first RAG project.
