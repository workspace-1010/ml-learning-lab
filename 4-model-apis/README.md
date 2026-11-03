# Part 4 — Use a model through an API

## Why this part exists

Parts 2 and 3 run DistilGPT-2 on the learner's own computer. That is useful for understanding model internals, but a computer with only 4 GB RAM may struggle to run larger models.

A model API lets a program send a request to a model hosted on another computer and receive the generated text. The learner's computer runs the Python application; the provider's server runs the model.

## What is an API?

An API is a defined way for one program to ask another program to do something. For a model API, the basic cycle is:

1. Your Python program prepares a request.
2. It sends the request over the internet, usually with HTTPS.
3. The provider checks authentication and processes the request.
4. The provider returns a response, commonly formatted as JSON.
5. Your program reads the response and uses the generated text.

An API key is a secret credential that identifies or authorizes the caller. Never put a real key in source code, commit it to Git, or share it in screenshots. Store it in an environment variable. API calls may cost money and need an internet connection; check the provider's current pricing and limits.

## Exercise path

| Exercise | Idea |
|---:|---|
| 25 | Understand request, response, endpoint, JSON, and API key |
| 26 | Make a harmless HTTP request and inspect a JSON response |
| 27 | Read API settings from environment variables |
| 28 | Send a prompt to a hosted language model |
| 29 | Handle errors and keep the API call separate from the rest of the program |

## Provider for this course: Gemini Developer API

We use the Gemini Developer API as the course's first provider because Google offers a free tier for selected models, subject to eligibility, quotas, and changing availability. Do not enable paid billing just to finish this course. If the free tier is unavailable or a quota is exhausted, pause live-call exercises and continue with offline exercises. Check the [current pricing and free-tier terms](https://ai.google.dev/gemini-api/docs/pricing) before making requests.

Privacy note: content submitted through the free tier may be used to improve Google's products. Use only fictional or public practice documents. Never send private school records, personal information, passwords, or confidential files. See Google's [API key guidance](https://ai.google.dev/gemini-api/docs/api-key).

## Setup: one account and key per learner

1. Open [Google AI Studio](https://aistudio.google.com/) and sign in with your own account.
2. Create an API key and follow the current on-screen instructions.
3. Check that your account has access to a model in the free tier. Examples default to `gemini-2.5-flash`; if unavailable, set `GEMINI_MODEL` to a model currently available to you on the free tier.
4. Set the key as an environment variable named `GEMINI_API_KEY`. On Windows, use **Edit environment variables for your account**, add a user variable named `GEMINI_API_KEY`, then open a new terminal. On macOS/Linux, you can temporarily use `export GEMINI_API_KEY=your_key` in the current terminal.
5. From the repository root, install the optional SDK: `python -m pip install -e ".[api]"`.
6. Check configuration without revealing the secret: `python 4-model-apis/27_environment_variables.py`.

Never paste a real key into code, commit it, share it in screenshots, or use a shared class key. Do not enable paid billing to complete a required exercise. API calls require internet and available quota; the local exercises remain useful when live calls are unavailable.

This part does not require downloading model weights. The local program sends text to the provider and receives a response.

## Finish line

You can explain what an API does, distinguish the local application from the remote model, send a prompt, read the returned text, protect an API key, and recognize basic request errors.

Next: [Part 5 — Build a tiny RAG application](../5-simple-rag/README.md).
