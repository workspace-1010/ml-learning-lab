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

## Before you run a model request

Choose a provider that offers a text-generation API and read its current documentation. Providers differ in endpoint URLs, model names, authentication, request formats, free quotas, and pricing. The later exercises should use one provider consistently, configured through environment variables. Do not assume an API is free or that every provider accepts the same request format.

This part does not require downloading model weights. The local program sends text to the provider and receives a response.

## Finish line

You can explain what an API does, distinguish the local application from the remote model, send a prompt, read the returned text, protect an API key, and recognize basic request errors.

Next: [Part 5 — Build a tiny RAG application](../5-simple-rag/README.md).
