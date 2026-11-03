"""Exercise 19 — Define what an inference request contains.

Large systems make request settings explicit before work begins.
This tiny example validates a request without loading the model.
"""

request = {
    "prompt": "A tiny computer can",
    "max_new_tokens": 12,
    "temperature": 0.8,
    "top_k": 20,
}

if not request["prompt"].strip():
    raise ValueError("The prompt cannot be empty.")

if not 1 <= request["max_new_tokens"] <= 32:
    raise ValueError("For this lesson, request 1–32 new tokens.")

if request["temperature"] <= 0:
    raise ValueError("Temperature must be greater than zero.")

if request["top_k"] < 1:
    raise ValueError("top_k must be at least 1.")

print("Validated inference request:")
for key, value in request.items():
    print(f"  {key}: {value}")

print("\nThe engine will turn this request into tokens, run the model,")
print("choose new tokens, stop, and return decoded text.")
