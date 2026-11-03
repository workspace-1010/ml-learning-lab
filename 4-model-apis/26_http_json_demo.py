"""Exercise 26: inspect a JSON response without calling an online service.

Run: python 4-model-apis/26_http_json_demo.py
This uses only Python's standard library and deliberately makes no network request.
"""

import json

example_response = {
    "status": "ok",
    "model": "example-model",
    "text": "A model response would appear here.",
    "usage": {"input_tokens": 12, "output_tokens": 8},
}

if __name__ == "__main__":
    print("A JSON response is structured data, not just a block of text:")
    print(json.dumps(example_response, indent=2))
    parsed = json.loads(json.dumps(example_response))
    print("\nRead individual fields:")
    print("Status:", parsed["status"])
    print("Generated text:", parsed["text"])
    print("Input token count:", parsed["usage"]["input_tokens"])
    print("\nThis is sample data, not a real model response.")
