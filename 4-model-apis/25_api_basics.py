"""Exercise 25: understand the pieces of a model API.

Run: python 4-model-apis/25_api_basics.py
No internet, API key, or extra package is required.
"""

parts = {
    "client": "Your Python program; it prepares a request and reads the response.",
    "endpoint": "The address where a service accepts requests.",
    "request": "The information you send, such as a prompt and model name.",
    "response": "The information returned by the service, often JSON.",
    "authentication": "A way to prove the caller is allowed to make the request.",
    "API key": "A secret credential; keep it out of source control.",
}

if __name__ == "__main__":
    print("A model API lets your program ask a remotely hosted model for output.\n")
    for name, meaning in parts.items():
        print(f"{name.title()}: {meaning}")
    print("\nImportant: your computer runs this Python program; the provider runs the model.")
