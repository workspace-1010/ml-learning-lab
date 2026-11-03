"""Exercise 27: read configuration from environment variables.

Run: python 4-model-apis/27_environment_variables.py
It never prints the API key itself.
"""

import os

if __name__ == "__main__":
    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if key:
        print("GEMINI_API_KEY is set.")
    else:
        print("GEMINI_API_KEY is not set yet.")
        print("Create a key in Google AI Studio, then set it as an environment variable.")
    print(f"Selected model name: {model}")
    print("\nNever paste a real key into this file or commit it to GitHub.")
