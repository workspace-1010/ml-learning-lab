"""Exercise 28: send one prompt to the Gemini API.

Install the optional SDK first: python -m pip install -e ".[api]"
Set GEMINI_API_KEY in your environment. API availability and quotas vary.
"""

import os
import sys


def main() -> int:
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("Missing API key. Set GEMINI_API_KEY, then open a new terminal.")
        return 1

    try:
        from google import genai
    except ImportError:
        print('SDK missing. Run: python -m pip install -e ".[api]"')
        return 1

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = input("Enter a short question: ").strip()
    if not prompt:
        print("Please enter a question.")
        return 1

    try:
        client = genai.Client()
        response = client.models.generate_content(model=model, contents=prompt)
        print("\nModel response:\n")
        print(response.text or "(The response did not contain text.)")
    except Exception as exc:
        # Show the error class, but avoid printing secrets or request details.
        print(f"Request failed ({type(exc).__name__}).")
        print("Check the key, model availability, internet connection, and quota.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
