"""Exercise 29: separate a model call from the rest of your program.

This is a safe scaffold: the function reports common failure categories without
printing an API key. Exercise 28 contains the first real request.
"""

def explain_error(error: Exception) -> str:
    name = type(error).__name__.lower()
    if "auth" in name or "permission" in name:
        return "Authentication or permission problem: check your key and project."
    if "rate" in name or "quota" in name or "resourceexhausted" in name:
        return "Rate limit or quota reached: wait and retry later; do not enable billing."
    if "timeout" in name or "connection" in name:
        return "Network problem: check internet access and retry."
    if "notfound" in name:
        return "The selected model or endpoint may not be available."
    return "Unexpected error: inspect the error type and provider documentation."


def main() -> None:
    examples = [
        ValueError("missing input"),
        RuntimeError("illustrative rate limit"),
        ConnectionError("illustrative network issue"),
    ]
    for error in examples:
        print(f"{type(error).__name__}: {explain_error(error)}")
    print("\nThese are local examples, not live API failures.")


if __name__ == "__main__":
    main()
