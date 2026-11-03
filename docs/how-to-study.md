# How to study this repository

1. Open a numbered exercise and read the short challenge at the top.
2. Before running it, predict what it will print or what you expect the model to return.
3. Run the reference file without changing it.
4. Trace each important variable. Add temporary print statements when useful.
5. Make one change on your own branch, predict the effect, run it, and record the result.
6. Explain the concept in your own words before moving on.

For Part 3, draw the request lifecycle before running the code. Try changing one setting at a time: prompt, token budget, temperature, or top-k. For the cache lesson, compare what is sent to the model on the first step with what is sent on later steps. For streaming, notice when the first output becomes visible.

## How to learn the RAG project rather than only finish it

Keep the existing exercises and build on them in stages:

1. **Retrieval only:** run the document splitting and keyword retrieval exercises; inspect and explain every selected passage before adding a model.
2. **Basic RAG:** send those passages with the question through the Part 4 API helper. Trace the request from input to final response.
3. **Inspectable RAG:** print the source and text of retrieved passages, and look for cases where the system has weak or no evidence. Ask whether each important answer claim is supported.
4. **Semantic RAG:** only after you understand the keyword baseline, compare it with embedding-based retrieval. Record which questions improve and which get worse.
5. **Usable application:** add document handling, a small interface, clear errors, and a repeatable evaluation set.

At every stage, make a simple architecture sketch and follow one question through each component. Then deliberately break one thing at a time: ask about a fact absent from the documents, use wording that differs from the source, try an ambiguous query, or simulate an unavailable API. Record the observed result and your explanation. Do not judge the system by a fluent answer alone; inspect the retrieved evidence.

Do not add an external dependency casually. Before a new library is used in a lesson, update the root README and setup guide with what it does, why it is needed, and the exact installation command. Keep early retrieval work local and lightweight; no embedding model or vector database is needed for the first version.

Do not memorize the entire library API. Learn how to identify an object's role, find the right documented method, inspect its inputs and outputs, and connect the steps into a working program.

The code files are reference solutions. Try each task before reading the implementation.
