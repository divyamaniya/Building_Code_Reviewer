# Building Code Reviewer (Ontario)

Building Code Reviewer is an AI-powered assistant designed to verify construction projects against the **Ontario Building Code (OBC)**. 

It helps users navigate complex regulations and identifies potential code violations in project descriptions and (eventually) architectural plans.

## What it does
* **Code Search:** Ask any question about Ontario building standards and get the exact clause as an answer.
* **Violation Check:** Tell the bot your situation (e.g., "I'm building a deck 3 feet off the ground"), and it will tell you if you are violating the code and how to fix it.
* **Plan Review (Future):** Upload a blueprint/drawing, and the AI will analyze the dimensions against OBC guidelines.

## How it works
This project uses a **RAG (Retrieval-Augmented Generation)** pipeline:
1.  **The Brain:** Uses advanced LLMs to understand construction context.
2.  **The Knowledge:** Specifically trained on the **2024 Ontario Building Code**.