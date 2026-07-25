"""
compare_providers.py

Runs the same set of test questions through both Gemini and Nemotron,
using identical RAG-retrieved context for each question, and saves a
side-by-side comparison report.

Run:
  python scripts/compare_providers.py
"""

import os
import sys

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from llm_providers import get_provider
from rag_query import load_meta, retrieve, build_prompt, INDEX_PATH, META_PATH, MODEL_NAME

TEST_QUESTIONS = [
    "A train travels 120km at 60km/h. Find the time taken.",
    "State the principle of calorimetry and write its formula.",
    "Explain why the effective resistance of resistors in parallel is less than the smallest individual resistance.",
    "RT is a tangent to a circle, touching it at S. Given angle PST = 30 degrees and angle SPQ = 60 degrees, find angle PSQ.",
    "Calculate the heat lost when hot water is mixed with ice, standard calorimetry setup.",
]

REPORT_PATH = os.path.join("dataset", "provider_comparison.md")


def main():
    if not os.getenv("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY not set in .env. Add a real key from build.nvidia.com first.")
        sys.exit(1)

    print("Loading index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    meta = load_meta(META_PATH)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    gemini = get_provider("gemini")
    nemotron = get_provider("nemotron")

    lines = ["# Gemini vs Nemotron Comparison\n\n"]

    for i, q in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n=== Question {i}/{len(TEST_QUESTIONS)} ===")
        print(q)

        retrieved = retrieve(q, model, index, meta, top_k=3)
        prompt = build_prompt(q, retrieved)

        print("Calling Gemini...")
        try:
            gemini_answer = gemini.generate(prompt)
        except Exception as e:
            gemini_answer = f"ERROR: {e}"

        print("Calling Nemotron...")
        try:
            nemotron_answer = nemotron.generate(prompt)
        except Exception as e:
            nemotron_answer = f"ERROR: {e}"

        lines.append(f"## Question {i}: {q}\n\n")
        lines.append(f"### Gemini 3.5 Flash\n{gemini_answer}\n\n")
        lines.append(f"### Nemotron 3 Super\n{nemotron_answer}\n\n")
        lines.append("---\n\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nComparison saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
