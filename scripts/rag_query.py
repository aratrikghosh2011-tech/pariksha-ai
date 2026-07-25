"""
rag_query.py

CLI to test the RAG pipeline end-to-end:
  1. Embed the question with the same local model used to build the index
  2. Retrieve top-k similar ICSE Q&A pairs from FAISS
  3. Build a prompt with those pairs as context
  4. Send to the chosen LLM provider (gemini or nemotron)
  5. Print the answer

Run:
  python rag_query.py --query "A train travels 120km at 60km/h. Find the time taken." --provider gemini
  python rag_query.py --query "..." --provider nemotron --top_k 5
"""

import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from llm_providers import get_provider

INDEX_PATH = os.path.join("vector_store", "icse.index")
META_PATH = os.path.join("vector_store", "icse_meta.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are Pariksha AI, an ICSE Class 10 Maths and Physics tutor.
Answer strictly in ICSE board exam style: numbered steps, correct formulas,
correct units, final answer clearly marked. Use the reference examples below
only as style/method guidance — solve the actual question asked, do not copy
numbers from the examples."""


def load_meta(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def retrieve(query, model, index, meta, top_k):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append((float(score), meta[idx]))
    return results


def build_prompt(query, retrieved):
    context_blocks = []
    for score, r in retrieved:
        if r.get("type") == "textbook":
            context_blocks.append(
                f"Textbook excerpt (similarity {score:.2f}, {r['book']} p.{r['page']}):\n{r['text']}"
            )
        else:
            context_blocks.append(
                f"Example (similarity {score:.2f}):\nQ: {r['instruction']}\nA: {r['response']}"
            )
    context = "\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

Reference material:
{context}

Now answer this question:
{query}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="The ICSE Maths/Physics question to ask")
    parser.add_argument("--provider", default=None, help="gemini or nemotron. Defaults to PROVIDER in .env")
    parser.add_argument("--top_k", type=int, default=3, help="Number of reference examples to retrieve")
    args = parser.parse_args()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("ERROR: vector store not found. Run build_vector_index.py first.")
        return

    print("Loading index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    meta = load_meta(META_PATH)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Retrieving top {args.top_k} similar examples...")
    retrieved = retrieve(args.query, model, index, meta, args.top_k)
    for score, r in retrieved:
        if r.get("type") == "textbook":
            label = f"{r['book']} p.{r['page']}"
        else:
            label = r['instruction'][:70]
        print(f"  [{score:.3f}] {label}...")

    prompt = build_prompt(args.query, retrieved)

    print(f"Calling provider: {args.provider or os.getenv('PROVIDER', 'gemini')}")
    provider = get_provider(args.provider)
    answer = provider.generate(prompt)

    print("\n--- ANSWER ---")
    print(answer)


if __name__ == "__main__":
    main()
