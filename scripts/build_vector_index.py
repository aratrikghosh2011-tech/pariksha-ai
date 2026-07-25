"""
build_vector_index.py

Reads dataset/icse_train.jsonl, embeds every instruction-response pair
locally using sentence-transformers (all-MiniLM-L6-v2, CPU only),
builds a FAISS flat index, and saves:
  - vector_store/icse.index      (FAISS index)
  - vector_store/icse_meta.jsonl (one JSON object per vector, same order as index)

Run:
  python build_vector_index.py
"""

import json
import os
import sys
import time

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATASET_PATH = os.path.join("dataset", "icse_train.jsonl")
OUTPUT_DIR = "vector_store"
INDEX_PATH = os.path.join(OUTPUT_DIR, "icse.index")
META_PATH = os.path.join(OUTPUT_DIR, "icse_meta.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_dataset(path):
    if not os.path.exists(path):
        print(f"ERROR: dataset not found at {path}")
        print("Run this script from the repo root, or fix DATASET_PATH.")
        sys.exit(1)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {line_num}: {e}")
                continue
            if "instruction" not in obj or "response" not in obj:
                print(f"WARNING: skipping line {line_num}, missing instruction/response key")
                continue
            records.append(obj)
    return records


def build_index(records, model):
    # Embed instruction + response together so retrieval matches on the
    # full worked example, not just the question phrasing.
    texts = [f"{r['instruction']}\n{r['response']}" for r in records]

    print(f"Embedding {len(texts)} pairs on CPU (all-MiniLM-L6-v2, 384-dim)...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings.astype(np.float32))

    return index


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    records = load_dataset(DATASET_PATH)
    print(f"Loaded {len(records)} pairs")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    index = build_index(records, model)

    print(f"Writing index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"Writing metadata to {META_PATH}")
    with open(META_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Index built: {index.ntotal} vectors, dim={index.d}")
    print("Done.")


if __name__ == "__main__":
    main()
