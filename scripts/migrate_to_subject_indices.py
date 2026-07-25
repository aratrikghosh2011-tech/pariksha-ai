"""
migrate_to_subject_indices.py

ONE-TIME migration: reads the existing combined vector_store/icse_meta.jsonl
and splits every record into separate FAISS indices per (subject, bucket):

  vector_store/<subject>/textbook.index + textbook_meta.jsonl
  vector_store/<subject>/qa.index + qa_meta.jsonl   (Q&A pairs + pyq_pattern records)

Does NOT re-run OCR or re-parse raw PYQ files - re-embeds the ALREADY
EXTRACTED text from existing metadata (fast) and writes it into the new
structure. Nothing is lost, only reorganized.

Run:
  python scripts/migrate_to_subject_indices.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

OLD_META_PATH = os.path.join("vector_store", "icse_meta.jsonl")
NEW_ROOT = "vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_subject(r):
    if r.get("subject"):
        return r["subject"].strip().lower()
    instr = r.get("instruction", "")
    if instr.startswith("["):
        end = instr.find("]")
        if end != -1:
            return instr[1:end].strip().lower()
    return "unsorted"


def get_bucket(r):
    if r.get("type") == "textbook":
        return "textbook"
    return "qa"


def get_embed_text(r):
    if r.get("type") in ("textbook", "pyq_pattern"):
        return r.get("text", "")
    return f"{r.get('instruction', '')}\n{r.get('response', '')}"


def main():
    if not os.path.exists(OLD_META_PATH):
        print(f"ERROR: {OLD_META_PATH} not found. Nothing to migrate.")
        sys.exit(1)

    print(f"Loading old metadata from {OLD_META_PATH}")
    records = []
    with open(OLD_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} total records")

    groups = defaultdict(list)
    for r in records:
        groups[(get_subject(r), get_bucket(r))].append(r)

    print(f"\nFound {len(groups)} (subject, bucket) groups:")
    for (subject, bucket), recs in sorted(groups.items()):
        print(f"  {subject}/{bucket}: {len(recs)} records")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    total_written = 0
    for (subject, bucket), recs in groups.items():
        subject_dir = os.path.join(NEW_ROOT, subject)
        os.makedirs(subject_dir, exist_ok=True)

        index_path = os.path.join(subject_dir, f"{bucket}.index")
        meta_path = os.path.join(subject_dir, f"{bucket}_meta.jsonl")

        texts = [get_embed_text(r) for r in recs]
        print(f"\nEmbedding {len(texts)} records for {subject}/{bucket}...")
        embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, index_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  Wrote {index.ntotal} vectors to {index_path}")
        total_written += index.ntotal

    print(f"\nDone. Total vectors written: {total_written}")
    print(f"Original combined store had: {len(records)}")

    if total_written != len(records):
        print("WARNING: totals don't match. Do NOT delete old files. Investigate before proceeding.")
    else:
        print("Totals match exactly. Safe to proceed.")


if __name__ == "__main__":
    main()
