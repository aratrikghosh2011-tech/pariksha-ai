"""
enrich_repeated_patterns.py

Clusters PYQ data restricted to a recent-years window (default 2023
onward, matching ICSE's post-2022 shift toward HOTS/Assertion-Reason
questions), then adds one searchable record per repeated pattern
(cluster size 2+) to vector_store/icse.index as type "pyq_pattern".

Does NOT attempt to link back to icse_train.jsonl (different format,
no year tags there - would need its own fragile matching pass). Patterns
are added as standalone retrievable records instead.

Dedup-safe: each pattern gets a content-hash ID, re-running after future
data additions only adds genuinely new patterns.

Run:
  python scripts/enrich_repeated_patterns.py
  python scripts/enrich_repeated_patterns.py --min_year 2023
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_PATH = os.path.join("dataset", "parsed_raw_questions.json")
INDEX_PATH = os.path.join("vector_store", "icse.index")
META_PATH = os.path.join("vector_store", "icse_meta.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"
STEM_WORDS = 150
DISTANCE_THRESHOLD = 0.3  # verified against known repeats in the earlier full-history analysis


def load_questions(path, min_year):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = []
    for q in data:
        if q.get("source_type") == "generated":
            continue
        year = q.get("year")
        if year is None or year >= min_year:
            filtered.append(q)
    return filtered


def pattern_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_existing_pattern_ids(meta_path):
    seen = set()
    if not os.path.exists(meta_path):
        return seen
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "pyq_pattern":
                seen.add(obj.get("pattern_id"))
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_year", type=int, default=2023,
                         help="Only cluster real-year questions from this year onward. "
                              "Default 2023, matching the post-2022 ICSE pattern shift.")
    args = parser.parse_args()

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run parse_raw_pyq.py first.")
        sys.exit(1)
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("ERROR: existing index not found. Run build_vector_index.py first.")
        sys.exit(1)

    questions = load_questions(INPUT_PATH, args.min_year)
    print(f"Loaded {len(questions)} questions from {args.min_year} onward "
          f"(specimen/important always included regardless of year filter)")

    texts = [" ".join(q["text"].split()[:STEM_WORDS]) for q in questions]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Embedding {len(texts)} questions on CPU...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    print(f"Clustering with distance_threshold={DISTANCE_THRESHOLD} (verified value)...")
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=DISTANCE_THRESHOLD,
        metric="cosine", linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    cluster_embeddings = defaultdict(list)
    for label, q, emb in zip(labels, questions, embeddings):
        clusters[label].append(q)
        cluster_embeddings[label].append(emb)

    repeated = {label: members for label, members in clusters.items() if len(members) >= 2}
    print(f"Found {len(repeated)} repeated patterns from {args.min_year} onward "
          f"(compare to 149 in the earlier full 2016-2026 analysis)")

    print(f"Loading existing index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    print(f"Existing index has {index.ntotal} vectors")

    existing_ids = load_existing_pattern_ids(META_PATH)
    print(f"Found {len(existing_ids)} existing pyq_pattern records already in the index")

    new_records = []
    new_embeddings = []

    for label, members in repeated.items():
        rep_text = members[0]["text"][:600]
        pid = pattern_id(rep_text)
        if pid in existing_ids:
            continue

        years = sorted(set(m["year"] for m in members if m["year"]))
        subject = members[0]["subject"]

        summary_text = (
            f"[{subject}] Recurring ICSE question pattern, appeared {len(members)} times "
            f"({', '.join(str(y) for y in years)}). Example: {rep_text}"
        )

        new_records.append({
            "type": "pyq_pattern",
            "pattern_id": pid,
            "subject": subject,
            "repetition_count": len(members),
            "years_seen": years,
            "text": summary_text,
        })
        avg_emb = np.mean(cluster_embeddings[label], axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        new_embeddings.append(avg_emb)

    if not new_records:
        print("No new patterns to add (all already in the index, or none found). Nothing to add.")
        sys.exit(0)

    print(f"Adding {len(new_records)} new pattern records to the index")
    new_embeddings = np.array(new_embeddings, dtype=np.float32)
    index.add(new_embeddings)

    with open(META_PATH, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    faiss.write_index(index, INDEX_PATH)

    print(f"Done. Index now has {index.ntotal} vectors total ({len(new_records)} new pattern records).")


if __name__ == "__main__":
    main()
