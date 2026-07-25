"""
analyze_repetition.py

Loads dataset/parsed_raw_questions.json, excludes "generated" questions,
embeds each question (first 150 words, to weight the question stem over
answer-specific numbers) using all-MiniLM-L6-v2, and clusters them by
cosine similarity to find recurring question patterns across years.

Run:
  python scripts/analyze_repetition.py
  python scripts/analyze_repetition.py --threshold 0.2   # stricter matching
  python scripts/analyze_repetition.py --threshold 0.4   # looser matching
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_PATH = os.path.join("dataset", "parsed_raw_questions.json")
REPORT_PATH = os.path.join("dataset", "repetition_report.md")
MODEL_NAME = "all-MiniLM-L6-v2"
STEM_WORDS = 150


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data if q.get("source_type") != "generated"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3,
                         help="Cosine distance threshold (lower = stricter). Default 0.3 "
                              "means roughly 0.70+ similarity required to group together.")
    args = parser.parse_args()

    questions = load_questions(INPUT_PATH)
    print(f"Loaded {len(questions)} questions (excluding 'generated')")

    texts = []
    for q in questions:
        words = q["text"].split()
        stem = " ".join(words[:STEM_WORDS])
        texts.append(stem)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Embedding {len(texts)} questions on CPU...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Clustering with distance_threshold={args.threshold} (cosine)...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=args.threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    for label, q in zip(labels, questions):
        clusters[label].append(q)

    repeated_clusters = {label: members for label, members in clusters.items() if len(members) >= 2}
    sorted_clusters = sorted(repeated_clusters.items(), key=lambda kv: len(kv[1]), reverse=True)

    unique_count = sum(1 for m in clusters.values() if len(m) == 1)
    print(f"\nTotal clusters: {len(clusters)}")
    print(f"Clusters with 2+ questions (repeated patterns): {len(repeated_clusters)}")
    print(f"Questions with no match (unique): {unique_count}")

    lines = []
    lines.append("# PYQ Repetition Report\n\n")
    lines.append(f"Clustering threshold: {args.threshold} (cosine distance)\n\n")
    lines.append(f"Total questions analyzed: {len(questions)} (excludes 'generated')\n\n")
    lines.append(f"Repeated patterns found: {len(repeated_clusters)}\n\n")
    lines.append("---\n")

    for label, members in sorted_clusters:
        lines.append(f"\n## Repeated {len(members)}x\n\n")
        for m in members:
            year_str = m["year"] if m["year"] else "?"
            preview = m["text"][:200].replace("\n", " ").strip()
            lines.append(f"- **[{m['subject']} {year_str}, {m['source_type']}]** ({m['file']}): {preview}...\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nFull report saved to {REPORT_PATH}")

    print("\n--- TOP 15 MOST-REPEATED PATTERNS ---")
    for label, members in sorted_clusters[:15]:
        years = sorted(set(str(m["year"]) for m in members if m["year"]))
        subject = members[0]["subject"]
        preview = members[0]["text"][:120].replace("\n", " ").strip()
        print(f"\n[{len(members)}x, {subject}, years: {', '.join(years)}]")
        print(f"  {preview}...")


if __name__ == "__main__":
    main()
