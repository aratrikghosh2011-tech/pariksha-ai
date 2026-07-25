"""
parse_raw_pyq.py

Parses every dataset/raw_qa_p10_*.txt file into individual question
blocks, tags each with subject/year/type from the filename, and saves
everything to dataset/parsed_raw_questions.json.

Filename tagging rules:
  raw_qa_p10_{mth|phy}_{YYYY}.txt        -> type "real_year", that year
  raw_qa_p10_{mth|phy}_spec{YYYY}.txt    -> type "specimen", that year
  raw_qa_p10_{mth|phy}_imp{YYYY}.txt     -> type "important", that year
  raw_qa_p10_{mth|phy}_generated.txt     -> type "generated", no year

Run:
  python scripts/parse_raw_pyq.py
"""

import glob
import json
import os
import re
import sys

RAW_GLOB = os.path.join("dataset", "raw_qa_p10_*.txt")
OUTPUT_PATH = os.path.join("dataset", "parsed_raw_questions.json")

QUESTION_MARKER = re.compile(r'^Question\s+\d+', re.IGNORECASE)
SECTION_MARKER = re.compile(r'^SECTION\s', re.IGNORECASE)

FILENAME_PATTERN = re.compile(
    r'raw_qa_p10_(?P<subject>mth|phy)_(?P<tag>.+)\.txt$', re.IGNORECASE
)


def classify_filename(fname):
    m = FILENAME_PATTERN.search(fname)
    if not m:
        return {"subject": None, "year": None, "source_type": "unknown"}

    subject = "Maths" if m.group("subject").lower() == "mth" else "Physics"
    tag = m.group("tag").lower()

    if tag.isdigit():
        return {"subject": subject, "year": int(tag), "source_type": "real_year"}
    if tag.startswith("spec"):
        year_part = tag.replace("spec", "")
        year = int(year_part) if year_part.isdigit() else None
        return {"subject": subject, "year": year, "source_type": "specimen"}
    if tag.startswith("imp"):
        year_part = tag.replace("imp", "")
        year = int(year_part) if year_part.isdigit() else None
        return {"subject": subject, "year": year, "source_type": "important"}
    if tag == "generated":
        return {"subject": subject, "year": None, "source_type": "generated"}

    return {"subject": subject, "year": None, "source_type": "unknown"}


def parse_file(path):
    """Splits a raw file into question-block strings on 'Question N' markers."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    blocks = []
    current = []

    def flush():
        if current:
            text = "\n".join(current).strip()
            if text:
                blocks.append(text)

    for line in lines:
        stripped = line.strip()
        if SECTION_MARKER.match(stripped):
            continue
        if QUESTION_MARKER.match(stripped):
            flush()
            current = []
            continue
        current.append(line.rstrip("\n"))

    flush()
    return blocks


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        print(f"ERROR: no files matched {RAW_GLOB}")
        return

    print(f"Found {len(files)} raw files")

    all_questions = []
    per_file_counts = {}

    for path in files:
        fname = os.path.basename(path)
        meta = classify_filename(fname)
        blocks = parse_file(path)
        per_file_counts[fname] = len(blocks)

        for i, block in enumerate(blocks):
            all_questions.append({
                "file": fname,
                "subject": meta["subject"],
                "year": meta["year"],
                "source_type": meta["source_type"],
                "block_index": i,
                "text": block,
            })

    print("\nPer-file question counts:")
    for fname, count in per_file_counts.items():
        print(f"  {fname}: {count}")

    print(f"\nTotal questions parsed: {len(all_questions)}")

    by_type = {}
    for q in all_questions:
        by_type[q["source_type"]] = by_type.get(q["source_type"], 0) + 1
    print("\nBy source type:")
    for t, count in by_type.items():
        print(f"  {t}: {count}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n--- SAMPLE: first 3 parsed questions from a real_year file ---")
    real_year_samples = [q for q in all_questions if q["source_type"] == "real_year"][:3]
    for q in real_year_samples:
        print(f"\n[{q['file']} | {q['subject']} | {q['year']}]")
        print(q["text"][:300])
        print("...")


if __name__ == "__main__":
    main()
