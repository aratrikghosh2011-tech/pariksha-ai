"""
fix_unsorted_physics.py

ONE-TIME fix for the unsorted(221) bug from the per-subject index
migration. Moves textbook records from vector_store/unsorted/ into
vector_store/physics/, IF (and only if) every record in unsorted is
confirmed to be one of the 4 known physics textbook PDFs.

Does NOT touch vector_store/physics/qa.index - only adds/creates
vector_store/physics/textbook.index (physics had no textbook.index
before this fix, only qa.index from the 801 Q&A pairs).

Safety:
- Prints every distinct "book" value found in unsorted/textbook_meta.jsonl
  and STOPS without writing anything if any of them is not in the
  known physics filename list. You must re-run after confirming.
- Re-embeds text using the same model/settings as the rest of the
  pipeline (all-MiniLM-L6-v2, CPU, normalized).
- Appends to physics/textbook.index if it already exists (it shouldn't,
  but this is safe either way) - never overwrites.
- Only deletes vector_store/unsorted/ after a verified successful write
  and a re-read confirming the new physics/textbook.index count is
  correct.

Run:
  python scripts/fix_unsorted_physics.py
"""

import json
import os
import shutil
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

VECTOR_STORE_ROOT = "vector_store"
UNSORTED_DIR = os.path.join(VECTOR_STORE_ROOT, "unsorted")
UNSORTED_META = os.path.join(UNSORTED_DIR, "textbook_meta.jsonl")
PHYSICS_DIR = os.path.join(VECTOR_STORE_ROOT, "physics")
PHYSICS_INDEX = os.path.join(PHYSICS_DIR, "textbook.index")
PHYSICS_META = os.path.join(PHYSICS_DIR, "textbook_meta.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"

KNOWN_PHYSICS_BOOKS = {
    "electromagnetism", "calorimetry", "current electricity", "household circuits",
}


def looks_like_known_physics_book(book_name):
    lower = book_name.lower()
    return any(keyword.replace(" ", "") in lower.replace(" ", "").replace("_", "").replace("-", "")
               for keyword in KNOWN_PHYSICS_BOOKS)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    if not os.path.exists(UNSORTED_META):
        print(f"ERROR: {UNSORTED_META} not found. Nothing to fix. "
              f"Check whether the unsorted folder or filename is different than expected.")
        sys.exit(1)

    records = load_jsonl(UNSORTED_META)
    print(f"Loaded {len(records)} records from {UNSORTED_META}")

    distinct_books = sorted(set(r.get("book", "UNKNOWN") for r in records))
    print(f"\nDistinct 'book' values found in unsorted:")
    for b in distinct_books:
        count = sum(1 for r in records if r.get("book") == b)
        flag = "OK (matches known physics book)" if looks_like_known_physics_book(b) else "!!! UNRECOGNIZED !!!"
        print(f"  {b}  ({count} records)  -> {flag}")

    unrecognized = [b for b in distinct_books if not looks_like_known_physics_book(b)]
    if unrecognized:
        print(f"\nSTOPPING. Found {len(unrecognized)} book name(s) that do not match the "
              f"4 known physics textbooks (Electromagnetism, Calorimetry, Current Electricity, "
              f"Household Circuits): {unrecognized}")
        print("Do NOT proceed blindly. Check these records manually - they may belong to a "
              "different subject entirely, or the filename just doesn't contain an expected "
              "keyword. Report this output back before re-running.")
        sys.exit(1)

    print(f"\nAll {len(distinct_books)} book(s) confirmed as known physics textbooks. Proceeding.")

    non_textbook = [r for r in records if r.get("type") != "textbook"]
    if non_textbook:
        print(f"\nWARNING: {len(non_textbook)} records in unsorted are NOT type='textbook' "
              f"(unexpected). Their type values: {sorted(set(r.get('type') for r in non_textbook))}")
        print("Stopping - this script only handles textbook records. Investigate these separately.")
        sys.exit(1)

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = [r.get("text", "") for r in records]
    print(f"Embedding {len(texts)} records...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    os.makedirs(PHYSICS_DIR, exist_ok=True)

    if os.path.exists(PHYSICS_INDEX):
        print(f"\n{PHYSICS_INDEX} already exists - appending to it.")
        index = faiss.read_index(PHYSICS_INDEX)
        existing_count = index.ntotal
    else:
        print(f"\n{PHYSICS_INDEX} does not exist yet - creating new index.")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        existing_count = 0

    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, PHYSICS_INDEX)

    # Fix the subject field on each record before writing (was "unsorted", now "physics")
    for r in records:
        r["subject"] = "physics"

    mode = "a" if os.path.exists(PHYSICS_META) and existing_count > 0 else "w"
    with open(PHYSICS_META, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records. {PHYSICS_INDEX} now has {index.ntotal} vectors "
          f"(was {existing_count} before this run).")

    # Verify by re-reading from disk
    print("\nVerifying by re-reading index and meta file from disk...")
    reread_index = faiss.read_index(PHYSICS_INDEX)
    reread_meta = load_jsonl(PHYSICS_META)

    expected_total = existing_count + len(records)
    if reread_index.ntotal != expected_total:
        print(f"ERROR: verification FAILED. Index has {reread_index.ntotal} vectors, "
              f"expected {expected_total}. Do NOT delete vector_store/unsorted/. Stop here.")
        sys.exit(1)
    if len(reread_meta) != expected_total:
        print(f"ERROR: verification FAILED. Meta file has {len(reread_meta)} records, "
              f"expected {expected_total}. Do NOT delete vector_store/unsorted/. Stop here.")
        sys.exit(1)

    print(f"Verification passed: {reread_index.ntotal} vectors, {len(reread_meta)} meta records match.")

    print(f"\nDeleting {UNSORTED_DIR} (now empty of useful data, migration confirmed complete)...")
    shutil.rmtree(UNSORTED_DIR)
    print("Done. vector_store/unsorted/ removed.")

    print("\nSummary:")
    print(f"  Moved {len(records)} records from unsorted -> physics/textbook.index")
    print(f"  physics/textbook.index: {reread_index.ntotal} total vectors")
    print(f"  Run scripts/verify_vector_counts.py next to confirm the grand total across "
          f"all subjects is unchanged, and test a Physics-filtered query in app.py.")


if __name__ == "__main__":
    main()
