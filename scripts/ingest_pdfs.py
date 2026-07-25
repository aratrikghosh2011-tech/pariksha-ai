"""
ingest_pdfs.py (OCR version, subject-folder aware, incremental / dedup-safe)

Expects textbooks/ to contain subject subfolders, e.g.:
  textbooks/physics/Chapter 8 Current Electricity.pdf
  textbooks/chemistry/...
  textbooks/robotics/...
  textbooks/literature/...

Each chunk is tagged with "subject" (the subfolder name) and "book"
(the PDF filename). PDFs left directly in textbooks/ with no subject
folder are tagged subject="unsorted" and a warning is printed.

Dedup key is (book, page) only - moving an already-ingested PDF into a
subject folder does not cause it to be re-processed.

Run:
  python scripts/ingest_pdfs.py
  python scripts/ingest_pdfs.py --max_pages 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import faiss
import pdfplumber
import pytesseract
from sentence_transformers import SentenceTransformer

TEXTBOOK_DIR = "textbooks"
INDEX_PATH = os.path.join("vector_store", "icse.index")
META_PATH = os.path.join("vector_store", "icse_meta.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_WORDS = 350
OVERLAP_WORDS = 50
MIN_CHARS_PER_PAGE = 20
OCR_RESOLUTION = 150
OCR_TIMEOUT_SECONDS = 60

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def check_tesseract():
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print("ERROR: Tesseract executable not found at:")
        print(f"  {pytesseract.pytesseract.tesseract_cmd}")
        print("Install it from https://github.com/UB-Mannheim/tesseract/wiki")
        print("or fix the tesseract_cmd path at the top of this script.")
        sys.exit(1)


def find_pdfs(root):
    """Returns list of (subject, filename, full_path). Loose PDFs in root get subject='unsorted'."""
    found = []
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            subject = entry
            for fname in os.listdir(full):
                if fname.lower().endswith(".pdf"):
                    found.append((subject, fname, os.path.join(full, fname)))
        elif entry.lower().endswith(".pdf"):
            found.append(("unsorted", entry, full))
    return found


def load_already_ingested(meta_path):
    """Returns a set of (book, page) tuples already embedded as textbook chunks."""
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
            if obj.get("type") == "textbook":
                seen.add((obj.get("book"), obj.get("page")))
    return seen


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def ocr_pdf(path, fname, already_ingested, max_pages=None):
    """Returns (list of (page_number, text), count_skipped_already_ingested)."""
    results = []
    skipped_already = 0
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        if max_pages:
            pages = pages[:max_pages]

        total = len(pages)
        for i, page in enumerate(pages, start=1):
            if (fname, i) in already_ingested:
                skipped_already += 1
                continue

            print(f"  [page {i}/{total}] rendering page to image...", flush=True)
            render_start = time.time()
            page_image = page.to_image(resolution=OCR_RESOLUTION).original
            render_elapsed = time.time() - render_start
            print(f"  [page {i}/{total}] render done in {render_elapsed:.1f}s, "
                  f"now running OCR (timeout {OCR_TIMEOUT_SECONDS}s)...", flush=True)

            ocr_start = time.time()
            try:
                text = pytesseract.image_to_string(page_image, timeout=OCR_TIMEOUT_SECONDS)
            except RuntimeError as e:
                print(f"  [page {i}/{total}] OCR TIMED OUT or FAILED after "
                      f"{OCR_TIMEOUT_SECONDS}s: {e}. Skipping this page.")
                continue
            ocr_elapsed = time.time() - ocr_start
            print(f"  [page {i}/{total}] OCR done in {ocr_elapsed:.1f}s, "
                  f"{len(text.strip())} chars extracted", flush=True)

            if len(text.strip()) < MIN_CHARS_PER_PAGE:
                print(f"    WARNING: page {i} produced almost no text. Likely a blank page, "
                      f"a diagram-only page, or a bad scan. Skipping.")
                continue

            results.append((i, text))
    return results, skipped_already


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_pages", type=int, default=None,
                         help="Only look at this many pages per PDF (for testing)")
    args = parser.parse_args()

    check_tesseract()

    if not os.path.isdir(TEXTBOOK_DIR):
        print(f"ERROR: folder '{TEXTBOOK_DIR}' not found.")
        sys.exit(1)

    pdfs = find_pdfs(TEXTBOOK_DIR)
    if not pdfs:
        print(f"ERROR: no PDF files found anywhere under '{TEXTBOOK_DIR}'.")
        sys.exit(1)

    unsorted = [p for p in pdfs if p[0] == "unsorted"]
    if unsorted:
        print(f"WARNING: {len(unsorted)} PDF(s) found directly in '{TEXTBOOK_DIR}' with no "
              f"subject subfolder. Tagged subject='unsorted'. Move them into a subject folder "
              f"before the next run:")
        for _, fname, _ in unsorted:
            print(f"    {fname}")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("ERROR: existing index not found. Run build_vector_index.py first.")
        sys.exit(1)

    print(f"Loading existing index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    print(f"Existing index has {index.ntotal} vectors")

    already_ingested = load_already_ingested(META_PATH)
    print(f"Found {len(already_ingested)} already-ingested textbook pages. These will be skipped.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("Embedding model loaded OK")

    new_records = []
    all_chunks = []

    for subject, fname, path in pdfs:
        print(f"Reading [{subject}] {fname}...")
        pages, skipped_already = ocr_pdf(path, fname, already_ingested, max_pages=args.max_pages)
        print(f"  {len(pages)} new pages OCR'd, {skipped_already} pages already in the index (skipped)")

        for page_num, text in pages:
            chunks = chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                new_records.append({
                    "type": "textbook",
                    "subject": subject,
                    "book": fname,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                    "source": "ocr",
                })

    if not all_chunks:
        print("No new pages found. Everything under textbooks/ is already in the index. Nothing to add.")
        sys.exit(0)

    print(f"Embedding {len(all_chunks)} new textbook chunks on CPU...")
    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    index.add(embeddings.astype(np.float32))

    print(f"Appending {len(new_records)} new metadata records to {META_PATH}")
    with open(META_PATH, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saving updated index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"Done. Index now has {index.ntotal} vectors total ({len(new_records)} new).")


if __name__ == "__main__":
    main()
