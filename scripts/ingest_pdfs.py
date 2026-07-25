"""
ingest_pdfs.py (OCR version, incremental / dedup-safe)

Before OCR'ing anything, this reads vector_store/icse_meta.jsonl and
builds a set of (book filename, page number) pairs already embedded.
Any page matching one of those pairs is skipped without rendering or
OCR. This means you can:
  - drop new PDFs into textbooks/ and just re-run this script
  - it will only process pages it hasn't seen before
  - the existing 801 Q&A pairs and any already-ingested textbook pages
    are never touched, re-embedded, or duplicated

If you ever need to force a re-ingestion of a book that's already in
the index (for example, you replaced it with a better-quality scan),
you need to manually remove its old lines from icse_meta.jsonl first
and rebuild the index from the Q&A baseline before re-running this —
ask before doing that, don't guess at it.

Run:
  python scripts/ingest_pdfs.py
  python scripts/ingest_pdfs.py --max_pages 5     # only look at first 5 pages of each PDF
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
        print(f"ERROR: folder '{TEXTBOOK_DIR}' not found. Create it and put PDF files inside.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(TEXTBOOK_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"ERROR: no PDF files found in '{TEXTBOOK_DIR}'.")
        sys.exit(1)

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

    for fname in pdf_files:
        path = os.path.join(TEXTBOOK_DIR, fname)
        print(f"Reading {fname}...")
        pages, skipped_already = ocr_pdf(path, fname, already_ingested, max_pages=args.max_pages)
        print(f"  {len(pages)} new pages OCR'd, {skipped_already} pages already in the index (skipped)")

        for page_num, text in pages:
            chunks = chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                new_records.append({
                    "type": "textbook",
                    "book": fname,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                    "source": "ocr",
                })

    if not all_chunks:
        print("No new pages found. Everything in textbooks/ is already in the index. Nothing to add.")
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
