"""
ingest_pdfs.py (per-subject output, PDF + HTML support, exercises truncation)

Walks textbooks/<subject>/ folders. Handles two file types:
  .pdf: renders each page to an image, OCRs it with Tesseract
  .html/.htm: extracts text directly, no OCR needed

Truncates ingested text at an EXERCISES-style section header - unsolved
practice questions have no recoverable answer in the source (even if
answered by hand in pencil, OCR cannot read that), and would pollute
retrieval with unanswered prompts instead of useful reference material.

Writes into vector_store/<subject>/textbook.index + textbook_meta.jsonl
- one pair of files per subject, never shared across subjects.

For files with "caesar" in the filename, only content from the first
Act 3 marker onward is kept (Acts 1-2 excluded).

Run:
  python scripts/ingest_pdfs.py
  python scripts/ingest_pdfs.py --max_pages 5
  python scripts/ingest_pdfs.py --subjects geography,biology
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import faiss
import pdfplumber
import pytesseract
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TEXTBOOK_DIR = "textbooks"
VECTOR_STORE_ROOT = "vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_WORDS = 350
OVERLAP_WORDS = 50
MIN_CHARS_PER_PAGE = 20
OCR_RESOLUTION = 150
OCR_TIMEOUT_SECONDS = 60

EXERCISE_MARKERS = re.compile(
    r'\b(EXERCISES?|EXERCISE\s+\d|UNSOLVED\s+QUESTIONS?|PRACTICE\s+QUESTIONS?)\b',
    re.IGNORECASE
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def check_tesseract():
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print("ERROR: Tesseract executable not found at:")
        print(f"  {pytesseract.pytesseract.tesseract_cmd}")
        print("Install it from https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)


def truncate_at_exercises(text):
    match = EXERCISE_MARKERS.search(text)
    if match:
        return text[:match.start()].strip()
    return text


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


def find_subject_files(root):
    found = []
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            subject = entry.strip().lower()
            for fname in os.listdir(full):
                lower = fname.lower()
                if lower.endswith((".pdf", ".html", ".htm")):
                    found.append((subject, fname, os.path.join(full, fname)))
        elif entry.lower().endswith((".pdf", ".html", ".htm")):
            found.append(("unsorted", entry, full))
    return found


def load_already_ingested(meta_path):
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
            seen.add((obj.get("book"), obj.get("page")))
    return seen


def ocr_pdf_pages(path, fname, already_ingested, max_pages=None):
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
                print(f"  [page {i}/{total}] OCR TIMED OUT or FAILED: {e}. Skipping.")
                continue
            ocr_elapsed = time.time() - ocr_start

            text = truncate_at_exercises(text)

            print(f"  [page {i}/{total}] OCR done in {ocr_elapsed:.1f}s, "
                  f"{len(text.strip())} chars (after exercises truncation)", flush=True)

            if len(text.strip()) < MIN_CHARS_PER_PAGE:
                print(f"    WARNING: page {i} produced almost no usable text "
                      f"(blank, diagram-only, bad scan, or entirely an exercises section). Skipping.")
                continue

            results.append((i, text))
    return results, skipped_already


def extract_html_text(path):
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_html = f.read()

    parser = TextExtractor()
    parser.feed(raw_html)
    text = " ".join(parser.parts)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ingest_html(path, fname, already_ingested, act_filter=None):
    if (fname, 1) in already_ingested:
        return [], 1

    text = extract_html_text(path)

    if act_filter == "act_3_onward":
        match = re.search(r'\bACT\s+(III|3|THREE)\s*,?\s*Scene\b', text, re.IGNORECASE)
        if match:
            print(f"  Found Act 3 marker at character {match.start()}, discarding Acts 1-2 before it")
            text = text[match.start():]
        else:
            print(f"  WARNING: no 'Act 3' marker found in {fname} - could not restrict to "
                  f"Act 3 onward. Ingesting the FULL text instead - check this manually, "
                  f"this may include Acts 1-2 which were not wanted.")

    return [(1, text)], 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subjects to process. Default: all found.")
    args = parser.parse_args()

    check_tesseract()

    if not os.path.isdir(TEXTBOOK_DIR):
        print(f"ERROR: folder '{TEXTBOOK_DIR}' not found.")
        sys.exit(1)

    all_files = find_subject_files(TEXTBOOK_DIR)
    if not all_files:
        print(f"ERROR: no PDF/HTML files found under '{TEXTBOOK_DIR}'.")
        sys.exit(1)

    if args.subjects:
        wanted = set(s.strip().lower() for s in args.subjects.split(","))
        all_files = [f for f in all_files if f[0] in wanted]
        print(f"Filtering to subjects: {sorted(wanted)}")

    by_subject = {}
    for subject, fname, path in all_files:
        by_subject.setdefault(subject, []).append((fname, path))

    print(f"Found files in {len(by_subject)} subjects: {sorted(by_subject.keys())}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    for subject, files in by_subject.items():
        print(f"\n=== Subject: {subject} ===")
        subject_dir = os.path.join(VECTOR_STORE_ROOT, subject)
        os.makedirs(subject_dir, exist_ok=True)
        index_path = os.path.join(subject_dir, "textbook.index")
        meta_path = os.path.join(subject_dir, "textbook_meta.jsonl")

        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)

        already_ingested = load_already_ingested(meta_path)
        print(f"{index.ntotal} existing vectors, {len(already_ingested)} already-ingested pages")

        new_records = []
        all_chunks = []

        for fname, path in files:
            print(f"Reading {fname}...")
            lower = fname.lower()

            if lower.endswith(".pdf"):
                pages, skipped = ocr_pdf_pages(path, fname, already_ingested, max_pages=args.max_pages)
                print(f"  {len(pages)} new pages, {skipped} already ingested (skipped)")
            elif lower.endswith((".html", ".htm")):
                act_filter = "act_3_onward" if "caesar" in fname.lower() else None
                pages, skipped = ingest_html(path, fname, already_ingested, act_filter=act_filter)
                if skipped:
                    print(f"  Already ingested, skipping")
            else:
                continue

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
                        "source": "ocr" if lower.endswith(".pdf") else "html",
                    })

        if not all_chunks:
            print(f"No new content for {subject}. Nothing to add.")
            continue

        print(f"Embedding {len(all_chunks)} new chunks for {subject}...")
        embeddings = model.encode(
            all_chunks, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        index.add(embeddings.astype(np.float32))

        with open(meta_path, "a", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        faiss.write_index(index, index_path)
        print(f"{subject}: index now has {index.ntotal} vectors total ({len(new_records)} new)")

    print("\nAll subjects processed.")


if __name__ == "__main__":
    main()
