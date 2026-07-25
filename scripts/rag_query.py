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

SYSTEM_PROMPT = """You are Pariksha AI, an ICSE Class 10 tutor covering Maths,
Physics, Chemistry, Robotics, and Literature.

Answer in ICSE board exam style appropriate to the subject of the question:
- Maths/Physics/Chemistry: numbered steps, correct formulas, correct units,
  final answer clearly marked.
- Literature: structured analysis with reference to the text, clear points,
  no invented quotes beyond what's given in the reference material.
- Robotics/Computer Applications: clear explanation of the concept or process,
  step-by-step where the question calls for a procedure.

Use the reference examples below only as style/method guidance — solve the
actual question asked, do not copy numbers or specifics from the examples."""


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


def get_subject(r):
    """
    Returns the subject of a record. Textbook and pyq_pattern records
    already have a "subject" field. Older Q&A pairs don't - their
    subject is embedded as a "[Maths]" or "[Physics]" prefix in the
    instruction text, so parse it out for those.
    """
    if r.get("subject"):
        return r["subject"]
    instr = r.get("instruction", "")
    if instr.startswith("["):
        end = instr.find("]")
        if end != -1:
            return instr[1:end]
    return "unknown"


def retrieve_filtered(query, model, index, meta, top_k, subject_filter=None, over_fetch=50):
    """
    Like retrieve(), but if subject_filter is set (not None and not "All"),
    prefers matches from that subject. Fetches a larger candidate pool
    first, then filters. If there aren't enough same-subject matches to
    fill top_k, fills remaining slots with the next-best overall matches
    - callers can check each result's subject via get_subject() to know
    which slots were same-subject vs fallback.
    """
    if not subject_filter or subject_filter == "All":
        return retrieve(query, model, index, meta, top_k)

    candidates = retrieve(query, model, index, meta, over_fetch)

    matching = [(score, r) for score, r in candidates
                if get_subject(r).lower() == subject_filter.lower()]

    if len(matching) >= top_k:
        return matching[:top_k]

    matching_ids = set(id(r) for _, r in matching)
    fallback = [(score, r) for score, r in candidates if id(r) not in matching_ids]
    return matching + fallback[:top_k - len(matching)]


def build_prompt(query, retrieved):
    context_blocks = []
    for score, r in retrieved:
        if r.get("type") == "textbook":
            context_blocks.append(
                f"Textbook excerpt (similarity {score:.2f}, {r['book']} p.{r['page']}):\n{r['text']}"
            )
        elif r.get("type") == "pyq_pattern":
            context_blocks.append(
                f"Exam pattern note (similarity {score:.2f}): {r['text']}"
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


def list_available_subjects(vector_store_root="vector_store"):
    """Returns subjects that have at least one index file, excluding legacy combined files."""
    subjects = []
    if not os.path.isdir(vector_store_root):
        return subjects
    for entry in os.listdir(vector_store_root):
        full = os.path.join(vector_store_root, entry)
        if os.path.isdir(full):
            has_index = any(f.endswith(".index") for f in os.listdir(full))
            if has_index:
                subjects.append(entry)
    return sorted(subjects)


def load_subject_store(subject, vector_store_root="vector_store"):
    """Loads every (index, meta) pair for one subject's folder."""
    subject_dir = os.path.join(vector_store_root, subject)
    indices = []
    metas = []
    if not os.path.isdir(subject_dir):
        return indices, metas
    for fname in sorted(os.listdir(subject_dir)):
        if fname.endswith(".index"):
            bucket = fname[:-len(".index")]
            meta_path = os.path.join(subject_dir, f"{bucket}_meta.jsonl")
            if os.path.exists(meta_path):
                indices.append(faiss.read_index(os.path.join(subject_dir, fname)))
                metas.append(load_meta(meta_path))
    return indices, metas


def retrieve_subject_aware(query, model, top_k, subject_filter=None, vector_store_root="vector_store"):
    """
    Searches only the chosen subject's index files if subject_filter is
    set. If None or "All", searches every subject and merges by score.
    This is TRUE isolation - a subject that isn't selected is never
    touched, not just deprioritized.
    """
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    subjects_to_search = (
        list_available_subjects(vector_store_root)
        if not subject_filter or subject_filter == "All"
        else [subject_filter.strip().lower()]
    )

    all_results = []
    for subject in subjects_to_search:
        indices, metas = load_subject_store(subject, vector_store_root)
        for index, meta in zip(indices, metas):
            k = min(top_k, index.ntotal)
            if k == 0:
                continue
            scores, idxs = index.search(q_emb, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                all_results.append((float(score), meta[idx]))

    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results[:top_k]


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
        elif r.get("type") == "pyq_pattern":
            label = f"pattern: {r['subject']} ({r['repetition_count']}x)"
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
