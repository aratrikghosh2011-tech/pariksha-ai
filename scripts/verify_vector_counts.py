"""
verify_vector_counts.py

Prints the vector count of every .index file under vector_store/,
grouped by subject folder, plus a grand total. Used to confirm no
vectors are lost or duplicated during a migration/fix.

Run:
  python scripts/verify_vector_counts.py
"""

import os
import faiss

VECTOR_STORE_ROOT = "vector_store"


def main():
    if not os.path.isdir(VECTOR_STORE_ROOT):
        print(f"ERROR: {VECTOR_STORE_ROOT} not found.")
        return

    grand_total = 0
    print(f"{'subject':<20} {'bucket':<15} {'vectors':>10}")
    print("-" * 47)

    for entry in sorted(os.listdir(VECTOR_STORE_ROOT)):
        full = os.path.join(VECTOR_STORE_ROOT, entry)
        if not os.path.isdir(full):
            continue
        for fname in sorted(os.listdir(full)):
            if fname.endswith(".index"):
                bucket = fname[:-len(".index")]
                index = faiss.read_index(os.path.join(full, fname))
                print(f"{entry:<20} {bucket:<15} {index.ntotal:>10}")
                grand_total += index.ntotal

    print("-" * 47)
    print(f"{'GRAND TOTAL':<36} {grand_total:>10}")


if __name__ == "__main__":
    main()
