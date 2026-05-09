import json
import os
import re
import sys
import glob

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_QA_DIR   = "dataset"
OUTPUT_JSONL = "dataset/icse_train.jsonl"
FLAGGED_LOG  = "dataset/flagged_diagram_questions.txt"
FILE_PATTERN = "raw_qa_*.txt"
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_HEADER = re.compile(r'^Question\s+\d+[\(\w\)]*', re.IGNORECASE)
ANSWER_MARKER   = re.compile(r'^Answer\s*$', re.IGNORECASE)
REASON_MARKER   = re.compile(r'^(Reason\s*[—\-]?|Explanation\s*:?)', re.IGNORECASE)
INSTRUCTION_RE  = re.compile(r'^INSTRUCTION\s*:\s*', re.IGNORECASE)
SEPARATOR       = '---'

# Phrases that signal a diagram-dependent question
DIAGRAM_SIGNALS = [
    r'from the (figure|diagram|graph|circuit|table) (given |shown |below|above)',
    r'(refer|referring) to the (figure|diagram|graph|circuit)',
    r'shown (below|above|in the figure|in the diagram)',
    r'the (figure|diagram|graph|circuit) (below|above|shows|given)',
    r'given (figure|diagram|graph|circuit)',
    r'redraw the diagram',
    r'study the (path|diagram|figure|graph)',
    r'in the circuit given below',
    r'identify the (lamp|component|element|device) in the (figure|circuit|diagram)',
    r'from the graph',
    r'the graph (below|above|shows)',
    r'plot (a|the) graph',
    r'draw (a|the) (ray diagram|diagram|graph|circuit)',
]
DIAGRAM_RE = re.compile('|'.join(DIAGRAM_SIGNALS), re.IGNORECASE)

NOISE_PATTERNS = [
    re.compile(r'^\s*$'),
    re.compile(r'^[a-z]\s*\)\s*$'),
    re.compile(r'^\(?\s*[A-D]\s*\)?\s*$'),
    re.compile(r'^ICSE\s+\d{4}', re.IGNORECASE),
    re.compile(r'^www\.', re.IGNORECASE),
]


def is_noise(line: str) -> bool:
    for pat in NOISE_PATTERNS:
        if pat.match(line.strip()):
            return True
    return False


def is_diagram_question(text: str) -> bool:
    return bool(DIAGRAM_RE.search(text))


def clean_text(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if re.match(r'^[\s⇒=×÷+\-*/^{}\\∘°₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹μΩλπ\d\(\)\[\]\.]+$', line.strip()):
            continue
        if is_noise(line):
            continue
        cleaned.append(line.rstrip())
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_response(answer_lines: list, reason_lines: list) -> str:
    parts = []
    if answer_lines:
        parts.append('\n'.join(answer_lines).strip())
    if reason_lines:
        parts.append('\n'.join(reason_lines).strip())
    return '\n\n'.join(parts)


def parse_format_a(content: str) -> tuple:
    pairs, flagged = [], []
    blocks = [b.strip() for b in content.split(SEPARATOR) if b.strip()]
    for block in blocks:
        instr_match = re.search(r'INSTRUCTION\s*:\s*(.*?)(?=RESPONSE\s*:)', block, re.DOTALL | re.IGNORECASE)
        resp_match  = re.search(r'RESPONSE\s*:\s*(.*)', block, re.DOTALL | re.IGNORECASE)
        if not instr_match or not resp_match:
            continue
        instruction = instr_match.group(1).strip()
        response    = resp_match.group(1).strip()
        if len(instruction) < 10 or len(response) < 20:
            continue
        if is_diagram_question(instruction):
            flagged.append(instruction)
            continue
        pairs.append({"instruction": instruction, "response": response})
    return pairs, flagged


def parse_format_b(content: str, source_file: str) -> tuple:
    pairs, flagged = [], []
    lines = content.split('\n')

    STATE_IDLE     = 0
    STATE_QUESTION = 1
    STATE_ANSWER   = 2
    STATE_REASON   = 3

    state   = STATE_IDLE
    q_lines = []
    a_lines = []
    r_lines = []
    current_qnum = ""

    def flush(q, a, r, qnum):
        instruction = clean_text('\n'.join(q))
        response    = clean_text(build_response(a, r))
        if len(instruction) < 10 or len(response) < 20:
            return None, None
        if is_diagram_question(instruction):
            return None, f"[{source_file} | {qnum}]\n{instruction[:200]}\n"
        return {"instruction": instruction, "response": response}, None

    for line in lines:
        stripped = line.strip()

        m = QUESTION_HEADER.match(stripped)
        if m:
            if state != STATE_IDLE and q_lines:
                pair, flag = flush(q_lines, a_lines, r_lines, current_qnum)
                if pair:
                    pairs.append(pair)
                if flag:
                    flagged.append(flag)
            q_lines, a_lines, r_lines = [], [], []
            current_qnum = stripped
            state = STATE_QUESTION
            continue

        if ANSWER_MARKER.match(stripped):
            state = STATE_ANSWER
            continue

        if REASON_MARKER.match(stripped):
            state = STATE_REASON
            rest = REASON_MARKER.sub('', stripped).strip()
            if rest:
                r_lines.append(rest)
            continue

        if state == STATE_QUESTION:
            if not is_noise(line):
                q_lines.append(line)
        elif state == STATE_ANSWER:
            if not is_noise(line):
                a_lines.append(line)
        elif state == STATE_REASON:
            if not is_noise(line):
                r_lines.append(line)

    if q_lines:
        pair, flag = flush(q_lines, a_lines, r_lines, current_qnum)
        if pair:
            pairs.append(pair)
        if flag:
            flagged.append(flag)

    return pairs, flagged


def detect_format(content: str) -> str:
    if INSTRUCTION_RE.search(content[:500]):
        return 'A'
    if QUESTION_HEADER.search(content[:1000]):
        return 'B'
    return 'B'


def parse_file(filepath: str) -> tuple:
    fname = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract subject from filename for tagging
    if '_phy_' in fname:
        subject_tag = "[Physics]"
    elif '_mth_' in fname:
        subject_tag = "[Maths]"
    else:
        subject_tag = ""

    fmt = detect_format(content)
    if fmt == 'A':
        pairs, flagged = parse_format_a(content)
    else:
        pairs, flagged = parse_format_b(content, fname)

    # Inject subject tag into every instruction
    if subject_tag:
        for pair in pairs:
            if not pair['instruction'].startswith('['):
                pair['instruction'] = f"{subject_tag} {pair['instruction']}"

    return pairs, flagged


def deduplicate(pairs: list) -> list:
    seen = set()
    unique = []
    for pair in pairs:
        key = pair['instruction'][:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def remove_bad_pairs(pairs: list) -> tuple:
    clean = []
    removed = 0
    for pair in pairs:
        instr = pair['instruction']

        # Only check the INSTRUCTION for broken content, not the response
        # Zero-width spaces and invisible unicode = broken copy-paste
        invisible_chars = '\u200b\u200c\u200d\ufeff\u200e\u200f'
        invisible_count = len([c for c in instr if c in invisible_chars])

        # Excessive newlines in question = broken matrix/table question
        newline_count = instr.count('\n')

        if invisible_count > 3 or newline_count > 15:
            removed += 1
            continue
        clean.append(pair)
    return clean, removed


def write_jsonl(pairs: list, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def write_flagged(flagged: list, filepath: str) -> None:
    if not flagged:
        return
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("DIAGRAM-DEPENDENT QUESTIONS — SKIP OR REWRITE MANUALLY\n")
        f.write("="*60 + "\n\n")
        for entry in flagged:
            if isinstance(entry, str):
                f.write(entry + "\n" + "-"*40 + "\n")
            else:
                f.write(str(entry)[:300] + "\n" + "-"*40 + "\n")


def validate_and_report(all_pairs: list, per_file: dict, flagged_count: int) -> None:
    print(f"\n{'='*55}")
    print("  PER FILE BREAKDOWN:")
    for fname, count in sorted(per_file.items()):
        print(f"    {fname:<40} {count:>4} pairs")
    print(f"{'='*55}")
    print(f"  Total before dedup  : {sum(per_file.values())}")
    print(f"  Total after dedup   : {len(all_pairs)}")
    print(f"  Diagram questions   : {flagged_count} (saved to flagged_diagram_questions.txt)")
    print(f"  Target              : 800–1200")
    remaining = max(0, 800 - len(all_pairs))
    if remaining > 0:
        print(f"  Still needed        : {remaining} more pairs")
    else:
        print(f"  STATUS              : READY FOR FINE-TUNING ✓")
    print(f"{'='*55}\n")

    print("[PREVIEW] First 2 pairs:\n")
    for pair in all_pairs[:2]:
        print(f"  Q: {pair['instruction'][:120]}...")
        print(f"  A: {pair['response'][:120]}...")
        print()


def main():
    pattern = os.path.join(RAW_QA_DIR, FILE_PATTERN)
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERROR] No files matching '{pattern}' found.")
        print(f"        Name your files like: raw_qa_p10_phy_2025.txt")
        sys.exit(1)

    all_pairs    = []
    all_flagged  = []
    per_file     = {}

    for filepath in files:
        fname        = os.path.basename(filepath)
        pairs, flagged = parse_file(filepath)
        per_file[fname] = len(pairs)
        all_pairs.extend(pairs)
        all_flagged.extend(flagged)
        print(f"[READ] {fname} → {len(pairs)} pairs, {len(flagged)} diagram questions skipped")

    all_pairs = deduplicate(all_pairs)
    all_pairs, removed = remove_bad_pairs(all_pairs)
    print(f"[CLEAN] Removed {removed} bad pairs with broken LaTeX")
    write_jsonl(all_pairs, OUTPUT_JSONL)
    write_flagged(all_flagged, FLAGGED_LOG)
    validate_and_report(all_pairs, per_file, len(all_flagged))


if __name__ == '__main__':
    main()