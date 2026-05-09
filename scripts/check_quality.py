import json

bad_pairs = []
ok = 0

with open('dataset/icse_train.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        instr = obj['instruction']
        resp  = obj['response']

        # Flag if too many math symbols (broken LaTeX copy-paste)
        symbol_count = len([c for c in instr if c in '⇒×÷μΩλπ∘'])
        newline_count = instr.count('\n')

        if symbol_count > 5 or newline_count > 15:
            bad_pairs.append((i, instr[:120]))
        else:
            ok += 1

print(f"Good pairs : {ok}")
print(f"Bad pairs  : {len(bad_pairs)}")
print()
if bad_pairs:
    print("BAD PAIRS PREVIEW:")
    for line_num, preview in bad_pairs[:5]:
        print(f"  Line {line_num}: {preview[:100]}")
        print()