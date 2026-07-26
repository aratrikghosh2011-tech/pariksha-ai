"""
calculator.py

Post-hoc arithmetic verification for LLM-generated answers. Does NOT
change how the model writes its solution - instead, after the model
answers, this scans the answer text for lines of the form
  <expression> = <claimed_result>
re-evaluates <expression> with sympy, and flags any line where the
model's arithmetic doesn't match. This catches the actual failure mode
(LLM does mental math wrong) without needing full function-calling
infrastructure wired into the provider layer.

Also exposes calculate(expr) directly, for the CLI's standalone
calculator command ("calc: 2^10 + sqrt(144)").

Usage in the RAG pipeline:
  answer = provider.generate(prompt)
  verified_answer, warnings = verify_arithmetic(answer)
  # verified_answer has [check: recomputed as X] appended after any
  # line where the model's number didn't match. warnings is a list of
  # human-readable strings for logging/debugging.
"""

import re

import sympy
from sympy import sympify
from sympy.core.sympify import SympifyError

# Matches lines like "120 / 60 = 2" or "5 * 3.2 = 16" - a standalone
# arithmetic expression (numbers, + - * / ^ () . and whitespace only,
# NO variable names) followed by "=" and a number. Deliberately
# conservative: only fires on pure-number expressions, so it never
# tries to "verify" an algebraic step like "x = 2y + 3" where matching
# a claimed number wouldn't make sense.
ARITHMETIC_LINE_PATTERN = re.compile(
    r"([\d\s\.\+\-\*/\^\(\)]{3,})=\s*([\-]?\d+\.?\d*)\s*$"
)

TOLERANCE = 1e-6


def calculate(expression: str):
    """
    Evaluates a math expression with sympy. Accepts standard notation
    plus ^ for exponentiation (converted to ** since sympy uses **
    natively and ICSE-style answers often use ^).

    Returns (result, error). result is a sympy value (or None on
    error), error is None on success or a short human-readable string
    on failure. Never raises - always returns a tuple.
    """
    try:
        normalized = expression.strip().replace("^", "**")
        result = sympify(normalized)
        return result, None
    except (SympifyError, TypeError, ValueError) as e:
        return None, f"Could not evaluate '{expression}': {e}"


def verify_arithmetic(answer_text: str):
    """
    Scans answer_text line by line for pure-numeric "expr = result"
    statements and re-checks each one with sympy. Returns
    (annotated_text, warnings):
      - annotated_text: same as input, but any line where the model's
        claimed result doesn't match the recomputed value gets a
        " [check: recomputed as X]" note appended to that line.
      - warnings: list of strings, one per mismatch found, for
        logging. Empty list means every checked line was correct (or
        no checkable lines were found - this is NOT a guarantee the
        whole answer is right, only that the checkable numeric lines
        were verified).
    """
    lines = answer_text.split("\n")
    warnings = []

    for i, line in enumerate(lines):
        match = ARITHMETIC_LINE_PATTERN.search(line)
        if not match:
            continue

        expr_str, claimed_str = match.groups()

        # Skip if the "expression" side is just a single number
        # (e.g. "Answer: 2" would otherwise match "2" as the
        # expression and try to verify 2 == nothing meaningful).
        if not re.search(r"[\+\-\*/\^]", expr_str):
            continue

        result, error = calculate(expr_str)
        if error:
            continue

        try:
            claimed = sympify(claimed_str)
        except (SympifyError, TypeError, ValueError):
            continue

        try:
            diff = abs(float(result) - float(claimed))
        except (TypeError, ValueError):
            continue

        if diff > TOLERANCE:
            warning = (
                f"Line {i+1}: model claimed '{expr_str.strip()} = {claimed_str}', "
                f"recomputed value is {result}"
            )
            warnings.append(warning)
            lines[i] = line + f"  [check: recomputed as {result}]"

    return "\n".join(lines), warnings


if __name__ == "__main__":
    # Quick manual test
    sample = """Step 1: Time = Distance/Speed
Step 2: Time = 120/60 = 3
Answer: 3 hours"""
    annotated, warns = verify_arithmetic(sample)
    print(annotated)
    print()
    print("Warnings:", warns)
