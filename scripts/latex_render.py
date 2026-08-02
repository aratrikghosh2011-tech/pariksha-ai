"""
latex_render.py

Converts LaTeX math delimiters ($...$ and $$...$$) plus common LaTeX
commands into readable plain text for a terminal, since rich's
Markdown renderer does NOT render LaTeX - it just prints the raw
"$$F = m \times a$$" text with the dollar signs and backslashes
showing.

This is deliberately NOT trying to render real math typesetting in a
terminal (that would need a much bigger dependency for marginal
benefit). Instead it strips the LaTeX wrapper and translates the small
set of commands that show up constantly in ICSE Maths/Physics answers
(fractions, powers, roots, times, greek letters, subscripts) into
plain Unicode that reads naturally: $$F = m \times a$$ becomes
F = m × a, $\frac{a}{b}$ becomes (a/b), etc.

Deliberately conservative: an unrecognized LaTeX command is left as
plain text with the backslash stripped, rather than the script trying
to guess-translate everything and risk mangling gibberish into
something that reads as authoritative but wrong.
"""

import re

# Order matters - do multi-character commands before single symbols
# that could be substrings of them.
LATEX_REPLACEMENTS = [
    (r"\\times", "×"),
    (r"\\div", "÷"),
    (r"\\pm", "±"),
    (r"\\cdot", "·"),
    (r"\\approx", "≈"),
    (r"\\neq", "≠"),
    (r"\\leq", "≤"),
    (r"\\geq", "≥"),
    (r"\\infty", "∞"),
    (r"\\rightarrow", "→"),
    (r"\\Rightarrow", "⇒"),
    (r"\\degree", "°"),
    (r"\\circ", "°"),
    (r"\\alpha", "α"),
    (r"\\beta", "β"),
    (r"\\gamma", "γ"),
    (r"\\delta", "δ"),
    (r"\\Delta", "Δ"),
    (r"\\theta", "θ"),
    (r"\\lambda", "λ"),
    (r"\\mu", "μ"),
    (r"\\pi", "π"),
    (r"\\sigma", "σ"),
    (r"\\Omega", "Ω"),
    (r"\\omega", "ω"),
    (r"\\text\{([^}]*)\}", r"\1"),   # \text{N} -> N
    (r"\\mathrm\{([^}]*)\}", r"\1"), # \mathrm{kg} -> kg
    (r"\\left", ""),
    (r"\\right", ""),
]

FRACTION_PATTERN = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
SQRT_PATTERN = re.compile(r"\\sqrt\{([^{}]*)\}")
SUPERSCRIPT_PATTERN = re.compile(r"\^\{([^{}]*)\}")
SUPERSCRIPT_SINGLE_PATTERN = re.compile(r"\^([a-zA-Z0-9])")
SUBSCRIPT_PATTERN = re.compile(r"_\{([^{}]*)\}")
SUBSCRIPT_SINGLE_PATTERN = re.compile(r"_([a-zA-Z0-9])")

SUPERSCRIPT_DIGITS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻",
}
SUBSCRIPT_DIGITS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
}


def _to_superscript(text):
    return "".join(SUPERSCRIPT_DIGITS.get(c, f"^{c}") for c in text)


def _to_subscript(text):
    return "".join(SUBSCRIPT_DIGITS.get(c, f"_{c}") for c in text)


def _convert_expression(expr: str) -> str:
    """Converts the inside of a single $...$ or $$...$$ block to plain text."""
    result = expr

    while FRACTION_PATTERN.search(result):
        result = FRACTION_PATTERN.sub(r"(\1/\2)", result)
    while SQRT_PATTERN.search(result):
        result = SQRT_PATTERN.sub(r"√(\1)", result)

    result = SUPERSCRIPT_PATTERN.sub(lambda m: _to_superscript(m.group(1)), result)
    result = SUPERSCRIPT_SINGLE_PATTERN.sub(lambda m: _to_superscript(m.group(1)), result)
    result = SUBSCRIPT_PATTERN.sub(lambda m: _to_subscript(m.group(1)), result)
    result = SUBSCRIPT_SINGLE_PATTERN.sub(lambda m: _to_subscript(m.group(1)), result)

    for pattern, replacement in LATEX_REPLACEMENTS:
        result = re.sub(pattern, replacement, result)

    result = re.sub(r"\\([a-zA-Z]+)", r"\1", result)
    result = result.replace("~", " ")

    return result.strip()


def render_latex_for_terminal(text: str) -> str:
    """
    Finds every $$...$$ and $...$ block in text and replaces it with a
    plain-text rendering. Leaves everything outside math delimiters
    untouched. Safe to call on text with no LaTeX at all (no-op).
    """
    def replace_block(m):
        return _convert_expression(m.group(1))

    text = re.sub(r"\$\$(.+?)\$\$", replace_block, text, flags=re.DOTALL)

    def replace_inline(m):
        return _convert_expression(m.group(1))

    text = re.sub(r"\$(.+?)\$", replace_inline, text)

    return text


if __name__ == "__main__":
    sample = (
        "Mathematical Expression: According to Newton's Second Law of Motion: "
        "$$F = m \\times a$$\n"
        "Where $F$ = Force, $m$ = Mass, $a$ = Acceleration.\n"
        "SI Unit: $1\\text{ N} = 1\\text{ kg m/s}^2$\n"
        "Example fraction: $\\frac{1}{2} m v^2$\n"
        "Example root: $\\sqrt{2gh}$\n"
        "Subscript example: $v_0 + a t$"
    )
    print(render_latex_for_terminal(sample))
