"""
flashcards.py

Generates flashcards or quiz questions from a subject's retrieved
content, using the same RAG retrieval as the main chat (so cards are
grounded in the actual ICSE source material, not the LLM's general
knowledge), and stores them persistently in the same chat_history.db
SQLite file so they survive restarts and can be reviewed later.

Two content types:
  - flashcard: {front, back}
  - quiz: {question, options (4), correct_index, explanation}

Generation flow:
  1. Retrieve N relevant chunks for the subject (+ optional topic
     keyword to narrow within the subject, e.g. "current electricity")
  2. Ask the LLM to produce a JSON array of cards grounded in that
     retrieved content
  3. Parse the JSON defensively (LLMs sometimes wrap JSON in prose or
     markdown fences) - if parsing fails entirely, return an error
     rather than silently returning nothing
  4. Save every successfully parsed card to the database immediately
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS flashcards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    topic TEXT,
    card_type TEXT NOT NULL CHECK (card_type IN ('flashcard', 'quiz')),
    front TEXT,
    back TEXT,
    question TEXT,
    options_json TEXT,
    correct_index INTEGER,
    explanation TEXT,
    times_reviewed INTEGER NOT NULL DEFAULT 0,
    times_correct INTEGER NOT NULL DEFAULT 0,
    last_reviewed_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_flashcards_subject ON flashcards(subject);
"""

FLASHCARD_GENERATION_PROMPT = """You are generating {count} {card_type_label} for an
ICSE Class 10 student studying {subject}{topic_clause}, based ONLY on
the reference material below. Every card must be answerable from this
material - do not invent facts outside it.

Reference material:
{context}

{format_instructions}

Respond with ONLY a JSON array, no other text, no markdown code fences,
no explanation before or after. Just the raw JSON array."""

FLASHCARD_FORMAT_INSTRUCTIONS = """Each item in the array must be an object with exactly these keys:
  "front": a short question or term (string)
  "back": the answer or definition (string)

Example: [{"front": "What is the SI unit of force?", "back": "Newton (N)"}]"""

QUIZ_FORMAT_INSTRUCTIONS = """Each item in the array must be an object with exactly these keys:
  "question": the question text (string)
  "options": an array of exactly 4 answer choices (strings)
  "correct_index": the 0-based index of the correct option in "options" (integer)
  "explanation": a short explanation of why that answer is correct (string)

Example: [{"question": "What is the SI unit of force?", "options": ["Joule", "Newton", "Watt", "Pascal"], "correct_index": 1, "explanation": "Force is measured in Newtons, defined as kg*m/s^2 by Newton's second law."}]"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _extract_json_array(raw_text: str):
    """
    LLMs frequently wrap JSON in markdown fences or add a sentence
    before/after despite instructions. This strips common wrappers and
    finds the first '[' to the matching last ']' as a fallback, rather
    than failing outright on a near-miss response.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost [ ... ] span
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def generate_flashcards(subject, provider, retrieve_fn, top_k=8, topic=None, count=8, card_type="flashcard"):
    """
    Generates flashcards or quiz questions for a subject.

    subject: the subject name (e.g. "physics")
    provider: an LLMProvider instance (from llm_providers.get_provider())
    retrieve_fn: a callable(query, top_k, subject_filter) -> list of
      (score, record) tuples - pass rag_query.retrieve_subject_aware
      partially applied with the embedding model, so this module
      doesn't need to import sentence-transformers directly
    topic: optional keyword to narrow retrieval within the subject
      (e.g. "current electricity"). Used as the retrieval query.
    count: how many cards to generate
    card_type: "flashcard" or "quiz"

    Returns (cards, error): cards is a list of dicts ready to pass to
    save_cards(), error is None on success or a short string on
    failure (e.g. the LLM's response couldn't be parsed as JSON at
    all - this is reported rather than silently returning an empty list,
    since an empty list on its own looks identical to "no relevant
    content found" from the caller's side).
    """
    query = topic if topic else f"{subject} important concepts and definitions"
    retrieved = retrieve_fn(query, top_k, subject)

    if not retrieved:
        return [], f"No reference material found for subject '{subject}'" + (f" and topic '{topic}'" if topic else "")

    context_blocks = []
    for score, r in retrieved:
        if r.get("type") == "textbook":
            context_blocks.append(f"{r['book']} p.{r['page']}: {r['text']}")
        elif r.get("type") == "pyq_pattern":
            context_blocks.append(f"Exam pattern: {r['text']}")
        else:
            context_blocks.append(f"Q: {r.get('instruction', '')}\nA: {r.get('response', '')}")
    context = "\n\n".join(context_blocks)

    format_instructions = FLASHCARD_FORMAT_INSTRUCTIONS if card_type == "flashcard" else QUIZ_FORMAT_INSTRUCTIONS
    card_type_label = "flashcards" if card_type == "flashcard" else "multiple-choice quiz questions"
    topic_clause = f", focused on '{topic}'" if topic else ""

    prompt = FLASHCARD_GENERATION_PROMPT.format(
        count=count, card_type_label=card_type_label, subject=subject,
        topic_clause=topic_clause, context=context, format_instructions=format_instructions,
    )

    try:
        raw_response = provider.generate(prompt)
    except Exception as e:
        return [], f"Generation failed: {e}"

    parsed = _extract_json_array(raw_response)
    if parsed is None:
        return [], f"Could not parse the AI's response as a JSON array. Raw response started with: {raw_response[:200]!r}"

    if not isinstance(parsed, list):
        return [], f"Expected a JSON array, got {type(parsed).__name__} instead"

    cards = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if card_type == "flashcard":
            if "front" in item and "back" in item:
                cards.append({
                    "subject": subject, "topic": topic, "card_type": "flashcard",
                    "front": str(item["front"]), "back": str(item["back"]),
                })
        else:  # quiz
            if all(k in item for k in ("question", "options", "correct_index", "explanation")):
                options = item["options"]
                if isinstance(options, list) and len(options) == 4 and isinstance(item["correct_index"], int):
                    cards.append({
                        "subject": subject, "topic": topic, "card_type": "quiz",
                        "question": str(item["question"]), "options": [str(o) for o in options],
                        "correct_index": item["correct_index"], "explanation": str(item["explanation"]),
                    })

    if not cards:
        return [], "AI response was valid JSON but contained no items matching the expected card format"

    return cards, None


def save_cards(cards, db_path=DB_PATH):
    """Saves a list of card dicts (from generate_flashcards) to the database. Returns the number saved."""
    now = _now()
    conn = sqlite3.connect(db_path)
    try:
        for card in cards:
            if card["card_type"] == "flashcard":
                conn.execute(
                    "INSERT INTO flashcards (subject, topic, card_type, front, back, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (card["subject"], card.get("topic"), "flashcard", card["front"], card["back"], now),
                )
            else:
                conn.execute(
                    "INSERT INTO flashcards (subject, topic, card_type, question, options_json, "
                    "correct_index, explanation, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (card["subject"], card.get("topic"), "quiz", card["question"],
                     json.dumps(card["options"], ensure_ascii=False), card["correct_index"],
                     card["explanation"], now),
                )
        conn.commit()
        return len(cards)
    finally:
        conn.close()


def get_cards(subject=None, card_type=None, db_path=DB_PATH):
    """Returns saved cards, optionally filtered by subject and/or card_type, as a list of dicts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM flashcards WHERE 1=1"
        params = []
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if card_type:
            query += " AND card_type = ?"
            params.append(card_type)
        query += " ORDER BY id ASC"
        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("options_json"):
                d["options"] = json.loads(d.pop("options_json"))
            results.append(d)
        return results
    finally:
        conn.close()


def record_review(card_id, was_correct, db_path=DB_PATH):
    """
    Records that a card was reviewed, and whether the answer was
    correct. Feeds the weak-topic tracking planned as a follow-up -
    this is the raw signal that a "weak topics" feature would later
    aggregate over.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE flashcards SET times_reviewed = times_reviewed + 1, "
            "times_correct = times_correct + ?, last_reviewed_at = ? WHERE id = ?",
            (1 if was_correct else 0, _now(), card_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_cards(subject=None, card_type=None, db_path=DB_PATH):
    """Deletes cards matching the given filters. Returns the number deleted. No filters = deletes ALL cards."""
    conn = sqlite3.connect(db_path)
    try:
        query = "DELETE FROM flashcards WHERE 1=1"
        params = []
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if card_type:
            query += " AND card_type = ?"
            params.append(card_type)
        cur = conn.execute(query, params)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
