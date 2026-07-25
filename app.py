"""
app.py

Local Streamlit chat interface for Pariksha AI. Reuses the existing
RAG pipeline (scripts/rag_query.py, scripts/llm_providers.py) instead
of reimplementing retrieval or generation.

This is LOCAL ONLY - not deployed to Streamlit Community Cloud. See
project notes on why (copyrighted textbook content can't be on a
public server without a separate plan for that).

Run:
  streamlit run app.py
"""

import os
import sys

import streamlit as st
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from rag_query import get_subject, build_prompt, MODEL_NAME, list_available_subjects, retrieve_subject_aware
from llm_providers import get_provider
from st_copy import copy_button

st.set_page_config(page_title="Pariksha AI", page_icon="📚", layout="centered")


@st.cache_resource
def load_pipeline():
    """
    Loads the embedding model ONCE per app run. Per-subject index files
    are small enough to load on demand per query - no need to cache
    them all in memory upfront.
    """
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return model


def format_source(r, score):
    """Turns one retrieved record into a human-readable source line."""
    subject = get_subject(r)
    if r.get("type") == "textbook":
        return f"[{subject}] {r['book']} p.{r['page']} (similarity {score:.2f})"
    elif r.get("type") == "pyq_pattern":
        years = ", ".join(str(y) for y in r.get("years_seen", []))
        return f"[{subject}] Recurring exam pattern, seen {r['repetition_count']}x since {years} (similarity {score:.2f})"
    else:
        preview = r.get("instruction", "")[:80]
        return f"[{subject}] Past-paper example: {preview}... (similarity {score:.2f})"


def main():
    st.title("📚 Pariksha AI")
    st.caption("ICSE Class 10 tutor — Maths, Physics, Chemistry, Robotics, Literature")

    with st.sidebar:
        st.header("Settings")
        provider_name = st.selectbox("AI provider", ["gemini", "nemotron"], index=0)
        available = list_available_subjects()
        subject_options = ["All"] + [s.capitalize() for s in available]
        subject_filter = st.selectbox(
            "Subject",
            subject_options,
            index=0,
            help="Restrict retrieval to one subject's files only - true isolation, not just re-ranking.",
        )
        top_k = st.slider("Number of reference examples", min_value=1, max_value=10, value=3)
        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    try:
        model = load_pipeline()
    except Exception as e:
        st.error(f"Could not load the RAG pipeline: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                copy_button(msg["content"], tooltip="Copy answer", copied_label="Copied!")
                if msg.get("sources"):
                    with st.expander("Sources used"):
                        for s in msg["sources"]:
                            st.write(f"- {s}")

    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                sources = []
                try:
                    retrieved = retrieve_subject_aware(user_input, model, top_k, subject_filter)
                    print(f"DEBUG: retrieved {len(retrieved)} results")
                    for score, r in retrieved:
                        print(f"DEBUG:   score={score:.3f} type={r.get('type')} keys={list(r.keys())}")
                    prompt = build_prompt(user_input, retrieved)
                    provider = get_provider(provider_name)
                    answer = provider.generate(prompt)
                    sources = []
                    for score, r in retrieved:
                        try:
                            sources.append(format_source(r, score))
                        except Exception as fmt_error:
                            print(f"DEBUG: format_source FAILED on record {r}: {fmt_error}")
                            sources.append(f"[source formatting error: {fmt_error}]")
                except Exception as e:
                    answer = f"Something went wrong: {e}"

            st.markdown(answer)
            copy_button(answer, tooltip="Copy answer", copied_label="Copied!")
            if sources:
                with st.expander("Sources used"):
                    for s in sources:
                        st.write(f"- {s}")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()
