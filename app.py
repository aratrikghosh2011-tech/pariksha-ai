"""
app.py

Local Streamlit chat interface for Pariksha AI. Reuses the existing
RAG pipeline (scripts/rag_query.py, scripts/llm_providers.py) instead
of reimplementing retrieval or generation.

Persistence: chat_history.db (SQLite, via scripts/chat_store.py) -
sidebar chat list, new chat button, search box, edit-and-resubmit.
Local file, gitignored, same treatment as vector_store/.

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
import chat_store

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


def run_rag(user_input, model, provider_name, subject_filter, top_k):
    """
    Runs retrieval + generation for one question. Returns (answer, sources).
    Shared by both the normal chat-input path and the edit-and-resubmit path
    so they can never drift out of sync with each other.
    """
    try:
        retrieved = retrieve_subject_aware(user_input, model, top_k, subject_filter)
        prompt = build_prompt(user_input, retrieved)
        provider = get_provider(provider_name)
        answer = provider.generate(prompt)
        sources = []
        for score, r in retrieved:
            try:
                sources.append(format_source(r, score))
            except Exception as fmt_error:
                sources.append(f"[source formatting error: {fmt_error}]")
        return answer, sources
    except Exception as e:
        return f"Something went wrong: {e}", []


def ensure_active_chat():
    """
    Makes sure st.session_state.active_chat_id points at a real chat.
    If there are no chats yet, creates one. Runs once per session,
    not once per rerun, thanks to the session_state guard in main().
    """
    chats = chat_store.list_chats()
    if not chats:
        return chat_store.create_chat()
    return chats[0]["id"]


def render_sidebar():
    """
    Renders the sidebar: settings, search box, new-chat button, and the
    chat list. Returns (provider_name, subject_filter, top_k) for use
    by the main chat area.
    """
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

        if st.button("New chat", use_container_width=True):
            new_id = chat_store.create_chat()
            st.session_state.active_chat_id = new_id
            st.session_state.edit_message_id = None
            st.rerun()

        search_query = st.text_input("Search chats", placeholder="Search messages and titles...")
        if search_query:
            results = chat_store.search_chats(search_query)
            st.caption(f"{len(results)} match(es)")
            for res in results[:20]:
                label = f"{res['chat_title']} — {res['snippet']}"
                if st.button(label, key=f"search_result_{res['message_id']}", use_container_width=True):
                    st.session_state.active_chat_id = res["chat_id"]
                    st.session_state.edit_message_id = None
                    st.rerun()
            st.divider()

        st.subheader("Chats")
        chats = chat_store.list_chats()
        for chat in chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            col1, col2 = st.columns([5, 1])
            with col1:
                label = ("▶ " if is_active else "") + chat["title"]
                if st.button(label, key=f"chat_select_{chat['id']}", use_container_width=True):
                    st.session_state.active_chat_id = chat["id"]
                    st.session_state.edit_message_id = None
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"chat_delete_{chat['id']}"):
                    chat_store.delete_chat(chat["id"])
                    if st.session_state.active_chat_id == chat["id"]:
                        remaining = chat_store.list_chats()
                        st.session_state.active_chat_id = remaining[0]["id"] if remaining else chat_store.create_chat()
                    st.rerun()

    return provider_name, subject_filter, top_k


def render_message(msg, model, provider_name, subject_filter, top_k):
    """
    Renders one message. User messages get an Edit button that, when
    clicked, swaps in a text_area + Resubmit button in place of the
    static text. Assistant messages get copy-to-clipboard + sources
    expander, same as before.
    """
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if st.session_state.edit_message_id == msg["id"]:
                edited = st.text_area("Edit your question", value=msg["content"], key=f"edit_box_{msg['id']}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Resubmit", key=f"resubmit_{msg['id']}"):
                        chat_id = chat_store.edit_message_and_truncate(msg["id"], edited)
                        with st.spinner("Thinking..."):
                            answer, sources = run_rag(edited, model, provider_name, subject_filter, top_k)
                        chat_store.add_message(chat_id, "assistant", answer, sources)
                        st.session_state.edit_message_id = None
                        st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_edit_{msg['id']}"):
                        st.session_state.edit_message_id = None
                        st.rerun()
            else:
                st.markdown(msg["content"])
                if st.button("✏️ Edit", key=f"edit_btn_{msg['id']}"):
                    st.session_state.edit_message_id = msg["id"]
                    st.rerun()
        else:
            st.markdown(msg["content"])
            copy_button(msg["content"], tooltip="Copy answer", copied_label="Copied!")
            if msg.get("sources"):
                with st.expander("Sources used"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")


def main():
    st.title("📚 Pariksha AI")
    st.caption("ICSE Class 10 tutor — Maths, Physics, Chemistry, Robotics, Literature")

    chat_store.init_db()

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = ensure_active_chat()
    if "edit_message_id" not in st.session_state:
        st.session_state.edit_message_id = None

    provider_name, subject_filter, top_k = render_sidebar()

    try:
        model = load_pipeline()
    except Exception as e:
        st.error(f"Could not load the RAG pipeline: {e}")
        st.stop()

    messages = chat_store.get_messages(st.session_state.active_chat_id)

    for msg in messages:
        render_message(msg, model, provider_name, subject_filter, top_k)

    user_input = st.chat_input("Ask a question...")

    if user_input:
        chat_id = st.session_state.active_chat_id
        was_empty = len(messages) == 0

        chat_store.add_message(chat_id, "user", user_input)
        if was_empty:
            chat_store.rename_chat(chat_id, chat_store.generate_title_from_first_message(user_input))

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = run_rag(user_input, model, provider_name, subject_filter, top_k)
            st.markdown(answer)
            copy_button(answer, tooltip="Copy answer", copied_label="Copied!")
            if sources:
                with st.expander("Sources used"):
                    for s in sources:
                        st.write(f"- {s}")

        chat_store.add_message(chat_id, "assistant", answer, sources)
        st.rerun()


if __name__ == "__main__":
    main()
