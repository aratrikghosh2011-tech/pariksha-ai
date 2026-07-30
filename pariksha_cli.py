"""
pariksha_cli.py

Terminal interface for Pariksha AI. No browser, no Streamlit - a
distraction-free chat loop, same RAG pipeline and SQLite persistence
as app.py underneath.

Run:
  python pariksha_cli.py

Commands inside the chat (type any of these instead of a question):
  /new              Start a new chat
  /list             List all past chats
  /switch <id>      Switch to a past chat by its number from /list
  /search <text>    Search all chats for matching text
  /subject <name>   Set the subject filter (or "all")
  /model            Show current model and pick a different one
  /image <path>     Attach a local image file to your NEXT question
  /calc <expr>      Standalone calculator, e.g. /calc 2^10 + sqrt(144)
  /edit <n>         Edit message number n in the current chat and resubmit
  /help             Show this list again
  /quit             Exit
"""

import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from rag_query import get_subject, build_prompt, MODEL_NAME, list_available_subjects, retrieve_subject_aware
from llm_providers import get_provider, GEMINI_MODEL_FALLBACK_CHAIN
from calculator import verify_arithmetic, calculate
import chat_store

console = Console()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif",
}


def format_source(r, score):
    subject = get_subject(r)
    if r.get("type") == "textbook":
        return f"[{subject}] {r['book']} p.{r['page']} (similarity {score:.2f})"
    elif r.get("type") == "pyq_pattern":
        years = ", ".join(str(y) for y in r.get("years_seen", []))
        return f"[{subject}] Recurring exam pattern, seen {r['repetition_count']}x since {years} (similarity {score:.2f})"
    else:
        preview = r.get("instruction", "")[:80]
        return f"[{subject}] Past-paper example: {preview}... (similarity {score:.2f})"


class CliState:
    """Holds everything that persists across the input loop."""

    def __init__(self):
        self.provider_name = "gemini"
        self.model_name = None  # None = use automatic fallback chain
        self.subject_filter = "All"
        self.top_k = 3
        self.pending_image_path = None
        self.chat_id = None
        self.model = None  # embedding model, loaded once


def print_help():
    console.print(Panel(
        "[bold]/new[/bold]              Start a new chat\n"
        "[bold]/list[/bold]             List all past chats\n"
        "[bold]/switch <id>[/bold]      Switch to a past chat by number\n"
        "[bold]/search <text>[/bold]    Search all chats\n"
        "[bold]/subject <name>[/bold]   Set subject filter (or \"all\")\n"
        "[bold]/provider[/bold]         Show/change the AI provider (gemini, nemotron, openai-oauth)\n"
        "[bold]/model[/bold]            Show/change the current model\n"
        "[bold]/image <path>[/bold]     Attach an image to your next question\n"
        "[bold]/calc <expr>[/bold]      Standalone calculator\n"
        "[bold]/edit <n>[/bold]         Edit message n and resubmit\n"
        "[bold]/help[/bold]             Show this again\n"
        "[bold]/quit[/bold]             Exit",
        title="Commands", border_style="cyan",
    ))


def print_chat_history(state: CliState):
    messages = chat_store.get_messages(state.chat_id)
    if not messages:
        console.print("[dim]New chat - ask a question to begin.[/dim]")
        return
    for i, msg in enumerate(messages, start=1):
        if msg["role"] == "user":
            console.print(f"\n[bold cyan]#{i} You:[/bold cyan] {msg['content']}")
        else:
            console.print(f"\n[bold green]Pariksha AI:[/bold green]")
            console.print(Markdown(msg["content"]))
            if msg.get("sources"):
                console.print("[dim]Sources:[/dim]")
                for s in msg["sources"]:
                    console.print(f"[dim]  - {s}[/dim]")


def run_rag(state: CliState, question: str, image_bytes=None, image_mime_type=None):
    """Runs retrieval + generation. Returns (answer, sources, model_used)."""
    try:
        retrieved = retrieve_subject_aware(question, state.model, state.top_k, state.subject_filter)
        prompt = build_prompt(question, retrieved)
        provider = get_provider(state.provider_name, model_name=state.model_name)
        answer = provider.generate(prompt, image_bytes=image_bytes, image_mime_type=image_mime_type)

        annotated_answer, calc_warnings = verify_arithmetic(answer)
        if calc_warnings:
            console.print(f"[yellow]Note: {len(calc_warnings)} arithmetic step(s) auto-corrected by calculator check[/yellow]")

        sources = []
        for score, r in retrieved:
            try:
                sources.append(format_source(r, score))
            except Exception as fmt_error:
                sources.append(f"[source formatting error: {fmt_error}]")

        model_used = getattr(provider, "last_model_used", state.model_name or "unknown")
        return annotated_answer, sources, model_used
    except Exception as e:
        return f"Something went wrong: {e}", [], None


def handle_command(state: CliState, raw_input: str) -> bool:
    """
    Handles a /command. Returns False if the loop should exit, True to continue.
    Returns True for everything except /quit.
    """
    parts = raw_input.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit":
        return False

    elif cmd == "/help":
        print_help()

    elif cmd == "/new":
        state.chat_id = chat_store.create_chat()
        console.print("[green]Started a new chat.[/green]")

    elif cmd == "/list":
        chats = chat_store.list_chats()
        table = Table(title="Your chats")
        table.add_column("#", style="cyan")
        table.add_column("Title")
        table.add_column("Last active")
        for i, c in enumerate(chats, start=1):
            marker = " (current)" if c["id"] == state.chat_id else ""
            table.add_row(str(c["id"]), c["title"] + marker, c["updated_at"][:19])
        console.print(table)

    elif cmd == "/switch":
        try:
            target_id = int(arg.strip())
        except ValueError:
            console.print("[red]Usage: /switch <chat id from /list>[/red]")
            return True
        chat = chat_store.get_chat(target_id)
        if not chat:
            console.print(f"[red]No chat with id {target_id}[/red]")
            return True
        state.chat_id = target_id
        console.print(f"[green]Switched to: {chat['title']}[/green]")
        print_chat_history(state)

    elif cmd == "/search":
        if not arg.strip():
            console.print("[red]Usage: /search <text>[/red]")
            return True
        results = chat_store.search_chats(arg.strip())
        if not results:
            console.print("[dim]No matches.[/dim]")
            return True
        table = Table(title=f"Search results for '{arg.strip()}'")
        table.add_column("Chat ID", style="cyan")
        table.add_column("Chat title")
        table.add_column("Snippet")
        for r in results[:20]:
            table.add_row(str(r["chat_id"]), r["chat_title"], r["snippet"])
        console.print(table)

    elif cmd == "/subject":
        if not arg.strip():
            console.print("[red]Usage: /subject <name or 'all'>[/red]")
            return True
        available = list_available_subjects()
        chosen = arg.strip().lower()
        if chosen == "all":
            state.subject_filter = "All"
        elif chosen in available:
            state.subject_filter = chosen
        else:
            console.print(f"[red]Unknown subject '{chosen}'. Available: {', '.join(available)}, or 'all'[/red]")
            return True
        console.print(f"[green]Subject filter set to: {state.subject_filter}[/green]")

    elif cmd == "/provider":
        valid_providers = ("gemini", "nemotron", "openai-oauth")
        if not arg.strip():
            console.print(f"Current provider: [bold]{state.provider_name}[/bold]")
            console.print(f"To change: /provider <name>, options: {', '.join(valid_providers)}")
            return True
        chosen = arg.strip().lower()
        if chosen not in valid_providers:
            console.print(f"[red]Unknown provider '{chosen}'. Use: {', '.join(valid_providers)}[/red]")
            return True
        state.provider_name = chosen
        state.model_name = None
        console.print(f"[green]Provider set to: {state.provider_name}[/green] (model reset to default for this provider)")
        if chosen == "openai-oauth":
            console.print("[dim]The local openai-oauth proxy will auto-start on your next question if it isn't already running.[/dim]")

    elif cmd == "/model":
        if not arg.strip():
            if state.provider_name == "gemini":
                current = state.model_name or f"auto ({' -> '.join(GEMINI_MODEL_FALLBACK_CHAIN)})"
                console.print(f"Current: [bold]{current}[/bold]")
                console.print("To change: /model <name>, or /model auto to use automatic fallback")
                console.print(f"Free-tier options: {', '.join(GEMINI_MODEL_FALLBACK_CHAIN)}")
            elif state.provider_name == "openai-oauth":
                default_model = os.getenv("OPENAI_OAUTH_MODEL", "gpt-5.4-mini")
                current = state.model_name or f"{default_model} (from OPENAI_OAUTH_MODEL / default)"
                console.print(f"Current: [bold]{current}[/bold]")
                console.print("To change: /model <name> - must be one of the models your local openai-oauth proxy lists on startup")
            else:
                console.print(f"Current: [bold]{state.model_name or 'provider default'}[/bold]")
                console.print("To change: /model <name>")
            return True
        chosen = arg.strip()
        state.model_name = None if chosen.lower() == "auto" else chosen
        console.print(f"[green]Model set to: {state.model_name or 'auto (fallback chain)'}[/green]")

    elif cmd == "/image":
        path = arg.strip().strip('"')
        if not path:
            console.print("[red]Usage: /image <path to image file>[/red]")
            return True
        if not os.path.exists(path):
            console.print(f"[red]File not found: {path}[/red]")
            return True
        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            console.print(f"[red]Unsupported image type '{ext}'. Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}[/red]")
            return True
        state.pending_image_path = path
        console.print(f"[green]Image attached: {path}[/green] - it will be sent with your next question.")

    elif cmd == "/calc":
        if not arg.strip():
            console.print("[red]Usage: /calc <expression>[/red]")
            return True
        result, error = calculate(arg.strip())
        if error:
            console.print(f"[red]{error}[/red]")
        else:
            console.print(f"[bold]{arg.strip()} = {result}[/bold]")

    elif cmd == "/edit":
        if not arg.strip():
            console.print("[red]Usage: /edit <message number from the chat history>[/red]")
            return True
        try:
            msg_number = int(arg.strip())
        except ValueError:
            console.print("[red]Usage: /edit <message number>[/red]")
            return True
        messages = chat_store.get_messages(state.chat_id)
        if msg_number < 1 or msg_number > len(messages):
            console.print(f"[red]No message #{msg_number} in this chat.[/red]")
            return True
        target = messages[msg_number - 1]
        if target["role"] != "user":
            console.print("[red]You can only edit your own questions, not the AI's answers.[/red]")
            return True
        console.print(f"[dim]Original: {target['content']}[/dim]")
        new_content = Prompt.ask("New version")
        if not new_content.strip():
            console.print("[dim]Cancelled (empty input).[/dim]")
            return True
        chat_store.edit_message_and_truncate(target["id"], new_content)
        console.print("[green]Edited. Regenerating answer...[/green]")
        with console.status("Thinking..."):
            answer, sources, model_used = run_rag(state, new_content)
        chat_store.add_message(state.chat_id, "assistant", answer, sources)
        console.print(f"\n[bold green]Pariksha AI[/bold green] [dim](via {model_used})[/dim]:")
        console.print(Markdown(answer))
        if sources:
            console.print("[dim]Sources:[/dim]")
            for s in sources:
                console.print(f"[dim]  - {s}[/dim]")

    else:
        console.print(f"[red]Unknown command: {cmd}. Type /help for the list.[/red]")

    return True


def main():
    console.print(Panel(
        "[bold]Pariksha AI[/bold] - ICSE Class 10 tutor (Maths, Physics, Chemistry, Robotics, Literature)\n"
        "Type a question, or /help for commands.",
        border_style="cyan",
    ))

    chat_store.init_db()

    state = CliState()
    chats = chat_store.list_chats()
    state.chat_id = chats[0]["id"] if chats else chat_store.create_chat()

    console.print("[dim]Loading embedding model...[/dim]")
    try:
        state.model = SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as e:
        console.print(f"[red]Could not load embedding model: {e}[/red]")
        return

    print_chat_history(state)

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Ask[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue

        if user_input.strip().startswith("/"):
            should_continue = handle_command(state, user_input)
            if not should_continue:
                console.print("[dim]Goodbye.[/dim]")
                break
            continue

        was_empty = len(chat_store.get_messages(state.chat_id)) == 0
        chat_store.add_message(state.chat_id, "user", user_input)
        if was_empty:
            chat_store.rename_chat(state.chat_id, chat_store.generate_title_from_first_message(user_input))

        image_bytes, image_mime_type = None, None
        if state.pending_image_path:
            ext = os.path.splitext(state.pending_image_path)[1].lower()
            with open(state.pending_image_path, "rb") as f:
                image_bytes = f.read()
            image_mime_type = MIME_TYPES[ext]
            console.print(f"[dim](including attached image: {state.pending_image_path})[/dim]")
            state.pending_image_path = None  # one-shot, consumed after this question

        with console.status("Thinking..."):
            answer, sources, model_used = run_rag(state, user_input, image_bytes, image_mime_type)

        chat_store.add_message(state.chat_id, "assistant", answer, sources)

        console.print(f"\n[bold green]Pariksha AI[/bold green] [dim](via {model_used})[/dim]:")
        console.print(Markdown(answer))
        if sources:
            console.print("[dim]Sources:[/dim]")
            for s in sources:
                console.print(f"[dim]  - {s}[/dim]")


if __name__ == "__main__":
    main()
