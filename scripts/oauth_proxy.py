"""
oauth_proxy.py

Lazy auto-start helper for the local openai-oauth proxy (`npx openai-oauth`).

Design goal: the proxy should only ever get spawned at the moment something
actually resolves the "openai-oauth" provider - never unconditionally when
app.py / pariksha_cli.py start up. If you never touch that provider in a
session, no extra `node` process gets spawned and no login browser tab pops
open. You never need to manually run `npx openai-oauth` in a separate
terminal again.

Usage: call ensure_running() right before constructing OpenAIOAuthProvider
(this is wired into llm_providers.get_provider(), not something you should
need to call directly). It's safe to call on every request:
  - if the proxy is already answering on its /v1/models endpoint, this is a
    single fast local HTTP GET and returns immediately.
  - if it isn't up yet, it spawns `npx openai-oauth` once (guarded by a
    module-level lock so two questions asked back-to-back can't
    double-spawn) and polls until the endpoint responds, the process exits
    early, or a timeout is hit.
"""

import atexit
import os
import shlex
import shutil
import subprocess
import threading
import time

import requests

# Overridable by tests so they don't have to shell out to a real `npx`.
_SPAWN_CMD = ["npx", "openai-oauth"]

_STARTUP_TIMEOUT_S = 45  # generous - first-ever `npx` run has to fetch the package
_POLL_INTERVAL_S = 0.5
_HEALTH_CHECK_TIMEOUT_S = 1.5

_lock = threading.Lock()
_process = None  # subprocess.Popen we spawned, if any - only ours gets killed at exit
_log_path = None


def _base_url() -> str:
    return os.getenv("OPENAI_OAUTH_BASE_URL", "http://127.0.0.1:10531/v1")


def _is_up(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/models", timeout=_HEALTH_CHECK_TIMEOUT_S)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _default_log_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, ".openai_oauth_proxy.log")


def _tail_log(n_chars: int = 800) -> str:
    if not _log_path or not os.path.exists(_log_path):
        return "(no log captured)"
    with open(_log_path, "r", errors="replace") as f:
        content = f.read()
    return content[-n_chars:].strip() or "(log is empty)"


def _cleanup():
    """atexit hook - only touches a process this module spawned itself."""
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()


atexit.register(_cleanup)


def ensure_running(timeout_s: float = _STARTUP_TIMEOUT_S) -> None:
    """
    Make sure the openai-oauth proxy is reachable at OPENAI_OAUTH_BASE_URL,
    starting it with `npx openai-oauth` if it isn't already up. Raises
    RuntimeError with an actionable message if it can't be reached within
    timeout_s (not logged in, npx/node missing, package failed to start,
    etc.) instead of letting a raw connection error bubble up.
    """
    base_url = _base_url()

    if _is_up(base_url):
        return  # already running (ours from an earlier call, or started manually) - no-op

    global _process, _log_path
    with _lock:
        # Re-check after acquiring the lock - another question asked a
        # moment earlier may have already finished starting it while we
        # were waiting our turn.
        if _is_up(base_url):
            return

        if _process is None or _process.poll() is not None:
            _log_path = _default_log_path()
            log_file = open(_log_path, "w")
            try:
                _process = subprocess.Popen(
                    _SPAWN_CMD,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "Couldn't start the openai-oauth proxy: `npx` was not found. "
                    "Install Node.js (which bundles npx) and try again."
                )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if _process.poll() is not None:
                # Exited before ever coming up - surface why instead of
                # leaving the caller with a bare ConnectionError later.
                raise RuntimeError(
                    "openai-oauth exited before its endpoint came up.\n"
                    f"Last output:\n{_tail_log()}\n\n"
                    "If that mentions a missing auth file, run "
                    "`npx openai-oauth login` once and try again."
                )
            if _is_up(base_url):
                return
            time.sleep(_POLL_INTERVAL_S)

        raise RuntimeError(
            f"openai-oauth did not become ready at {base_url} within "
            f"{timeout_s}s. Check {_log_path} for details."
        )


def _reset_for_tests():
    """Test-only: clear module state between test cases."""
    global _process, _log_path
    _cleanup()
    _process = None
    _log_path = None