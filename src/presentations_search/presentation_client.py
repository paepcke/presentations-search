#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-30 19:13:38
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-30 19:13:56
"""
presentation_client.py  --  Interactive client for the remote presentation
search server.

Connects to a ``presentation_server.py`` instance (e.g. on sextus) and
provides the same interactive REPL experience as running
``presentation_query.py`` locally.  Conversation history is maintained on
the client and sent with every request; the server is fully stateless.

Usage
-----
    python presentation_client.py                         # connect to sextus:58009
    python presentation_client.py --host 192.168.1.111   # explicit IP
    python presentation_client.py --host sextus.local    # mDNS
    python presentation_client.py "opening question"     # answer then prompt
    python presentation_client.py --no-interactive "..."  # single-shot

Options
-------
    --host   HOST   Hostname or IP of the server.  Default: sextus.local
    --port   N      Server port.                   Default: 58009
    --no-interactive   Single-shot mode: answer one question and exit.

Special commands (interactive mode)
------------------------------------
    new  /new       Clear conversation history — start a fresh topic.
    quit exit /q    Exit.

Dependencies
------------
    pip install requests        # only dependency — no Qdrant, no Ollama
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HOST = "sextus.local"
DEFAULT_PORT = 58009

RULE      = "─" * 72
THIN_RULE = "╌" * 72

CMD_NEW  = {"new", "/new"}
CMD_QUIT = {"quit", "exit", "/q"}


# ---------------------------------------------------------------------------
# PresentationSearchClient
# ---------------------------------------------------------------------------

class PresentationSearchClient:
    """HTTP client for the remote presentation search server.

    Maintains conversation history locally and sends it with every request
    so the server can produce contextually aware follow-up answers.

    :param host: Hostname or IP address of the server.
    :param port: TCP port the server listens on.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._base_url = f"http://{host}:{port}"
        self._history: list[dict] = []
        self._check_server()

    # ------------------------------------------------------------------
    def _check_server(self) -> None:
        """Verify the server is reachable.

        :raises SystemExit: If the server cannot be reached.
        """
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=5)
            resp.raise_for_status()
            print(f"Connected to {self._base_url}")
        except requests.exceptions.ConnectionError:
            print(
                f"ERROR: Cannot connect to presentation search server at "
                f"{self._base_url}.\n"
                f"Is the service running on the remote host?\n"
                f"  sudo systemctl status presentation-search",
                file=sys.stderr,
            )
            sys.exit(1)
        except requests.exceptions.HTTPError as exc:
            print(f"ERROR: Server returned {exc}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    def query(self, question: str) -> None:
        """Send *question* to the server, print the answer and results.

        :param question: Natural-language question.
        """
        try:
            resp = requests.post(
                f"{self._base_url}/query",
                json={"question": question, "history": self._history},
                timeout=180,   # LLM calls can be slow
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            print()
            print(RULE)
            print("  Request timed out — the server may be busy with a long LLM call.")
            print(RULE)
            print()
            return
        except Exception as exc:
            print()
            print(RULE)
            print(f"  Error: {exc}")
            print(RULE)
            print()
            return

        if "error" in data:
            print()
            print(RULE)
            print(f"  Server error: {data['error']}")
            print(RULE)
            print()
            return

        # Update local history from server response
        self._history = data.get("history", self._history)

        chunks      = data.get("chunks", [])
        answer      = data.get("answer")
        self._print_results(question, chunks, answer)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear conversation history to start a new topic."""
        self._history = []

    # ------------------------------------------------------------------
    def run_interactive(self, initial_question: str | None = None) -> None:
        """Enter the interactive REPL loop.

        If *initial_question* is provided it is answered immediately before
        the first prompt, matching the behaviour of ``presentation_query.py``.

        :param initial_question: Optional first question from the CLI.
        """
        # Emacs-style readline + persistent history
        try:
            import readline
            readline.parse_and_bind("set editing-mode emacs")
            hist_file = Path.home() / ".pres_client_history"
            try:
                readline.read_history_file(str(hist_file))
            except FileNotFoundError:
                pass
            import atexit
            atexit.register(readline.write_history_file, str(hist_file))
        except ImportError:
            pass

        print()
        print(RULE)
        print("  Presentation Search  —  interactive mode")
        print(f"  Server: {self._base_url}")
        print("  Commands:  new / /new        →  fresh topic")
        print("             quit / exit / /q  →  exit")
        print(RULE)
        print()

        if initial_question:
            self.query(initial_question)

        while True:
            try:
                raw = input("❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue
            if raw.lower() in CMD_QUIT:
                break
            if raw.lower() in CMD_NEW:
                self.reset()
                print()
                print("  [conversation history cleared — new topic]")
                print()
                continue

            self.query(raw)

    # ------------------------------------------------------------------
    def _print_results(
        self,
        question:    str,
        chunks:      list[dict],
        answer:      str | None,
    ) -> None:
        """Format and print results to stdout.

        :param question: Original question.
        :param chunks:   Retrieved slide payload dicts from the server.
        :param answer:   LLM-generated answer, or ``None``.
        """
        print()
        print(RULE)

        if not chunks and not answer:
            print("  No results found.")
            print(RULE)
            print()
            return

        for i, chunk in enumerate(chunks, 1):
            path      = chunk.get("file_path", "unknown")
            slide_pos = chunk.get("slide_position") or (chunk.get("slide_index", 0) + 1)
            fmt       = chunk.get("source_format", "")
            title     = (chunk.get("title") or "").strip()

            fmt_tag   = f"[{fmt}]" if fmt else ""
            title_tag = f"  \"{title}\"" if title else ""
            print(f"  {i}.  {path}{title_tag}")
            print(f"       Slide {slide_pos}  {fmt_tag}")
            print()

        if answer is not None:
            print(RULE)
            print("  Answer:")
            print()
            for line in answer.splitlines():
                if line.strip():
                    filled = textwrap.fill(line, width=68,
                                          subsequent_indent="      ")
                    print(textwrap.indent(filled, "    "))
                else:
                    print()

        print(RULE)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    :return: Configured :class:`argparse.ArgumentParser` instance.
    """
    p = argparse.ArgumentParser(
        prog="presentation_client",
        description=(
            "Interactive client for the remote presentation search server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "question",
        nargs="*",
        help="Opening question (answered immediately, then drops into REPL).",
    )
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        metavar="HOST",
        help=f"Server hostname or IP. Default: {DEFAULT_HOST}",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        metavar="N",
        help=f"Server port. Default: {DEFAULT_PORT}",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Single-shot mode: answer one question and exit.",
    )
    return p


def main() -> None:
    """Entry point for the presentation search client."""
    parser = _build_parser()
    args   = parser.parse_args()

    client = PresentationSearchClient(host=args.host, port=args.port)

    if args.no_interactive:
        if not args.question:
            parser.error("--no-interactive requires a question as a positional argument.")
        client.query(" ".join(args.question))
    else:
        initial = " ".join(args.question) if args.question else None
        client.run_interactive(initial_question=initial)


if __name__ == "__main__":
    main()
