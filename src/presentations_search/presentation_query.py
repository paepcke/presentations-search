#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-28 16:16:32
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-28 19:04:56

"""
presentation_query.py  --  Natural-language query interface for the
presentation index built by ``presentation_indexer.py``.

Embeds natural-language questions with ``nomic-embed-text`` via Ollama,
retrieves the most relevant slide chunks from the Qdrant collection, and
asks a local LLM to synthesise an answer.

By default, runs as an interactive REPL.  The LLM sees full conversation
history so follow-up questions work naturally.  Special commands:

    new  /new       Clear conversation history and start a fresh topic.
    quit exit /q    Exit the program.

Usage
-----
    python presentation_query.py                        # interactive (default)
    python presentation_query.py --no-interactive "..."  # single-shot
    python presentation_query.py --slide-text           # show extracted text

Options
-------
    --no-interactive     Single-shot mode: answer one question and exit.
    --slide-text         Show extracted slide text in results (default: off).
    --index-dir   DIR    Same directory passed to presentation_indexer.py.
                         Default: ~/.presentation_index
    --collection  NAME   Qdrant collection name.  Default: presentation_index
    --top-k       N      Number of slides to retrieve per question.  Default: 5
    --model       NAME   Ollama model tag.  Default: llama3:8b
    --ollama-url  URL    Ollama base URL.  Default: http://localhost:11434
    --no-llm             Skip LLM step; print retrieved slide locations only.

Dependencies
------------
    pip install qdrant-client requests
    ollama pull nomic-embed-text
    ollama pull llama3:8b
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import requests
from logging_service import LoggingService
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIM      = 768
COLLECTION_NAME    = "presentation_index"
DEFAULT_INDEX_DIR  = Path.home() / ".presentation_index"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL      = "llama3:8b"
DEFAULT_TOP_K      = 5
SLIDE_TEXT_LINES   = 30   # max lines of slide text shown when --slide-text is set

# Visual formatting
RULE      = "─" * 72
THIN_RULE = "╌" * 72

# Interactive-mode special commands
CMD_NEW  = {"new", "/new"}
CMD_QUIT = {"quit", "exit", "/q"}


# ---------------------------------------------------------------------------
# AggregateHandler  —  answers metadata/count/title questions directly from
# Qdrant payload data, bypassing embedding and the LLM entirely.
# ---------------------------------------------------------------------------

# Patterns are tried in order; first match wins.
# Each entry: (compiled_regex, handler_method_name)
_AGG_PATTERNS: list[tuple] = []   # populated after class definition


class AggregateHandler:
    """Answer aggregate and introspective questions directly from Qdrant.

    Handles questions like:
      * "How many slides do I have?"
      * "How many presentations are indexed?"
      * "What files are indexed?"
      * "Show me the titles of this presentation"
      * "Show titles of the presentation that mentions 'ISTDP'"

    Returns a formatted string answer, or ``None`` if the question does not
    match any aggregate pattern (caller should fall through to normal RAG).

    Accepts an already-open ``QdrantClient`` rather than opening its own, to
    avoid the single-instance file lock on local Qdrant storage.

    :param client:     An open :class:`QdrantClient` instance.
    :param collection: Qdrant collection name.
    """

    def __init__(
        self,
        client:     QdrantClient,
        collection: str = COLLECTION_NAME,
    ) -> None:
        self._collection      = collection
        self._client          = client
        self._last_agg_handler: str | None = None   # tracks last fired handler

    # Regex for bare conversational number references, e.g. "what about 3?",
    # "how about 5", "and 12?", "what about slide 7?"
    _BARE_NUM = re.compile(
        r'^(?:(?:how|what)\s+about\s+(?:slide\s+)?|and\s+(?:slide\s+)?)?(\d+)\??$',
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    def try_answer(self, question: str) -> str | None:
        """Try to answer *question* as an aggregate query.

        If the question looks like a bare number reference (e.g. "what about 3?"
        or just "5?") and the previous turn was a slide-title query, the number
        is treated as a slide position and the title handler is re-invoked.

        :param question: Raw user question string.
        :return: Formatted answer string, or ``None`` if not an aggregate query.
        """
        q = question.strip()

        # Continuation: bare number after a slide-title turn
        m_bare = self._BARE_NUM.match(q)
        if m_bare and self._last_agg_handler == "_answer_title_of_slide":
            try:
                result = self._answer_title_of_slide(question, m_bare)
                # _last_agg_handler stays the same — allows chaining
                return result
            except Exception as exc:
                log.warn(f"Bare-number continuation failed: {exc}")
                return f"  [Error answering that question: {exc}]"

        for pattern, method_name in _AGG_PATTERNS:
            m = pattern.search(q.lower())
            if m:
                try:
                    result = getattr(self, method_name)(question, m)
                    self._last_agg_handler = method_name
                    return result
                except Exception as exc:
                    log.warn(f"Aggregate handler '{method_name}' failed: {exc}")
                    return f"  [Error answering that question: {exc}]"

        # No aggregate match — reset context so a later bare number
        # doesn't incorrectly continue a stale slide-title session.
        self._last_agg_handler = None
        return None

    # ------------------------------------------------------------------
    # --- count handlers -----------------------------------------------

    def _answer_slide_count(self, question: str, match: re.Match) -> str:
        """Return total number of indexed slide points.

        :param question: Original question (unused, for signature consistency).
        :param match:    Regex match object (unused).
        :return: Formatted answer string.
        """
        result = self._client.count(
            collection_name=self._collection,
            exact=True,
        )
        n = result.count
        files = self._distinct_files()
        n_files = len(files)
        noun = "presentation" if n_files == 1 else "presentations"
        return (
            f"You have {n} indexed slide(s) across "
            f"{n_files} {noun}."
        )

    def _answer_file_count(self, question: str, match: re.Match) -> str:
        """Return number of distinct indexed presentation files.

        :param question: Original question (unused).
        :param match:    Regex match object (unused).
        :return: Formatted answer string.
        """
        files = self._distinct_files()
        n = len(files)
        noun = "presentation" if n == 1 else "presentations"
        return f"You have {n} indexed {noun}."

    def _answer_file_list(self, question: str, match: re.Match) -> str:
        """List all indexed presentation file paths.

        :param question: Original question (unused).
        :param match:    Regex match object (unused).
        :return: Formatted answer string.
        """
        files = self._distinct_files()
        if not files:
            return "No presentations are currently indexed."
        lines = [f"  {i+1}.  {Path(f).name}  ({f})"
                 for i, f in enumerate(sorted(files))]
        noun = "presentation" if len(files) == 1 else "presentations"
        return f"{len(files)} indexed {noun}:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # --- title handlers -----------------------------------------------

    def _answer_titles_all(self, question: str, match: re.Match) -> str:
        """List slide titles for all indexed presentations, or the only one.

        :param question: Original question (unused).
        :param match:    Regex match object (unused).
        :return: Formatted answer string.
        """
        files = self._distinct_files()
        if not files:
            return "No presentations are currently indexed."
        parts = []
        for f in sorted(files):
            titles = self._titles_for_file(f)
            parts.append(self._format_title_block(f, titles))
        return "\n\n".join(parts)

    def _answer_titles_filtered(self, question: str, match: re.Match) -> str:
        """List titles of presentations whose text contains a keyword.

        The keyword is the first quoted string found in *question*, or the
        word following "mention(s)" / "contain(s)" / "about" if no quotes
        are present.

        :param question: Original question containing the search term.
        :param match:    Regex match object (unused).
        :return: Formatted answer string.
        """
        keyword = self._extract_keyword(question)
        if not keyword:
            return (
                "I couldn't identify a search term in your question. "
                "Try: \"Show titles of the presentation that mentions 'ISTDP'\""
            )

        matching_files = self._files_mentioning(keyword)
        if not matching_files:
            return f"No indexed presentations mention '{keyword}'."

        parts = []
        for f in sorted(matching_files):
            titles = self._titles_for_file(f)
            parts.append(self._format_title_block(f, titles))
        noun = "presentation" if len(matching_files) == 1 else "presentations"
        header = (
            f"{len(matching_files)} {noun} mention '{keyword}':\n"
        )
        return header + "\n\n".join(parts)

    # ------------------------------------------------------------------
    # --- internal helpers ---------------------------------------------

    def _distinct_files(self) -> list[str]:
        """Return a sorted list of distinct ``file_path`` values in the index.

        Scrolls through all points collecting unique file paths.

        :return: List of absolute path strings.
        """
        seen:   set[str]  = set()
        offset: int | None = None

        while True:
            response, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=["file_path"],
                with_vectors=False,
            )
            for point in response:
                fp = (point.payload or {}).get("file_path")
                if fp:
                    seen.add(fp)
            if next_offset is None:
                break
            offset = next_offset

        return sorted(seen)

    def _titles_for_file(self, file_path: str) -> list[tuple[int, str]]:
        """Return ``(slide_position, title)`` pairs for all slides in *file_path*.

        Slides with an empty title are included with a placeholder so the
        slide numbering remains continuous.

        :param file_path: Absolute path string matching the ``file_path`` payload key.
        :return: List of ``(slide_position, title)`` tuples, sorted by position.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results: list[tuple[int, str]] = []
        offset: int | None = None

        while True:
            response, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path),
                    )]
                ),
                limit=256,
                offset=offset,
                with_payload=["slide_index", "slide_position", "title"],
                with_vectors=False,
            )
            for point in response:
                payload  = point.payload or {}
                # Prefer slide_position (1-based); fall back to slide_index+1
                # for points indexed before this field was added.
                pos   = payload.get("slide_position") or (payload.get("slide_index", 0) + 1)
                title = (payload.get("title") or "").strip()
                results.append((pos, title or "—"))
            if next_offset is None:
                break
            offset = next_offset

        results.sort(key=lambda t: t[0])
        return results

    def _files_mentioning(self, keyword: str) -> list[str]:
        """Return distinct file paths where any text field contains *keyword*.

        Uses Qdrant ``MatchText`` for case-insensitive substring search across
        the ``title``, ``body``, and ``notes`` payload fields.

        :param keyword: Substring to search for (case-insensitive).
        :return: List of matching absolute path strings.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchText

        seen:   set[str]   = set()
        offset: int | None = None

        text_filter = Filter(
            should=[
                FieldCondition(key="title", match=MatchText(text=keyword)),
                FieldCondition(key="body",  match=MatchText(text=keyword)),
                FieldCondition(key="notes", match=MatchText(text=keyword)),
            ]
        )

        while True:
            response, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=text_filter,
                limit=256,
                offset=offset,
                with_payload=["file_path"],
                with_vectors=False,
            )
            for point in response:
                fp = (point.payload or {}).get("file_path")
                if fp:
                    seen.add(fp)
            if next_offset is None:
                break
            offset = next_offset

        return sorted(seen)

    @staticmethod
    def _extract_keyword(question: str) -> str:
        """Extract a search keyword from *question*.

        Tries quoted strings first, then words after trigger verbs.

        :param question: Raw user question.
        :return: Extracted keyword string, or empty string if none found.
        """
        # Prefer quoted term: 'ISTDP' or "ISTDP"
        m = re.search(r"""['\"]([\w\s\-]+)['\"]""", question)
        if m:
            return m.group(1).strip()
        # Fall back to word after trigger verb
        m = re.search(
            r'\b(?:mention(?:ing|s)?|contain(?:ing|s)?|about|with|including)\s+(\w+)',
            question, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return ""

    def _answer_title_of_slide(self, question: str, match: re.Match) -> str:
        """Return the title of a specific slide number.

        :param question: Original question (used to extract slide number).
        :param match:    Regex match containing the slide number in group 1.
        :return: Formatted answer string.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Extract slide number from whichever capture group matched
        num_str = match.group(1) or match.group(2) or match.group(3)
        if not num_str:
            return "I couldn't identify a slide number in your question."
        slide_num = int(num_str)
        files     = self._distinct_files()

        if not files:
            return "No presentations are currently indexed."

        results = []
        for file_path in files:
            response, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="file_path",
                                       match=MatchValue(value=file_path)),
                        FieldCondition(key="slide_position",
                                       match=MatchValue(value=slide_num)),
                    ]
                ),
                limit=5,
                with_payload=["slide_position", "title", "image_only"],
                with_vectors=False,
            )
            for point in response:
                payload   = point.payload or {}
                title     = (payload.get("title") or "").strip()
                img_only  = payload.get("image_only", False)
                file_name = Path(file_path).name
                results.append((file_name, title, img_only))

        if not results:
            return f"No slide {slide_num} found in the index."

        if len(results) == 1:
            file_name, title, img_only = results[0]
            if img_only:
                return f"Slide {slide_num} ({file_name}) is an image-only slide with no text title."
            return f"Slide {slide_num} ({file_name}): \"{title}\""

        # Multiple presentations — list all
        lines = [f"Slide {slide_num} across {len(results)} presentation(s):"]
        for file_name, title, img_only in results:
            label = "[image-only]" if img_only else f"\"{title}\""
            lines.append(f"  {file_name}: {label}")
        return "\n".join(lines)

    @staticmethod
    def _format_title_block(file_path: str, titles: list[tuple[int, str]]) -> str:
        """Format a titled list of slides for one presentation file.

        :param file_path: Absolute path string of the presentation.
        :param titles:    List of ``(slide_position, title)`` tuples.
        :return: Formatted multi-line string.
        """
        name  = Path(file_path).name
        lines = [f"  {file_path}", f"  ({name})  —  {len(titles)} slide(s):"]
        for pos, title in titles:
            lines.append(f"    Slide {pos:>3}:  {title}")
        return "\n".join(lines)


# Populate pattern table now that the class is defined.
# Patterns are tried in order; first match wins.
_AGG_PATTERNS = [
    # Title queries with keyword filter — must come before bare title query.
    # Require either an explicit mention/contain verb, a quoted term, or
    # "about" only when preceded by title/presentation context words.
    # Does NOT match bare conversational "how about slide 4?".
    (re.compile(
        r'\b(?:mention(?:ing|s)?|contain(?:ing|s)?|with\s+the\s+word|including)\b'
        r'|show.*titl.*(?:mention|contain)'
        r"|titl.*\babout\b.{0,30}['\"]"
        r"|presentation.*\babout\b\s+\w",
        re.IGNORECASE),
     "_answer_titles_filtered"),

    # Title of a specific slide number — must come before bare title listing
    (re.compile(
        r'\btitle\b.{0,30}\bslide\s+(\d+)\b'
        r'|\bslide\s+(\d+)\b.{0,30}\btitle\b'
        r'|\bwhat\b.{0,20}\bslide\s+(\d+)\b',
        re.IGNORECASE),
     "_answer_title_of_slide"),

    # Bare title listing
    (re.compile(
        r'\b(?:show|list|give|what\s+are).*\btitles?\b'
        r'|\btitles?\b.*(?:of|in|for)\b',
        re.IGNORECASE),
     "_answer_titles_all"),

    # File / presentation listing
    (re.compile(
        r'\b(?:what|which|list|show)\b.{0,30}\b(?:files?|presentations?|decks?)\b'
        r'.*\b(?:indexed|index|have|do i have)\b'
        r'|\b(?:files?|presentations?|decks?)\b.*\b(?:indexed|index|available)\b',
        re.IGNORECASE),
     "_answer_file_list"),

    # Presentation / file count
    (re.compile(
        r'\bhow\s+many\b.{0,30}\b(?:presentations?|files?|decks?)\b',
        re.IGNORECASE),
     "_answer_file_count"),

    # Slide count
    (re.compile(
        r'\bhow\s+many\b.{0,30}\bslides?\b'
        r'|\bslide\s+count\b'
        r'|\bnumber\s+of\s+slides?\b',
        re.IGNORECASE),
     "_answer_slide_count"),
]


# ---------------------------------------------------------------------------
# PresentationRetriever
# ---------------------------------------------------------------------------

class PresentationRetriever:
    """Embed a query and retrieve the top-k matching slide chunks from Qdrant.

    :param index_dir:  Directory containing the Qdrant file-based storage.
    :param collection: Qdrant collection name.
    :param top_k:      Number of results to return.
    :param ollama_url: Base URL for the Ollama REST API.
    """

    def __init__(
        self,
        index_dir:  Path = DEFAULT_INDEX_DIR,
        collection: str  = COLLECTION_NAME,
        top_k:      int  = DEFAULT_TOP_K,
        ollama_url: str  = DEFAULT_OLLAMA_URL,
    ) -> None:
        self._collection = collection
        self._top_k      = top_k
        self._embed_url  = f"{ollama_url.rstrip('/')}/api/embed"
        self._client     = QdrantClient(path=str(index_dir / "qdrant"))

    @property
    def client(self) -> QdrantClient:
        """The underlying :class:`QdrantClient` instance.

        Exposed so that :class:`AggregateHandler` can share the same open
        client rather than acquiring the file lock a second time.

        :return: Open ``QdrantClient``.
        """
        return self._client

    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> list[dict]:
        """Embed *question* and return the top-k slide chunk payloads.

        :param question: Natural-language query string.
        :return: List of Qdrant payload dicts, ordered by relevance.
        :raises RuntimeError: If the Ollama embed API call fails.
        """
        vec      = self._embed(question)
        response = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=self._top_k,
            with_payload=True,
        )
        return [hit.payload for hit in response.points]

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        """Embed *text* using the Ollama nomic-embed-text REST API.

        Uses the ``query:`` prefix for asymmetric retrieval — the indexer
        used ``passage:`` on the stored side.

        :param text: Text to embed.
        :return: Embedding vector as a list of floats.
        :raises RuntimeError: If the API call fails.
        """
        resp = requests.post(
            self._embed_url,
            json={"model": OLLAMA_EMBED_MODEL, "input": f"query: {text}"},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama embed API returned {resp.status_code}: {resp.text}"
            )
        return resp.json()["embeddings"][0]


# ---------------------------------------------------------------------------
# PresentationSynthesiser
# ---------------------------------------------------------------------------

class PresentationSynthesiser:
    """Generate a natural-language answer from retrieved slide chunks via Ollama.

    Maintains conversation history so follow-up questions within the same
    session have context from prior turns.

    :param model:      Ollama model tag, e.g. ``llama3:8b``.
    :param ollama_url: Base URL for the Ollama REST API.
    """

    _SYSTEM = (
        "You are a concise assistant that helps a researcher find information "
        "across their Keynote and PowerPoint presentations.  Answer in as few "
        "words as possible.  Always cite the file path and slide number when "
        "referring to a specific slide.  If the provided slide excerpts do not "
        "answer the question, say so in one sentence."
    )

    def __init__(
        self,
        model:      str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self._model    = model
        self._chat_url = f"{ollama_url.rstrip('/')}/api/chat"
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear conversation history to start a new topic."""
        self._history = []

    # ------------------------------------------------------------------
    def explain(self, question: str, chunks: list[dict]) -> str:
        """Generate a synthesised answer given *question* and *chunks*.

        Appends the exchange to internal history so follow-up questions
        have context.

        :param question: The user's natural-language question.
        :param chunks:   Retrieved slide chunk payload dicts.
        :return: LLM response text, or an error message string.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            path       = chunk.get("file_path", "unknown")
            slide_idx  = chunk.get("slide_index", "?")
            title      = chunk.get("title", "")
            body       = chunk.get("body", "")
            notes      = chunk.get("notes", "")
            fmt        = chunk.get("source_format", "")
            context_parts.append(
                f"[{i}] {path}  slide {slide_idx}  ({fmt})\n"
                f"Title: {title}\n"
                f"Body: {body}\n"
                f"Notes: {notes}"
            )
        context = "\n\n".join(context_parts)

        user_msg = (
            f"Slide excerpts:\n\n{context}\n\n"
            f"Question: {question}"
        )

        self._history.append({"role": "user", "content": user_msg})

        payload = {
            "model":    self._model,
            "stream":   False,
            "messages": [{"role": "system", "content": self._SYSTEM}]
                        + self._history,
        }

        try:
            resp = requests.post(self._chat_url, json=payload, timeout=120)
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]
        except Exception as exc:
            answer = f"[LLM error: {exc}]"

        self._history.append({"role": "assistant", "content": answer})
        return answer


# ---------------------------------------------------------------------------
# ResultPrinter
# ---------------------------------------------------------------------------

class ResultPrinter:
    """Format and print retrieved slide results to stdout.

    :param show_slide_text: If ``True``, print extracted slide text under
        each result (up to ``SLIDE_TEXT_LINES`` lines).
    """

    def __init__(self, show_slide_text: bool = False) -> None:
        self._show_slide_text = show_slide_text

    # ------------------------------------------------------------------
    def print_results(
        self,
        question:    str,
        chunks:      list[dict],
        explanation: str | None,
    ) -> None:
        """Print a formatted result block to stdout.

        :param question:    The original user question.
        :param chunks:      Retrieved slide chunk payload dicts.
        :param explanation: LLM-generated answer, or ``None`` if LLM disabled.
        """
        print()
        print(RULE)

        if not chunks:
            print("  No results found.")
            print(RULE)
            return

        for i, chunk in enumerate(chunks, 1):
            path      = chunk.get("file_path", "unknown")
            slide_pos = chunk.get("slide_position") or (chunk.get("slide_index", 0) + 1)
            fmt       = chunk.get("source_format", "")
            title     = chunk.get("title", "").strip()
            body      = chunk.get("body", "").strip()
            notes     = chunk.get("notes", "").strip()

            fmt_tag   = f"[{fmt}]" if fmt else ""
            title_tag = f"  \"{title}\"" if title else ""
            print(f"  {i}.  {path}{title_tag}")
            print(f"       Slide {slide_pos}  {fmt_tag}")

            if self._show_slide_text:
                print(THIN_RULE)
                for field_label, field_text in [
                    ("Title", title), ("Body", body), ("Notes", notes)
                ]:
                    if field_text:
                        truncated = self._truncate(field_text)
                        indented  = textwrap.indent(truncated, "    ")
                        print(f"  {field_label}:")
                        print(indented)

            print()

        if explanation is not None:
            print(RULE)
            print("  Answer:")
            print()
            answer_indented = textwrap.indent(
                textwrap.fill(explanation, width=68), "    "
            )
            print(answer_indented)

        print(RULE)
        print()

    # ------------------------------------------------------------------
    @staticmethod
    def _truncate(text: str, max_lines: int = SLIDE_TEXT_LINES) -> str:
        """Truncate *text* to *max_lines* lines, appending an ellipsis if cut.

        :param text:      Text to truncate.
        :param max_lines: Maximum number of lines to retain.
        :return: Truncated string.
        """
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[:max_lines]) + f"\n  … ({len(lines) - max_lines} more lines)"


# ---------------------------------------------------------------------------
# PresentationQuerier  (top-level orchestrator)
# ---------------------------------------------------------------------------

class PresentationQuerier:
    """Orchestrate retrieval, synthesis, and printing for one query turn.

    This is the class the Flask API will instantiate directly, passing
    ``use_llm=True`` and calling :meth:`query` per HTTP request.

    :param index_dir:       Qdrant storage directory.
    :param collection:      Qdrant collection name.
    :param top_k:           Number of slides to retrieve.
    :param model:           Ollama LLM model tag.
    :param ollama_url:      Ollama base URL.
    :param use_llm:         If ``False``, skip the LLM step entirely.
    :param show_slide_text: If ``True``, include extracted slide text in output.
    """

    def __init__(
        self,
        index_dir:       Path = DEFAULT_INDEX_DIR,
        collection:      str  = COLLECTION_NAME,
        top_k:           int  = DEFAULT_TOP_K,
        model:           str  = DEFAULT_MODEL,
        ollama_url:      str  = DEFAULT_OLLAMA_URL,
        use_llm:         bool = True,
        show_slide_text: bool = False,
    ) -> None:
        index_dir = index_dir.expanduser().resolve()

        self._retriever = PresentationRetriever(
            index_dir=index_dir,
            collection=collection,
            top_k=top_k,
            ollama_url=ollama_url,
        )
        self._synthesiser: PresentationSynthesiser | None = (
            PresentationSynthesiser(model=model, ollama_url=ollama_url)
            if use_llm else None
        )
        self._printer   = ResultPrinter(show_slide_text=show_slide_text)
        self._aggregate = AggregateHandler(
            client=self._retriever.client,
            collection=collection,
        )
        self._use_llm   = use_llm

    # ------------------------------------------------------------------
    def query(self, question: str) -> None:
        """Run one full retrieve-synthesise-print cycle for *question*.

        Aggregate/introspective questions (slide counts, title listings, etc.)
        are answered directly from Qdrant metadata without invoking the
        retriever or LLM.  All exceptions are caught and reported gracefully
        so the interactive session survives unexpected errors.

        :param question: Natural-language question from the user.
        """
        try:
            agg = self._aggregate.try_answer(question)
            if agg is not None:
                print()
                print(RULE)
                print(agg)
                print(RULE)
                print()
                return

            chunks = self._retriever.retrieve(question)

            explanation: str | None = None
            if self._use_llm and self._synthesiser is not None:
                print("  Asking LLM …")
                explanation = self._synthesiser.explain(question, chunks)

            self._printer.print_results(question, chunks, explanation)

        except Exception as exc:
            log.warn(f"Error processing question: {exc}")
            print()
            print(RULE)
            print(f"  Error: {exc}")
            print(f"  The session is still active — please try another question.")
            print(RULE)
            print()

    # ------------------------------------------------------------------
    def query_raw(self, question: str) -> dict:
        """Retrieve slides and optionally synthesise an answer; return as dict.

        Aggregate questions are answered directly and returned under the
        ``"answer"`` key with an empty ``"chunks"`` list.

        Intended for use by the Flask HTTP API — returns structured data
        rather than printing to stdout.  Exceptions are caught and returned
        as ``{"chunks": [], "answer": None, "error": "..."}`` so the API
        never raises an unhandled 500.

        :param question: Natural-language question.
        :return: Dict with keys ``"chunks"``, ``"answer"``, and optionally
            ``"error"``.
        """
        try:
            agg = self._aggregate.try_answer(question)
            if agg is not None:
                return {"chunks": [], "answer": agg}

            chunks = self._retriever.retrieve(question)
            answer: str | None = None
            if self._use_llm and self._synthesiser is not None:
                answer = self._synthesiser.explain(question, chunks)
            return {"chunks": chunks, "answer": answer}

        except Exception as exc:
            log.warn(f"query_raw error: {exc}")
            return {"chunks": [], "answer": None, "error": str(exc)}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear LLM conversation history for a new topic.

        Safe to call even when LLM is disabled.
        """
        if self._synthesiser is not None:
            self._synthesiser.reset()

    # ------------------------------------------------------------------
    def run_interactive(self, initial_question: str | None = None) -> None:
        """Enter the interactive REPL loop.

        Reads questions from stdin, maintains conversation history across
        turns, and honours the ``new``/``quit`` special commands.

        If *initial_question* is provided it is answered immediately before
        the first prompt, so a question passed on the CLI becomes the first
        turn of the session rather than being discarded.

        Readline is configured for Emacs-style key bindings (Ctrl-a, Ctrl-e,
        Ctrl-d, Ctrl-k, etc.) and persistent history via ``~/.pres_query_history``.

        :param initial_question: Optional question to answer before prompting.
        """
        # ---- readline setup (Emacs bindings + history) ----
        try:
            import readline
            readline.parse_and_bind("set editing-mode emacs")
            hist_file = Path.home() / ".pres_query_history"
            try:
                readline.read_history_file(str(hist_file))
            except FileNotFoundError:
                pass
            import atexit
            atexit.register(readline.write_history_file, str(hist_file))
        except ImportError:
            pass  # readline not available on all platforms

        print()
        print(RULE)
        print("  Presentation Search  —  interactive mode")
        print(f"  Index: {DEFAULT_INDEX_DIR}")
        print("  Commands:  new / /new        →  fresh topic")
        print("             quit / exit / /q  →  exit")
        print(RULE)
        print()

        # Answer the CLI-supplied question as the first turn, then loop
        if initial_question:
            self.query(initial_question)  # exceptions caught inside query()

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

            self.query(raw)  # exceptions caught inside query()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    :return: Configured :class:`argparse.ArgumentParser` instance.
    """
    p = argparse.ArgumentParser(
        prog="presentation_query",
        description="Natural-language query tool for the presentation index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "question",
        nargs="*",
        help="Question text for --no-interactive mode (omit in interactive mode).",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Single-shot mode: answer one question and exit.",
    )
    p.add_argument(
        "--slide-text",
        action="store_true",
        help="Show extracted slide text alongside each result (default: off).",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        metavar="DIR",
        help=f"Index directory (same as presentation_indexer.py). Default: {DEFAULT_INDEX_DIR}",
    )
    p.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        metavar="NAME",
        help=f"Qdrant collection name. Default: {COLLECTION_NAME}",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"Number of slides to retrieve. Default: {DEFAULT_TOP_K}",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=f"Ollama LLM model tag. Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"Ollama base URL. Default: {DEFAULT_OLLAMA_URL}",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM synthesis; print retrieved slide locations only.",
    )
    return p


def main() -> None:
    """Entry point for the presentation query CLI."""
    parser = _build_parser()
    args   = parser.parse_args()

    querier = PresentationQuerier(
        index_dir=args.index_dir,
        collection=args.collection,
        top_k=args.top_k,
        model=args.model,
        ollama_url=args.ollama_url,
        use_llm=not args.no_llm,
        show_slide_text=args.slide_text,
    )

    if args.no_interactive:
        if not args.question:
            parser.error("--no-interactive requires a question as a positional argument.")
        querier.query(" ".join(args.question))
    else:
        # If a question was passed on the CLI, use it as the first turn
        # of the interactive session rather than discarding it.
        initial = " ".join(args.question) if args.question else None
        querier.run_interactive(initial_question=initial)


if __name__ == "__main__":
    main()