#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-30 18:46:19
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-30 18:46:40
"""
presentation_server.py  --  HTTP API server for the semantic presentation search index.

Exposes two endpoints:

    POST /query
        Body:  {"question": "...", "history": [...]}
        Returns: {"answer": "...", "chunks": [...], "history": [...]}

    GET /health
        Returns: {"status": "ok"}

The server is fully stateless — all conversation history is held by the client
and sent with each request.  The server instantiates ``PresentationQuerier``
once at startup and reuses it across requests (Qdrant client, Ollama connection,
etc. are all shared safely for single-worker use).

Start via Gunicorn (see presentation-search.service):

    gunicorn --workers 1 --bind 0.0.0.0:58009 --timeout 180 presentation_server:app

Or for quick testing:

    python presentation_server.py          # Flask dev server, port 58009

Environment overrides (set in the systemd service or shell):

    PRES_INDEX_DIR     Path to Qdrant + manifest directory. Default: ~/.presentation_index
    PRES_COLLECTION    Qdrant collection name.              Default: presentation_index
    PRES_LLM_MODEL     Ollama model tag.                    Default: llama3:8b
    PRES_TOP_K         Number of slides to retrieve.        Default: 5
    PRES_OLLAMA_URL    Ollama base URL.                     Default: http://localhost:11434
    PRES_PORT          Port for the dev-server fallback.    Default: 58009
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from logging_service import LoggingService

from presentation_query import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TOP_K,
    COLLECTION_NAME,
    PresentationQuerier,
    PresentationSynthesiser,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
log = LoggingService()

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
INDEX_DIR  = Path(os.environ.get("PRES_INDEX_DIR",  str(DEFAULT_INDEX_DIR)))
COLLECTION = os.environ.get("PRES_COLLECTION", COLLECTION_NAME)
MODEL      = os.environ.get("PRES_LLM_MODEL",  DEFAULT_MODEL)
TOP_K      = int(os.environ.get("PRES_TOP_K",  str(DEFAULT_TOP_K)))
OLLAMA_URL = os.environ.get("PRES_OLLAMA_URL", DEFAULT_OLLAMA_URL)
PORT       = int(os.environ.get("PRES_PORT",   "58009"))

# ---------------------------------------------------------------------------
# Flask app + shared querier (initialised once at startup)
# ---------------------------------------------------------------------------
app = Flask(__name__)

log.info(
    f"Initialising PresentationQuerier — "
    f"index={INDEX_DIR}  model={MODEL}  top_k={TOP_K}"
)
_querier = PresentationQuerier(
    index_dir=INDEX_DIR,
    collection=COLLECTION,
    top_k=TOP_K,
    model=MODEL,
    ollama_url=OLLAMA_URL,
    use_llm=True,
)
log.info("PresentationQuerier ready.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> tuple:
    """Simple liveness check.

    :return: JSON ``{"status": "ok"}`` with HTTP 200.
    """
    return jsonify({"status": "ok"}), 200


@app.route("/query", methods=["POST"])
def query() -> tuple:
    """Answer a natural-language question about indexed presentations.

    The client supplies the question and the full conversation history from
    prior turns.  The server restores that history into the synthesiser,
    runs retrieval + LLM, and returns the answer plus the updated history.

    Request body (JSON)
    -------------------
    question : str
        The user's question.
    history : list[dict]
        Conversation history as a list of ``{"role": ..., "content": ...}``
        dicts (may be empty for the first turn).

    Response body (JSON)
    --------------------
    answer : str | None
        LLM-generated answer, or ``None`` if LLM is disabled.
    chunks : list[dict]
        Retrieved slide payload dicts.
    history : list[dict]
        Updated conversation history including this turn.
    error : str
        Present only on failure.

    :return: JSON response tuple with HTTP status code.
    """
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in request body."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question must not be empty."}), 400

    history: list[dict] = data.get("history", [])

    # Restore client-supplied history into the synthesiser so follow-on
    # questions have context from prior turns.
    if _querier._synthesiser is not None:
        _querier._synthesiser.reset()
        _querier._synthesiser._history = list(history)

    try:
        result = _querier.query_raw(question)
    except Exception as exc:
        log.warn(f"query_raw error: {exc}")
        return jsonify({"error": str(exc)}), 500

    # Return updated history so the client can send it back next turn
    updated_history = (
        list(_querier._synthesiser._history)
        if _querier._synthesiser is not None
        else history
    )

    return jsonify({
        "answer":  result.get("answer"),
        "chunks":  result.get("chunks", []),
        "history": updated_history,
    }), 200


# ---------------------------------------------------------------------------
# Dev-server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(f"Starting Flask dev server on port {PORT} …")
    app.run(host="0.0.0.0", port=PORT, debug=False)
