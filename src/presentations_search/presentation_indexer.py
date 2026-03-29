#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-28 15:58:06
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-28 18:52:20
"""
presentation_indexer.py  --  Index Keynote (.key) and PowerPoint (.pptx) files
into a Qdrant vector store for semantic search.

Each presentation is chunked into per-slide documents carrying:
  * slide title
  * body text (all text boxes / bullet runs)
  * speaker notes
  * file path, slide index, and content type as metadata

Keynote files are unpacked via ``keynote-parser`` into a temporary directory;
text is extracted by walking the resulting YAML archives.  PowerPoint files are
handled directly by ``python-pptx``.

Only changed or new files are re-indexed on subsequent runs (mtime manifest).
Deleted files are pruned from the collection automatically.

Usage
-----
    python presentation_indexer.py presentations/ ~/Dropbox/Talks MySlides.key
    python presentation_indexer.py ~/Talks --no-recursive
    python presentation_indexer.py ~/Talks --force --verbose

Options
-------
    inputs             One or more directories or .key/.pptx files.
    --index-dir  DIR   Qdrant storage + manifest directory.
                       Default: ~/.presentation_index
    --collection NAME  Qdrant collection name.  Default: presentation_index
    --batch-size N     Embedding batch size.  Default: 32
    --force            Ignore mtime manifest; re-index everything.
    --no-recursive     Only look one level deep in directories (no descent).
    --verbose          Log each file path as it is processed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Generator

import yaml
from logging_service import LoggingService
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Module-level logger (mirrors code-search project convention)
# ---------------------------------------------------------------------------
log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR  = Path("~/.presentation_index")
COLLECTION_NAME    = "presentation_index"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIM      = 768
DEFAULT_OLLAMA_URL = "http://localhost:11434"
SUPPORTED_SUFFIXES = {".key", ".pptx"}

# Protobuf type names that carry actual slide text in Keynote YAML
_STORAGE_PBTYPE = "TSWP.StorageArchive"
# Object-replacement character used as image/table placeholder in text runs
_ORC = "\uFFFC"


# ---------------------------------------------------------------------------
# SlideChunk dataclass (plain dict so it serialises easily into Qdrant payload)
# ---------------------------------------------------------------------------

def _make_chunk(
    file_path: str,
    slide_index: int,
    slide_position: int,
    title: str,
    body: str,
    notes: str,
    source_format: str,
    image_only: bool = False,
) -> dict:
    """Return a payload dict for a single slide chunk.

    :param file_path:      Absolute path to the source presentation file.
    :param slide_index:    0-based index derived from sorted YAML filename order.
                           Not meaningful as a human-readable slide number.
    :param slide_position: 1-based slide position matching the presentation
                           application's numbering (derived from slideTree order
                           for Keynote; from slide order for PPTX).
    :param title:          Slide title text (may be empty).
    :param body:           All non-title, non-note text joined with newlines.
    :param notes:          Speaker-note text (may be empty).
    :param source_format:  ``"keynote"`` or ``"pptx"``.
    :param image_only:     ``True`` if no text was extractable (image/diagram slide).
    :return: Payload dict suitable for a Qdrant ``PointStruct``.
    """
    return {
        "file_path":      file_path,
        "slide_index":    slide_index,
        "slide_position": slide_position,
        "title":          title,
        "body":           body,
        "notes":          notes,
        "source_format":  source_format,
        "image_only":     image_only,
        # Combined text field used for embedding
        "text": "\n".join(filter(None, [title, body, notes])),
    }


# ---------------------------------------------------------------------------
# Keynote extractor
# ---------------------------------------------------------------------------

class KeynoteExtractor:
    """Extract per-slide text chunks from a Keynote ``.key`` file.

    Uses ``keynote-parser`` to unpack the file into a temporary directory of
    YAML archives, then walks each ``Slide-*.iwa.yaml`` file to collect
    ``TSWP.StorageArchive`` objects, distinguishing title placeholders,
    body text, and speaker notes.

    :param key_path: Path to the ``.key`` file to extract.
    """

    def __init__(self, key_path: Path) -> None:
        self._key_path = key_path

    # ------------------------------------------------------------------
    def extract(self) -> list[dict]:
        """Unpack the Keynote file and return a list of slide chunk dicts.

        Creates a temporary directory for unpacking that is cleaned up
        automatically after extraction regardless of success or failure.

        :return: List of slide payload dicts (see :func:`_make_chunk`).
        :raises RuntimeError: If ``keynote-parser`` is not installed or
            the unpack command fails.
        """
        import subprocess

        chunks: list[dict] = []

        with tempfile.TemporaryDirectory(prefix="kn_unpack_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            unpack_dir = tmp_path / "unpacked"

            result = subprocess.run(
                [
                    "keynote-parser", "unpack",
                    str(self._key_path),
                    "--output", str(unpack_dir),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"keynote-parser failed on '{self._key_path}': "
                    f"{result.stderr.strip()}"
                )

            index_dir = unpack_dir / "Index"
            if not index_dir.is_dir():
                log.warn(
                    f"No Index/ directory after unpacking '{self._key_path}' "
                    f"— skipping."
                )
                return chunks

            # Build archive-id → 1-based Keynote position map from Document.iwa.yaml.
            # Also get total slide count so we can stub positions with no YAML file.
            position_map, total_positions = self._build_position_map(index_dir)

            # Collect slide YAML files, sort by name for stable slide order
            slide_files = sorted(index_dir.glob("Slide-*.iwa.yaml"))
            for slide_idx, yaml_file in enumerate(slide_files):
                # Extract archive id from filename: Slide-XXXXXX.iwa.yaml
                archive_id   = int(yaml_file.stem.split("Slide-")[1].replace(".iwa", ""))
                slide_pos    = position_map.get(archive_id, slide_idx + 1)
                slide_chunks = self._parse_slide_yaml(yaml_file, slide_idx, slide_pos)
                chunks.extend(slide_chunks)

            # Emit image-only stubs for slideTree positions that have no YAML file
            # (keynote-parser produces no output for these slides — typically pure
            # image slides created outside Keynote, e.g. in Affinity Photo).
            mapped_positions = set(position_map.values())
            for pos in range(1, total_positions + 1):
                if pos not in mapped_positions:
                    chunks.append(_make_chunk(
                        file_path=str(self._key_path.resolve()),
                        slide_index=-1,
                        slide_position=pos,
                        title="[image-only slide]",
                        body="",
                        notes="",
                        source_format="keynote",
                        image_only=True,
                    ))

        return chunks

    # ------------------------------------------------------------------
    @staticmethod
    def _build_position_map(index_dir: Path) -> dict[int, int]:
        """Parse ``Document.iwa.yaml`` to map YAML archive ids to 1-based
        Keynote slide positions.

        The ``KN.ShowArchive`` in ``Document.iwa.yaml`` contains a
        ``slideTree.slides`` list giving the presentation order.  Each entry
        references a slideTree identifier that is slightly larger than the
        corresponding ``Slide-XXXXXX.iwa.yaml`` archive id.  We match each
        slideTree id to the largest yaml archive id that is ≤ the tree id
        with a delta ≤ ``_MAX_ID_DELTA`` (empirically ~40; we use 100 for
        safety).

        :param index_dir: Path to the unpacked ``Index/`` directory.
        :return: Dict mapping yaml archive id (int) → 1-based slide position.
        """
        _MAX_ID_DELTA = 100

        doc_yaml = index_dir / "Document.iwa.yaml"
        if not doc_yaml.exists():
            return {}

        try:
            with open(doc_yaml, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            log.warn(f"Could not parse Document.iwa.yaml: {exc} — slide positions unavailable.")
            return {}

        # Locate KN.ShowArchive (referenced as 'show' from KN.DocumentArchive)
        # Find its identifier first
        show_id: str | None = None
        for chunk in data.get("chunks", []):
            for archive in chunk.get("archives", []):
                for obj in archive.get("objects", []):
                    if obj.get("_pbtype") == "KN.DocumentArchive":
                        show_id = obj.get("show", {}).get("identifier")
                        break

        if not show_id:
            return {}

        # Now find the KN.ShowArchive with that identifier
        ordered_tree_ids: list[int] = []
        for chunk in data.get("chunks", []):
            for archive in chunk.get("archives", []):
                if archive.get("header", {}).get("identifier") == show_id:
                    for obj in archive.get("objects", []):
                        slides = obj.get("slideTree", {}).get("slides", [])
                        ordered_tree_ids = [int(s["identifier"]) for s in slides]
                        break

        if not ordered_tree_ids:
            return {}

        # Collect all yaml archive ids from Slide-*.iwa.yaml filenames
        yaml_ids = sorted(
            int(f.stem.split("Slide-")[1].replace(".iwa", ""))
            for f in index_dir.glob("Slide-*.iwa.yaml")
        )

        # Match each tree id to the largest yaml id ≤ tree id with delta ≤ MAX.
        # Each yaml_id may only be claimed once (first match in slideTree order
        # wins).  Tree ids with no unclaimed candidate have no YAML file —
        # keynote-parser dropped them — and are silently skipped.
        position_map: dict[int, int] = {}
        claimed: set[int] = set()

        for position, tree_id in enumerate(ordered_tree_ids, 1):
            candidates = [y for y in yaml_ids
                          if y <= tree_id and (tree_id - y) <= _MAX_ID_DELTA
                          and y not in claimed]
            if candidates:
                matched_yaml_id = max(candidates)
                position_map[matched_yaml_id] = position
                claimed.add(matched_yaml_id)

        return position_map, len(ordered_tree_ids)

    # ------------------------------------------------------------------
    def _parse_slide_yaml(self, yaml_file: Path, slide_idx: int, slide_position: int) -> list[dict]:
        """Parse one ``Slide-XXX.iwa.yaml`` file into zero or more chunks.

        A single YAML file contains one logical slide.  We collect one
        combined chunk per slide with separate title, body, and notes fields.

        :param yaml_file:      Path to the unpacked YAML file.
        :param slide_idx:      0-based index from sorted filename order.
        :param slide_position: 1-based Keynote slide number from slideTree.
        :return: List with 0 or 1 chunk dicts for this slide.
        """
        try:
            with open(yaml_file, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            log.warn(f"Could not parse '{yaml_file}': {exc} — skipping slide.")
            return []

        title_texts: list[str] = []
        body_texts:  list[str] = []
        note_texts:  list[str] = []

        # Walk all archive objects in the chunk list
        for chunk in data.get("chunks", []):
            for archive in chunk.get("archives", []):
                for obj in archive.get("objects", []):
                    if obj.get("_pbtype") != _STORAGE_PBTYPE:
                        continue

                    raw_texts = [
                        t for t in obj.get("text", [])
                        if isinstance(t, str) and t.strip(_ORC).strip()
                    ]
                    if not raw_texts:
                        continue

                    combined = "\n".join(raw_texts).replace(_ORC, "").strip()
                    if not combined:
                        continue

                    kind = obj.get("kind", "")
                    if kind == "NOTE":
                        note_texts.append(combined)
                    else:
                        # Treat the first non-note, single-line short text as
                        # the title.  Multi-line text (newlines present) is
                        # body content even if it appears first.
                        single_line = combined.replace("\n", " ").strip()
                        if not title_texts and "\n" not in combined and len(single_line) < 200:
                            title_texts.append(single_line)
                        else:
                            body_texts.append(combined)

        title = " ".join(title_texts)
        body  = "\n".join(body_texts)
        notes = "\n".join(note_texts)

        if not any([title, body, notes]):
            # Slide contains no extractable text (image/diagram only).
            # Index a stub so the slide is counted and findable by position.
            return [_make_chunk(
                file_path=str(self._key_path.resolve()),
                slide_index=slide_idx,
                slide_position=slide_position,
                title="[image-only slide]",
                body="",
                notes="",
                source_format="keynote",
                image_only=True,
            )]

        return [_make_chunk(
            file_path=str(self._key_path.resolve()),
            slide_index=slide_idx,
            slide_position=slide_position,
            title=title,
            body=body,
            notes=notes,
            source_format="keynote",
        )]


# ---------------------------------------------------------------------------
# PowerPoint extractor
# ---------------------------------------------------------------------------

class PptxExtractor:
    """Extract per-slide text chunks from a PowerPoint ``.pptx`` file.

    Uses ``python-pptx`` to iterate over slides, collecting title placeholder
    text, all other shape text, and speaker notes.

    :param pptx_path: Path to the ``.pptx`` file to extract.
    """

    def __init__(self, pptx_path: Path) -> None:
        self._pptx_path = pptx_path

    # ------------------------------------------------------------------
    def extract(self) -> list[dict]:
        """Return a list of per-slide chunk dicts for the presentation.

        :return: List of slide payload dicts (see :func:`_make_chunk`).
        :raises RuntimeError: If ``python-pptx`` is not installed or the
            file cannot be opened.
        """
        from pptx import Presentation
        from pptx.enum.text import PP_ALIGN  # noqa: F401 (imported for type hints)
        from pptx.util import Pt             # noqa: F401

        try:
            prs = Presentation(str(self._pptx_path))
        except Exception as exc:
            raise RuntimeError(
                f"python-pptx could not open '{self._pptx_path}': {exc}"
            ) from exc

        chunks: list[dict] = []

        for slide_idx, slide in enumerate(prs.slides):
            title = ""
            body_parts: list[str] = []
            notes = ""

            # ---- slide body shapes ----
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                text = shape.text_frame.text.strip()
                if not text:
                    continue
                if shape.is_placeholder:
                    ph_type = shape.placeholder_format.type
                    # TITLE = 1, CENTER_TITLE = 3
                    if ph_type in (1, 3) and not title:
                        title = text
                        continue
                body_parts.append(text)

            # ---- speaker notes ----
            if slide.has_notes_slide:
                notes_tf = slide.notes_slide.notes_text_frame
                notes = notes_tf.text.strip() if notes_tf else ""

            body = "\n".join(body_parts)

            if not any([title, body, notes]):
                # Slide contains no extractable text (image/diagram only).
                chunks.append(_make_chunk(
                    file_path=str(self._pptx_path.resolve()),
                    slide_index=slide_idx,
                    slide_position=slide_idx + 1,
                    title="[image-only slide]",
                    body="",
                    notes="",
                    source_format="pptx",
                    image_only=True,
                ))
                continue

            chunks.append(_make_chunk(
                file_path=str(self._pptx_path.resolve()),
                slide_index=slide_idx,
                slide_position=slide_idx + 1,
                title=title,
                body=body,
                notes=notes,
                source_format="pptx",
            ))

        return chunks


# ---------------------------------------------------------------------------
# Mtime manifest  (identical pattern to code-search project)
# ---------------------------------------------------------------------------

class MtimeManifest:
    """Persist file modification times to detect changed presentations.

    Stores a JSON mapping of ``absolute_path -> mtime_float`` next to the
    Qdrant storage directory.

    :param manifest_path: Path to the JSON manifest file.
    """

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._data: dict[str, float] = {}
        if manifest_path.exists():
            try:
                self._data = json.loads(manifest_path.read_text())
            except Exception as exc:
                log.warn(f"Could not read manifest '{manifest_path}': {exc} — starting fresh.")

    # ------------------------------------------------------------------
    def is_stale(self, path: Path) -> bool:
        """Return ``True`` if *path* is new or has been modified since last index.

        :param path: Absolute path to the file to check.
        :return: ``True`` if the file should be (re-)indexed.
        """
        key   = str(path.resolve())
        mtime = path.stat().st_mtime
        return self._data.get(key) != mtime

    # ------------------------------------------------------------------
    def mark(self, path: Path) -> None:
        """Record the current mtime for *path*.

        :param path: Absolute path whose mtime should be saved.
        """
        self._data[str(path.resolve())] = path.stat().st_mtime

    # ------------------------------------------------------------------
    def remove(self, path: Path) -> None:
        """Remove *path* from the manifest (called when a file is deleted).

        :param path: Absolute path to remove.
        """
        self._data.pop(str(path.resolve()), None)

    # ------------------------------------------------------------------
    def known_paths(self) -> set[str]:
        """Return all absolute path strings currently tracked in the manifest.

        :return: Set of path strings.
        """
        return set(self._data.keys())

    # ------------------------------------------------------------------
    def save(self) -> None:
        """Flush the manifest to disk.

        Creates parent directories if they do not exist.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))


# ---------------------------------------------------------------------------
# Main indexer
# ---------------------------------------------------------------------------

class PresentationIndexer:
    """Orchestrate discovery, extraction, embedding, and Qdrant upsert.

    Walks the supplied input paths (files or directories), extracts slide
    chunks from each supported presentation file, embeds the text with
    ``nomic-embed-text``, and upserts into a local Qdrant collection.

    :param index_dir:   Directory for Qdrant storage and the mtime manifest.
    :param collection:  Qdrant collection name.
    :param batch_size:  Number of chunks to embed and upsert per batch.
    :param force:       If ``True``, ignore the mtime manifest and re-index all.
    :param recursive:   If ``True`` (default), descend into subdirectories.
    :param verbose:     If ``True``, log each file path as it is processed.
    """

    def __init__(
        self,
        index_dir:   Path  = DEFAULT_INDEX_DIR,
        collection:  str   = COLLECTION_NAME,
        batch_size:  int   = 32,
        force:       bool  = False,
        recursive:   bool  = True,
        verbose:     bool  = False,
        ollama_url:  str   = DEFAULT_OLLAMA_URL,
    ) -> None:
        self._index_dir  = index_dir.expanduser().resolve()
        self._collection = collection
        self._batch_size = batch_size
        self._force      = force
        self._recursive  = recursive
        self._verbose    = verbose
        self._embed_url  = f"{ollama_url.rstrip('/')}/api/embed"

        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._client   = QdrantClient(path=str(self._index_dir / "qdrant"))
        self._manifest = MtimeManifest(self._index_dir / "manifest.json")

        self._ensure_collection()

    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            log.info(f"Created Qdrant collection '{self._collection}'.")

    # ------------------------------------------------------------------
    def index_inputs(self, inputs: list[Path]) -> None:
        """Index all presentations reachable from *inputs*.

        Iterates over discovered files, skips unchanged ones (unless
        ``--force``), extracts slide chunks, embeds them in batches, and
        upserts into Qdrant.  After indexing, prunes any files that are
        tracked in the manifest but no longer exist on disk.

        :param inputs: List of file or directory paths supplied on the CLI.
        """
        all_files   = list(self._discover(inputs))
        current_set = {str(p.resolve()) for p in all_files}

        total_files   = 0
        total_slides  = 0
        total_skipped = 0

        pending_points: list[PointStruct] = []

        for pres_path in all_files:
            abs_str = str(pres_path.resolve())

            if not self._force and not self._manifest.is_stale(pres_path):
                total_skipped += 1
                continue

            if self._verbose:
                log.info(f"Indexing: {pres_path}")

            try:
                chunks = self._extract(pres_path)
            except Exception as exc:
                log.warn(f"Skipping '{pres_path}': {exc}")
                continue

            if not chunks:
                log.warn(f"No text extracted from '{pres_path}' — skipping.")
                continue

            # Delete any previously indexed points for this file before
            # upserting the fresh set, so stale slide points are replaced.
            self._delete_file_points(abs_str)

            for chunk in chunks:
                vec = self._embed(chunk["text"])
                point_id = self._stable_id(abs_str, chunk["slide_index"])
                pending_points.append(
                    PointStruct(id=point_id, vector=vec, payload=chunk)
                )

            if len(pending_points) >= self._batch_size:
                self._flush(pending_points)
                pending_points = []

            self._manifest.mark(pres_path)
            total_files  += 1
            total_slides += len(chunks)

        # Flush remainder
        if pending_points:
            self._flush(pending_points)

        # Prune deleted files
        pruned = self._prune(current_set)

        self._manifest.save()
        log.info(
            f"Done. Indexed {total_files} file(s), {total_slides} slide(s). "
            f"Skipped {total_skipped} unchanged. Pruned {pruned} deleted."
        )

    # ------------------------------------------------------------------
    def _discover(self, inputs: list[Path]) -> Generator[Path, None, None]:
        """Yield all supported presentation files reachable from *inputs*.

        Handles a mix of individual files and directories.  When *recursive*
        is ``True`` (the default), descends into all subdirectories.

        :param inputs: List of file or directory paths.
        :return: Generator of ``Path`` objects for each presentation file found.
        """
        for inp in inputs:
            if inp.is_file():
                if inp.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield inp
                else:
                    log.warn(f"'{inp}' is not a supported format — skipping.")
            elif inp.is_dir():
                if self._recursive:
                    for suffix in SUPPORTED_SUFFIXES:
                        yield from sorted(inp.rglob(f"*{suffix}"))
                else:
                    for suffix in SUPPORTED_SUFFIXES:
                        yield from sorted(inp.glob(f"*{suffix}"))
            else:
                log.warn(f"'{inp}' does not exist — skipping.")

    # ------------------------------------------------------------------
    def _extract(self, pres_path: Path) -> list[dict]:
        """Dispatch extraction to the appropriate format handler.

        :param pres_path: Path to a ``.key`` or ``.pptx`` file.
        :return: List of slide chunk dicts.
        :raises ValueError: If the file suffix is unrecognised.
        """
        suffix = pres_path.suffix.lower()
        if suffix == ".key":
            return KeynoteExtractor(pres_path).extract()
        elif suffix == ".pptx":
            return PptxExtractor(pres_path).extract()
        else:
            raise ValueError(f"Unsupported format: '{suffix}'")

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        """Embed *text* via the Ollama nomic-embed-text REST API.

        Uses the ``passage:`` prefix expected by nomic-embed-text's
        asymmetric retrieval mode.

        :param text: Text to embed.
        :return: Embedding vector as a list of floats.
        :raises RuntimeError: If the Ollama API call fails.
        """
        prefixed = f"passage: {text}"
        resp = requests.post(
            self._embed_url,
            json={"model": OLLAMA_EMBED_MODEL, "input": prefixed},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama embed API returned {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        # Ollama returns {"embeddings": [[...float...]]}
        return data["embeddings"][0]

    # ------------------------------------------------------------------
    def _flush(self, points: list[PointStruct]) -> None:
        """Upsert a batch of points into Qdrant.

        :param points: List of ``PointStruct`` objects to upsert.
        """
        self._client.upsert(collection_name=self._collection, points=points)
        log.info(f"Upserted {len(points)} point(s).")

    # ------------------------------------------------------------------
    def _delete_file_points(self, abs_str: str) -> None:
        """Delete all Qdrant points previously indexed from *abs_str*.

        Uses a payload filter so only points from that file are removed.

        :param abs_str: Absolute path string used as the ``file_path`` payload key.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="file_path", match=MatchValue(value=abs_str))]
            ),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _stable_id(abs_str: str, slide_index: int) -> int:
        """Generate a stable integer point ID from file path and slide index.

        Uses a hash so IDs are deterministic across runs, enabling upsert
        semantics (re-indexing the same slide overwrites the old point).

        :param abs_str:     Absolute path string of the presentation file.
        :param slide_index: 0-based slide position.
        :return: Non-negative integer suitable for a Qdrant point ID.
        """
        import hashlib
        raw = f"{abs_str}::{slide_index}"
        return int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**53)

    # ------------------------------------------------------------------
    def _prune(self, current_set: set[str]) -> int:
        """Remove manifest entries and Qdrant points for files no longer on disk.

        :param current_set: Set of absolute path strings found in the current run.
        :return: Number of files pruned.
        """
        pruned = 0
        for known in list(self._manifest.known_paths()):
            if known not in current_set and not Path(known).exists():
                log.info(f"Pruning deleted file: {known}")
                self._delete_file_points(known)
                self._manifest.remove(Path(known))
                pruned += 1
        return pruned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    :return: Configured :class:`argparse.ArgumentParser` instance.
    """
    p = argparse.ArgumentParser(
        prog="presentation_indexer",
        description=(
            "Index Keynote (.key) and PowerPoint (.pptx) presentations "
            "into a local Qdrant vector store for semantic search."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help="One or more .key/.pptx files or directories to index.",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        metavar="DIR",
        help=f"Qdrant storage + manifest directory. Default: {DEFAULT_INDEX_DIR}",
    )
    p.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        metavar="NAME",
        help=f"Qdrant collection name. Default: {COLLECTION_NAME}",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Embedding batch size. Default: 32",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore mtime manifest and re-index all files.",
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Limit directory search to top level only (no subdirectory descent).",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"Ollama base URL. Default: {DEFAULT_OLLAMA_URL}",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log each file path as it is processed.",
    )
    return p


def main() -> None:
    """Entry point for the presentation indexer CLI."""
    parser = _build_parser()
    args   = parser.parse_args()

    # Validate inputs; warn on missing paths but continue with the rest
    inputs: list[Path] = []
    for item in args.inputs:
        p = Path(item).expanduser()
        if not p.exists():
            log.warn(f"'{item}' does not exist — skipping.")
            continue
        inputs.append(p)

    if not inputs:
        parser.error("No valid inputs found.")

    indexer = PresentationIndexer(
        index_dir=args.index_dir.expanduser().resolve(),
        collection=args.collection,
        batch_size=args.batch_size,
        force=args.force,
        recursive=not args.no_recursive,
        verbose=args.verbose,
        ollama_url=args.ollama_url,
    )
    indexer.index_inputs(inputs)


if __name__ == "__main__":
    main()
    