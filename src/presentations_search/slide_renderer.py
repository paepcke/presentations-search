#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-30 17:43:10
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-30 18:05:49
"""
slide_renderer.py  --  Convert presentation slides to PNG and display them.

Uses LibreOffice headless to convert ``.key`` and ``.pptx`` files to PDF,
then ``pdf2image`` (poppler) to extract individual slide pages as PNG images,
and ``xdg-open`` to display them in the system image viewer.

Converted PDFs and extracted PNGs are cached under the presentation index
directory to avoid re-running LibreOffice on subsequent requests.  Cache
entries are invalidated when the source file's mtime changes.

Dependencies
------------
    LibreOffice (system): sudo apt install libreoffice
    pdf2image (Python):   pip install pdf2image
    poppler (system):     sudo apt install poppler-utils

Usage (standalone)
------------------
    python slide_renderer.py /path/to/deck.pptx 5
    python slide_renderer.py /path/to/deck.key 3 7 12
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR = Path("~/.presentation_index")
DPI               = 150    # PNG render resolution — good quality, reasonable size
LIBREOFFICE       = "libreoffice"


# ---------------------------------------------------------------------------
# SlideRenderer
# ---------------------------------------------------------------------------

class SlideRenderer:
    """Convert presentation slides to PNG and open them in the system viewer.

    Maintains two disk caches under *index_dir*:

    * ``pdf_cache/``  — one PDF per source file (LibreOffice output)
    * ``png_cache/``  — one PNG per (file, slide_position) pair

    Cache entries are keyed by a SHA-256 hash of the source file's absolute
    path.  A JSON manifest tracks source file mtimes so stale entries are
    automatically re-generated.

    :param index_dir: Root directory of the presentation index.  Cache
        subdirectories are created here automatically.
    """

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
        self._index_dir  = index_dir.expanduser().resolve()
        self._pdf_dir    = self._index_dir / "pdf_cache"
        self._png_dir    = self._index_dir / "png_cache"
        self._mtime_file = self._index_dir / "render_manifest.json"

        self._pdf_dir.mkdir(parents=True, exist_ok=True)
        self._png_dir.mkdir(parents=True, exist_ok=True)

        self._mtime_map: dict[str, float] = {}
        if self._mtime_file.exists():
            try:
                self._mtime_map = json.loads(self._mtime_file.read_text())
            except Exception:
                pass

    # ------------------------------------------------------------------
    def show(self, file_path: str, slide_positions: list[int]) -> list[int]:
        """Render and display one or more slides from *file_path*.

        Converts the source file to PDF (if not already cached), extracts
        each requested slide as a PNG, and opens each PNG in the system
        image viewer via ``xdg-open``.

        Progress messages are printed to stdout so the user knows what is
        happening during the (potentially slow) first conversion.

        :param file_path:       Absolute path to the ``.key`` or ``.pptx`` file.
        :param slide_positions: List of 1-based slide positions to display.
        :return: List of slide positions that were successfully opened.
        """
        src = Path(file_path).resolve()
        if not src.exists():
            log.warn(f"Source file not found: {src}")
            print(f"  File not found: {src}")
            return []

        pdf = self._ensure_pdf(src)
        if pdf is None:
            return []

        opened = []
        for pos in slide_positions:
            png = self._ensure_png(src, pdf, pos)
            if png is None:
                continue
            print(f"  Opening slide {pos} …")
            try:
                subprocess.Popen(
                    ["xdg-open", str(png)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                opened.append(pos)
            except Exception as exc:
                log.warn(f"Could not open '{png}': {exc}")
                print(f"  Could not open viewer: {exc}")
                print(f"  PNG saved at: {png}")

        self._save_manifest()
        return opened

    # ------------------------------------------------------------------
    def _ensure_pdf(self, src: Path) -> Path | None:
        """Return a cached PDF for *src*, converting if necessary.

        :param src: Resolved path to the source presentation file.
        :return: Path to the PDF file, or ``None`` on conversion failure.
        """
        key     = self._file_key(src)
        pdf     = self._pdf_dir / f"{key}.pdf"
        current = src.stat().st_mtime

        if pdf.exists() and self._mtime_map.get(str(src)) == current:
            return pdf   # cache hit — no conversion needed

        # Cache miss or stale — (re-)convert
        fname = src.name
        print(f"  Converting '{fname}' to PDF (first time — please wait) …")
        result = subprocess.run(
            [
                LIBREOFFICE, "--headless",
                "--convert-to", "pdf",
                str(src),
                "--outdir", str(self._pdf_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.warn(f"LibreOffice failed on '{src}': {result.stderr.strip()}")
            print(f"  Conversion failed: {result.stderr.strip()[:120]}")
            return None

        # LibreOffice names the output after the source stem
        lo_output = self._pdf_dir / (src.stem + ".pdf")
        if not lo_output.exists():
            log.warn(f"Expected PDF not found: {lo_output}")
            print(f"  Conversion produced no output.")
            return None

        # Rename to our cache key so multiple files with the same stem
        # don't collide (e.g. Lec3 - STAT60.pptx vs Lec3 - STAT60.key)
        if lo_output != pdf:
            lo_output.rename(pdf)

        # Invalidate any cached PNGs for this file since the PDF changed
        self._evict_pngs(key)
        self._mtime_map[str(src)] = current
        return pdf

    # ------------------------------------------------------------------
    def _ensure_png(self, src: Path, pdf: Path, slide_pos: int) -> Path | None:
        """Return a cached PNG for slide *slide_pos* of *pdf*.

        :param src:       Source presentation file (used for cache key).
        :param pdf:       Cached PDF for the source file.
        :param slide_pos: 1-based slide position.
        :return: Path to the PNG file, or ``None`` on failure.
        """
        key = self._file_key(src)
        png = self._png_dir / f"{key}_slide{slide_pos:04d}.png"

        if png.exists():
            return png   # cache hit

        try:
            from pdf2image import convert_from_path
        except ImportError:
            print("  pdf2image not installed. Run: pip install pdf2image")
            return None

        print(f"  Rendering slide {slide_pos} …")
        try:
            pages = convert_from_path(
                str(pdf),
                dpi=DPI,
                first_page=slide_pos,
                last_page=slide_pos,
            )
        except Exception as exc:
            log.warn(f"pdf2image failed for slide {slide_pos} of '{pdf}': {exc}")
            print(f"  Render failed: {exc}")
            return None

        if not pages:
            print(f"  Slide {slide_pos} not found in PDF "
                  f"(presentation may have fewer slides).")
            return None

        pages[0].save(str(png), format="PNG")
        return png

    # ------------------------------------------------------------------
    def _evict_pngs(self, key: str) -> None:
        """Delete all cached PNGs for the given file key.

        Called when the source file has changed and its PDF was re-generated,
        making any previously cached per-slide PNGs stale.

        :param key: SHA-256 hex key for the source file.
        """
        for stale in self._png_dir.glob(f"{key}_slide*.png"):
            try:
                stale.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _save_manifest(self) -> None:
        """Persist the mtime manifest to disk."""
        try:
            self._mtime_file.write_text(json.dumps(self._mtime_map, indent=2))
        except Exception as exc:
            log.warn(f"Could not save render manifest: {exc}")

    # ------------------------------------------------------------------
    @staticmethod
    def _file_key(path: Path) -> str:
        """Return a short SHA-256 hex digest of *path*'s absolute string.

        :param path: Resolved absolute path.
        :return: 16-character hex string suitable for use in filenames.
        """
        return hashlib.sha256(str(path).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CLI (standalone usage)
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for standalone slide display."""
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <presentation_file> <slide_num> [slide_num ...]")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        positions = [int(a) for a in sys.argv[2:]]
    except ValueError:
        print("Slide numbers must be integers.")
        sys.exit(1)

    renderer = SlideRenderer()
    renderer.show(file_path, positions)


if __name__ == "__main__":
    main()
