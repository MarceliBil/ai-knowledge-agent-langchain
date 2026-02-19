from __future__ import annotations

import re


_JOIN_IN_WORD_NEWLINES = re.compile(r"(?<=\S)\n(?=\S)")
_TRIM_AROUND_NEWLINES = re.compile(r"[ \t]*\n[ \t]*")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_extracted_text(text: str) -> str:
    """Normalize text extracted from PDFs.

    Some PDF extractors produce output with newlines between letters
    (e.g. "Podró\nż\ne"), which can break retrieval/judging.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _JOIN_IN_WORD_NEWLINES.sub("", text)
    text = _TRIM_AROUND_NEWLINES.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)

    return text.strip()
