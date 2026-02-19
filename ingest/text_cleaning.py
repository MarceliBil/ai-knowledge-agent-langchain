from __future__ import annotations

import re


_MANY_NEWLINES = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[\t\u00A0 ]{2,}")

_DEHYPHENATE = re.compile(r"(?i)(?<=\w)-\n(?=\w)")

_BULLET_LINE = re.compile(r"^[-•*‣–—]\s+\S")
_NUMBERED_LINE = re.compile(r"^\d{1,3}(?:\.\d{1,3})*[\.)]\s+\S")


def _looks_like_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) < 4 and not line.endswith(":"):
        return False
    if len(line) > 100:
        return False
    if line.endswith(":"):
        return True
    if line[0].isupper() and not line.endswith((".", "?", "!")) and len(line) <= 80:
        if line.count(",") <= 1:
            return True
    return False


def normalize_extracted_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")

    text = _DEHYPHENATE.sub("", text)

    text = _MANY_NEWLINES.sub("\n\n", text)

    split_lines = text.split("\n")
    raw_lines: list[str] = []
    whitespace_only: list[bool] = []
    for ln in split_lines:
        stripped = ln.strip()
        raw_lines.append(stripped)
        whitespace_only.append((ln != "") and (stripped == ""))

    def can_append_suffix(prev: str, suffix: str) -> bool:
        if not prev or not suffix:
            return False
        if not prev[-1].isalpha():
            return False
        if prev.endswith((".", "?", "!", ":")):
            return False
        if _BULLET_LINE.match(prev):
            return False
        return True

    def is_single_letter_alpha(line: str) -> bool:
        return len(line) == 1 and line.isalpha()

    def is_short_alpha_suffix(line: str) -> bool:
        return 2 <= len(line) <= 4 and line.isalpha() and line.islower()

    lines: list[str] = []
    i = 0
    while i < len(raw_lines):
        ln = raw_lines[i]

        if ln == "":
            if whitespace_only[i]:
                lines.append("")
            else:
                lines.append("")
            i += 1
            continue

        if is_single_letter_alpha(ln):
            letters: list[str] = [ln]
            j = i + 1
            while j < len(raw_lines):
                if raw_lines[j] == "":
                    if whitespace_only[j]:
                        break
                    j += 1
                    continue
                if is_single_letter_alpha(raw_lines[j]):
                    letters.append(raw_lines[j])
                    j += 1
                    continue
                break

            word = "".join(letters)

            attach_idx = None
            if len(word) <= 2 and word.islower() and lines:
                k = len(lines) - 1
                while k >= 0 and lines[k] == "":
                    k -= 1
                if k >= 0 and " " not in lines[k] and can_append_suffix(lines[k], word):
                    attach_idx = k

            if attach_idx is not None:
                while len(lines) - 1 > attach_idx and lines[-1] == "":
                    lines.pop()
                lines[attach_idx] = f"{lines[attach_idx]}{word}"
            else:
                lines.append(word)

            i = j
            continue

        if is_short_alpha_suffix(ln) and lines and lines[-1] != "" and can_append_suffix(lines[-1], ln) and lines[-1][-1].islower():
            lines[-1] = f"{lines[-1]}{ln}"
            i += 1
            continue

        lines.append(ln)
        i += 1

    out_lines: list[str] = []
    paragraph: list[str] = []

    def is_structural(line: str) -> bool:
        return bool(_BULLET_LINE.match(line) or _NUMBERED_LINE.match(line) or _looks_like_heading(line))

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        merged = " ".join(paragraph).strip()
        merged = _MANY_SPACES.sub(" ", merged)
        out_lines.append(merged)
        paragraph = []

    def next_non_empty(start: int) -> str:
        for k in range(start, len(lines)):
            if lines[k] != "":
                return lines[k]
        return ""

    for idx, ln in enumerate(lines):
        if ln == "":
            prev = paragraph[-1] if paragraph else (
                out_lines[-1] if out_lines else "")
            nxt = next_non_empty(idx + 1)

            hard_break = False
            if not prev:
                hard_break = False
            elif prev.endswith((".", "?", "!", ":")):
                hard_break = True
            elif is_structural(prev) or is_structural(nxt):
                hard_break = True
            elif len(prev) >= 80:
                hard_break = True

            if hard_break:
                flush_paragraph()
                if out_lines and out_lines[-1] != "":
                    out_lines.append("")
            continue

        ln = _MANY_SPACES.sub(" ", ln)
        if is_structural(ln):
            flush_paragraph()
            out_lines.append(ln)
        else:
            paragraph.append(ln)

    flush_paragraph()

    text = "\n".join(out_lines)
    text = _MANY_NEWLINES.sub("\n\n", text)
    text = _MANY_SPACES.sub(" ", text)

    return text.strip()
