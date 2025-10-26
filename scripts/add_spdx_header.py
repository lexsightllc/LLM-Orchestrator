#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Utility to ensure files carry an SPDX license identifier."""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

SPDX_LINE = "SPDX-License-Identifier: MPL-2.0"

COMMENT_STYLES = {
    ".py": "#",
    ".pyi": "#",
    ".pyx": "#",
    ".pxd": "#",
    ".sh": "#",
    ".bash": "#",
    ".zsh": "#",
    ".ps1": "#",
    ".psm1": "#",
    ".psd1": "#",
    ".toml": "#",
    ".cfg": "#",
    ".ini": "#",
    ".in": "#",
    ".txt": "#",
    ".md": "<!-- -->",
    ".rst": "..",
    ".yml": "#",
    ".yaml": "#",
    ".mk": "#",
    ".make": "#",
    ".mkd": "<!-- -->",
    ".mkdn": "<!-- -->",
    ".cjs": "//",
    ".js": "//",
    ".json": None,
    ".lock": None,
    ".gitignore": "#",
    ".gitattributes": "#",
    ".editorconfig": ";",
    ".env": "#",
    ".example": "#",
    ".service": "#",
    ".conf": "#",
    ".ps": "#",
    ".bat": "REM",
    ".cfg": "#",
}

SPECIAL_FILENAMES = {
    "Makefile": "#",
    "Dockerfile": "#",
    "MANIFEST.in": "#",
    "NOTICE": None,
    "LICENSE": None,
    "THIRD_PARTY_NOTICES": None,
}

HTML_STYLE = "<!-- -->"
RST_STYLE = ".."

SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".lock", ".json"}


def detect_comment_style(path: pathlib.Path) -> str | None:
    if path.name in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[path.name]
    if path.suffix == "" and path.name.startswith("."):
        # dotfiles without suffix
        if path.name in {".gitignore", ".gitattributes", ".gitmodules"}:
            return "#"
        if path.name in {".env", ".env.example"}:
            return "#"
        if path.name == ".gitmessage":
            return "#"
        return "#"
    style = COMMENT_STYLES.get(path.suffix)
    if style is None and path.suffix:
        if path.suffix.lower() in SKIP_EXTENSIONS:
            return None
    if style:
        return style
    if path.suffix:
        return COMMENT_STYLES.get(path.suffix.lower())
    return "#"


def spdx_header(style: str) -> str:
    if style == "#":
        return f"# {SPDX_LINE}\n"
    if style == "//":
        return f"// {SPDX_LINE}\n"
    if style == "REM":
        return f"REM {SPDX_LINE}\n"
    if style == HTML_STYLE:
        return f"<!-- {SPDX_LINE} -->\n"
    if style == RST_STYLE:
        return f".. {SPDX_LINE}\n\n"
    if style == ";":
        return f"; {SPDX_LINE}\n"
    # fallback to hash
    return f"# {SPDX_LINE}\n"


def add_header(path: pathlib.Path) -> bool:
    style = detect_comment_style(path)
    if style is None:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    header = spdx_header(style)
    header_stripped = header.strip()
    lines = text.splitlines(keepends=True)
    target_index = 1 if lines and lines[0].startswith("#!") else 0
    indices = [i for i, line in enumerate(lines) if line.strip() == header_stripped]
    changed = False
    if indices:
        for idx in reversed(indices[1:]):
            del lines[idx]
            changed = True
        first_idx = indices[0]
        if first_idx != target_index:
            line = lines.pop(first_idx)
            if not line.endswith("\n"):
                line = line + "\n"
            lines.insert(target_index, line)
            changed = True
        if changed:
            path.write_text("".join(lines), encoding="utf-8")
        return changed
    if target_index == 1:
        lines.insert(1, header)
        new_text = "".join(lines)
    else:
        new_text = header + text
    path.write_text(new_text, encoding="utf-8")
    return True


def iter_files(paths: Iterable[str]) -> Iterable[pathlib.Path]:
    if not paths:
        yield from (pathlib.Path(p) for p in sys.stdin.read().split())
    for p in paths:
        path = pathlib.Path(p)
        if path.is_dir():
            for sub in path.rglob("*"):
                if ".git" in sub.parts:
                    continue
                if sub.is_file():
                    yield sub
        elif path.is_file():
            yield path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Insert SPDX headers where missing.")
    parser.add_argument("paths", nargs="*", help="Files or directories to process")
    args = parser.parse_args(argv)
    changed = False
    for path in iter_files(args.paths):
        if add_header(path):
            changed = True
    return 0 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
