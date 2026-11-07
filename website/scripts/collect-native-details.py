#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read native configuration declarations without importing or building a model.

This intentionally is not a C++ parser or a capability detector. Simple member
declarations become reference rows; the entire source declaration and any
unparsed members remain available so a more complex declaration is never lost.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys


_RECORD = re.compile(
    r"\b(?P<kind>struct|class)\s+(?P<name>[A-Za-z_]\w*(?:Config|Options)\w*|Config|Options)"
    r"\s*(?:final\s*)?(?::[^;{}]+)?\{"
)
_FIELD = re.compile(
    r"(?P<type>[A-Za-z_][\w:\s<>,*&]*?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?P<array>\[[^\]]*\])?\s*"
    r"(?P<initializer>[={][\s\S]*)?;\s*\Z"
)
_COMMENT = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_NATIVE_DESCRIPTION = (
    "Native source declaration. These are family runtime implementation fields, "
    "not a list of CLI-adjustable options or verified checkpoint capabilities. "
    "Defaults shown here are C++ member initializers; bundle parsing and request "
    "validation may override or constrain them."
)


def _mask_cpp(source: str, *, strings: bool = True, comments: bool = True) -> str:
    """Mask comments and optionally literals while preserving character offsets."""
    output = list(source)
    index = 0
    while index < len(source):
        end = None
        comment = False
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end < 0:
                end = len(source)
            comment = True
        elif source.startswith("/*", index):
            closing = source.find("*/", index + 2)
            end = len(source) if closing < 0 else closing + 2
            comment = True
        elif source.startswith('R"', index):
            opening = source.find("(", index + 2, index + 19)
            if opening >= 0:
                delimiter = source[index + 2 : opening]
                terminator = ")" + delimiter + '"'
                closing = source.find(terminator, opening + 1)
                end = len(source) if closing < 0 else closing + len(terminator)
        elif source[index] in ('"', "'"):
            # C++ digit separators are not character literals.
            if source[index] == "'" and index > 0 and source[index - 1].isdigit():
                index += 1
                continue
            quote = source[index]
            end = index + 1
            while end < len(source):
                if source[end] == "\\":
                    end += 2
                elif source[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
        if end is None:
            index += 1
            continue
        if (comment and comments) or (not comment and strings):
            for cursor in range(index, min(end, len(source))):
                if source[cursor] != "\n":
                    output[cursor] = " "
        index = end
    return "".join(output)


def _closing_brace(masked: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _members(source: str, masked: str, start: int, end: int):
    """Yield top-level member spans, keeping method bodies separate from fields."""
    cursor = start
    index = start
    while index < end:
        if masked[index] == "{":
            closing = _closing_brace(masked, index)
            if closing is None or closing > end:
                break
            prefix = masked[cursor:index]
            # Inline methods do not need a trailing semicolon. An initializer
            # with a call or lambda has '=' before its body and remains a field.
            if ")" in prefix and "=" not in prefix:
                yield cursor, closing + 1
                cursor = closing + 1
            index = closing + 1
            continue
        if masked[index] == ";":
            yield cursor, index + 1
            cursor = index + 1
        index += 1
    if masked[cursor:end].strip():
        yield cursor, end


def _comment_text(source: str) -> str:
    comments = []
    for match in _COMMENT.finditer(_mask_cpp(source, comments=False)):
        body = match.group().removeprefix("//").removeprefix("/*").removesuffix("*/")
        comments.extend(line.strip().lstrip("*/ ").strip() for line in body.splitlines())
    return " ".join(line for line in comments if line)


def _has_outer_comma(source: str) -> bool:
    depth = 0
    for character in source:
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif character == "," and depth == 0:
            return True
    return False


def _source_ref(repo_root: Path, path: Path, source: str, offset: int) -> dict:
    return {"path": path.relative_to(repo_root).as_posix(), "line": source.count("\n", 0, offset) + 1}


def _parse_records(repo_root: Path, path: Path) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    masked = _mask_cpp(source)
    without_comments = _mask_cpp(source, strings=False)
    records = []
    for match in _RECORD.finditer(masked):
        if re.search(r"\benum\s*$", masked[:match.start()]):
            continue
        opening = match.end() - 1
        closing = _closing_brace(masked, opening)
        if closing is None:
            raise ValueError(f"Unterminated native config declaration: {path}:{match['name']}")
        fields = []
        unparsed = []
        for begin, end in _members(source, masked, opening + 1, closing):
            code = without_comments[begin:end]
            code = re.sub(r"^\s*(?:public|protected|private)\s*:\s*", "", code)
            code = code.strip()
            if not code or code == ";":
                continue
            declaration_token = re.search(r"\S", masked[begin:end])
            offset = begin + (declaration_token.start() if declaration_token else 0)
            unparsed_member = {
                "declaration": source[begin:end].strip(),
                "source": _source_ref(repo_root, path, source, offset),
            }
            # Declarations other than data members remain in the full record.
            prefix = code.split("=", 1)[0].split("{", 1)[0]
            if code.startswith(("using ", "typedef ", "static_assert", "friend ", "enum ", "struct ", "class ")):
                unparsed.append(unparsed_member)
                continue
            if "(" in prefix or ")" in prefix:
                unparsed.append(unparsed_member)
                continue
            field = _FIELD.fullmatch(code)
            if field is None or _has_outer_comma(field["type"]):
                unparsed.append(unparsed_member)
                continue
            initializer = field["initializer"]
            default = None
            if initializer:
                if initializer.startswith("{"):
                    initializer_end = _closing_brace(_mask_cpp(initializer), 0)
                    if initializer_end is None or initializer[initializer_end + 1:].strip():
                        unparsed.append(unparsed_member)
                        continue
                default = initializer[1:].strip()
                if initializer.startswith("{"):
                    default = default[:-1].strip()
                    if not default:
                        default = "{}"
            field_type = " ".join(field["type"].split()) + (field["array"] or "")
            name_position = re.search(r"\b" + re.escape(field["name"]) + r"\b", masked[begin:end])
            offset = begin + name_position.start()
            description_start = begin
            if source[source.rfind("\n", 0, begin) + 1 : begin].strip():
                following_line = source.find("\n", begin, offset)
                if following_line >= 0:
                    description_start = following_line + 1
            description = _comment_text(source[description_start:end])
            # Include an adjacent same-line comment attached to this member.
            line_end = source.find("\n", end)
            trailing = source[end : len(source) if line_end < 0 else line_end]
            if trailing.lstrip().startswith("//"):
                description = " ".join(filter(None, [description, _comment_text(trailing)]))
            fields.append({
                "name": field["name"],
                "type": field_type,
                "required": False,
                "default": default,
                "description": description or "Native member declaration; no CLI option is implied.",
                "readOnly": bool(re.search(r"\b(?:const|constexpr)\b", field_type)),
                "source": _source_ref(repo_root, path, source, offset),
            })
        declaration_end = closing + 1
        if source[declaration_end : declaration_end + 1] == ";":
            declaration_end += 1
        records.append({
            "name": match["name"],
            "kind": "runtime",
            "description": _NATIVE_DESCRIPTION,
            "source": _source_ref(repo_root, path, source, match.start()),
            "fields": fields,
            "declaration": source[match.start() : declaration_end],
            "unparsedDeclarations": unparsed,
        })
    return records


def collect_native_details(repo_root: Path, family_root: Path) -> list[dict]:
    """Collect family-owned Config/Options records, with precise source links."""
    repo_root, family_root = Path(repo_root).resolve(), Path(family_root).resolve()
    runtime = family_root / "runtime"
    if not runtime.is_dir():
        return []
    records = []
    for path in sorted(runtime.rglob("*")):
        if path.suffix not in {".h", ".hpp", ".cpp", ".cc", ".cxx"} or not path.is_file():
            continue
        if any(part in {"tests", "third_party", "vendor"} for part in path.relative_to(runtime).parts):
            continue
        records.extend(_parse_records(repo_root, path))
    return records


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: collect-native-details.py REPO_ROOT FAMILY_ROOT")
    print(json.dumps(collect_native_details(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
