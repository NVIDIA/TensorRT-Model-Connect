# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check and install the repository's approved SPDX source headers.

The checker deliberately operates on ``git ls-files -z`` rather than a
filesystem walk.  This keeps generated build products out of the audit and
makes filenames containing whitespace (or other non-newline characters) safe.

``--fix`` only changes a managed header preamble.  It does not re-encode files,
normalize line endings, or change file modes.  An audit record proves that the
bytes below the managed preamble are identical before and after each rewrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


MANAGED_YEAR = 2026
COPYRIGHT_TEXT = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved."
)
LICENSE_TEXT = "SPDX-License-Identifier: Apache-2.0"


@dataclass(frozen=True)
class HeaderStyle:
    """One exact, Confluence-approved comment form."""

    name: str
    lines: tuple[str, ...]
    preserves_python_preamble: bool = False

    def render(self, newline: bytes) -> bytes:
        return newline.join(line.encode("ascii") for line in self.lines)


HASH_STYLE = HeaderStyle(
    name="hash",
    lines=(f"# {COPYRIGHT_TEXT}", f"# {LICENSE_TEXT}"),
    preserves_python_preamble=True,
)
NATIVE_STYLE = HeaderStyle(
    name="native-block",
    lines=("/*", f" * {COPYRIGHT_TEXT}", f" * {LICENSE_TEXT}", " */"),
)

# This table is intentionally explicit.  A new source-language suffix must be
# reviewed and added here rather than silently receiving an unsuitable comment
# form.  Template suffixes (for example ``foo.cpp.in``) are peeled before this
# table is consulted.
STYLE_BY_SUFFIX: Mapping[str, HeaderStyle] = {
    # Python and Python-adjacent sources.
    ".py": HASH_STYLE,
    ".pyi": HASH_STYLE,
    ".pyx": HASH_STYLE,
    ".pxd": HASH_STYLE,
    # Shell and build/configuration sources.
    ".sh": HASH_STYLE,
    ".bash": HASH_STYLE,
    ".zsh": HASH_STYLE,
    ".cmake": HASH_STYLE,
    ".mk": HASH_STYLE,
    ".mak": HASH_STYLE,
    ".yaml": HASH_STYLE,
    ".yml": HASH_STYLE,
    ".toml": HASH_STYLE,
    # C, C++, CUDA, and compatible native formats.
    ".c": NATIVE_STYLE,
    ".cc": NATIVE_STYLE,
    ".cpp": NATIVE_STYLE,
    ".cxx": NATIVE_STYLE,
    ".h": NATIVE_STYLE,
    ".hh": NATIVE_STYLE,
    ".hpp": NATIVE_STYLE,
    ".hxx": NATIVE_STYLE,
    ".cu": NATIVE_STYLE,
    ".cuh": NATIVE_STYLE,
    ".map": NATIVE_STYLE,
    ".proto": NATIVE_STYLE,
    # Web sources whose grammars accept C-style block comments.
    ".js": NATIVE_STYLE,
    ".jsx": NATIVE_STYLE,
    ".mjs": NATIVE_STYLE,
    ".cjs": NATIVE_STYLE,
    ".ts": NATIVE_STYLE,
    ".tsx": NATIVE_STYLE,
    ".css": NATIVE_STYLE,
}

TEMPLATE_SUFFIXES = frozenset({".in", ".j2", ".jinja", ".tmpl", ".template"})

HASH_SPECIAL_NAMES = frozenset(
    {
        ".clang-format",
        ".clang-tidy",
        ".dockerignore",
        ".gitattributes",
        ".gitignore",
        "CODEOWNERS",
        "CMakeLists.txt",
        "Dockerfile",
        "GNUmakefile",
        "Makefile",
    }
)

# These formats are deliberately outside the source-header migration.  Keeping
# the list explicit means a previously unseen tracked suffix fails closed as
# ``unclassified`` instead of being silently skipped.
NON_SOURCE_SUFFIXES = frozenset(
    {
        ".a3m",
        ".bin",
        ".csv",
        ".dat",
        ".diff",
        ".engine",
        ".f32",
        ".flac",
        ".gif",
        ".htm",
        ".html",
        ".ico",
        ".jpeg",
        ".jpg",
        ".json",
        ".jsonl",
        ".lock",
        ".md",
        ".model",
        ".mp3",
        ".npy",
        ".npz",
        ".onnx",
        ".patch",
        ".pdf",
        ".png",
        ".rst",
        ".safetensors",
        ".svg",
        ".bundle",
        ".tsv",
        ".txt",
        ".wav",
        ".webp",
        ".xml",
    }
)
NON_SOURCE_NAMES = frozenset(
    {
        ".last_scan_sha",
        "AUTHORS",
        "COPYING",
        "LICENSE",
        "NOTICE",
        "README",
    }
)

_CODING_COOKIE_RE = re.compile(rb"^[ \t\f]*\#.*?coding[:=][ \t]*[-_.a-zA-Z0-9]+")
_PLACEHOLDER_RE = re.compile(rb"(?:<|\[)(?:year|yyyy)(?:>|\])", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHEBANG_RE = re.compile(
    rb"^#![^\r\n]*(?:python(?:[0-9.]*)?|(?:ba|z|k|da)?sh)(?:[ \t\r\n]|$)",
    re.IGNORECASE,
)


class ManifestError(ValueError):
    """Raised when the exception manifest itself is not auditable."""


class UnsafeFixError(ValueError):
    """Raised when an SPDX fragment is too far into a file to move safely."""


@dataclass(frozen=True)
class HeaderException:
    path: str
    reason: str
    license: str
    source: str
    sha256: str


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.path}: {self.detail}"


@dataclass(frozen=True)
class BodyAudit:
    path: str
    before_sha256: str
    after_sha256: str


@dataclass
class AuditReport:
    tracked_files: int = 0
    managed_files: int = 0
    excepted_files: int = 0
    ignored_files: int = 0
    changed_files: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    body_audits: list[BodyAudit] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracked_files": self.tracked_files,
            "managed_files": self.managed_files,
            "excepted_files": self.excepted_files,
            "ignored_files": self.ignored_files,
            "changed_files": self.changed_files,
            "findings": [asdict(item) for item in self.findings],
            "body_audits": [asdict(item) for item in self.body_audits],
            "ok": self.ok,
        }


@dataclass(frozen=True)
class _Marker:
    kind: str
    text: bytes
    line_index: int
    offset: int


def _canonical_repo_path(path: str) -> str:
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != path
    ):
        raise ManifestError(f"exception path is not canonical repo-relative POSIX: {path!r}")
    return path


def load_exceptions(manifest_path: Path) -> dict[str, HeaderException]:
    """Load and strictly validate exact-path third-party exceptions."""

    try:
        raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"cannot read {manifest_path}: {exc}") from exc

    if raw.get("schema_version") != 1:
        raise ManifestError("exception manifest must declare schema_version = 1")
    entries = raw.get("exceptions", [])
    if not isinstance(entries, list):
        raise ManifestError("exception manifest must contain [[exceptions]] entries")

    required = {"path", "reason", "license", "source", "sha256"}
    result: dict[str, HeaderException] = {}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ManifestError(f"exceptions entry {index} is not a table")
        keys = set(entry)
        if keys != required:
            missing = sorted(required - keys)
            extra = sorted(keys - required)
            raise ManifestError(f"exceptions entry {index} has missing={missing} extra={extra}")
        if not all(isinstance(entry[key], str) and entry[key].strip() for key in required):
            raise ManifestError(f"exceptions entry {index} has an empty/non-string field")

        path = _canonical_repo_path(entry["path"])
        digest = entry["sha256"]
        if not _SHA256_RE.fullmatch(digest):
            raise ManifestError(f"exception {path} has an invalid lowercase SHA-256")
        if path in result:
            raise ManifestError(f"duplicate exception path: {path}")
        result[path] = HeaderException(
            path=path,
            reason=entry["reason"],
            license=entry["license"],
            source=entry["source"],
            sha256=digest,
        )
    return result


def tracked_files(repo_root: Path) -> list[str]:
    """Return tracked paths using Git's NUL-delimited representation."""

    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--cached"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise RuntimeError(f"git ls-files failed: {detail or exc}") from exc

    return [os.fsdecode(item) for item in completed.stdout.split(b"\0") if item]


def _peel_template_suffixes(name: str) -> str:
    while True:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in TEMPLATE_SUFFIXES:
            return name
        name = PurePosixPath(name).stem


def style_for_path(path: str, data: bytes | None = None) -> HeaderStyle | None:
    """Return the explicitly approved header style, if ``path`` is source."""

    name = PurePosixPath(path).name
    if name in HASH_SPECIAL_NAMES or name.startswith("Dockerfile.") or name.startswith("Makefile."):
        return HASH_STYLE

    base_name = _peel_template_suffixes(name)
    style = STYLE_BY_SUFFIX.get(PurePosixPath(base_name).suffix.lower())
    if style is not None:
        return style

    # Extensionless executable scripts remain classifiable, but only when the
    # shebang names one of the explicitly supported Python/shell interpreters.
    if not PurePosixPath(name).suffix and data is not None and _SHEBANG_RE.match(data):
        return HASH_STYLE
    return None


def _is_known_non_source(path: str) -> bool:
    name = PurePosixPath(path).name
    return name in NON_SOURCE_NAMES or PurePosixPath(name).suffix.lower() in NON_SOURCE_SUFFIXES


def _detect_newline(data: bytes) -> bytes:
    newline_at = data.find(b"\n")
    if newline_at > 0 and data[newline_at - 1 : newline_at + 1] == b"\r\n":
        return b"\r\n"
    return b"\n"


def _without_line_ending(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith((b"\n", b"\r")):
        return line[:-1]
    return line


def _header_offset(data: bytes, style: HeaderStyle) -> int:
    """Find the insertion point after BOM, shebang, and PEP 263 cookie."""

    bom_length = len(b"\xef\xbb\xbf") if data.startswith(b"\xef\xbb\xbf") else 0
    if not style.preserves_python_preamble:
        return bom_length

    payload = data[bom_length:]
    lines = payload.splitlines(keepends=True)
    if not lines:
        return bom_length

    last_reserved = -1
    if _without_line_ending(lines[0]).startswith(b"#!"):
        last_reserved = 0
    for index, line in enumerate(lines[:2]):
        if _CODING_COOKIE_RE.match(_without_line_ending(line)):
            last_reserved = max(last_reserved, index)
    return bom_length + sum(len(line) for line in lines[: last_reserved + 1])


def _marker_text(line: bytes) -> tuple[str, bytes] | None:
    content = _without_line_ending(line).lstrip()
    for prefix in (b"//", b"/*", b"#", b"*"):
        if content.startswith(prefix):
            content = content[len(prefix) :].lstrip()
            break
    else:
        return None

    if content.startswith(b"SPDX-FileCopyrightText:"):
        return "copyright", content
    if content.startswith(b"SPDX-License-Identifier:"):
        return "license", content
    return None


def _markers(data: bytes) -> list[_Marker]:
    records: list[_Marker] = []
    offset = 0
    for index, line in enumerate(data.splitlines(keepends=True)):
        marker = _marker_text(line)
        if marker is not None:
            records.append(
                _Marker(
                    kind=marker[0],
                    text=marker[1],
                    line_index=index,
                    offset=offset,
                )
            )
        offset += len(line)
    return records


def inspect_header(data: bytes, style: HeaderStyle, path: str = "<memory>") -> Finding | None:
    """Return the one primary compliance failure for ``data``, if any."""

    records = _markers(data)
    if not records:
        return Finding("missing", path, f"missing required {style.name} SPDX header")

    if any(_PLACEHOLDER_RE.search(record.text) for record in records):
        return Finding("placeholder", path, "SPDX header contains an unresolved year placeholder")

    copyrights = [item for item in records if item.kind == "copyright"]
    licenses = [item for item in records if item.kind == "license"]
    if len(copyrights) > 1 or len(licenses) > 1:
        return Finding("duplicate", path, "more than one managed SPDX directive is present")
    if not copyrights or not licenses:
        missing = "copyright" if not copyrights else "license"
        return Finding("partial", path, f"SPDX header is missing its {missing} directive")

    newline = _detect_newline(data)
    expected = style.render(newline)
    expected_offset = _header_offset(data, style)
    expected_end = expected_offset + len(expected)
    exact = data.startswith(expected, expected_offset) and (
        expected_end == len(data) or data.startswith(newline, expected_end)
    )
    if exact:
        return None

    exact_texts = {
        "copyright": COPYRIGHT_TEXT.encode("ascii"),
        "license": LICENSE_TEXT.encode("ascii"),
    }
    if all(item.text == exact_texts[item.kind] for item in records):
        if records[0].offset != expected_offset or data.find(expected) != expected_offset:
            return Finding(
                "misplaced",
                path,
                "exact SPDX directives are not at the approved preamble location",
            )
    return Finding(
        "malformed",
        path,
        f"SPDX directives do not match the exact 2026 {style.name} form",
    )


def _blank(line: bytes) -> bool:
    return not _without_line_ending(line).strip()


def _strip_leading_managed_fragments(data: bytes) -> bytes:
    """Remove repairable SPDX preamble fragments and one separator line."""

    records = _markers(data)
    if not records:
        return data
    if any(record.line_index >= 32 for record in records):
        raise UnsafeFixError("SPDX directive occurs after the 32-line repairable preamble")

    lines = data.splitlines(keepends=True)
    remove = {record.line_index for record in records}
    markers_in_header_only_blocks: set[int] = set()

    # Remove /* ... */ delimiters only when their block contains no content
    # other than SPDX directives and blank lines.  A marker embedded in a
    # larger copyright comment requires human review.
    marker_indexes = set(remove)
    for opener, line in enumerate(lines):
        if _without_line_ending(line).strip() != b"/*":
            continue
        closer = opener + 1
        while closer < len(lines):
            stripped = _without_line_ending(lines[closer]).strip()
            if stripped == b"*/":
                interior = set(range(opener + 1, closer))
                if interior & marker_indexes and all(
                    index in marker_indexes or _blank(lines[index]) for index in interior
                ):
                    remove.update(range(opener, closer + 1))
                    markers_in_header_only_blocks.update(interior & marker_indexes)
                break
            if closer - opener > 8:
                break
            closer += 1

    for record in records:
        stripped = _without_line_ending(lines[record.line_index]).lstrip()
        if stripped.startswith(b"*") and record.line_index not in markers_in_header_only_blocks:
            raise UnsafeFixError("SPDX directive is embedded in a non-header block comment")

    # Blank lines wholly between duplicate fragments belong to the malformed
    # preamble.  Also remove exactly one separator immediately after each
    # fragment so replacing a partial header cannot accumulate blank lines.
    if remove:
        low, high = min(remove), max(remove)
        for index in range(low, high + 1):
            if _blank(lines[index]):
                remove.add(index)

        for end in sorted(remove, reverse=True):
            next_index = end + 1
            if next_index not in remove:
                if next_index < len(lines) and _blank(lines[next_index]):
                    remove.add(next_index)
                break

    return b"".join(line for index, line in enumerate(lines) if index not in remove)


def _install_header(core: bytes, style: HeaderStyle) -> bytes:
    newline = _detect_newline(core)
    offset = _header_offset(core, style)
    prefix, suffix = core[:offset], core[offset:]
    if prefix and not prefix.endswith((b"\n", b"\r")):
        raise UnsafeFixError("shebang or encoding-cookie line has no terminating newline")
    separator = newline if not suffix else newline + newline
    return prefix + style.render(newline) + separator + suffix


def strip_managed_header(data: bytes, style: HeaderStyle) -> bytes:
    """Remove one exact managed header and its optional blank separator."""

    newline = _detect_newline(data)
    offset = _header_offset(data, style)
    expected = style.render(newline)
    if not data.startswith(expected, offset):
        raise ValueError("data does not contain an exact managed header")
    end = offset + len(expected)
    if data.startswith(newline + newline, end):
        end += 2 * len(newline)
    elif data.startswith(newline, end):
        end += len(newline)
    return data[:offset] + data[end:]


def fix_header(data: bytes, style: HeaderStyle) -> tuple[bytes, str, str]:
    """Return normalized bytes plus before/after body SHA-256 values."""

    finding = inspect_header(data, style)
    if finding is None:
        digest = hashlib.sha256(strip_managed_header(data, style)).hexdigest()
        return data, digest, digest

    core = _strip_leading_managed_fragments(data)
    before = hashlib.sha256(core).hexdigest()
    fixed = _install_header(core, style)
    after_core = strip_managed_header(fixed, style)
    after = hashlib.sha256(after_core).hexdigest()
    if before != after:
        raise AssertionError("managed-header rewrite changed source-body bytes")
    return fixed, before, after


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_repository(repo_root: Path, manifest_path: Path, *, fix: bool = False) -> AuditReport:
    """Check (and optionally repair) every tracked file in ``repo_root``."""

    repo_root = repo_root.resolve()
    exceptions = load_exceptions(manifest_path.resolve())
    paths = tracked_files(repo_root)
    tracked_set = set(paths)
    report = AuditReport(tracked_files=len(paths))

    for missing_exception in sorted(set(exceptions) - tracked_set):
        report.findings.append(
            Finding(
                "exception-not-tracked",
                missing_exception,
                "exception path is absent from git ls-files",
            )
        )

    for relative_path in paths:
        full_path = repo_root / relative_path
        try:
            data = full_path.read_bytes()
        except OSError as exc:
            report.findings.append(Finding("unreadable", relative_path, str(exc)))
            continue

        exception = exceptions.get(relative_path)
        if exception is not None:
            report.excepted_files += 1
            actual = _sha256(data)
            if actual != exception.sha256:
                report.findings.append(
                    Finding(
                        "exception-hash-mismatch",
                        relative_path,
                        f"expected {exception.sha256}, found {actual}",
                    )
                )
            continue

        style = style_for_path(relative_path, data)
        if style is None:
            if _is_known_non_source(relative_path):
                report.ignored_files += 1
            else:
                report.findings.append(
                    Finding(
                        "unclassified",
                        relative_path,
                        "tracked suffix/name is neither managed nor explicitly non-source",
                    )
                )
            continue

        report.managed_files += 1
        finding = inspect_header(data, style, relative_path)
        if finding is None or not fix:
            if finding is not None:
                report.findings.append(finding)
            continue

        if full_path.is_symlink():
            report.findings.append(
                Finding("unsafe-fix", relative_path, "refusing to rewrite a symbolic link")
            )
            continue

        try:
            fixed, before_body, after_body = fix_header(data, style)
        except (UnsafeFixError, ValueError) as exc:
            report.findings.append(Finding("unsafe-fix", relative_path, str(exc)))
            continue

        if fixed != data:
            mode = stat.S_IMODE(full_path.stat().st_mode)
            full_path.write_bytes(fixed)
            if stat.S_IMODE(full_path.stat().st_mode) != mode:
                os.chmod(full_path, mode)
            report.changed_files.append(relative_path)
            report.body_audits.append(BodyAudit(relative_path, before_body, after_body))

        remaining = inspect_header(fixed, style, relative_path)
        if remaining is not None:
            report.findings.append(remaining)

    return report


def _write_report(path: Path, report: AuditReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_const", const="check", dest="mode")
    mode.add_argument("--fix", action="store_const", const="fix", dest="mode")
    parser.set_defaults(mode="check")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to this script's parent repository)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="exception manifest (defaults to tools/legal_header_exceptions.toml)",
    )
    parser.add_argument("--report", type=Path, help="write a machine-readable JSON audit report")
    parser.add_argument("--verbose", action="store_true", help="list files changed by --fix")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else repo_root / "tools" / "legal_header_exceptions.toml"
    )
    try:
        report = audit_repository(repo_root, manifest, fix=args.mode == "fix")
    except (ManifestError, RuntimeError) as exc:
        print(f"legal header audit failed: {exc}", file=sys.stderr)
        return 2

    for finding in report.findings:
        print(finding, file=sys.stderr)
    if args.verbose:
        for path in report.changed_files:
            print(f"[fixed] {path}")
    if args.report is not None:
        _write_report(args.report, report)

    print(
        "legal headers: "
        f"tracked={report.tracked_files} managed={report.managed_files} "
        f"excepted={report.excepted_files} ignored={report.ignored_files} "
        f"changed={len(report.changed_files)} findings={len(report.findings)}"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
