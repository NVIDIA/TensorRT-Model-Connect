# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render recorded pytest observations without supplying model pass criteria."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Any

_LIMIT = 32 * 1024 * 1024
_INLINE_BUDGET = 256 * 1024 * 1024
_CSS = """
:root{font:15px/1.55 system-ui,sans-serif;color:#18202a;background:#f3f5f7}
body{max-width:1320px;margin:30px auto;padding:0 20px}h1{font-size:28px}h2{font-size:21px}
h3{font-size:17px}.case,.overview{padding:22px;border:1px solid #d5dde4;border-radius:12px;background:white;margin:20px 0}
.meta,.note{color:#566473}.passed{color:#21662d}.failed,.error{color:#ad2727}.skipped,.partial{color:#885400}
table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #dce2e6}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px;border-radius:6px;font-size:12px}
code{overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
figure{margin:0}figcaption{font-weight:650;padding:8px 0}img,video{width:100%;height:auto}audio{width:100%}
details{margin:10px 0}summary{cursor:pointer}.filters{display:flex;gap:10px;position:sticky;top:0;padding:12px;background:#f3f5f7}
input,select{font:inherit;padding:8px;border:1px solid #b5c0ca;border-radius:6px}input{flex:1}[hidden]{display:none!important}
@media(max-width:720px){.pair{grid-template-columns:1fr}.case{padding:12px}body{padding:0 10px}.filters{position:static}}
"""
_JS = """
function filterCases(){const q=document.getElementById('search').value.toLowerCase();const s=document.getElementById('status').value;document.querySelectorAll('.case').forEach(e=>{e.hidden=!(e.dataset.name.includes(q)&&(!s||e.dataset.status===s));});}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json(value: Any) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, indent=2))


def _media(path: str, root: Path, media_type: str, budget: list[int]) -> tuple[str, str | None]:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        return "", "Artifact path must remain within its testcase"
    source = root / relative
    if any(part.is_symlink() for part in (source, *source.parents) if part != root.parent):
        return "", "Symlink artifacts are not embedded"
    if not source.resolve().is_relative_to(root.resolve()) or not source.is_file():
        return "", "Artifact file is unavailable"
    if source.stat().st_size > _LIMIT:
        return "", "Artifact exceeds the inline size bound"
    allowed = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    mime = allowed.get(source.suffix.lower())
    if mime is None and source.suffix.lower() != ".ppm":
        return "", None
    if source.stat().st_size > budget[0]:
        return (
            "",
            "Aggregate inline media bound reached; original retained in the evidence directory",
        )
    data = source.read_bytes()
    if source.suffix.lower() == ".ppm":
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                converted = io.BytesIO()
                image.save(converted, format="PNG")
                data = converted.getvalue()
                mime = "image/png"
        except (ImportError, OSError, ValueError):
            return "", "PPM preview needs Pillow; the original remains in the evidence directory"
    if mime is None:
        return "", None
    if len(data) > min(_LIMIT, budget[0]):
        return "", "Converted preview exceeds the remaining inline bound"
    budget[0] -= len(data)
    uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    if mime.startswith("image/"):
        return f'<img loading="lazy" alt="Recorded output" src="{uri}">', None
    element = "audio" if mime.startswith("audio/") else "video"
    return f'<{element} controls preload="none" src="{uri}"></{element}>', None


def _checks(data: dict[str, Any]) -> str:
    checks = data.get("checks", [])
    if not checks:
        return '<p class="note">No assertion measurements were recorded. Consult the execution outcome and failure stage.</p>'
    rows = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = _escape(check.get("status", "unknown"))
        rows.append(
            f'<tr><td class="{status}">{status}</td><td><code>{_escape(check.get("expression", ""))}</code>'
            f"<details><summary>Evaluated operands / threshold</summary><pre>{_escape(check.get('explanation', ''))}</pre></details></td></tr>"
        )
    return (
        "<table><thead><tr><th>Result</th><th>Original assertion and recorded values</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _artifacts(data: dict[str, Any], root: Path, budget: list[int]) -> str:
    titles, captions, notes = {}, {}, []
    for view in data.get("views", []) if isinstance(data.get("views"), list) else []:
        if not isinstance(view, dict):
            continue
        referenced = False
        for kind in ("image", "audio", "video"):
            media = view.get(kind)
            if isinstance(media, dict) and media.get("artifact"):
                titles[media["artifact"]] = view.get("title", "Recorded view")
                captions[media["artifact"]] = view.get("caption", "")
                referenced = True
        if not referenced and view.get("caption"):
            notes.append(
                f"<p class='note'><strong>{_escape(view.get('title', 'Evidence note'))}</strong>: {_escape(view['caption'])}</p>"
            )
    figures, files = [], []
    for artifact in data.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path", ""))
        label = (
            titles.get(path) or f"{artifact.get('role', 'output')} / {artifact.get('label', path)}"
        )
        rendered, issue = _media(path, root, str(artifact.get("media_type", "")), budget)
        if rendered:
            caption = f"<p class='note'>{_escape(captions[path])}</p>" if captions.get(path) else ""
            figures.append(
                f"<figure><figcaption>{_escape(label)}</figcaption>{rendered}{caption}</figure>"
            )
        else:
            suffix = f" — {_escape(issue)}" if issue else " — retained in the evidence directory"
            files.append(f"<li><code>{_escape(path)}</code>{suffix}</li>")
    return (
        "".join(notes)
        + ('<div class="pair">' + "".join(figures) + "</div>" if figures else "")
        + (
            "<details><summary>Raw evidence files</summary><ul>"
            + "".join(files)
            + "</ul></details>"
            if files
            else ""
        )
    )


def _content(data: dict[str, Any], root: Path, budget: list[int]) -> str:
    family, case = str(data.get("family", "unknown")), str(data.get("case", "unknown"))
    status = str(data.get("status", "unknown"))
    parts = [
        f'<section class="case" data-name="{_escape((family + " " + case).lower())}" data-status="{_escape(status)}">',
        f'<h2>{_escape(family)} / {_escape(case)} <span class="{_escape(status)}">{_escape(status.upper())}</span></h2>',
        f'<p class="meta">Source: <code>{_escape(data.get("source_revision", "unknown"))}</code> · Evidence: {_escape(data.get("evidence_status", "unknown"))}</p>',
    ]
    if data.get("failure"):
        parts.append(
            f'<p class="failed">Failure stage: {_escape(data.get("failure_stage", "unknown"))}</p><pre>{_json(data["failure"])}</pre>'
        )
    if data.get("issues"):
        parts.append(
            '<p class="partial">Evidence is incomplete.</p><pre>' + _json(data["issues"]) + "</pre>"
        )
    if "inputs" in data:
        parts.append(
            "<h3>Inputs and checkpoint contract</h3><pre>" + _json(data["inputs"]) + "</pre>"
        )
    parts.append('<h3>Recorded outputs</h3><div class="pair">')
    for key, title in (("native", "Native output"), ("reference", "Reference output")):
        value = data.get(
            key,
            {"unavailable": "No output recorded; inspect the family contract and failure stage."},
        )
        parts.append(f"<div><h3>{title}</h3><pre>{_json(value)}</pre></div>")
    parts.append("</div>" + _artifacts(data, root, budget))
    parts.append(
        '<h3>Checks</h3><p class="note">These are the existing family assertions and pytest\'s evaluated operands. The report does not invent thresholds or recompute the model verdict.</p>'
        + _checks(data)
    )
    for key, title in (
        ("thresholds", "Configured thresholds"),
        ("timing", "Stage timings"),
        ("repro", "Reproduction"),
        ("captured_output", "Captured process output"),
        ("environment", "Environment"),
        ("checkpoint", "Resolved checkpoint"),
        ("observations", "All recorded observations"),
    ):
        if key in data:
            parts.append(
                f"<details><summary>{title}</summary><pre>{_json(data[key])}</pre></details>"
            )
    parts.append("</section>")
    return "".join(parts)


def render_report(
    cases: list[tuple[dict[str, Any], Path]], title: str = "Model correctness evidence"
) -> str:
    budget = [_INLINE_BUDGET]
    counts = {
        status: sum(data.get("status") == status for data, _ in cases)
        for status in ("passed", "failed", "error", "skipped")
    }
    summary = " · ".join(f"{status}: {count}" for status, count in counts.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(title)}</title><style>{_CSS}</style></head><body>
<h1>{_escape(title)}</h1><div class="overview"><p>{_escape(summary)}</p><p class="note">Execution results and family-owned observations. Missing evidence is not a passing model comparison.</p></div>
<div class="filters"><input id="search" aria-label="Search family or case" placeholder="Search family or case" oninput="filterCases()"><select id="status" aria-label="Filter status" onchange="filterCases()"><option value="">All results</option><option>passed</option><option>failed</option><option>error</option><option>skipped</option></select></div>
{"".join(_content(data, root, budget) for data, root in cases)}<script>{_JS}</script></body></html>"""


def render_case(data: dict[str, Any], root: Path) -> str:
    return render_report([(data, root)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_root", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    cases = []
    for path in sorted(arguments.artifacts_root.rglob("evidence.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(
            arguments.artifacts_root.resolve()
        ):
            raise ValueError("evidence file must remain within its root")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError(f"unsupported evidence schema: {path}")
        cases.append((data, path.parent))
    if not cases:
        parser.error("no testcase evidence.json was found")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_report(cases), encoding="utf-8")
    print(f"Wrote {len(cases)} testcase reports to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
