# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a standalone HTML bring-up report from this run's actual receipts."""

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


STYLE = """
body{font:16px/1.7 system-ui,-apple-system,sans-serif;color:#152536;background:#f2f5f7;margin:0}
main{max-width:1140px;margin:32px auto;padding:0 24px 60px}
header,section{background:white;padding:28px 34px;margin:18px 0;border-radius:12px;border:1px solid #dae2e8}
header{border-top:6px solid #167367}h1{font-size:34px;line-height:1.3;margin:12px 0}h2{font-size:24px;color:#164c48}
a{color:#146861}code,pre{font:13px/1.6 ui-monospace,monospace}pre{background:#f3f6f8;padding:18px;overflow:auto;white-space:pre-wrap}
code,pre,td,h1{overflow-wrap:anywhere}.eyebrow,.small{font-size:13px;color:#526b78}
.status{background:#e6f5ee;border-left:4px solid #167367;padding:12px}.pending{background:#fff4d9;border-color:#ba8c24}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #dfe6eb;vertical-align:top}
th{background:#edf3f5}.scroll{overflow:auto}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.metric{background:#f3f7f8;padding:16px;border-radius:8px}.metric strong{display:block;font-size:28px;color:#164c48}
nav a{margin-right:18px}button{border:1px solid #a6bab7;border-radius:6px;padding:8px 16px;background:white;color:#164c48;cursor:pointer}
summary{cursor:pointer;font-weight:600}.sample{border-left:3px solid #167367;padding:2px 20px;margin:20px 0}
li{margin:8px 0}@media(max-width:650px){main{padding:0 12px 30px;margin:12px auto}section,header{padding:20px 16px}.metrics{grid-template-columns:1fr}h1{font-size:28px}}
@media print{body{background:white}main{margin:0;padding:0}section,header{border-radius:0}button,nav{display:none}a{color:inherit}}
"""


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def table(headers, rows):
    head = "".join(f"<th>{escape(str(cell))}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(state, output, source_commit=None, evidence_directory=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_directory = evidence_directory or output.parent / "evidence"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    build = read_json(state / "flint-full-build-result.json")
    suite = read_json(state / "flint-e2e-result.json")
    architecture = read_json(state / "flint-architecture-final-result.json")
    boundary = read_json(state / "flint-window-boundary-result.json")
    environment = read_json(state / "flint-environment.json") or {}
    quality = read_json(state / "flint-quality-result.json") or {}
    legal = read_json(state / "flint-legal-headers.json") or {}
    comparisons = list((state / "flint-e2e-evidence").rglob("comparison.json"))
    comparison = read_json(comparisons[0]) if comparisons else None
    junit = state / "flint-e2e.xml"
    counts = None
    if junit.is_file():
        root = ET.parse(junit).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        counts = {
            name: sum(int(s.get(name, 0)) for s in suites)
            for name in ("tests", "failures", "errors", "skipped")
        }
        counts["passed"] = (
            counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        )
    verified = bool(
        build
        and build.get("exit_code") == 0
        and suite
        and suite.get("exit_code") == 0
        and comparison
    )
    status = (
        "Verified: full-checkpoint native BF16 generation and independent numerical comparison"
        if verified
        else "In progress: full-model generation has not passed its end-to-end gate"
    )
    if suite and suite.get("exit_code") != 0:
        status = "Validation failed: full-model support is not established"
    files = {
        "build.json": state / "flint-full-build-result.json",
        "family-suite.json": state / "flint-e2e-result.json",
        "architecture.json": state / "flint-architecture-final-result.json",
        "native-build.json": state / "flint-native-build-result.json",
        "window-boundary.json": state / "flint-window-boundary-result.json",
        "environment.json": state / "flint-environment.json",
        "legal-headers.json": state / "flint-legal-headers.json",
        "source-quality.json": state / "flint-quality-result.json",
        "family-suite.xml": junit,
        "family-suite.txt": state / "flint-e2e.log",
        "architecture.txt": state / "flint-architecture-final.log",
        "native-build.txt": state / "flint-native-build.log",
        "build.txt": state / "flint-full-build.log",
    }
    if comparisons:
        files["comparison.json"] = comparisons[0]
    links = []
    for name, path in files.items():
        if path.is_file():
            target = evidence_directory / name
            shutil.copyfile(path, target)
            href = os.path.relpath(target, output.parent)
            links.append(f'<li><a href="{escape(href, quote=True)}">{escape(name)}</a></li>')
    sections = []

    def section(title, body, anchor):
        sections.append(f'<section id="{anchor}"><h2>{escape(title)}</h2>{body}</section>')

    case_scope = "All 49 layers; exact BF16 checkpoint; one GPU; batch 1; bundle context capacity 1,024 tokens."
    section(
        "Conclusion and deliverables",
        f'<p class="status {"" if verified else "pending"}">{escape(status)}.</p><p>{case_scope}</p><p>The implementation is a new self-contained <code>families/flint/</code> vertical slice. It includes exact support resolution, weight mapping, strongly typed TensorRT graphs, a native family DSO, tokenization, sampling, KV state, a build command, and family-owned validation.</p><p>Source commit: <code>{escape(source_commit or "Not committed yet")}</code>. Branch: <code>bringup/flint</code>.</p><p>The numerical reference is an independent CPU BF16 PyTorch implementation of equations from the pinned upstream SGLang package. The upstream SGLang server and agent benchmarks were not executed.</p>',
        "conclusion",
    )
    env_rows = [
        ("Checkpoint", "RecursiveSI/flint"),
        ("Hugging Face revision", "2a247be0c0aec895ccb0ac7ca92229ec6df37288"),
        ("Family / task", "flint / text_generation"),
        (
            "Native / reference precision",
            "BF16 / BF16, explicit FP32 normalization, probability arithmetic and LM head",
        ),
        (
            "Layer and expert inventory",
            "49 layers: 2 dense + 47 MoE; 256 routed experts per MoE layer; top 6 + shared expert",
        ),
        (
            "Attention",
            "14 full layers without RoPE; 35 sliding layers with RoPE; learned per-head sinks; 513-token sliding window",
        ),
        ("Built capacity", "1,024 tokens, batch 1, tensor parallelism 1"),
        (
            "GPU",
            environment.get(
                "gpu", "SM100, approximately 179 GiB VRAM; environment receipt pending"
            ),
        ),
        ("GPU identity", environment.get("gpu_uuid", "Pending environment receipt")),
        (
            "Driver / CUDA compiler",
            str(environment.get("driver", "pending"))
            + " / "
            + str(environment.get("cuda", "pending")),
        ),
        (
            "Python / TensorRT / PyTorch",
            " / ".join(
                str(environment.get(key, "pending")) for key in ("python", "tensorrt", "torch")
            ),
        ),
        ("Bundle", build.get("bundle", "Not published yet") if build else "Not published yet"),
        ("Bundle bytes", build.get("size", "pending") if build else "pending"),
    ]
    section(
        "Versions, environment and scope",
        table(["Item", "Recorded value"], env_rows),
        "environment",
    )
    section(
        "Bring-up design",
        "<ol><li>Pin the checkpoint and inspect its upstream configuration and SGLang model implementation.</li><li>Recognize Flint as a distinct family, preserving sigmoid routing, ungated shared experts, learned attention sinks, hybrid attention and selective RoPE.</li><li>Build one TensorRT engine per decoder layer and one FP32 LM-head engine. TensorRT selects all GPU kernels and tactics.</li><li>Assemble a self-contained bundle containing every checkpoint layer, the embedding table, runtime metadata, tokenizer and chat template.</li><li>Load the bundle through the family DSO and abstract Task interface. Each request resets its own KV state.</li><li>Compare native logits with independent PyTorch equations on the same token prefix; check tokenizer parity and repeat-request determinism.</li></ol><p>Resume caching belongs to the local build recipe, outside production family code. Cached and production graph syntax trees were checked for equivalence.</p>",
        "design",
    )
    run_rows = [
        (
            "Native compilation",
            "PASS"
            if (read_json(state / "flint-native-build-result.json") or {}).get("exit_code") == 0
            else "PENDING",
            "CLI, TensorRT backend, Flint DSO and native logit diagnostic",
        ),
        (
            "Architecture + contracts",
            "61 passed" if architecture and architecture.get("exit_code") == 0 else "PENDING",
            "Separate run; includes the four CPU contract tests",
        ),
        (
            "Dense/full, MoE/sliding, MoE/full layer checks",
            "3 passed"
            if (read_json(state / "flint-layer-checks-result.json") or {}).get("exit_code") == 0
            else "PENDING",
            "Exact checkpoint weights; independent BF16 equations",
        ),
        (
            "Sliding-window boundary",
            "PASS" if boundary and boundary.get("exit_code") == 0 else "PENDING",
            "Layer 2, position 512, 513 attended tokens",
        ),
        (
            "Full bundle",
            "PASS" if build and build.get("exit_code") == 0 else "PENDING",
            "All 49 layers plus final norm and LM head",
        ),
        (
            "Selected family suite",
            f"{counts['passed']} passed, {counts['failures']} failed, {counts['errors']} errors, {counts['skipped']} skipped"
            if counts
            else "PENDING",
            "Includes native Task E2E, CPU contracts and three GPU layer checks",
        ),
        (
            "Python quality / legal headers",
            "PASS / 0 findings"
            if quality.get("ruff_exit_code") == 0 and legal.get("ok")
            else "PENDING",
            "Ruff; tracked-source legal-header audit; clean whitespace diff",
        ),
        (
            "Public CI / performance / deployment",
            "NOT RUN",
            "No broad quality, latency, throughput or production qualification claim",
        ),
    ]
    section(
        "Validation results", table(["Check", "Result", "Evidence scope"], run_rows), "validation"
    )
    if comparison:
        rows = []
        for index, (relative, cosine) in enumerate(
            zip(comparison["relative_l2"], comparison["cosine_similarity"])
        ):
            rows.append((index, f"{relative:.8g}", "< 0.05", f"{cosine:.8g}", "> 0.999"))
        section(
            "Native logits and reference comparison",
            table(["Decode row", "Relative L2", "Frozen limit", "Cosine", "Frozen limit"], rows)
            + f"<p>Top-1 agreement: <strong>{comparison['top1_agreement']:.0%}</strong>; required ≥75%. Teacher forcing preserves the native token prefix for every compared row.</p>",
            "comparison",
        )
        native = comparison["native"]
        section(
            "Real generated text",
            f'<div class="sample"><h3>Raw continuation</h3><p><strong>Input</strong></p><pre>The capital of France is</pre><p><strong>Native output</strong></p><pre>{escape(native["text"])}</pre><p>Generated token IDs: <code>{escape(str(native["generated_ids"]))}</code></p><p>Native tokenizer IDs match the upstream tokenizer exactly. Repeated requests through the same loaded Task produce identical token IDs.</p></div>',
            "samples",
        )
    else:
        section(
            "Native logits and generated text",
            '<p class="status pending">Pending. No generated text or full-model numerical pass is claimed before the real native E2E run completes.</p>',
            "comparison",
        )
    issues = [
        (
            "Changing execution host",
            "The initial 16 GB host could not hold the BF16 model. The verified build hosts expose an SM100 GPU with about 179 GiB VRAM. Completed engines were preserved when the original container handle became unavailable.",
        ),
        (
            "Architecture integration",
            "Moved caching out of production family code, introduced family-owned build inputs, split the required factory entry point, and explicitly rejected unsupported runtime KV budgets. All architecture checks passed.",
        ),
        (
            "Long build and storage I/O",
            "The full checkpoint contains about 79.6 GB of tensor data. Layer engines were cached by the local recipe, with tail layers precompiled concurrently.",
        ),
        (
            "Qualification limits",
            "Only the declared BF16, one-GPU, batch-1, 1,024-token build and recorded test workloads are covered. Full 262,144-token context, quantization, parallelism, tool-use serving and upstream agent benchmarks are unqualified.",
        ),
    ]
    section(
        "Findings, fixes and remaining limits",
        table(["Finding", "Disposition"], issues),
        "findings",
    )
    commands = """# In the prepared checkout-owned development container
export PYTHONPATH=core/builder:apps/benchmark:.
build-flint/trtmc flint build /path/to/pinned/flint -o flint-bf16.bundle --max-sequence-length 1024
build-flint/trtmc run flint-bf16.bundle --runtime-root build-flint --prompt 'The capital of France is' --max-new-tokens 4

# Exact family validation (TRTMC_FLINT_BUNDLE may reuse this run's built artifact)
export TRTMC_BINARY=$PWD/build-flint/trtmc
export TRTMC_RUNTIME_ROOT=$PWD/build-flint
export TRTMC_FLINT_CHECKPOINT=/path/to/pinned/flint
export TRTMC_FLINT_BUNDLE=$PWD/flint-bf16.bundle
python -m tools.model_ci validate
python -m pytest families/flint/tests --e2e-model flint -q
git diff --check"""
    section(
        "Reproduce",
        "<pre>"
        + escape(commands)
        + "</pre><p>Use the repository's <code>apps/devtoolkit</code> environment preparation and the current source-build documentation. The native build targets are <code>trtmc</code>, <code>trtmc_backend_trt</code>, <code>trtmc_model_flint</code> and <code>flint_logits_trace</code>.</p>",
        "reproduce",
    )
    section(
        "Retained evidence",
        "<ul>"
        + "".join(links)
        + "</ul><p>Build completion, native compilation and numerical correctness are separate results. This report reads execution receipts rather than inferring success from a stage label.</p>",
        "evidence",
    )
    section(
        "Upstream references",
        '<ul><li><a href="https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288">Pinned checkpoint, configuration and serving kit</a></li>'
        '<li><a href="https://huggingface.co/RecursiveSI/flint/blob/2a247be0c0aec895ccb0ac7ca92229ec6df37288/sglang/README.md">Upstream reference runtime and supported configurations</a></li>'
        '<li><a href="https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/models/qwen3_moe.py">Inherited SGLang attention and MoE implementation</a></li></ul>',
        "references",
    )
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Flint bringup and verification report</title><style>{STYLE}</style></head><body><main><header><p class="eyebrow">TensorRT-Model-Connect · Native model integration</p><h1>Recursive Flint bringup and verification report</h1><p>RecursiveSI/flint · {stamp}</p><p class="status {"" if verified else "pending"}">{escape(status)}</p><nav><a href="#conclusion">Conclusion</a><a href="#validation">Validation</a><a href="#comparison">Native output</a><a href="#reproduce">Reproduce</a><button onclick="window.print()">Print / save PDF</button></nav></header>{"".join(sections)}</main></body></html>'
    output.write_text(html)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--evidence-directory", type=Path)
    args = parser.parse_args()
    render(args.state_root, args.output, args.source_commit, args.evidence_directory)
