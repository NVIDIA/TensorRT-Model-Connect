# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controlled exact prefix/suffix inputs, without implying native cache reuse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import ConfigError
from .text_corpus import select_prompts
from .text_profile import digest, git


def construct(spec, capacity, output, tokenize):
    prefix, suffixes = spec.get("prefix"), spec.get("suffixes")
    if (not isinstance(prefix, str) or not prefix or not isinstance(suffixes, list) or not suffixes or
            any(not isinstance(suffix, str) or not suffix.strip() for suffix in suffixes) or
            len(set(suffixes)) != len(suffixes) or spec.get("count") != len(suffixes)):
        raise ConfigError("controlled prefix requires a nonempty prefix and count distinct nonempty suffixes")
    texts = [prefix + suffix for suffix in suffixes]
    selected, ledger = select_prompts([{"conversations": [{"from": "human", "value": text}]} for text in texts],
                                     len(texts), capacity, output, tokenize)
    tokenized = [tokenize(text) for text in texts]
    shared = 0
    for tokens in zip(*tokenized):
        if len(set(tokens)) != 1:
            break
        shared += 1
    for row in selected:
        row.update(suffix=suffixes[row["source_index"]], shared_prefix_tokens=shared,
                   shared_fraction=shared / row["input_tokens"] if row["input_tokens"] else 0)
    return selected, ledger, {"prefix": prefix, "suffixes": suffixes,
        "configured_prefix_tokens": len(tokenize(prefix)), "common_prefix_tokens": shared,
        "overlap_policy": "longest identical token-ID prefix across all complete prompts; includes tokenization at the prefix/suffix boundary"}


def prepare(config, capacity: int, output: int, out: Path, *, repo: Path | None = None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["name"],
        revision=config["tokenizer"]["revision"], trust_remote_code=False)
    spec = config["corpus"]
    selected, ledger, layout = construct(spec, capacity, output,
        lambda text: tokenizer.encode(text, add_special_tokens=False))
    path = out / "inputs.jsonl"
    path.write_text("".join(json.dumps({"text": row["prompt"]}, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    source_bytes = json.dumps({"prefix": layout["prefix"], "suffixes": layout["suffixes"]},
                              ensure_ascii=False, sort_keys=True).encode("utf-8")
    warmup = config["workload"].get("warmup_requests", 3)
    audit = spec.get("cache_audit")
    if not isinstance(audit, dict) or not audit.get("conclusion") or not audit.get("source_file"):
        raise ConfigError("controlled-prefix profiles require a documented native cache capability audit")
    repo = Path(git(repo or Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    source = (repo / audit["source_file"]).resolve()
    if not source.is_relative_to(repo) or not source.is_file():
        raise ConfigError("cache audit source_file must identify an existing file in the repository")
    audit = {**audit, "source_sha256": digest(source), "repo_commit": git(repo, "rev-parse", "HEAD"),
             "evidence_scope": "manual source audit, recorded separately from runtime cache-hit observations"}
    report = {"format": "controlled_prefix", "source": "literal prefix/suffix layout in the recorded configuration",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "source_records": len(spec["suffixes"]),
        "considered_records": len(ledger), "unexamined_records": 0, "selected": selected, "ledger": ledger,
        "layout": layout, "context_limit": capacity, "reserved_output_tokens": output,
        "tokenizer": config["tokenizer"], "input_file": str(path), "input_file_sha256": digest(path),
        "execution_policy": {"server": "new server process per text-profile invocation",
            "warmup_requests": warmup, "measured_requests": config["workload"].get("requests", 20),
            "warmup_scope": "same prompt corpus; excluded from measured request metrics",
            "interpretation": "process warmup comparison; prompt overlap alone does not establish KV-cache reuse"},
        "cache_audit": audit, "cache_hit_telemetry": "unavailable; no cache hits inferred from overlap or timings"}
    (out / "corpus.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if len(selected) != spec["count"]:
        raise ConfigError("controlled prefix corpus does not fit the bundle; see corpus.json; no truncation applied")
    return report


def verify(run, joined, corpus):
    from .text_corpus import verify as verify_prompts
    result = verify_prompts(run, joined, corpus)
    raw = run.raw_records(phase=None)
    warmup = [row for row in raw if row["metadata"].get("benchmark_phase") == "warmup"]
    measured = [row for row in raw if row["metadata"].get("benchmark_phase") == "profiling"]
    policy = corpus["execution_policy"]
    if (len(warmup) != policy["warmup_requests"] or len(measured) != policy["measured_requests"] or
            len(raw) != len(warmup) + len(measured)):
        raise ConfigError("controlled prefix phases do not match the recorded fresh/warm policy")
    result.update(layout=corpus["layout"], execution_policy=policy,
                  cache_audit=corpus["cache_audit"], cache_hit_telemetry=corpus["cache_hit_telemetry"])
    return result
