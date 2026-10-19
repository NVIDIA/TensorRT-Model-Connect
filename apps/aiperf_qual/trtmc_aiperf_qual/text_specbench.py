# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact SpecBench corpus selection for an ordinary native generation baseline."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from .config import ConfigError
from .text_corpus import select_prompts
from .text_profile import SHA, digest, git


def select(records, count, capacity, output, tokenize):
    converted = []
    for row in records:
        turns = row.get("turns") if isinstance(row, dict) else None
        prompt = turns[0] if isinstance(turns, list) and turns and isinstance(turns[0], str) else None
        converted.append({"conversations": [{"from": "human", "value": prompt}]})
    selected, ledger = select_prompts(converted, count, capacity, output, tokenize)
    for entry in ledger:
        source = records[entry["source_index"]]
        if isinstance(source, dict):
            entry.update(question_id=source.get("question_id"), category=source.get("category"))
    return selected, ledger


def prepare(config, capacity: int, output: int, out: Path, *, repo: Path | None = None):
    spec, source = config["corpus"], config["corpus"]["source"]
    revision = str(source.get("revision", ""))
    if spec.get("runtime_mode", "ordinary_native_generation") != "ordinary_native_generation":
        raise ConfigError("this SpecBench workload qualifies ordinary generation only")
    if not SHA.fullmatch(revision) or f"/{revision}/" not in source["url"]:
        raise ConfigError("SpecBench source URL must contain its immutable revision")
    with urllib.request.urlopen(source["url"], timeout=60) as response:
        data = response.read()
    source_hash = hashlib.sha256(data).hexdigest()
    if source_hash != source.get("sha256"):
        raise ConfigError("SpecBench source checksum differs from its pinned configuration")
    (out / "source.jsonl").write_bytes(data)
    records = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["name"],
        revision=config["tokenizer"]["revision"], trust_remote_code=False)
    selected, ledger = select(records, spec["count"], capacity, output,
        lambda text: tokenizer.encode(text, add_special_tokens=False))
    path = out / "inputs.jsonl"
    path.write_text("".join(json.dumps({"text": row["prompt"]}, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    audit = spec.get("runtime_audit")
    if not isinstance(audit, dict) or not audit.get("conclusion") or not audit.get("source_file"):
        raise ConfigError("SpecBench baseline requires a documented native execution audit")
    repo = Path(git(repo or Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    source_path = (repo / audit["source_file"]).resolve()
    if not source_path.is_relative_to(repo) or not source_path.is_file():
        raise ConfigError("runtime audit must refer to an existing repository file")
    audit = {**audit, "source_sha256": digest(source_path), "repo_commit": git(repo, "rev-parse", "HEAD"),
             "evidence_scope": "manual source audit for the qualified pilot; no speculative counters observed"}
    execution = {"mode": "ordinary_native_generation", "draft_model": None, "target_model": config["checkpoint"],
        "speculative_execution_qualified": False, "acceptance_metrics": None,
        "interpretation": "qualified corpus input only; no draft/target execution or speculative acceleration claim"}
    report = {"format": "spec_bench", "source": source, "source_sha256": source_hash,
        "source_records": len(records), "considered_records": len(ledger),
        "unexamined_records": len(records) - len(ledger), "selected": selected, "ledger": ledger,
        "context_limit": capacity, "reserved_output_tokens": output, "tokenizer": config["tokenizer"],
        "selection_policy": "first distinct fitting original first-turn strings in source file order; no trimming or truncation; selected subset only",
        "client_dataset_type": "single_turn", "runtime_mode": "ordinary_native_generation",
        "runtime_audit": audit, "speculative_execution": execution,
        "input_file": str(path), "input_file_sha256": digest(path)}
    (out / "corpus.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if len(selected) != spec["count"]:
        raise ConfigError("not enough SpecBench prompts fit; see corpus.json; no truncation applied")
    return report


def verify(run, joined, corpus):
    from .text_corpus import verify as verify_prompts
    result = verify_prompts(run, joined, corpus)
    output = corpus["reserved_output_tokens"]
    for row in joined:
        server = row["server"]
        if (server.get("timing_scope") != "public_task_call_wall" or
                server.get("generation_config", {}).get("max_new_tokens") != output or
                type(server.get("completion_tokens")) is not int or not 0 <= server["completion_tokens"] <= output):
            raise ConfigError("SpecBench baseline has changed generation work or invalid native Task timing")
    result.update(runtime_mode=corpus["runtime_mode"], speculative_execution=corpus["speculative_execution"],
                  runtime_audit=corpus["runtime_audit"])
    return result
