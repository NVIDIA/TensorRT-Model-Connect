# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize a fixed real-prompt selection for AIPerf's existing loader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .config import ConfigError
from .text_profile import SHA, digest, response_text


def select_prompts(records: list[Any], count: int, context: int, output: int,
                   tokenize: Callable[[str], list[int]]) -> tuple[list[dict], list[dict]]:
    """Take distinct fitting first-user turns, preserving every original string."""
    if any(type(v) is not int or v < 1 for v in (count, context, output)) or output >= context:
        raise ConfigError("corpus count/context/output must be positive integers fitting the bundle")
    selected, ledger, texts = [], [], set()
    for index, row in enumerate(records):
        turns = row.get("conversations") if isinstance(row, dict) else None
        first = turns[0] if isinstance(turns, list) and turns else None
        prompt = first.get("value") if isinstance(first, dict) and first.get("from") == "human" else None
        entry: dict[str, Any] = {"source_index": index}
        if not isinstance(prompt, str) or not prompt.strip():
            entry.update(status="rejected", reason="no nonempty first human turn")
        else:
            tokens = len(tokenize(prompt))
            entry.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                         prompt_bytes=len(prompt.encode()), input_tokens=tokens)
            if prompt in texts:
                entry.update(status="rejected", reason="duplicate selected prompt text")
            elif tokens + output > context:
                entry.update(status="rejected", reason="prompt plus output exceeds bundle context")
            else:
                entry.update(status="selected", prompt=prompt)
                selected.append(entry)
                texts.add(prompt)
        ledger.append(entry)
        if len(selected) == count:
            break
    return selected, ledger


def prepare(config: dict[str, Any], capacity: int, output: int, out: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from .suites import _hf_file, _read_records

    spec = config["corpus"]
    source = spec["source"]
    if spec.get("format") != "sharegpt" or not SHA.fullmatch(str(source.get("revision", ""))):
        raise ConfigError("real prompts require ShareGPT format and an immutable dataset revision")
    path = _hf_file(source)
    source_hash = digest(path)
    if source.get("sha256") and source["sha256"] != source_hash:
        raise ConfigError("real-prompt source checksum differs from its pinned configuration")
    records = _read_records(path)
    if not isinstance(records, list):
        raise ConfigError("ShareGPT source must contain a list of conversations")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["name"],
        revision=config["tokenizer"]["revision"], trust_remote_code=False)
    selected, ledger = select_prompts(records, spec["count"], capacity, output,
                                      lambda text: tokenizer.encode(text, add_special_tokens=False))
    prepared = out / "inputs.jsonl"
    prepared.write_text("".join(json.dumps({"text": row["prompt"]}, ensure_ascii=False) + "\n"
                                 for row in selected), encoding="utf-8")
    report = {"source": source, "source_sha256": source_hash, "source_records": len(records),
              "selection_policy": "first distinct fitting first-human turns in source file order; no rewriting or truncation",
              "considered_records": len(ledger), "unexamined_records": len(records) - len(ledger),
              "context_limit": capacity, "reserved_output_tokens": output,
              "tokenizer": config["tokenizer"], "token_count_scope": "pinned tokenizer, add_special_tokens=false; native input count unavailable",
              "selected": selected, "ledger": ledger, "input_file": str(prepared),
              "input_file_sha256": digest(prepared)}
    (out / "corpus.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if len(selected) != spec["count"]:
        raise ConfigError(f"only {len(selected)} corpus prompts fit; see corpus.json; no truncation applied")
    return report


def verify(run, joined: list[dict[str, Any]], corpus: dict[str, Any]) -> dict[str, Any]:
    """Join original dataset strings, actual wire payloads and native dispatch records."""
    expected = {row["prompt"]: row for row in corpus["selected"]}
    by_id = {row["request_id"]: row for row in joined}
    seen, measured, identities = set(), set(), []
    for raw in run.raw_records(phase=None):
        request_id = raw["metadata"].get("x_request_id")
        payload = raw.get("payload")
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        if (not isinstance(request_id, str) or request_id in seen or request_id not in by_id
                or not isinstance(prompt, str) or prompt not in expected):
            raise ConfigError("real-prompt request has unknown/duplicate identity or changed text")
        seen.add(request_id)
        selected = expected[prompt]
        server = by_id[request_id]["server"]
        text = response_text(raw)
        if (server is None or server.get("status") != 200 or not isinstance(text, str) or
                server.get("prompt_sha256") != selected["prompt_sha256"] or
                server.get("prompt_bytes") != selected["prompt_bytes"] or
                server.get("completion_sha256") != hashlib.sha256(text.encode()).hexdigest()):
            raise ConfigError("real-prompt native input/output does not match the original corpus and HTTP response")
        if raw["metadata"].get("benchmark_phase") == "profiling":
            measured.add(prompt)
        identities.append({"request_id": request_id, "phase": raw["metadata"].get("benchmark_phase"),
                           "source_index": selected["source_index"], "prompt_sha256": selected["prompt_sha256"],
                           "input_tokens": selected["input_tokens"], "completion_sha256": server["completion_sha256"]})
    if not seen or measured != set(expected):
        raise ConfigError("real-prompt profiling did not cover every selected prompt")
    return {"verified": True, "measured_unique_prompts": len(measured), "requests": identities}


def verify_and_record(run, joined: list[dict[str, Any]], corpus: dict[str, Any]) -> None:
    """Preserve a verification verdict for missing, changed or malformed inputs."""
    path = run.directory / "corpus-verification.json"
    try:
        verified = verify(run, joined, corpus)
        try:
            unchanged = digest(Path(corpus["input_file"])) == corpus["input_file_sha256"]
        except OSError as error:
            raise ConfigError("real-prompt input file is unavailable") from error
        if not unchanged:
            raise ConfigError("real-prompt inputs changed during profiling")
    except ConfigError as error:
        path.write_text(json.dumps({"verified": False, "error": str(error),
                                    "request_records": "joined.jsonl"}, indent=2) + "\n")
        raise
    path.write_text(json.dumps(verified, indent=2) + "\n")
