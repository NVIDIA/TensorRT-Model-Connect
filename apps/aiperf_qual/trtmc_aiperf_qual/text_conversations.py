# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed original user turns using AIPerf's existing live-response history."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import ConfigError
from .text_corpus import load_source
from .text_profile import digest, response_text


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def template_tokens(tokenizer, messages) -> int:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def select(records, count, capacity, output, count_messages):
    if any(type(value) is not int or value < 1 for value in (count, capacity, output)) or output >= capacity:
        raise ConfigError("conversation count/context/output must be positive integers fitting the bundle")
    selected, ledger, seen = [], [], set()
    for index, row in enumerate(records):
        turns = row.get("conversations", []) if isinstance(row, dict) else []
        entry = {"source_index": index}
        if (not isinstance(turns, list) or len(turns) < 3 or
                any(not isinstance(turns[i], dict) or turns[i].get("from") != role or
                    not isinstance(turns[i].get("value"), str) or not turns[i]["value"].strip()
                    for i, role in enumerate(("human", "gpt", "human")))):
            entry.update(status="rejected", reason="no original human/assistant/human prefix")
        else:
            texts = (turns[0]["value"], turns[2]["value"])
            base = count_messages([{"role": "user", "content": texts[0]},
                                   {"role": "assistant", "content": ""},
                                   {"role": "user", "content": texts[1]}])
            entry["template_tokens_without_live_answer"] = base
            if texts in seen or base + 2 * output > capacity:
                entry.update(status="rejected", reason="duplicate turns or conversation/output reserve exceeds context")
            else:
                entry.update(status="selected", session_id=f"sharegpt-{index}", user_turns=list(texts))
                selected.append(entry)
                seen.add(texts)
        ledger.append(entry)
        if len(selected) == count:
            break
    return selected, ledger


def prepare(config, capacity: int, output: int, out: Path):
    source, source_hash, records, tokenizer = load_source(config)
    spec = config["corpus"]
    delay = spec.get("think_time_ms", 50)
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        raise ConfigError("think_time_ms must be a finite nonnegative number")
    def count_messages(messages):
        return template_tokens(tokenizer, messages)
    selected, ledger = select(records, spec["count"], capacity, output, count_messages)
    path = out / "inputs.jsonl"
    path.write_text("".join(json.dumps({"session_id": row["session_id"], "turns": [
        {"text": text, "delay": 0 if index == 0 else delay, "output_length": output}
        for index, text in enumerate(row["user_turns"])]}, ensure_ascii=False) + "\n"
        for row in selected), encoding="utf-8")
    report = {"format": "sharegpt_multi_turn", "source": source, "source_sha256": source_hash,
              "tokenizer": config["tokenizer"], "context_limit": capacity, "output_tokens": output,
              "think_time_ms": delay, "source_records": len(records), "considered_records": len(ledger),
              "unexamined_records": len(records) - len(ledger), "selected": selected, "ledger": ledger,
              "selection_policy": "first distinct fitting original human/assistant/human prefixes; send the first two human strings unchanged",
              "assistant_history_policy": "AIPerf deltas_without_responses captures actual native answers; source assistant answers are not replayed",
              "token_count_scope": "pinned tokenizer chat template; native input count unavailable; actual wire history counted after the run",
              "input_file": str(path), "input_file_sha256": digest(path)}
    (out / "corpus.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if len(selected) != spec["count"]:
        raise ConfigError("not enough original conversations fit; see corpus.json; no history was truncated")
    return report


def messages_from_payload(payload):
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        raise ConfigError("conversation payload is missing messages")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ConfigError("conversation message is not an object")
        content = message.get("content")
        if isinstance(content, list) and all(isinstance(p, dict) and p.get("type") == "text" and
                                             isinstance(p.get("text"), str) for p in content):
            content = "".join(p["text"] for p in content)
        if not isinstance(content, str):
            raise ConfigError("conversation content is not text")
        normalized.append({"role": message.get("role"), "content": content})
    return normalized


def verify(run, joined, corpus):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(corpus["tokenizer"]["name"],
        revision=corpus["tokenizer"]["revision"], trust_remote_code=False)
    return verify_records(run, joined, corpus, lambda messages: template_tokens(tokenizer, messages))


def verify_records(run, joined, corpus, count_messages):
    expected = {row["session_id"]: row for row in corpus["selected"]}
    servers = {row["request_id"]: row["server"] for row in joined}
    groups, seen, measured, identities = {}, set(), set(), []
    raw_records = run.raw_records(phase=None)
    for raw in raw_records:
        meta = raw.get("metadata", {})
        if (not isinstance(meta, dict) or not isinstance(meta.get("x_request_id"), str) or
                not isinstance(meta.get("x_correlation_id"), str) or not meta["x_correlation_id"] or
                not isinstance(meta.get("conversation_id"), str) or type(meta.get("session_num")) is not int or
                type(meta.get("turn_index")) is not int or meta["turn_index"] not in (0, 1) or
                any(type(meta.get(name)) is not int or meta[name] < 1 for name in ("request_start_ns", "request_end_ns"))):
            raise ConfigError("conversation request metadata is missing identity or timing")
    for raw in sorted(raw_records, key=lambda r: r["metadata"]["request_start_ns"]):
        meta = raw["metadata"]
        request_id, session, turn = meta.get("x_request_id"), meta.get("conversation_id"), meta.get("turn_index")
        if request_id in seen or session not in expected or type(turn) is not int or turn not in (0, 1):
            raise ConfigError("unknown or repeated conversation request identity")
        seen.add(request_id)
        key = (meta.get("benchmark_phase"), meta["x_correlation_id"])
        previous = groups.setdefault(key, [])
        if len(previous) != turn or (previous and previous[0]["session"] != session):
            raise ConfigError("conversation turns are missing, repeated or out of order")
        source = expected[session]
        wanted = [{"role": "user", "content": source["user_turns"][0]}]
        if turn:
            wanted += [{"role": "assistant", "content": previous[0]["text"]},
                       {"role": "user", "content": source["user_turns"][1]}]
        messages = messages_from_payload(raw.get("payload"))
        text = response_text(raw)
        server = servers.get(request_id)
        if (messages != wanted or not isinstance(text, str) or server is None or server.get("status") != 200 or
                server.get("completion_sha256") != hashlib.sha256(text.encode()).hexdigest()):
            raise ConfigError("conversation lost original text, roles, actual assistant history or native output")
        if turn and (server.get("native_operation") != "converse" or server.get("conversation_sha256") != fingerprint(messages)):
            raise ConfigError("native conversation dispatch changed history")
        if not turn and server.get("prompt_sha256") != hashlib.sha256(source["user_turns"][0].encode()).hexdigest():
            raise ConfigError("native first-turn dispatch changed the original text")
        tokens = count_messages(messages)
        if tokens + corpus["output_tokens"] > corpus["context_limit"]:
            raise ConfigError("actual conversation exceeds bundle context; no history may be dropped")
        gap = ((meta["request_start_ns"] - previous[0]["end_ns"]) / 1e6) if turn else None
        if gap is not None and gap < 0:
            raise ConfigError("conversation dispatched a turn before its previous response completed")
        previous.append({"text": text, "end_ns": meta["request_end_ns"], "session": session})
        if meta.get("benchmark_phase") == "profiling":
            measured.add((session, turn))
        identities.append({"request_id": request_id, "phase": meta.get("benchmark_phase"),
            "session_id": session, "session_num": meta["session_num"], "turn_index": turn,
            "correlation_id": meta.get("x_correlation_id"), "input_tokens": tokens,
            "messages_sha256": fingerprint(messages), "completion_sha256": server["completion_sha256"],
            "configured_think_time_ms": corpus["think_time_ms"] if turn else 0, "observed_gap_ms": gap})
    if (any(len(group) != 2 for (phase, _), group in groups.items() if phase == "profiling") or
            measured != {(session, turn) for session in expected for turn in (0, 1)}):
        raise ConfigError("profiling did not cover both turns of every selected conversation")
    return {"verified": True, "measured_conversations": len(expected), "requests": identities}
