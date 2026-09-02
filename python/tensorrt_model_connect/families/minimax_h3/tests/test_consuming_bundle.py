# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tensorrt_model_connect.bundle_writer import (
    BundleInfo,
    BundleSection,
    write_bundle,
)
from tensorrt_model_connect.families.minimax_h3 import consuming_bundle
from tests.builder.conftest import read_bundle_file


def _file_section(
    path: Path,
    name: str,
    payload: bytes,
    *,
    consume: bool = True,
) -> consuming_bundle.ConsumingBundleSection:
    path.write_bytes(payload)
    return consuming_bundle.ConsumingBundleSection.from_file(
        name,
        path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        consume_source=consume,
    )


def _info() -> BundleInfo:
    return BundleInfo(
        model_id="MiniMaxAI/MiniMax-H3",
        model_type="minimax_h3",
        family="minimax_h3",
        trt_version="1.6.1.120",
        trt_abi="1.6",
        runtime_strategy="diffusion_minimax_h3",
        precision="bf16",
    )


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"file symlinks are unavailable to this test process: {error}")


def test_consuming_finalize_keeps_old_final_until_audited_and_consumes_only_plans(
    tmp_path: Path,
) -> None:
    output = tmp_path / "h3.bundle"
    output.write_bytes(b"old-final")
    plan = tmp_path / "engine.plan"
    tokenizer = tmp_path / "tokenizer.json"
    sections = [
        _file_section(plan, "engine_plan", b"qualified-plan"),
        _file_section(
            tokenizer,
            "tokenizer.json",
            b'{"model": {}}',
            consume=False,
        ),
        consuming_bundle.ConsumingBundleSection.from_bytes(
            "config.json", b'{"plan_sha256": {}}'
        ),
    ]

    assert consuming_bundle.write_consuming_bundle(output, _info(), sections) == output
    header, payloads = read_bundle_file(output)
    assert header["model_type"] == "minimax_h3"
    assert payloads["engine_plan"] == b"qualified-plan"
    assert not plan.exists()
    assert tokenizer.is_file()
    partial, journal = consuming_bundle.assembly_paths(output)
    assert not partial.exists()
    assert not journal.exists()

    # A complete final is fully validated and reused even though the consumed
    # source no longer exists.
    assert consuming_bundle.write_consuming_bundle(output, _info(), sections) == output


def test_consuming_bundle_is_byte_compatible_with_traditional_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTMC_ENGINE_BUILD_REVISION", raising=False)
    traditional = tmp_path / "traditional.bundle"
    consuming = tmp_path / "consuming.bundle"
    payloads = (
        ("engine_plan", b"qualified-plan"),
        ("tokenizer.json", b'{"model": {}}'),
        ("config.json", b"{}"),
    )
    write_bundle(
        traditional,
        _info(),
        [BundleSection(name, data) for name, data in payloads],
    )
    consuming_bundle.write_consuming_bundle(
        consuming,
        _info(),
        [
            consuming_bundle.ConsumingBundleSection.from_bytes(name, data)
            for name, data in payloads
        ],
    )

    assert consuming.read_bytes() == traditional.read_bytes()


@pytest.mark.parametrize(
    "event",
    (
        "after_section_write_fsync:first_plan",
        "after_section_range_verify:first_plan",
        "after_journal_commit:first_plan",
        "after_source_unlink:first_plan",
        "before_final_replace",
        "after_final_replace",
    ),
)
def test_each_durable_failure_window_resumes_without_rebuilding_committed_prefix(
    tmp_path: Path,
    event: str,
) -> None:
    output = tmp_path / "h3.bundle"
    output.write_bytes(b"old-final")
    first = tmp_path / "first.plan"
    second = tmp_path / "second.plan"
    sections = [
        _file_section(first, "first_plan", b"first-qualified-plan"),
        _file_section(second, "second_plan", b"second-qualified-plan"),
        consuming_bundle.ConsumingBundleSection.from_bytes("config.json", b"{}"),
    ]

    def fail(actual: str) -> None:
        if actual == event:
            raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected interruption"):
        consuming_bundle.write_consuming_bundle(
            output,
            _info(),
            sections,
            failure_injector=fail,
        )

    if event != "after_final_replace":
        assert output.read_bytes() == b"old-final"
    assert consuming_bundle.write_consuming_bundle(output, _info(), sections) == output
    _header, payloads = read_bundle_file(output)
    assert payloads["first_plan"] == b"first-qualified-plan"
    assert payloads["second_plan"] == b"second-qualified-plan"
    assert not first.exists()
    assert not second.exists()


def test_uncommitted_tail_is_verified_then_truncated_on_resume(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")

    def fail(event: str) -> None:
        if event == "after_section_write_fsync:engine_plan":
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        consuming_bundle.write_consuming_bundle(
            output,
            _info(),
            [section],
            failure_injector=fail,
        )
    partial, _journal = consuming_bundle.assembly_paths(output)
    with partial.open("ab") as stream:
        stream.write(b"torn-tail")

    consuming_bundle.write_consuming_bundle(output, _info(), [section])
    _header, payloads = read_bundle_file(output)
    assert payloads == {"engine_plan": b"qualified-plan"}


def test_exact_uncommitted_range_recovers_even_if_its_source_is_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")

    def fail(event: str) -> None:
        if event == "after_section_range_verify:engine_plan":
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        consuming_bundle.write_consuming_bundle(
            output,
            _info(),
            [section],
            failure_injector=fail,
        )
    plan.unlink()

    consuming_bundle.write_consuming_bundle(output, _info(), [section])
    _header, payloads = read_bundle_file(output)
    assert payloads == {"engine_plan": b"qualified-plan"}


def test_changed_source_is_never_deleted_or_published(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    output.write_bytes(b"old-final")
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")
    plan.write_bytes(b"tampered-plan!")

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="SHA-256"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])

    assert plan.read_bytes() == b"tampered-plan!"
    assert output.read_bytes() == b"old-final"


def test_changed_source_after_journal_commit_is_not_unlinked(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")

    def fail(event: str) -> None:
        if event == "after_journal_commit:engine_plan":
            raise RuntimeError("stop before unlink")

    with pytest.raises(RuntimeError, match="stop before unlink"):
        consuming_bundle.write_consuming_bundle(
            output,
            _info(),
            [section],
            failure_injector=fail,
        )
    plan.write_bytes(b"unrelated-file")

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="Refusing to remove"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert plan.read_bytes() == b"unrelated-file"
    assert not output.exists()


def test_invalid_existing_final_stays_untouched_when_source_is_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "h3.bundle"
    output.write_bytes(b"invalid-existing-final")
    missing = tmp_path / "missing.plan"
    section = consuming_bundle.ConsumingBundleSection.from_file(
        "engine_plan",
        missing,
        size=4,
        sha256=hashlib.sha256(b"plan").hexdigest(),
        consume_source=True,
    )

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="unavailable"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert output.read_bytes() == b"invalid-existing-final"


def test_complete_fast_h3_ref2va_layout_has_61_plan_sections_and_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "h3-61.bundle"
    plan_sections = []
    for index in range(61):
        payload = f"plan-{index:02d}".encode()
        plan_sections.append(
            _file_section(
                tmp_path / f"plan-{index:02d}.plan",
                f"plan_{index:02d}",
                payload,
            )
        )
    sections = [
        *plan_sections,
        consuming_bundle.ConsumingBundleSection.from_bytes(
            "tokenizer.json", b'{"model": {}}'
        ),
        consuming_bundle.ConsumingBundleSection.from_bytes("config.json", b"{}"),
    ]

    consuming_bundle.write_consuming_bundle(output, _info(), sections)
    header, payloads = read_bundle_file(output)
    assert len(header["sections"]) == 63
    assert len([name for name in payloads if name.startswith("plan_")]) == 61
    assert all(not (tmp_path / f"plan-{index:02d}.plan").exists() for index in range(61))


def test_partial_and_journal_reject_non_regular_files(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"plan")
    partial, journal = consuming_bundle.assembly_paths(output)
    partial.mkdir()

    with pytest.raises(consuming_bundle.ConsumingBundleError):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert plan.is_file()
    partial.rmdir()
    journal.mkdir()
    with pytest.raises(consuming_bundle.ConsumingBundleError, match="journal exists"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert plan.is_file()


def test_source_and_final_reject_non_regular_files(tmp_path: Path) -> None:
    source = tmp_path / "engine.plan"
    source.mkdir()
    section = consuming_bundle.ConsumingBundleSection.from_file(
        "engine_plan",
        source,
        size=4,
        sha256=hashlib.sha256(b"plan").hexdigest(),
        consume_source=True,
    )
    output = tmp_path / "h3.bundle"

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="wrong type"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert source.is_dir()

    output.mkdir()
    plan = tmp_path / "regular.plan"
    regular = _file_section(plan, "engine_plan", b"plan")
    with pytest.raises(consuming_bundle.ConsumingBundleError, match="not a regular"):
        consuming_bundle.write_consuming_bundle(output, _info(), [regular])
    assert plan.is_file()


def test_source_cannot_alias_output_or_another_section(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    output.write_bytes(b"plan")
    output_section = consuming_bundle.ConsumingBundleSection.from_file(
        "engine_plan",
        output,
        size=4,
        sha256=hashlib.sha256(b"plan").hexdigest(),
        consume_source=True,
    )
    with pytest.raises(consuming_bundle.ConsumingBundleError, match="aliases"):
        consuming_bundle.write_consuming_bundle(output, _info(), [output_section])
    assert output.read_bytes() == b"plan"

    source = tmp_path / "shared.plan"
    first = _file_section(source, "first_plan", b"shared-plan")
    second = consuming_bundle.ConsumingBundleSection.from_file(
        "second_plan",
        source,
        size=first.size,
        sha256=first.sha256,
        consume_source=True,
    )
    destination = tmp_path / "different.bundle"
    with pytest.raises(consuming_bundle.ConsumingBundleError, match="reuse"):
        consuming_bundle.write_consuming_bundle(destination, _info(), [first, second])
    assert source.read_bytes() == b"shared-plan"


def test_valid_final_rejects_non_regular_stale_journal(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")
    consuming_bundle.write_consuming_bundle(output, _info(), [section])
    original = output.read_bytes()
    _partial, journal = consuming_bundle.assembly_paths(output)
    journal.mkdir()

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="not a regular"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema_version", True),
        ("assembly_sha256", 7),
        ("committed_count", True),
        ("committed_count", "1"),
        ("committed_count", -1),
        ("committed_payload_bytes", False),
        ("committed_payload_bytes", "4"),
        ("committed_payload_bytes", -1),
        ("state_sha256", []),
    ),
)
def test_malformed_journal_types_are_reconstructed_without_raw_exceptions(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")

    def fail(event: str) -> None:
        if event == "after_section_write_fsync:engine_plan":
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        consuming_bundle.write_consuming_bundle(
            output,
            _info(),
            [section],
            failure_injector=fail,
        )
    _partial, journal = consuming_bundle.assembly_paths(output)
    value = json.loads(journal.read_text(encoding="utf-8"))
    value[field] = invalid_value
    if field != "state_sha256":
        unsigned = {key: item for key, item in value.items() if key != "state_sha256"}
        value["state_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    journal.write_text(json.dumps(value), encoding="utf-8")

    consuming_bundle.write_consuming_bundle(output, _info(), [section])
    _header, payloads = read_bundle_file(output)
    assert payloads == {"engine_plan": b"qualified-plan"}
    assert not plan.exists()


@pytest.mark.parametrize(
    "raw_journal",
    (
        b"\xff\xfe",
        b'{"schema_version": 1, "schema_version": 1}',
        b"{" + (b" " * ((1 << 20) + 1)),
    ),
    ids=("invalid-utf8", "duplicate-key", "oversized"),
)
def test_malformed_journal_encoding_duplicates_and_size_are_bounded(
    tmp_path: Path,
    raw_journal: bytes,
) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")

    def fail(event: str) -> None:
        if event == "after_section_write_fsync:engine_plan":
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        consuming_bundle.write_consuming_bundle(
            output,
            _info(),
            [section],
            failure_injector=fail,
        )
    _partial, journal = consuming_bundle.assembly_paths(output)
    journal.write_bytes(raw_journal)

    consuming_bundle.write_consuming_bundle(output, _info(), [section])
    _header, payloads = read_bundle_file(output)
    assert payloads == {"engine_plan": b"qualified-plan"}


def test_source_symlink_is_rejected_and_target_is_never_deleted(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    target = tmp_path / "actual.plan"
    target.write_bytes(b"qualified-plan")
    source = tmp_path / "engine.plan"
    _symlink_or_skip(source, target)
    section = consuming_bundle.ConsumingBundleSection.from_file(
        "engine_plan",
        source,
        size=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        consume_source=True,
    )

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="wrong type"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert target.read_bytes() == b"qualified-plan"
    assert source.is_symlink()


def test_partial_symlink_is_rejected_without_touching_its_target(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")
    partial, _journal = consuming_bundle.assembly_paths(output)
    target = tmp_path / "unrelated.partial"
    target.write_bytes(b"unrelated")
    _symlink_or_skip(partial, target)

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="not regular"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert target.read_bytes() == b"unrelated"
    assert plan.is_file()


def test_journal_symlink_is_rejected_without_touching_its_target(tmp_path: Path) -> None:
    output = tmp_path / "h3.bundle"
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")

    def fail(event: str) -> None:
        if event == "after_partial_header_fsync":
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        consuming_bundle.write_consuming_bundle(
            output,
            _info(),
            [section],
            failure_injector=fail,
        )
    _partial, journal = consuming_bundle.assembly_paths(output)
    target = tmp_path / "unrelated-journal.json"
    target.write_text("{}", encoding="utf-8")
    _symlink_or_skip(journal, target)

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="not a regular"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert target.read_text(encoding="utf-8") == "{}"
    assert plan.is_file()


def test_final_symlink_is_rejected_without_touching_its_target(tmp_path: Path) -> None:
    target = tmp_path / "unrelated-final"
    target.write_bytes(b"unrelated")
    output = tmp_path / "h3.bundle"
    _symlink_or_skip(output, target)
    plan = tmp_path / "engine.plan"
    section = _file_section(plan, "engine_plan", b"qualified-plan")

    with pytest.raises(consuming_bundle.ConsumingBundleError, match="not a regular"):
        consuming_bundle.write_consuming_bundle(output, _info(), [section])
    assert target.read_bytes() == b"unrelated"
    assert plan.is_file()
