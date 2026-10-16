# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise portable Community reports across evidence and recovery trust boundaries."""

import base64
import copy
import hashlib
from html.parser import HTMLParser
import io
import json
import os
import tarfile
import wave

import pytest

from tools import community_gpu_report as report

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMQMgkDAAD4AJ3MaiF4AAAAAElFTkSuQmCC"
)
# A 32x32 two-color VP8 clip generated with Chromium MediaRecorder. This is
# renderer test data, not an output or qualification receipt from a model.
WEBM = base64.b64decode(
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQRChYECGFOAZwH/////////FUmpZpkq17GDD0JATYCGQ2hy"
    "b21lV0GGQ2hyb21lFlSua6mup9eBAXPFhz61E8vMwEWDgQFV7oEBhoVWX1ZQOOCKsIEguoEgU8CBAR9DtnUB////"
    "/////+eBAKDtocCBAAAA8AIAnQEqIAAgAAAHCIWFiJmEiAEkEBOtUB/AEEEJTPOA/v5z5f/20D+2gf20D/7aB/+7"
    "J/tk/2yf+4QAdaGopqbugQGloVACAJ0BKiAAIAAABwiFhYiZhIgBJBAAPIeeTgD+/7qDAKDQoa+BAFAAMQIAABAJ"
    "IADAOcBBC9OWagHX+AD+/LhL/9Cfig6Mf9Ch//Uav1QG//qDwHWhmaaX7oEBpZLRAQAAEAkgAMAAwsF/oABDgAD7"
    "gQCg0aGwgQCfAPEBAAAQCSAAwADyv6BgAghwAP7+c+X/9s/8KQYv/toH/7skS/7JEv+yf+4QdaGZppfugQGlktEB"
    "AAAQCSAAwADCwX+gAEOAAPuBUKDOoa2BAO8A8QEAABAJIADAAPK/oGABqDgA/vy4S//Qn4oOjH/Qof/1Gr9UBv/6"
    "g8B1oZmml+6BAaWS0QEAABAJIADAAMLBf6AAQ4AA+4GfoNGhsIEBPwDxAQAAEAkgAMAA8r+gYAIIcAD+/nPl//bP"
    "/CkGL/7aB/+7JEv+yRL/sn/uEHWhmaaX7oEBpZLRAQAAEAkgAMAAwsF/oABDgAD7ge+gz6GtgQGPAPEBAAAQCSAA"
    "wADyv6BgAag4AP78uEv/0J+KDox/0KH/9Rq/VAb/+oPAdaGZppfugQGlktEBAAAQCSAAwADCwX+gAEOAAPuCAT+g"
    "0qGwgQHfAPEBAAAQCSAAwADyv6BgAghwAP7+c+X/9s/8KQYv/toH/7skS/7JEv+yf+4QdaGZppfugQGlktEBAAAQ"
    "CSAAwADCwX+gAEOAAPuCAY8="
)


@pytest.fixture
def metadata():
    return {
        "repository": "example/project",
        "run_id": 123,
        "run_attempt": 1,
        "workflow_sha": "a" * 40,
        "source_sha": "b" * 40,
        "head_sha": "c" * 40,
        "base_sha": "d" * 40,
        "pr_number": 17,
        "lane": "dev",
        "cpu_status": "success",
        "failure_class": "none",
        "cleanup_confirmed": True,
        "no_allocation": False,
    }


@pytest.fixture
def receipt(metadata):
    return {
        "schema_version": 1,
        **{key: metadata[key] for key in ("repository", "run_id", "run_attempt", "workflow_sha")},
        "leases": [
            {
                "schema_version": 1,
                "name": "trtmc-gpu-ci-123-1",
                "sku": "gpu-fixture.1",
                "organization_id": "org-fixture",
                "instance_id": "vm-fixture",
                "create_started": True,
                "create_accepted": True,
                "allocation_pending": False,
                "cleanup_confirmed": True,
                "delete_accepted": True,
                "phase": "deleted",
                "run_id": "123",
                "run_attempt": "1",
            }
        ],
    }


def summary(status="passed", *, entered=True, category=None, phase="complete", evidence=""):
    return {
        "schema_version": 1,
        "duration_seconds": 12,
        "passed": status == "passed",
        "complete": status in {"passed", "failed"},
        "families": [
            {
                "family": "alpha",
                "status": "passed" if status == "passed" else "failed",
                "phase": phase,
                "entrypoint_started": entered,
                "failure_class": category,
                "evidence": evidence,
                "cases": {"smoke": status},
                "requested_cases": ["smoke"],
                "deferred_cases": [],
            }
        ],
    }


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.scripts = []
        self.current = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "script":
            self.current = {"attrs": attrs, "text": ""}
            self.scripts.append(self.current)

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None

    def recovery(self):
        return json.loads(
            next(
                s["text"]
                for s in self.scripts
                if s["attrs"].get("id") == "trtmc-community-recovery"
            )
        )

    def outcomes(self):
        return [
            attrs["data-execution-status"]
            for tag, attrs in self.tags
            if tag == "section" and attrs.get("class") == "case"
        ]


def evidence(root, metadata, *, status="passed"):
    directory = root / "evidence/alpha/smoke"
    (directory / "artifacts").mkdir(parents=True)
    (directory / "artifacts/sample.png").write_bytes(PNG)
    data = {
        "schema_version": 1,
        "family": "alpha",
        "case": "smoke",
        "status": status,
        "source_revision": metadata["source_sha"],
        "workflow_run_attempt": 1,
        "inputs": {"prompt": "a recorded prompt"},
        "native": {"text": "actual output", "artifact": "artifacts/sample.png"},
        "reference": {"text": "expected output"},
        "checks": [
            {
                "status": "passed",
                "expression": "measured >= threshold",
                "explanation": "0.995 >= 0.99",
            }
        ],
        "repro": {
            "command": "python -m pytest families/alpha/tests/test_e2e.py --e2e-testcase smoke"
        },
        "artifacts": [
            {"path": "artifacts/sample.png", "role": "native", "media_type": "image/png"}
        ],
    }
    raw = json.dumps(data).encode()
    (directory / "evidence.json").write_bytes(raw)
    index = {
        "schema_version": 1,
        "source_revision": metadata["source_sha"],
        "workflow_run_id": 123,
        "workflow_run_attempt": 1,
        "cases": [
            {
                "family": "alpha",
                "case": "smoke",
                "status": "collected",
                "path": "evidence/alpha/smoke/evidence.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "artifacts": [
                    {
                        "path": "evidence/alpha/smoke/artifacts/sample.png",
                        "sha256": hashlib.sha256(PNG).hexdigest(),
                        "size_bytes": len(PNG),
                    }
                ],
                "issues": [],
            }
        ],
    }
    (root / "evidence-index.json").write_text(json.dumps(index))
    return data, index


def update_evidence(root, data, index):
    raw = json.dumps(data).encode()
    (root / "evidence/alpha/smoke/evidence.json").write_bytes(raw)
    index["cases"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "evidence-index.json").write_text(json.dumps(index))


def test_portable_report_reuses_recorded_checks_media_and_exact_recovery(
    tmp_path, metadata, receipt
):
    evidence(tmp_path, metadata)
    before = copy.deepcopy(receipt)
    text = report.render_report(summary(), metadata, tmp_path, receipt)
    parsed = Document(text)
    assert parsed.recovery() == before == receipt
    assert parsed.outcomes() == ["passed"]
    assert "data:image/png;base64," + base64.b64encode(PNG).decode() in text
    assert "measured &gt;= threshold" in text and "0.995 &gt;= 0.99" in text
    assert "python -m pytest families/alpha/tests/test_e2e.py" in text
    assert "actual output" in text and "expected output" in text
    assert "Selected GPU tests passed" in text and "VM release confirmed" in text
    assert not any(attrs.get("src", "").startswith(("http:", "https:")) for _, attrs in parsed.tags)
    assert len(parsed.scripts) == 2
    policy = next(
        attrs["content"]
        for tag, attrs in parsed.tags
        if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy"
    )
    assert "connect-src 'none'" in policy and "script-src 'sha256-" in policy
    assert len(text.encode()) < report.MAX_HTML_BYTES


@pytest.mark.parametrize(
    "filename,tag,media_type",
    [
        ("sample.png", "img", "image/png"),
        ("sample.wav", "audio", "audio/wav"),
        ("sample.webm", "video", "video/webm"),
    ],
)
def test_recorder_media_survives_collection_and_inline_rendering(
    tmp_path, metadata, receipt, filename, tag, media_type
):
    from tools.community_gpu_ci import collect_family_evidence
    from tools.e2e_evidence import Evidence

    media = tmp_path / filename
    if filename.endswith(".wav"):
        with wave.open(str(media), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\x00\x00" * 800)
    else:
        media.write_bytes(PNG if filename.endswith(".png") else WEBM)
    capture, collected = tmp_path / "capture", tmp_path / "collected"
    capture.mkdir()
    collected.mkdir()
    recorder = Evidence(
        capture / "evidence/alpha/smoke",
        family="alpha",
        case="smoke",
        source_revision=metadata["source_sha"],
        roots=(tmp_path,),
    )
    recorder.data["workflow_run_attempt"] = metadata["run_attempt"]
    recorder.record("inputs", {"prompt": "Synthetic renderer fixture"})
    recorder.record("native", {"artifact": media})
    recorder.data["checks"] = [
        {"status": "passed", "expression": "recorded >= threshold", "explanation": "0.995 >= 0.99"}
    ]
    recorder.finish("passed")
    original = (recorder.directory / "evidence.json").read_bytes()
    index = {
        "schema_version": 1,
        "source_revision": metadata["source_sha"],
        "workflow_run_id": metadata["run_id"],
        "workflow_run_attempt": metadata["run_attempt"],
        "cases": [{"family": "alpha", "case": "smoke"}],
        "issues": [],
        "file_count": 0,
        "total_bytes": 0,
    }
    collect_family_evidence(capture, collected, "alpha", index)
    assert index["cases"][0]["status"] == "collected"
    assert index["cases"][0]["issues"] == []
    (collected / "evidence-index.json").write_text(json.dumps(index))
    text = report.render_report(summary(), metadata, collected, receipt)
    parsed = Document(text)
    elements = [attrs for element, attrs in parsed.tags if element == tag]
    assert len(elements) == 1
    assert (
        elements[0]["src"]
        == f"data:{media_type};base64," + base64.b64encode(media.read_bytes()).decode()
    )
    if tag != "img":
        assert "controls" in elements[0] and "autoplay" not in elements[0]
    downloads = [attrs for element, attrs in parsed.tags if element == "a" and "download" in attrs]
    assert len(downloads) == 1
    assert downloads[0]["download"] == "0000-" + filename
    assert base64.b64decode(downloads[0]["href"].split(",", 1)[1]) == media.read_bytes()
    assert "recorded &gt;= threshold" in text and "0.995 &gt;= 0.99" in text
    assert parsed.outcomes() == ["passed"] and parsed.recovery() == receipt
    assert "Partial evidence" not in text
    assert (collected / index["cases"][0]["path"]).read_bytes() == original


@pytest.mark.parametrize("detail_error", [False, True])
def test_media_without_a_selected_preview_remains_downloadable(
    tmp_path, metadata, receipt, detail_error
):
    data, index = evidence(tmp_path, metadata, status="failed")
    # A case can fail after recording auxiliary media but before producing any
    # input/output preview. The rich layout has nowhere to display this legend.
    data.update(
        inputs={},
        native=None,
        reference=None,
        views=[{"title": "Legend", "image": {"artifact": "artifacts/sample.png"}}],
    )
    data["artifacts"][0]["role"] = "views"
    if detail_error:
        data["checks"] = 1
    update_evidence(tmp_path, data, index)
    text = report.render_report(
        summary("failed", phase="validation", category="pr_failure"), metadata, tmp_path, receipt
    )
    parsed = Document(text)
    assert not any(tag == "img" for tag, _ in parsed.tags)
    download = next(attrs for tag, attrs in parsed.tags if tag == "a" and "download" in attrs)
    assert download["download"] == "sample.png"
    assert base64.b64decode(download["href"].split(",", 1)[1]) == PNG
    assert parsed.outcomes() == ["failed"]
    assert ("Partial evidence" in text) is detail_error


@pytest.mark.parametrize("diagnostic", [False, "true", 1])
def test_missing_receipt_requires_explicit_diagnostic_mode(metadata, diagnostic):
    metadata["diagnostic_only"] = diagnostic
    with pytest.raises(ValueError):
        report.render_report(None, metadata, None, None)


@pytest.mark.parametrize("released", [False, True])
@pytest.mark.parametrize("no_allocation", [False, True])
def test_diagnostic_report_never_substitutes_for_results_or_recovery_authority(
    tmp_path, metadata, released, no_allocation
):
    evidence(tmp_path, metadata)
    metadata.update(
        diagnostic_only=True,
        cleanup_confirmed=released,
        no_allocation=no_allocation,
        owner_result="cancelled",
        report_upload_result="skipped",
        recovery_result="success" if released else "failure",
    )
    # Even available passing evidence cannot turn a fallback diagnostic into an
    # authoritative GPU result or a replacement recovery journal.
    text = report.render_report(summary(), metadata, tmp_path, None)
    parsed = Document(text)
    assert parsed.outcomes() == []
    assert len(parsed.scripts) == 1
    assert "trtmc-community-recovery" not in text
    assert "The GPU report is incomplete" in text and "No GPU pass is inferred" in text
    assert "Selected GPU tests passed" not in text and "data:image/" not in text
    assert "original operational journals remain necessary" in text
    assert "Resource owner" in text and "cancelled" in text and "skipped" in text
    assert ("Independent recovery confirmed release" in text) is released
    assert ("Resource recovery remains unconfirmed" in text) is not released
    assert ("This attempt did not request an allocation." in text) is (released and no_allocation)
    assert metadata["source_sha"] in text and metadata["head_sha"] in text


def test_diagnostic_mode_cannot_embed_a_recovery_receipt(metadata, receipt):
    metadata["diagnostic_only"] = True
    with pytest.raises(report.ReportError, match="cannot contain recovery authority"):
        report.render_report(None, metadata, None, receipt)


@pytest.mark.parametrize("field", ["summary", "detail"])
def test_lone_surrogates_only_change_display_not_original_evidence_or_recovery(
    tmp_path, metadata, receipt, field
):
    text_value = "café 中文 😀 lone surrogate: \ud800"
    data, index = evidence(tmp_path, metadata, status="failed")
    result = summary("failed", phase="validation", category="pr_failure")
    if field == "summary":
        result["families"][0]["evidence"] = text_value
    else:
        data["native"]["text"] = text_value
    update_evidence(tmp_path, data, index)
    receipt["leases"][0]["note"] = text_value
    original_receipt = copy.deepcopy(receipt)
    evidence_path = tmp_path / index["cases"][0]["path"]
    original_evidence = evidence_path.read_bytes()

    text = report.render_report(result, metadata, tmp_path, receipt)
    text.encode("utf-8", errors="strict")
    assert "café 中文 😀" in text and "\\ud800" in text and "\ud800" not in text
    parsed = Document(text)
    assert parsed.outcomes() == ["failed"]
    assert parsed.recovery() == original_receipt == receipt
    download = next(attrs for tag, attrs in parsed.tags if tag == "a" and "download" in attrs)
    assert base64.b64decode(download["href"].split(",", 1)[1]) == PNG
    assert evidence_path.read_bytes() == original_evidence
    assert hashlib.sha256(original_evidence).hexdigest() == index["cases"][0]["sha256"]


@pytest.mark.parametrize(
    "authoritative,detail", [("failed", "passed"), ("not_run", "passed"), ("passed", "failed")]
)
def test_case_verdict_cannot_be_overridden_by_detail(
    tmp_path, metadata, receipt, authoritative, detail
):
    data, index = evidence(tmp_path, metadata, status=detail)
    data["checks"][0]["status"] = detail
    update_evidence(tmp_path, data, index)
    text = report.render_report(
        summary(
            authoritative,
            category="pr_failure" if authoritative != "passed" else None,
            phase="validation",
        ),
        metadata,
        tmp_path,
        receipt,
    )
    assert Document(text).outcomes() == [authoritative]
    assert "Recorded detail status differs" in text
    if authoritative != "passed":
        assert "Selected GPU tests passed" not in text


@pytest.mark.parametrize(
    "field",
    [
        "index_source",
        "index_run",
        "index_attempt",
        "raw_source",
        "raw_attempt",
        "raw_family",
        "raw_case",
        "digest",
        "duplicate_case",
    ],
)
def test_stale_or_ambiguous_evidence_is_never_attached(tmp_path, metadata, receipt, field):
    data, index = evidence(tmp_path, metadata)
    data["native"]["text"] = "UNTRUSTED_STALE_DETAIL"
    if field == "index_source":
        index["source_revision"] = "f" * 40
    elif field == "index_run":
        index["workflow_run_id"] = 124
    elif field == "index_attempt":
        index["workflow_run_attempt"] = 2
    elif field == "raw_source":
        data["source_revision"] = "f" * 40
    elif field == "raw_attempt":
        data["workflow_run_attempt"] = 2
    elif field == "raw_family":
        data["family"] = "beta"
    elif field == "raw_case":
        data["case"] = "other"
    elif field == "duplicate_case":
        index["cases"].append(copy.deepcopy(index["cases"][0]))
    update_evidence(tmp_path, data, index)
    if field == "digest":
        index["cases"][0]["sha256"] = "f" * 64
        (tmp_path / "evidence-index.json").write_text(json.dumps(index))
    text = report.render_report(summary(), metadata, tmp_path, receipt)
    assert "UNTRUSTED_STALE_DETAIL" not in text and "data:image/png;base64," not in text
    assert Document(text).outcomes() == ["passed"]
    assert "Partial evidence" in text


def test_pr_strings_and_recovery_json_cannot_create_active_markup(tmp_path, metadata, receipt):
    data, index = evidence(tmp_path, metadata)
    attack = '</script><script src="https://attacker.invalid/run.js"></script><img src=x onerror=alert(1)>'
    data["native"]["text"] = attack
    data["checks"][0]["explanation"] = attack
    data["repro"]["command"] = "$(cat /root/secret) " + attack
    update_evidence(tmp_path, data, index)
    receipt["leases"][0]["note"] = attack
    text = report.render_report(summary(), metadata, tmp_path, receipt)
    parsed = Document(text)
    assert len(parsed.scripts) == 2
    assert parsed.recovery() == receipt
    assert "&lt;script" in text and "\\u003c/script\\u003e" in text
    assert not any(key.startswith("on") for _, attrs in parsed.tags for key in attrs)
    assert not any(attrs.get("src", "").startswith("https://attacker") for _, attrs in parsed.tags)
    assert "$(cat /root/secret)" in text


@pytest.mark.parametrize(
    "kind",
    [
        "symlink",
        "parent_symlink",
        "hardlink",
        "fifo",
        "traversal",
        "absolute",
        "html",
        "mislabeled_media",
    ],
)
def test_artifacts_cannot_read_host_files_or_embed_active_content(
    tmp_path, metadata, receipt, kind
):
    root = tmp_path / "evidence-root"
    root.mkdir()
    data, index = evidence(root, metadata)
    target = root / "evidence/alpha/smoke/artifacts/sample.png"
    external = tmp_path / "private.png"
    private_bytes = PNG + b"PRIVATE_HOST_FILE_MUST_NOT_BE_EMBEDDED"
    external.write_bytes(private_bytes)
    item = index["cases"][0]["artifacts"][0]
    item.update(sha256=hashlib.sha256(private_bytes).hexdigest(), size_bytes=len(private_bytes))
    target.unlink()
    if kind == "symlink":
        target.symlink_to(external)
    elif kind == "hardlink":
        os.link(external, target)
    elif kind == "fifo":
        os.mkfifo(target)
    elif kind == "parent_symlink":
        directory = target.parent
        directory.rmdir()
        directory.symlink_to(tmp_path, target_is_directory=True)
        item["path"] = "evidence/alpha/smoke/artifacts/private.png"
    elif kind == "traversal":
        item["path"] = "evidence/alpha/smoke/../../../../private.png"
    elif kind == "absolute":
        item["path"] = str(external)
    elif kind == "html":
        item["path"] = "evidence/alpha/smoke/artifacts/payload.html"
        (target.parent / "payload.html").write_bytes(private_bytes)
    elif kind == "mislabeled_media":
        target.write_bytes(b"<svg onload='alert(1)'/>")
        item.update(
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(), size_bytes=target.stat().st_size
        )
    data["native"]["artifact"] = str(external)
    data["reference"] = {"artifact": "../../../../private.npy", "shape": [1]}
    update_evidence(root, data, index)
    text = report.render_report(summary(), metadata, root, receipt)
    assert base64.b64encode(private_bytes).decode() not in text
    assert "PRIVATE_HOST_FILE_MUST_NOT_BE_EMBEDDED" not in text
    assert "data:image/png;base64," not in text
    assert Document(text).outcomes() == ["passed"]
    assert "artifact was unavailable" in text


def test_missing_oom_or_preentry_details_cannot_turn_into_pass(tmp_path, metadata, receipt):
    metadata["failure_class"] = "pr_failure"
    text = report.render_report(
        summary(
            "not_run",
            category="pr_failure",
            phase="validation",
            evidence="Docker reported OOMKilled for this container",
        ),
        metadata,
        None,
        receipt,
    )
    assert "OOMKilled" in text and "GPU validation failed or remains incomplete" in text
    assert Document(text).outcomes() == ["not_run"]
    metadata["failure_class"] = "infra_failure"
    text = report.render_report(
        summary(
            "not_run",
            entered=False,
            category="infra_failure",
            phase="checkpoints",
            evidence="checkpoint staging deadline reached",
        ),
        metadata,
        None,
        receipt,
    )
    assert "infra_failure" in text and "checkpoint staging deadline reached" in text
    assert "Selected GPU tests passed" not in text


@pytest.mark.parametrize(
    "gpu_status,cpu_status", [("not_started", "failure"), ("not_required", "success")]
)
@pytest.mark.parametrize("released", [False, True])
def test_cpu_only_no_allocation_is_not_a_gpu_pass(
    metadata, receipt, gpu_status, cpu_status, released
):
    metadata.update(
        gpu_status=gpu_status,
        cpu_status=cpu_status,
        failure_class="cpu_failure" if cpu_status == "failure" else "none",
        no_allocation=True,
        cleanup_confirmed=released,
    )
    receipt["leases"] = []
    # Even an accidentally supplied stale passing summary cannot claim GPU execution.
    text = report.render_report(summary(), metadata, None, receipt)
    assert "Selected GPU tests passed" not in text
    assert Document(text).outcomes() == []
    if not released:
        assert "earlier allocation recovery remains unconfirmed" in text
    assert Document(text).recovery() == receipt


def test_rejected_create_with_authenticated_absence_counts_as_release(metadata, receipt):
    lease = receipt["leases"][0]
    lease.update(
        phase="unallocated",
        instance_id=None,
        create_started=False,
        create_accepted=False,
        delete_accepted=False,
    )
    text = report.render_report(None, metadata, None, receipt)
    assert "Resource release confirmed: no instance was allocated" in text
    assert "VM release is not confirmed" not in text


@pytest.mark.parametrize("field", ["run_id", "run_attempt", "workflow_sha", "lease_name", "empty"])
def test_recovery_authority_cannot_be_replaced(metadata, receipt, field):
    if field == "lease_name":
        receipt["leases"][0]["name"] = "someone-elses-vm"
    elif field == "empty":
        receipt["leases"] = []
    elif field == "workflow_sha":
        receipt[field] = "f" * 40
    else:
        receipt[field] += 1
    with pytest.raises((ValueError, RuntimeError)):
        report.render_report(None, metadata, None, receipt)


def test_media_budget_and_final_size_limit_are_enforced(tmp_path, metadata, receipt, monkeypatch):
    evidence(tmp_path, metadata)
    monkeypatch.setattr(report, "MAX_ASSET_TOTAL", len(PNG) - 1)
    text = report.render_report(summary(), metadata, tmp_path, receipt)
    assert "data:image/png;base64," not in text and "Partial evidence" in text
    monkeypatch.setattr(report, "MAX_HTML_BYTES", 1024)
    with pytest.raises(report.ReportError, match="final HTML size"):
        report.render_report(summary(), metadata, tmp_path, receipt)


def archive(root, output, *, bad_member=None):
    with tarfile.open(output, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
        if bad_member is not None:
            if bad_member == "symlink":
                member = tarfile.TarInfo("evidence/alpha/smoke/artifacts/secret.png")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                tar.addfile(member)
            else:
                payload = b"<script>UNTRUSTED_ARCHIVE_HTML</script>"
                member = tarfile.TarInfo(bad_member)
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))


@pytest.mark.parametrize(
    "bad_member",
    [None, "symlink", "../../outside", "evidence/alpha/smoke/report.html", "evidence-index.json"],
)
def test_cli_uses_only_bound_regular_archive_members(tmp_path, metadata, receipt, bad_member):
    root = tmp_path / "input"
    root.mkdir()
    evidence(root, metadata)
    # The archive's summary must never replace the trusted separate summary.
    (root / "summary.json").write_text(json.dumps(summary("failed", category="pr_failure")))
    payload = tmp_path / "evidence.tar.gz"
    archive(root, payload, bad_member=bad_member)
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary()))
    run = tmp_path / "run.json"
    run.write_text(json.dumps(metadata))
    journal = tmp_path / "recovery.json"
    journal.write_text(json.dumps(receipt))
    destination = tmp_path / "report.html"
    assert (
        report.main(
            [
                "--summary",
                str(source),
                "--run-metadata",
                str(run),
                "--evidence-archive",
                str(payload),
                "--recovery-receipt",
                str(journal),
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    text = destination.read_text()
    assert Document(text).outcomes() == ["passed"]
    assert Document(text).recovery() == receipt
    assert "UNTRUSTED_ARCHIVE_HTML" not in text
    assert ("data:image/png;base64," in text) is (bad_member is None)
    if bad_member is not None:
        assert "archive is missing or invalid" in text
    assert not (tmp_path / "outside").exists()


def test_duplicate_or_nonfinite_json_is_rejected():
    for raw in (b'{"status":"passed","status":"failed"}', b'{"value":NaN}', b'{"value":1e999}'):
        with pytest.raises(report.ReportError):
            report._json(raw)


def test_existing_public_renderer_default_assessment_is_unchanged(tmp_path):
    from tools.e2e_report import render_case

    text = render_case({"family": "example", "case": "case", "status": "passed"}, tmp_path)
    assert "Not verified" in text


def test_plan_policy_has_actionable_headline_without_inventing_gpu_results(metadata, receipt):
    metadata.update(failure_class="plan_rejected", no_allocation=True)
    receipt["leases"] = []
    text = report.render_report(None, metadata, None, receipt)
    assert "GPU plan rejected before allocation" in text
    assert "at most five directly changed or added families" in text
    assert "Selected GPU tests passed" not in text
    assert Document(text).outcomes() == []


def test_execution_log_is_escaped_bounded_and_cannot_replace_recovery(metadata, receipt):
    attack = '</script><script id="trtmc-community-recovery">wrong</script>'
    text = report.render_report(None, metadata, None, receipt, execution_log=attack)
    assert len(Document(text).scripts) == 2 and Document(text).recovery() == receipt
    assert "&lt;/script&gt;" in text
    assert "Preparation or execution ended without a valid coordinator receipt" in text
    text = report.render_report(
        None,
        metadata,
        None,
        receipt,
        execution_log="DISCARDED_PREFIX" + "x" * report.MAX_EXECUTION_LOG_BYTES + "KEPT_TAIL",
    )
    assert "DISCARDED_PREFIX" not in text and "KEPT_TAIL" in text
    assert "Log truncated: only the final 1 MiB" in text
    assert len(text.encode()) < report.MAX_HTML_BYTES


@pytest.mark.parametrize("present", [False, True])
def test_cli_keeps_only_the_bounded_execution_log_tail(tmp_path, metadata, receipt, present):
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(receipt))
    log = tmp_path / "execution.log"
    if present:
        log.write_text("DO_NOT_KEEP_PREFIX" + "z" * report.MAX_EXECUTION_LOG_BYTES + "ACTUAL_TAIL")
    output = tmp_path / "report.html"
    assert (
        report.main(
            [
                "--summary",
                str(tmp_path / "missing-summary.json"),
                "--run-metadata",
                str(metadata_file),
                "--recovery-receipt",
                str(receipt_file),
                "--execution-log",
                str(log),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    text = output.read_text()
    assert Document(text).recovery() == receipt
    if present:
        assert "DO_NOT_KEEP_PREFIX" not in text and "ACTUAL_TAIL" in text
        assert "Log truncated" in text
    else:
        assert "Execution log is unavailable" in text


@pytest.mark.parametrize("kind", [tarfile.GNUTYPE_SPARSE, tarfile.CONTTYPE])
def test_archive_rejects_sparse_and_extended_contiguous_files(tmp_path, kind):
    path = tmp_path / "special.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("evidence/alpha/smoke/artifacts/output.raw")
        member.type = kind
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with report._archive_root(path) as (root, issues):
        assert root is None and issues


def test_archive_enforces_aggregate_logical_member_bytes(tmp_path, monkeypatch):
    path = tmp_path / "large.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in ("summary.json", "evidence-index.json"):
            member = tarfile.TarInfo(name)
            member.size = 2
            archive.addfile(member, io.BytesIO(b"{}"))
    monkeypatch.setattr(report, "MAX_ARCHIVE_MEMBER_BYTES", 3)
    with report._archive_root(path) as (root, issues):
        assert root is None and issues
