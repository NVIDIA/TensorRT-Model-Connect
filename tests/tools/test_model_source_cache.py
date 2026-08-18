# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.ci import model_source_cache as model_source_cache_module
from tools.ci.context import CiContext
from tools.ci.model_source_cache import (
    ModelSourcePackagePreparer,
    materialized_tree_sha256,
    parse_model_source_package_contract,
)
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parents[2]


def _archive(path: Path, members: list[tuple[str, str, bytes | str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for kind, name, value in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                payload = bytes(value)
                info.size = len(payload)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = str(value)
                archive.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = str(value)
                archive.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)
            elif kind == "device":
                info.type = tarfile.CHRTYPE
                archive.addfile(info)
            else:  # pragma: no cover - test helper misuse
                raise AssertionError(kind)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(digest: str) -> dict[str, str]:
    return {
        "cache_file": "sam2_hoi/package.tar.gz",
        "sha256": digest,
        "project_path": "artifacts/sam2_hoi/hoi",
        "entrypoint": "SOURCE_COMMIT",
    }


def _prepare(
    tmp_path: Path,
    members: list[tuple[str, str, bytes | str]],
    *,
    digest_override: str | None = None,
) -> tuple[Path, Path, Path]:
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive, members)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    private = tmp_path / "private"
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )
    result = ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
        _contract(digest_override or digest), private, artifacts
    )
    assert result is not None
    return result, artifacts, archive


def _valid_members() -> list[tuple[str, str, bytes | str]]:
    return [
        ("dir", "./sam2", b""),
        ("dir", "./sam2/configs", b""),
        ("dir", "./sam2/configs/sam2", b""),
        ("file", "./SOURCE_COMMIT", b"79ab25d6\n"),
        ("file", "./sam2/configs/sam2/sam2_hiera_s.yaml", b"model: small\n"),
        (
            "symlink",
            "./sam2/sam2_hiera_s.yaml",
            "configs/sam2/sam2_hiera_s.yaml",
        ),
    ]


def test_safe_relative_symlink_is_materialized_as_a_regular_file(tmp_path: Path) -> None:
    result, artifacts, archive = _prepare(tmp_path, _valid_members())

    materialized = result / "sam2/sam2_hiera_s.yaml"
    target = result / "sam2/configs/sam2/sam2_hiera_s.yaml"
    assert materialized.is_file() and not materialized.is_symlink()
    assert materialized.read_bytes() == target.read_bytes()
    assert not [path for path in result.rglob("*") if path.is_symlink()]
    evidence = json.loads((artifacts / "model-source-package.json").read_text(encoding="utf-8"))
    assert evidence == {
        "schema_version": 1,
        "model": "sam2_hoi",
        "isolation": "selected-digest-private",
        "cache_file": "sam2_hoi/package.tar.gz",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "project_path": "artifacts/sam2_hoi/hoi",
        "entrypoint": "SOURCE_COMMIT",
        "container_path": "/src/artifacts/sam2_hoi/hoi",
        "copy_method": "verified-tar-materialization",
        "member_count": 6,
        "regular_file_count": 2,
        "directory_count": 3,
        "materialized_symlink_count": 1,
        "materialized_tree_sha256": materialized_tree_sha256(result),
    }
    assert str(tmp_path / "cache") not in json.dumps(evidence)


def test_repeated_leading_dot_components_normalize_safely(tmp_path: Path) -> None:
    members = _valid_members()
    members[3] = ("file", "./././SOURCE_COMMIT", b"79ab25d6\n")

    result, _, _ = _prepare(tmp_path, members)

    assert (result / "SOURCE_COMMIT").is_file()


@pytest.mark.parametrize("root_name", [".", "./"])
def test_reviewed_archive_inventory_shape_and_four_config_links_are_supported(
    tmp_path: Path,
    root_name: str,
) -> None:
    variants = ("s", "t", "b+", "l")
    members: list[tuple[str, str, bytes | str]] = [
        ("dir", root_name, b""),
        ("dir", "./sam2", b""),
        ("dir", "./sam2/configs", b""),
        ("dir", "./sam2/configs/sam2", b""),
        *(("dir", f"./directory_{index}", b"") for index in range(16)),
        ("file", "./SOURCE_COMMIT", b"79ab25d6\n"),
        *(
            (
                "file",
                f"./sam2/configs/sam2/sam2_hiera_{variant}.yaml",
                f"model: {variant}\n".encode(),
            )
            for variant in variants
        ),
        *(("file", f"./file_{index:02d}.bin", b"x") for index in range(78)),
        *(
            (
                "symlink",
                f"./sam2/sam2_hiera_{variant}.yaml",
                f"configs/sam2/sam2_hiera_{variant}.yaml",
            )
            for variant in variants
        ),
    ]

    result, artifacts, _ = _prepare(tmp_path, members)

    evidence = json.loads((artifacts / "model-source-package.json").read_text(encoding="utf-8"))
    assert len(members) == 107
    assert evidence["member_count"] == 106
    assert evidence["regular_file_count"] == 83
    assert evidence["directory_count"] == 19
    assert evidence["materialized_symlink_count"] == 4
    for variant in variants:
        path = result / f"sam2/sam2_hiera_{variant}.yaml"
        assert path.is_file() and not path.is_symlink()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_root_non_directory_member_is_rejected(tmp_path: Path, kind: str) -> None:
    value: bytes | str = b"root" if kind == "file" else "SOURCE_COMMIT"
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive, [*_valid_members(), (kind, ".", value)])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )

    with pytest.raises(CiError, match="unsafe member path"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private", artifacts
        )


def test_root_directory_markers_count_toward_the_member_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_source_cache_module, "_MAX_ARCHIVE_MEMBERS", 4)
    members = [("dir", ".", b"") for _ in range(5)]
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive, [*members, ("file", "SOURCE_COMMIT", b"79ab25d6\n")])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )

    with pytest.raises(CiError, match="too many .*headers|too many members"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private", artifacts
        )


def test_oversized_pax_metadata_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_source_cache_module, "_MAX_PAX_HEADER_BYTES", 32)
    cache = tmp_path / "cache"
    archive_path = cache / "sam2_hoi/package.tar.gz"
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(
        archive_path,
        "w:gz",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": "x" * 64},
    ) as archive:
        payload = b"79ab25d6\n"
        info = tarfile.TarInfo("SOURCE_COMMIT")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    tar_opened = False
    real_tar_open = tarfile.open

    def reject_parser_entry(*args, **kwargs):
        nonlocal tar_opened
        tar_opened = True
        return real_tar_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", reject_parser_entry)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )

    with pytest.raises(CiError, match="metadata exceeds the size limit"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private", artifacts
        )
    assert not tar_opened


def test_consecutive_metadata_headers_are_rejected_before_tarfile_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        model_source_cache_module,
        "_MAX_CONSECUTIVE_TAR_METADATA_HEADERS",
        2,
    )
    header = bytearray(512)
    header[:4] = b"pax\0"
    header[124:136] = b"00000000000\0"
    header[156:157] = b"g"
    raw_tar = bytes(header) * 3 + b"\0" * 1024
    cache = tmp_path / "cache"
    archive_path = cache / "sam2_hoi/package.tar.gz"
    archive_path.parent.mkdir(parents=True)
    with gzip.open(archive_path, "wb") as archive:
        archive.write(raw_tar)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )
    tar_opened = False

    def unexpected_tar_open(*_args, **_kwargs):
        nonlocal tar_opened
        tar_opened = True
        raise AssertionError("raw preflight must reject metadata chains before tarfile")

    monkeypatch.setattr(tarfile, "open", unexpected_tar_open)

    with pytest.raises(CiError, match="too many consecutive metadata headers"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private", artifacts
        )
    assert not tar_opened


def test_pax_size_override_is_rejected_before_tarfile_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pax_payload = b"12 size=512\n"
    pax_header = bytearray(512)
    pax_header[:4] = b"pax\0"
    pax_header[124:136] = f"{len(pax_payload):011o}\0".encode("ascii")
    pax_header[156:157] = b"x"
    raw_tar = bytes(pax_header) + pax_payload.ljust(512, b"\0") + b"\0" * 1024
    cache = tmp_path / "cache"
    archive_path = cache / "sam2_hoi/package.tar.gz"
    archive_path.parent.mkdir(parents=True)
    with gzip.open(archive_path, "wb") as archive:
        archive.write(raw_tar)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )
    tar_opened = False

    def unexpected_tar_open(*_args, **_kwargs):
        nonlocal tar_opened
        tar_opened = True
        raise AssertionError("raw preflight must reject PAX size before tarfile")

    monkeypatch.setattr(tarfile, "open", unexpected_tar_open)

    with pytest.raises(CiError, match="must not override file size"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private", artifacts
        )
    assert not tar_opened


def test_hf_cache_warmer_emits_canonical_empty_evidence_for_selected_local_source(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "fake-python/huggingface_hub"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        "from . import constants\n"
        "def hf_hub_download(*args, **kwargs): raise AssertionError('unexpected HF download')\n"
        "def snapshot_download(*args, **kwargs): raise AssertionError('unexpected HF download')\n",
        encoding="utf-8",
    )
    (fake / "constants.py").write_text(
        "import os\nHF_HUB_CACHE = os.environ['HF_HUB_CACHE']\n",
        encoding="utf-8",
    )
    (fake / "file_download.py").write_text(
        "def repo_folder_name(*, repo_id, repo_type):\n"
        "    return f'{repo_type}s--' + repo_id.replace('/', '--')\n",
        encoding="utf-8",
    )
    models = tmp_path / "models.txt"
    models.write_text("sam2-hoi-tracking\n", encoding="utf-8")
    hub = tmp_path / "hub"
    hub.mkdir()
    evidence = tmp_path / "hf-cache-repos.json"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_CACHE": str(hub),
            "PYTHONPATH": f"{fake.parent}:{REPO_ROOT / 'python'}:{REPO_ROOT}",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/warm_hf_cache.py"),
            "--models-file",
            str(models),
            "--local-only",
            "--strict",
            "--emit-cache-repos",
            str(evidence),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(evidence.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "hub_cache": str(hub),
        "selected_models": ["sam2-hoi-tracking"],
        "local_source_only": True,
        "repositories": [],
    }


@pytest.mark.parametrize(
    "extra",
    [
        [("file", "/absolute", b"x")],
        [("file", "./../traversal", b"x")],
        [("file", "./bad\\path", b"x")],
        [("file", "SOURCE_COMMIT", b"duplicate")],
        [("hardlink", "./hard", "SOURCE_COMMIT")],
        [("fifo", "./pipe", b"")],
        [("device", "./device", b"")],
        [("symlink", "./absolute-link", "/etc/passwd")],
        [("symlink", "./traversal-link", "../SOURCE_COMMIT")],
        [("symlink", "./dangling", "missing")],
        [("symlink", "./directory-link", "sam2")],
        [
            ("symlink", "./first-link", "SOURCE_COMMIT"),
            ("symlink", "./second-link", "first-link"),
        ],
    ],
    ids=(
        "absolute-member",
        "traversal-member",
        "backslash-member",
        "normalized-duplicate",
        "hardlink",
        "fifo",
        "device",
        "absolute-link",
        "traversal-link",
        "dangling-link",
        "link-to-directory",
        "link-chain",
    ),
)
def test_malicious_or_unsupported_archive_members_fail_closed(
    tmp_path: Path,
    extra: list[tuple[str, str, bytes | str]],
) -> None:
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive, [*_valid_members(), *extra])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    private = tmp_path / "private"
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )

    with pytest.raises(CiError):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), private, artifacts
        )

    assert not (private / "payload").exists()
    assert not list(private.glob(".extract-*")) if private.exists() else True
    assert not (artifacts / "model-source-package.json").exists()


def test_digest_mismatch_fails_before_tar_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    _archive(archive, _valid_members())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )
    called = False

    def unexpected_open(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("tar parsing must follow digest verification")

    monkeypatch.setattr(tarfile, "open", unexpected_open)
    with pytest.raises(CiError, match="SHA-256 mismatch"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract("0" * 64), tmp_path / "private", artifacts
        )
    assert not called


def test_verified_private_snapshot_closes_the_hash_to_parse_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    archive_path = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive_path, _valid_members())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    private = tmp_path / "private"
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )
    real_open = tarfile.open

    def mutate_original_then_open(*args, **kwargs):
        archive_path.write_bytes(b"tampered after verified snapshot")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", mutate_original_then_open)

    result = ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
        _contract(digest), private, artifacts
    )

    assert result is not None
    assert (result / "SOURCE_COMMIT").read_bytes() == b"79ab25d6\n"
    assert not list(private.glob(".archive-*"))


def test_declared_expansion_limit_is_enforced_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive, _valid_members())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    private = tmp_path / "private"
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )
    monkeypatch.setattr(model_source_cache_module, "_MAX_MATERIALIZED_BYTES", 1)

    with pytest.raises(CiError, match="size limit"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), private, artifacts
        )

    assert not (private / "payload").exists()


def test_compressed_and_tar_stream_limits_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive, _valid_members())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )

    monkeypatch.setattr(model_source_cache_module, "_MAX_COMPRESSED_ARCHIVE_BYTES", 1)
    with pytest.raises(CiError, match="compressed size limit"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private-compressed", artifacts
        )

    monkeypatch.setattr(
        model_source_cache_module,
        "_MAX_COMPRESSED_ARCHIVE_BYTES",
        8 * 1024 * 1024 * 1024,
    )
    monkeypatch.setattr(model_source_cache_module, "_MAX_TAR_STREAM_BYTES", 1)
    with pytest.raises(CiError, match="tar stream size limit"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private-tar", artifacts
        )


def test_missing_regular_entrypoint_fails_closed(tmp_path: Path) -> None:
    members = [member for member in _valid_members() if member[1] != "./SOURCE_COMMIT"]
    cache = tmp_path / "cache"
    archive = cache / "sam2_hoi/package.tar.gz"
    digest = _archive(archive, members)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    private = tmp_path / "private"
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )

    with pytest.raises(CiError, match="missing its regular entrypoint"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), private, artifacts
        )

    assert not (private / "payload").exists()


def test_cache_archive_must_not_be_a_symlink(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    real_archive = tmp_path / "real.tar.gz"
    digest = _archive(real_archive, _valid_members())
    archive = cache / "sam2_hoi/package.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.symlink_to(real_archive)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    context = CiContext(
        REPO_ROOT,
        {"TRTMC_MODEL_SOURCE_CACHE_ROOT": str(cache)},
    )

    with pytest.raises(CiError, match="contains a symlink"):
        ModelSourcePackagePreparer(context, "sam2_hoi").prepare(
            _contract(digest), tmp_path / "private", artifacts
        )


def test_parser_emits_only_the_selected_canonical_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "MODEL.toml"
    owner = {
        "model_source_package": {
            "suites": ["premerge", "nightly"],
            "cache_file": "sam2_hoi/package.tar.gz",
            "sha256": "a" * 64,
            "project_path": "artifacts/sam2_hoi/hoi",
            "entrypoint": "SOURCE_COMMIT",
        }
    }

    contract = parse_model_source_package_contract(owner, "sam2_hoi", manifest, "premerge")

    assert contract is not None
    assert contract.as_payload() == {
        "cache_file": "sam2_hoi/package.tar.gz",
        "sha256": "a" * 64,
        "project_path": "artifacts/sam2_hoi/hoi",
        "entrypoint": "SOURCE_COMMIT",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cache_file", "../package.tar.gz"),
        ("cache_file", "/package.tar.gz"),
        ("cache_file", "other/package.tar.gz"),
        ("project_path", "artifacts/other/hoi"),
        ("project_path", "artifacts/sam2_hoi/../hoi"),
        ("project_path", "artifacts/sam2_hoi/hoi,dst=/tmp/injected"),
        ("entrypoint", "../SOURCE_COMMIT"),
        ("sha256", "A" * 64),
    ],
)
def test_parser_rejects_unsafe_or_cross_family_contract_fields(
    tmp_path: Path, field: str, value: str
) -> None:
    owner: dict[str, object] = {
        "model_source_package": {
            "suites": ["premerge", "nightly"],
            "cache_file": "sam2_hoi/package.tar.gz",
            "sha256": "a" * 64,
            "project_path": "artifacts/sam2_hoi/hoi",
            "entrypoint": "SOURCE_COMMIT",
        }
    }
    owner["model_source_package"][field] = value  # type: ignore[index]

    with pytest.raises(CiError):
        parse_model_source_package_contract(owner, "sam2_hoi", tmp_path / "MODEL.toml", "premerge")
