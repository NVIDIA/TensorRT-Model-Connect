# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the fail-closed SAM2 source capture command."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2 import capture_golden, golden_evidence
from tensorrt_model_connect.families.sam2.capture_golden import (
    DELIVERED_CONFIG_NAME,
    Sam2GoldenCaptureError,
    VerifiedCaptureInputs,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fake_composed_source(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, str], str]:
    source = tmp_path / "source"
    (source / "sam2/modeling/backbones").mkdir(parents=True)
    base_payloads = {
        "sam2/__init__.py": b"# base package\n",
        "sam2/modeling/backbones/__init__.py": b"",
        "sam2/replaced.py": b"BASE = True\n",
    }
    for relative, payload in base_payloads.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for relative, target in capture_golden._PUBLIC_CONFIG_SYMLINKS.items():
        target_path = source / "sam2" / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(f"# {relative}\n", encoding="utf-8")
        (source / relative).symlink_to(target)
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "capture-test@example.invalid")
    _git(source, "config", "user.name", "Capture Test")
    _git(source, "add", ".")
    _git(source, "commit", "-q", "-m", "public base")
    public_commit = _git(source, "rev-parse", "HEAD")
    public_files = {relative: _digest(payload) for relative, payload in base_payloads.items()}

    overlay_payloads = {
        "sam2/replaced.py": b"OVERLAY = True\n",
        "sam2/new.py": b"NEW = True\n",
    }
    for relative, payload in overlay_payloads.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    overlay_files = {relative: _digest(payload) for relative, payload in overlay_payloads.items()}
    config = source / "sam2" / DELIVERED_CONFIG_NAME
    config.parent.mkdir(parents=True)
    config.write_bytes(b"delivered config\n")
    (source / capture_golden.SOURCE_OVERLAY_COMMIT_RECEIPT).write_text(
        "1" * 40 + "\n", encoding="ascii"
    )
    return source, public_files, overlay_files, public_commit


def _verify_fake_source(source: Path, public, overlay, public_commit):
    composed = {**public, **overlay}
    return capture_golden._verify_source_tree(
        source,
        public_commit=public_commit,
        overlay_commit="1" * 40,
        public_files=public,
        overlay_files=overlay,
        composed_files=composed,
        delivered_config_sha256=_digest(b"delivered config\n"),
    )


def test_source_verification_binds_git_base_overlay_and_composed_tree(tmp_path: Path) -> None:
    source, public, overlay, public_commit = _fake_composed_source(tmp_path)

    verified_source, staged_config, composed = _verify_fake_source(
        source, public, overlay, public_commit
    )

    assert verified_source == source.resolve()
    assert staged_config == source.resolve() / "sam2" / DELIVERED_CONFIG_NAME
    assert composed == dict(sorted({**public, **overlay}.items()))


def test_source_verification_rejects_drifted_public_config_symlink(tmp_path: Path) -> None:
    source, public, overlay, public_commit = _fake_composed_source(tmp_path)
    path = source / "sam2/sam2_hiera_l.yaml"
    path.unlink()
    path.symlink_to("configs/sam2/sam2_hiera_s.yaml")

    with pytest.raises(Sam2GoldenCaptureError):
        _verify_fake_source(source, public, overlay, public_commit)


@pytest.mark.parametrize("mutation", ["extra_python", "executable", "symlink"])
def test_source_verification_rejects_unreviewed_import_artifacts(
    tmp_path: Path, mutation: str
) -> None:
    source, public, overlay, public_commit = _fake_composed_source(tmp_path)
    if mutation == "extra_python":
        (source / "sam2/hoi_head.py").write_text("PRIVATE = True\n", encoding="utf-8")
    elif mutation == "executable":
        (source / "sam2/new.py").chmod(0o755)
    else:
        target = source / "outside.py"
        target.write_text("OUTSIDE = True\n", encoding="utf-8")
        (source / "sam2/new.py").unlink()
        (source / "sam2/new.py").symlink_to(target)

    with pytest.raises(Sam2GoldenCaptureError):
        _verify_fake_source(source, public, overlay, public_commit)


def test_source_verification_rejects_private_backbones_init_even_if_declared(
    tmp_path: Path,
) -> None:
    source, public, overlay, public_commit = _fake_composed_source(tmp_path)
    relative = "sam2/modeling/backbones/__init__.py"
    payload = b"from .hoi_head import HOIHead\n"
    (source / relative).write_bytes(payload)
    overlay = {**overlay, relative: _digest(payload)}

    with pytest.raises(Sam2GoldenCaptureError, match="private backbones"):
        _verify_fake_source(source, public, overlay, public_commit)


def test_source_verification_rejects_runtime_dependency_shadow(tmp_path: Path) -> None:
    source, public, overlay, public_commit = _fake_composed_source(tmp_path)
    (source / "yaml.py").write_text("raise AssertionError('shadow executed')\n", encoding="utf-8")

    with pytest.raises(Sam2GoldenCaptureError, match="yaml.py"):
        _verify_fake_source(source, public, overlay, public_commit)


def test_input_images_require_exact_names_order_hashes_and_regular_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "inputs"
    image_dir.mkdir()
    payloads = {"000000.jpg": b"zero", "000001.jpg": b"one"}
    for name, payload in payloads.items():
        (image_dir / name).write_bytes(payload)
    expected = {name: _digest(payload) for name, payload in payloads.items()}

    assert capture_golden._verify_input_images(image_dir, expected) == expected

    (image_dir / "000002.jpg").write_bytes(b"extra")
    with pytest.raises(Sam2GoldenCaptureError, match="exact numeric order"):
        capture_golden._verify_input_images(image_dir, expected)


def test_input_images_reject_symlink(tmp_path: Path) -> None:
    image_dir = tmp_path / "inputs"
    image_dir.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"image")
    (image_dir / "000000.jpg").symlink_to(outside)
    with pytest.raises(Sam2GoldenCaptureError, match="regular non-symlink"):
        capture_golden._verify_input_images(image_dir, {"000000.jpg": _digest(b"image")})


class _FakeOpenedImage:
    def __init__(self, decoded: np.ndarray, resized: np.ndarray | None = None) -> None:
        self.decoded = decoded
        self.resized = decoded if resized is None else resized

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def convert(self, mode: str):
        assert mode == "RGB"
        return self

    def resize(self, size: tuple[int, int]):
        assert size == (
            capture_golden.MODEL_IMAGE_SHAPE_HW[1],
            capture_golden.MODEL_IMAGE_SHAPE_HW[0],
        )
        return _FakeOpenedImage(self.resized)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.decoded, dtype=dtype, copy=True if copy is None else copy)


def _install_fake_decoder(
    monkeypatch: pytest.MonkeyPatch,
    decoded_by_name,
    *,
    resized_by_name=None,
    turbo="3.1.4.1",
) -> None:
    package_root = Path(sys.prefix) / "lib/python3.12/site-packages/PIL"
    pil = SimpleNamespace(
        __version__="12.3.0",
        __file__=str(package_root / "__init__.py"),
    )
    image = SimpleNamespace(
        open=lambda path: _FakeOpenedImage(
            decoded_by_name[Path(path).name],
            (resized_by_name or decoded_by_name)[Path(path).name],
        ),
        __file__=str(package_root / "Image.py"),
    )
    features = SimpleNamespace(
        version_codec=lambda name: "6.2" if name == "jpg" else None,
        version_feature=lambda name: turbo if name == "libjpeg_turbo" else None,
        __file__=str(package_root / "features.py"),
    )
    modules = {"PIL": pil, "PIL.Image": image, "PIL.features": features}
    monkeypatch.setattr(
        capture_golden.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        capture_golden,
        "_dependency_origin",
        lambda _module, label: str(package_root / label / "__init__.py"),
    )
    monkeypatch.setattr(capture_golden.np, "__version__", "2.5.2")


def test_decoded_rgb_hashes_and_decoder_versions_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_golden, "ORIGINAL_IMAGE_SHAPE_HW", (2, 3))
    monkeypatch.setattr(capture_golden, "MODEL_IMAGE_SHAPE_HW", (4, 5))
    decoded = {
        "000000.jpg": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "000001.jpg": np.full((2, 3, 3), 17, dtype=np.uint8),
    }
    expected = {
        name: _digest(np.ascontiguousarray(value).tobytes()) for name, value in decoded.items()
    }
    resized = {
        name: np.resize(value, (4, 5, 3)).astype(np.uint8) for name, value in decoded.items()
    }
    resized_expected = {
        name: _digest(np.ascontiguousarray(value).tobytes()) for name, value in resized.items()
    }
    monkeypatch.setattr(
        capture_golden,
        "INPUT_IMAGES_DECODED_RGB_UINT8_SHA256",
        expected,
    )
    monkeypatch.setattr(
        capture_golden,
        "INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256",
        resized_expected,
    )
    _install_fake_decoder(monkeypatch, decoded, resized_by_name=resized)

    assert capture_golden._verify_decoded_input_images(tmp_path) == {
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "pillow_jpeg_codec": "6.2",
        "libjpeg_turbo": "3.1.4.1",
        "input_images_decoded_rgb_uint8_sha256": expected,
        "input_images_resized_1024_rgb_uint8_sha256": resized_expected,
    }


def test_decoded_rgb_or_codec_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_golden, "ORIGINAL_IMAGE_SHAPE_HW", (1, 1))
    monkeypatch.setattr(capture_golden, "MODEL_IMAGE_SHAPE_HW", (1, 1))
    decoded = {"000000.jpg": np.zeros((1, 1, 3), dtype=np.uint8)}
    monkeypatch.setattr(
        capture_golden,
        "INPUT_IMAGES_DECODED_RGB_UINT8_SHA256",
        {"000000.jpg": "0" * 64},
    )
    monkeypatch.setattr(
        capture_golden,
        "INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256",
        {"000000.jpg": _digest(decoded["000000.jpg"].tobytes())},
    )
    _install_fake_decoder(monkeypatch, decoded)
    with pytest.raises(Sam2GoldenCaptureError, match="decoded RGB uint8 hash mismatch"):
        capture_golden._verify_decoded_input_images(tmp_path)

    _install_fake_decoder(monkeypatch, decoded, turbo="3.1.4")
    with pytest.raises(Sam2GoldenCaptureError, match="decoder environment mismatch"):
        capture_golden._verify_decoded_input_images(tmp_path)


def test_source_resize_rgb_hash_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_golden, "ORIGINAL_IMAGE_SHAPE_HW", (1, 1))
    monkeypatch.setattr(capture_golden, "MODEL_IMAGE_SHAPE_HW", (1, 1))
    decoded = {"000000.jpg": np.zeros((1, 1, 3), dtype=np.uint8)}
    monkeypatch.setattr(
        capture_golden,
        "INPUT_IMAGES_DECODED_RGB_UINT8_SHA256",
        {"000000.jpg": _digest(decoded["000000.jpg"].tobytes())},
    )
    monkeypatch.setattr(
        capture_golden,
        "INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256",
        {"000000.jpg": "0" * 64},
    )
    _install_fake_decoder(monkeypatch, decoded)

    with pytest.raises(Sam2GoldenCaptureError, match="resized RGB uint8 hash mismatch"):
        capture_golden._verify_decoded_input_images(tmp_path)


def _fake_snapshot_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[VerifiedCaptureInputs, dict[str, bytes]]:
    source = tmp_path / "source"
    package = tmp_path / "delivery"
    payloads = {
        "source": b"# exact source\n",
        "config": b"model: exact\n",
        "checkpoint": b"exact checkpoint",
        "image": b"exact jpeg bytes",
    }
    source_file = source / "sam2/__init__.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(payloads["source"])
    staged_config = source / "sam2" / DELIVERED_CONFIG_NAME
    staged_config.parent.mkdir(parents=True)
    staged_config.write_bytes(payloads["config"])
    checkpoint = package / "checkpoint/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payloads["checkpoint"])
    image_dir = package / "inputs"
    image_dir.mkdir()
    (image_dir / "000000.jpg").write_bytes(payloads["image"])

    source_hashes = {"sam2/__init__.py": _digest(payloads["source"])}
    image_hashes = {"000000.jpg": _digest(payloads["image"])}
    decoder_environment = {"decoder": "exact"}
    monkeypatch.setattr(capture_golden, "COMPATIBLE_SOURCE_FILES_SHA256", source_hashes)
    monkeypatch.setattr(capture_golden, "INPUT_IMAGES_SHA256", image_hashes)
    monkeypatch.setattr(
        capture_golden,
        "REFERENCE_CONFIG_SHA256",
        _digest(payloads["config"]),
    )
    monkeypatch.setattr(
        capture_golden,
        "REFERENCE_CHECKPOINT_SHA256",
        _digest(payloads["checkpoint"]),
    )
    monkeypatch.setattr(
        capture_golden,
        "_verify_decoded_input_images",
        lambda _image_dir: decoder_environment,
    )
    return (
        VerifiedCaptureInputs(
            source_root=source,
            package_root=package,
            checkpoint=checkpoint,
            image_dir=image_dir,
            staged_config=staged_config,
            source_files_sha256=source_hashes,
            image_files_sha256=image_hashes,
            decoder_environment=decoder_environment,
            capture_code_sha256={"capture": "a" * 64},
            capture_code_raw_sha256={"capture": "a" * 64},
            tool_sha256="a" * 64,
        ),
        payloads,
    )


def test_private_snapshot_is_read_only_and_immune_to_original_input_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, payloads = _fake_snapshot_inputs(tmp_path, monkeypatch)

    with capture_golden._private_capture_snapshot(inputs) as snapshot:
        assert snapshot.source_root != inputs.source_root
        assert snapshot.checkpoint != inputs.checkpoint
        assert snapshot.image_dir != inputs.image_dir
        assert snapshot.staged_config != inputs.staged_config
        assert snapshot.checkpoint.stat().st_mode & 0o777 == 0o400
        assert snapshot.image_dir.stat().st_mode & 0o777 == 0o500

        (inputs.source_root / "sam2/__init__.py").write_bytes(b"mutated source")
        inputs.staged_config.write_bytes(b"mutated config")
        inputs.checkpoint.write_bytes(b"mutated checkpoint")
        (inputs.image_dir / "000000.jpg").write_bytes(b"mutated image")

        assert (snapshot.source_root / "sam2/__init__.py").read_bytes() == payloads["source"]
        assert snapshot.staged_config.read_bytes() == payloads["config"]
        assert snapshot.checkpoint.read_bytes() == payloads["checkpoint"]
        assert (snapshot.image_dir / "000000.jpg").read_bytes() == payloads["image"]


def test_private_snapshot_detects_mutation_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _payloads = _fake_snapshot_inputs(tmp_path, monkeypatch)

    with pytest.raises(Sam2GoldenCaptureError, match="checkpoint snapshot changed"):
        with capture_golden._private_capture_snapshot(inputs) as snapshot:
            snapshot.checkpoint.chmod(0o600)
            snapshot.checkpoint.write_bytes(b"mutated snapshot")


def test_capture_code_closure_records_every_executed_family_helper() -> None:
    closure, raw = capture_golden._capture_code_closure()
    capture = Path(capture_golden.__file__)
    archive = capture.parent / "archive_contract.py"
    golden = capture.parent / "golden_evidence.py"
    normalized, _tool_pin, _normalized_pin = capture_golden._normalized_golden_bytes(
        golden.read_bytes()
    )

    assert closure == {
        "tensorrt_model_connect.families.sam2.archive_contract": _digest(archive.read_bytes()),
        "tensorrt_model_connect.families.sam2.golden_evidence.normalized": _digest(normalized),
        "tensorrt_model_connect.families.sam2.capture_golden": _digest(capture.read_bytes()),
    }
    assert raw == {
        "tensorrt_model_connect.families.sam2.archive_contract": _digest(archive.read_bytes()),
        "tensorrt_model_connect.families.sam2.golden_evidence": _digest(golden.read_bytes()),
        "tensorrt_model_connect.families.sam2.capture_golden": _digest(capture.read_bytes()),
    }
    assert Path(capture_golden._archive_contract.__file__).parent != capture.parent
    assert Path(capture_golden._golden_evidence.__file__).parent != capture.parent


@pytest.mark.parametrize(
    ("helper", "message"),
    [
        ("archive_contract.py", "archive contract helper hash mismatch"),
        ("golden_evidence.py", "golden evidence normalized hash mismatch"),
        ("capture_golden.py", "capture runner does not match the verified golden tool pin"),
    ],
)
def test_direct_bootstrap_rejects_helper_mutation_before_import(
    tmp_path: Path, helper: str, message: str
) -> None:
    family = tmp_path / "sam2"
    family.mkdir()
    for name in ("archive_contract.py", "golden_evidence.py", "capture_golden.py"):
        shutil.copyfile(Path(capture_golden.__file__).parent / name, family / name)
    with (family / helper).open("ab") as stream:
        stream.write(b"\nraise AssertionError('mutated helper executed')\n")

    result = subprocess.run(
        [sys.executable, "-I", "-S", str(family / "capture_golden.py"), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert "mutated helper executed" not in result.stderr


def test_golden_normalization_masks_all_post_capture_pins() -> None:
    payload = Path(capture_golden._GOLDEN_EVIDENCE_PATH).read_bytes()
    normalized, _tool_pin, _normalized_pin = capture_golden._normalized_golden_bytes(payload)
    changed = capture_golden._replace_one_pin(
        payload,
        capture_golden._TOOL_PIN_PATTERN,
        b"1" * 64,
    )
    changed = capture_golden._replace_one_pin(
        changed,
        capture_golden._GOLDEN_NORMALIZED_PIN_PATTERN,
        b"2" * 64,
    )
    changed = capture_golden._replace_one_pin(
        changed,
        capture_golden._REFERENCE_MANIFEST_PIN_PATTERN,
        b"3" * 64,
    )

    assert capture_golden._normalized_golden_bytes(changed)[0] == normalized
    non_pin_change = payload.replace(
        b"Accuracy evidence for the exact five-frame SAM2 workload.",
        b"Accuracy evidence for one exact five-frame SAM2 workload.",
        1,
    )
    assert capture_golden._normalized_golden_bytes(non_pin_change)[0] != normalized


def test_pending_to_reviewed_manifest_pin_is_ruff_format_stable() -> None:
    payload = Path(capture_golden._GOLDEN_EVIDENCE_PATH).read_bytes()
    reviewed = capture_golden._replace_one_pin(
        payload,
        capture_golden._REFERENCE_MANIFEST_PIN_PATTERN,
        b"3" * 64,
    )
    assert (
        capture_golden._normalized_golden_bytes(reviewed)[0]
        == (capture_golden._normalized_golden_bytes(payload)[0])
    )

    result = subprocess.run(
        [
            "ruff",
            "format",
            "--check",
            "--stdin-filename",
            str(capture_golden._GOLDEN_EVIDENCE_PATH),
            "-",
        ],
        input=reviewed,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_golden_normalization_rejects_duplicate_or_malformed_pins() -> None:
    payload = Path(capture_golden._GOLDEN_EVIDENCE_PATH).read_bytes()
    tool_assignment = capture_golden._TOOL_PIN_PATTERN.search(payload)
    assert tool_assignment is not None
    duplicate = payload + b"\n" + tool_assignment.group(0) + b"\n"
    malformed = payload.replace(
        b"AUTHORITATIVE_CAPTURE_TOOL_SHA256: str | None = (",
        b"AUTHORITATIVE_CAPTURE_TOOL_SHA256: str | None = [",
        1,
    )

    for changed in (duplicate, malformed):
        with pytest.raises(Sam2GoldenCaptureError, match="pin layout changed"):
            capture_golden._normalized_golden_bytes(changed)


def test_loaded_python_customizers_are_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(capture_golden.sys.modules, "sitecustomize", SimpleNamespace())
    with pytest.raises(Sam2GoldenCaptureError, match="sitecustomize"):
        capture_golden._reject_customizers()


def test_distribution_metadata_is_discovered_only_in_controlled_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    distribution = SimpleNamespace(
        version="6.0.3",
        locate_file=lambda _relative: capture_golden._CONTROLLED_SITE_PACKAGES,
    )
    monkeypatch.setattr(
        capture_golden.importlib.metadata,
        "Distribution",
        SimpleNamespace(discover=lambda **kwargs: calls.append(kwargs) or [distribution]),
    )

    assert capture_golden._require_distribution_version("PyYAML", "6.0.3") == "6.0.3"
    assert calls == [
        {
            "name": "PyYAML",
            "path": [str(capture_golden._CONTROLLED_SITE_PACKAGES)],
        }
    ]


def test_capture_requires_isolated_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        capture_golden.sys,
        "flags",
        SimpleNamespace(isolated=0, safe_path=True, no_user_site=1, no_site=1),
    )
    with pytest.raises(Sam2GoldenCaptureError, match="Python -I -S mode"):
        capture_golden._require_isolated_interpreter()


def test_runtime_import_rejects_preinitialized_hydra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "snapshot-source"
    source.mkdir()
    initialized = SimpleNamespace(is_initialized=lambda: True)
    modules = {
        "torch": SimpleNamespace(),
        "torchvision": SimpleNamespace(),
        "antlr4": SimpleNamespace(),
        "hydra": SimpleNamespace(),
        "hydra.core.global_hydra": SimpleNamespace(
            GlobalHydra=SimpleNamespace(instance=lambda: initialized)
        ),
        "omegaconf": SimpleNamespace(),
        "yaml": SimpleNamespace(),
        "tqdm": SimpleNamespace(),
        "iopath": SimpleNamespace(),
        "portalocker": SimpleNamespace(),
    }
    monkeypatch.setattr(capture_golden, "_loaded_sam2_module_mismatch", lambda _source: None)
    monkeypatch.setattr(capture_golden, "_sam2_import_candidates", lambda _source: set())
    monkeypatch.setattr(
        capture_golden.importlib,
        "import_module",
        lambda name: modules[name],
    )

    with pytest.raises(Sam2GoldenCaptureError, match="Hydra was initialized before"):
        with capture_golden._isolated_runtime_imports(source):
            pytest.fail("preinitialized Hydra must prevent runtime import")


class _FakeTensor:
    def __init__(self, values, *, dtype=None) -> None:
        self.values = np.asarray(values)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype if self._dtype is not None else self.values.dtype

    @property
    def shape(self):
        return self.values.shape

    def detach(self):
        return self

    def clone(self):
        return _FakeTensor(self.values.copy(), dtype=self._dtype)

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.values

    def to(self, *, dtype=None):
        return _FakeTensor(self.values.astype(dtype, copy=False))

    def float(self):
        return _FakeTensor(self.values.astype(np.float32))

    def __getitem__(self, item):
        return _FakeTensor(self.values[item])

    def __gt__(self, value):
        return _FakeTensor(self.values > value)

    def __mul__(self, other):
        values = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values * values)


class _FakeCuda:
    def __init__(self) -> None:
        self.seeds = []
        self.synchronizations = 0
        self.empty_cache_calls = 0

    def manual_seed_all(self, value):
        self.seeds.append(value)

    def synchronize(self):
        self.synchronizations += 1

    def empty_cache(self):
        self.empty_cache_calls += 1

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def set_device(self, index):
        assert index == 0

    def get_device_name(self, index):
        assert index == 0
        return "NVIDIA L4"

    def get_device_capability(self, index):
        assert index == 0
        return (8, 9)


class _FakeTorch:
    __version__ = "2.7.1+cu128"
    bfloat16 = "bfloat16"
    uint8 = np.uint8

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.seeds = []
        self.version = SimpleNamespace(cuda="12.8")
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(
                allow_tf32=False,
                deterministic=True,
                benchmark=True,
                version=lambda: 90701,
            ),
        )

    def manual_seed(self, value):
        self.seeds.append(value)

    def inference_mode(self):
        return nullcontext()

    def autocast(self, device, *, dtype):
        assert (device, dtype) == ("cuda", self.bfloat16)
        return nullcontext()

    def are_deterministic_algorithms_enabled(self):
        return False


def test_runner_environment_satisfies_golden_authority_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "antlr4-python3-runtime": "4.9.3",
        "hydra-core": "1.3.2",
        "iopath": "0.1.10",
        "omegaconf": "2.3.1",
        "portalocker": "4.1.0",
        "PyYAML": "6.0.3",
        "torchvision": "0.22.1+cu128",
        "tqdm": "4.67.1",
    }
    monkeypatch.setattr(
        capture_golden,
        "_require_distribution_version",
        lambda name, expected: versions[name] if versions[name] == expected else pytest.fail(),
    )
    monkeypatch.setattr(capture_golden, "_driver_version", lambda: "595.58.03")
    monkeypatch.setattr(capture_golden, "_read_regular_bytes", lambda *_args: b"cfg")
    monkeypatch.setattr(
        capture_golden,
        "_raw_sha256",
        lambda _payload: capture_golden._REFERENCE_PYVENV_CFG_SHA256,
    )
    monkeypatch.setattr(
        capture_golden.sys,
        "flags",
        SimpleNamespace(isolated=1, safe_path=True, no_user_site=1, no_site=1),
    )
    origins = {
        name: f"{capture_golden._CONTROLLED_SITE_PACKAGES}/{name}/__init__.py"
        for name in (
            "antlr4",
            "hydra",
            "iopath",
            "numpy",
            "omegaconf",
            "pillow",
            "portalocker",
            "pyyaml",
            "torch",
            "torchvision",
            "tqdm",
        )
    }
    decoder_environment = {
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "pillow_jpeg_codec": "6.2",
        "libjpeg_turbo": "3.1.4.1",
        "input_images_decoded_rgb_uint8_sha256": dict(
            golden_evidence.INPUT_IMAGES_DECODED_RGB_UINT8_SHA256
        ),
        "input_images_resized_1024_rgb_uint8_sha256": dict(
            golden_evidence.INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256
        ),
    }
    environment = capture_golden._configure_and_record_environment(
        _FakeTorch(), decoder_environment, origins
    )
    environment["video_res_logits_dtypes"] = [["torch.bfloat16"] * 5 for _ in range(3)]
    provenance = {
        "source_commit": golden_evidence.PUBLIC_SAM2_BASE_COMMIT,
        "source_overlay_declared_commit": golden_evidence.COMPATIBLE_SOURCE_COMMIT,
        "source_files_sha256": dict(golden_evidence.COMPATIBLE_SOURCE_FILES_SHA256),
        "checkpoint_sha256": golden_evidence.REFERENCE_CHECKPOINT_SHA256,
        "config_sha256": golden_evidence.REFERENCE_CONFIG_SHA256,
        "image_files_sha256": dict(golden_evidence.INPUT_IMAGES_SHA256),
        "capture_tool_sha256": golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256,
        "environment": environment,
        "artifacts_sha256": {
            "capture_code/tensorrt_model_connect.families.sam2.archive_contract": (
                golden_evidence.AUTHORITATIVE_ARCHIVE_CONTRACT_SHA256
            ),
            "capture_code/tensorrt_model_connect.families.sam2.golden_evidence.normalized": (
                golden_evidence.AUTHORITATIVE_GOLDEN_EVIDENCE_NORMALIZED_SHA256
            ),
            "capture_code/tensorrt_model_connect.families.sam2.capture_golden": (
                golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256
            ),
        },
    }

    assert golden_evidence._authority_errors(provenance) == []


def test_capture_orchestration_uses_only_private_snapshot_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original"
    snapshot_root = tmp_path / "snapshot"
    original.mkdir()
    snapshot_root.mkdir()
    original_checkpoint = original / "checkpoint.pt"
    original_checkpoint.write_bytes(b"original checkpoint")
    original_images = original / "inputs"
    original_images.mkdir()
    original_config = original / "config.yaml"
    original_config.write_bytes(b"original config")
    snapshot_checkpoint = snapshot_root / "checkpoint.pt"
    snapshot_checkpoint.write_bytes(b"snapshot checkpoint")
    snapshot_images = snapshot_root / "inputs"
    snapshot_images.mkdir()
    snapshot_config = snapshot_root / "config.yaml"
    snapshot_config.write_bytes(b"snapshot config")
    source_root = original / "source"
    source_root.mkdir()
    (source_root / capture_golden.SOURCE_OVERLAY_COMMIT_RECEIPT).write_text(
        "receipt\n", encoding="ascii"
    )
    package_root = original / "delivery"
    package_root.mkdir()
    (package_root / capture_golden.SHA256SUMS_RELATIVE_PATH).write_text(
        "manifest\n", encoding="ascii"
    )
    inputs = VerifiedCaptureInputs(
        source_root=source_root,
        package_root=package_root,
        checkpoint=original_checkpoint,
        image_dir=original_images,
        staged_config=original_config,
        source_files_sha256={"sam2/a.py": "a" * 64},
        image_files_sha256={"000000.jpg": "b" * 64},
        decoder_environment={"decoder": "exact"},
        capture_code_sha256={"tensorrt_model_connect.families.sam2.capture_golden": "c" * 64},
        capture_code_raw_sha256={"tensorrt_model_connect.families.sam2.capture_golden": "c" * 64},
        tool_sha256="c" * 64,
    )
    snapshot = capture_golden.CaptureSnapshot(
        root=snapshot_root,
        source_root=snapshot_root / "source",
        checkpoint=snapshot_checkpoint,
        image_dir=snapshot_images,
        staged_config=snapshot_config,
    )
    snapshot.source_root.mkdir()
    torch = _FakeTorch()
    predictor = object()
    builder_arguments = {}
    capture_image_dirs = []
    verification_calls = []
    written = {}

    def verify(source, delivery):
        verification_calls.append((source, delivery))
        return inputs

    def builder(**kwargs):
        builder_arguments.update(kwargs)
        return predictor

    captures = [
        capture_golden.CapturedRun(
            workload=SimpleNamespace(run=index),
            video_res_logits_dtypes=("torch.bfloat16",) * 5,
        )
        for index in range(3)
    ]

    def capture_runs(actual_predictor, image_dir, actual_torch):
        assert actual_predictor is predictor
        assert actual_torch is torch
        capture_image_dirs.append(image_dir)
        return captures

    def write(destination, **kwargs):
        written.update(destination=destination, **kwargs)
        return {"capture_sha256": "d" * 64}

    monkeypatch.setattr(capture_golden, "_require_isolated_interpreter", lambda: None)
    monkeypatch.setattr(capture_golden, "_dependency_origin", lambda *_args: "origin")
    monkeypatch.setattr(capture_golden, "verify_capture_inputs", verify)
    monkeypatch.setattr(
        capture_golden,
        "_private_capture_snapshot",
        lambda actual: nullcontext(snapshot) if actual is inputs else pytest.fail(),
    )
    monkeypatch.setattr(
        capture_golden,
        "_isolated_runtime_imports",
        lambda source: (
            nullcontext((torch, builder, {"torch": "origin"}))
            if source == snapshot.source_root
            else pytest.fail()
        ),
    )
    monkeypatch.setattr(
        capture_golden,
        "_configure_and_record_environment",
        lambda *_args: {},
    )
    monkeypatch.setattr(capture_golden, "_assert_predictor_contract", lambda value: None)
    monkeypatch.setattr(capture_golden, "_capture_three_runs", capture_runs)
    monkeypatch.setattr(capture_golden, "sha256_file", lambda _path: "e" * 64)
    monkeypatch.setattr(capture_golden, "write_evidence", write)

    result = capture_golden.capture_authoritative_evidence(
        source_root, package_root, tmp_path / "evidence"
    )

    assert result == {"capture_sha256": "d" * 64}
    assert verification_calls == [(source_root, package_root), (source_root, package_root)]
    assert builder_arguments["ckpt_path"] == str(snapshot_checkpoint)
    assert builder_arguments["config_file"] == DELIVERED_CONFIG_NAME
    assert capture_image_dirs == [snapshot_images]
    assert written["capture"] is captures[0].workload
    assert written["replay_captures"] == [
        captures[1].workload,
        captures[2].workload,
    ]


class _FakePredictor:
    def __init__(
        self,
        *,
        detection_count: int = 1,
        frames=(0, 1, 2, 3, 4),
        nonfinite_logits: bool = False,
        logits_dtype: str = "torch.bfloat16",
        logits_shape: tuple[int, ...] = (1, 1, 1280, 1088),
        image_dir_override: Path | None = None,
    ) -> None:
        model_boxes = np.repeat([[100.0, 200.0, 300.0, 400.0]], detection_count, axis=0)
        self.model_boxes = model_boxes
        self.frames = frames
        self.nonfinite_logits = nonfinite_logits
        self.logits_dtype = logits_dtype
        self.logits_shape = logits_shape
        self.image_dir_override = image_dir_override
        self.reset_calls = 0
        self.prompts = []

    def _get_det_results(self, state, frame_idx):
        result = state["cached_features"][frame_idx][1]["det_results"][0]
        if not result.get("has_rescaled", False):
            result["bboxes"] = result["bboxes"] * _FakeTensor(
                [1088 / 1024, 1280 / 1024, 1088 / 1024, 1280 / 1024]
            )
            result["has_rescaled"] = True
        return result

    def init_state(self, image_dir, **kwargs):
        assert kwargs == {
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "async_loading_frames": True,
            "frame_idx": 0,
        }
        detection = {
            "bboxes": _FakeTensor(self.model_boxes),
            "scores": _FakeTensor(np.full(len(self.model_boxes), 0.5, dtype=np.float32)),
            "labels": _FakeTensor(np.ones(len(self.model_boxes), dtype=np.int64)),
        }
        state = {
            "num_frames": 5,
            "video_height": 1280,
            "video_width": 1088,
            "img_paths": [
                str((self.image_dir_override or Path(image_dir)) / name)
                for name in golden_evidence.INPUT_IMAGES_SHA256
            ],
            "cached_features": {0: (None, {"det_results": [detection]})},
        }
        self._get_det_results(state, 0)
        return state

    def add_new_points_or_box(self, state, frame_idx, object_id, *, box):
        self.prompts.append((state, frame_idx, object_id, box.values.copy()))

    def propagate_in_video(self, state, **kwargs):
        assert kwargs == {
            "start_frame_idx": 0,
            "max_frame_num_to_track": None,
            "reverse": False,
        }
        for frame_idx in self.frames:
            values = np.full(self.logits_shape, -1.0, dtype=np.float32)
            if self.logits_shape == (1, 1, 1280, 1088):
                values[:, :, 10 + frame_idx : 20 + frame_idx, 30:40] = 1.0
            if self.nonfinite_logits and frame_idx == 2:
                values[0, 0, 0, 0] = np.nan
            yield frame_idx, [0], _FakeTensor(values, dtype=self.logits_dtype), None, None

    def reset_state(self, state):
        self.reset_calls += 1


def _fake_exact_image_dir(tmp_path: Path) -> Path:
    image_dir = tmp_path / "exact-inputs"
    image_dir.mkdir()
    for name in golden_evidence.INPUT_IMAGES_SHA256:
        (image_dir / name).write_bytes(b"image")
    return image_dir


def test_three_runs_capture_pre_rescale_box_exact_frames_and_video_masks(
    tmp_path: Path,
) -> None:
    predictor = _FakePredictor()
    torch = _FakeTorch()
    image_dir = _fake_exact_image_dir(tmp_path)

    captures = capture_golden._capture_three_runs(predictor, image_dir, torch)

    assert len(captures) == 3
    assert predictor.reset_calls == 3
    assert [item[2] for item in predictor.prompts] == [0, 0, 0]
    assert captures[0].workload.frame_zero_bbox.model_xyxy_1024 == (
        100.0,
        200.0,
        300.0,
        400.0,
    )
    assert captures[0].workload.frame_zero_bbox.original_xyxy == (
        106.25,
        250.0,
        318.75,
        500.0,
    )
    assert captures[0].workload.masks.shape == (5, 1, 1280, 1088)
    assert captures[0].workload.masks.dtype == np.uint8
    assert [int(frame.sum()) for frame in captures[0].workload.masks] == [100] * 5
    assert captures[0].video_res_logits_dtypes == ("torch.bfloat16",) * 5
    for replay in captures[1:]:
        np.testing.assert_array_equal(replay.workload.masks, captures[0].workload.masks)


def test_numpy_converts_only_bfloat16_before_host_materialization() -> None:
    bfloat = _FakeTensor([1.5], dtype="torch.bfloat16")
    integer = _FakeTensor([7], dtype="torch.int64")

    assert capture_golden._numpy(bfloat).dtype == np.float32
    assert capture_golden._numpy(integer).dtype == np.int64


def test_capture_rejects_top1_selection_and_wrong_frame_order(tmp_path: Path) -> None:
    image_dir = _fake_exact_image_dir(tmp_path)
    with pytest.raises(Sam2GoldenCaptureError, match="exactly one"):
        capture_golden._capture_once(_FakePredictor(detection_count=2), image_dir, _FakeTorch())
    with pytest.raises(Sam2GoldenCaptureError, match="frame order"):
        capture_golden._capture_once(
            _FakePredictor(frames=(0, 1, 3, 2, 4)), image_dir, _FakeTorch()
        )
    with pytest.raises(Sam2GoldenCaptureError, match="finite before threshold"):
        capture_golden._capture_once(_FakePredictor(nonfinite_logits=True), image_dir, _FakeTorch())
    with pytest.raises(Sam2GoldenCaptureError, match="BF16 or FP32"):
        capture_golden._capture_once(
            _FakePredictor(logits_dtype="torch.float16"), image_dir, _FakeTorch()
        )
    with pytest.raises(Sam2GoldenCaptureError, match="must have shape"):
        capture_golden._capture_once(
            _FakePredictor(logits_shape=(1, 1, 256, 256)), image_dir, _FakeTorch()
        )


def test_capture_requires_async_loader_paths_from_private_snapshot(tmp_path: Path) -> None:
    image_dir = _fake_exact_image_dir(tmp_path)
    foreign = tmp_path / "foreign-inputs"
    foreign.mkdir()
    for name in golden_evidence.INPUT_IMAGES_SHA256:
        (foreign / name).write_bytes(b"foreign")

    with pytest.raises(Sam2GoldenCaptureError, match="snapshot-bound numeric frame order"):
        capture_golden._capture_once(
            _FakePredictor(image_dir_override=foreign), image_dir, _FakeTorch()
        )


def _predictor_contract(**changes):
    values = {
        "image_encoder": SimpleNamespace(bbox_head=object(), learnable_fpn_module=None),
        "fill_hole_area": 8,
        "binarize_mask_from_pts_for_mem_enc": True,
        "sam_mask_decoder": SimpleNamespace(dynamic_multimask_via_stability=True),
        "training": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_predictor_contract_requires_postprocessing_and_rejects_cspnext() -> None:
    capture_golden._assert_predictor_contract(_predictor_contract())
    with pytest.raises(Sam2GoldenCaptureError, match="CSPNeXt"):
        capture_golden._assert_predictor_contract(
            _predictor_contract(
                image_encoder=SimpleNamespace(bbox_head=object(), learnable_fpn_module=object())
            )
        )
    with pytest.raises(Sam2GoldenCaptureError, match="postprocessing"):
        capture_golden._assert_predictor_contract(_predictor_contract(fill_hole_area=0))


def test_capture_tool_hash_is_checked_in_and_byte_exact() -> None:
    assert golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256 is not None
    assert (
        hashlib.sha256(Path(capture_golden.__file__).read_bytes()).hexdigest()
        == golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256
    )


def test_capture_module_has_no_eager_torch_or_sam2_import() -> None:
    assert "torch" not in capture_golden.__dict__
    assert "sam2" not in capture_golden.__dict__
    assert capture_golden.sys.dont_write_bytecode is True
