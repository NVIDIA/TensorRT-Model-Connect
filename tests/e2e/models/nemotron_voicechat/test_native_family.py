# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import hashlib
import ast
import json
from pathlib import Path

import pytest


MODEL_ID = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"


def _config() -> dict:
    return {
        "model": {
            "stt": {
                "model": {
                    "pretrained_llm": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                    "perception": {"encoder": {}, "preprocessor": {}},
                }
            },
            "speech_generation": {
                "model": {
                    "tts_config": {"backbone_config": {}},
                    "codec_config": {"num_quantizers": 31, "codebook_size": 1024},
                }
            },
        }
    }


def test_plugin_discovery_does_not_import_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name == "tensorrt":
            raise AssertionError("family discovery must not import TensorRT")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    plugin_module = real_import("tensorrt_model_connect.families.nemotron_voicechat.plugin")
    assert plugin_module.plugin.name == "nemotron_voicechat"
    assert plugin_module.plugin.matches_config(_config())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("model", "stt", "model", "pretrained_llm"), "another/model"),
        (("model", "speech_generation", "model", "codec_config", "num_quantizers"), 8),
        (("model", "speech_generation", "model", "codec_config", "codebook_size"), 2048),
    ],
)
def test_structural_match_rejects_incompatible_checkpoints(
    path: tuple[str, ...], replacement: object
) -> None:
    from tensorrt_model_connect.families.nemotron_voicechat.model import matches

    config = _config()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement
    assert not matches(config)


def test_native_manifest_owns_build_and_exact_checkpoint() -> None:
    root = Path(__file__).parents[4]
    manifest = (
        root / "python/tensorrt_model_connect/families/nemotron_voicechat/MODEL.toml"
    ).read_text(encoding="utf-8")
    assert 'capabilities = ["model_owned_build"]' in manifest
    assert f'"{MODEL_ID}|model.safetensors"' in manifest


def test_checkpoint_config_is_sanitized_to_exact_native_dimensions(tmp_path: Path) -> None:
    from tensorrt_model_connect.families.nemotron_voicechat.model import _thinker_config

    config = _thinker_config(tmp_path, "fp32")
    assert json.loads(json.dumps(config.raw))["hybrid_override_pattern"].count("*") == 4
    assert (config.hidden_size, config.num_hidden_layers) == (4480, 56)
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (40, 8, 128)


def test_local_checkpoint_provenance_requires_exact_weight_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tensorrt_model_connect.families.nemotron_voicechat import model

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"exact voicechat checkpoint fixture")
    expected = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setattr(model, "VOICECHAT_WEIGHT_SHA256", expected)
    model._verify_exact_checkpoint(tmp_path)

    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model._verify_exact_checkpoint(tmp_path)


def test_all_small_pinned_assets_are_content_verified(tmp_path: Path) -> None:
    from tensorrt_model_connect.families.nemotron_voicechat import model

    (tmp_path / "nested").mkdir()
    asset = tmp_path / "nested/asset.json"
    asset.write_bytes(b"pinned asset")
    expected = {"nested/asset.json": hashlib.sha256(asset.read_bytes()).hexdigest()}
    model._verify_asset_set(tmp_path, expected, label="fixture")

    asset.write_bytes(b"mutated asset")
    with pytest.raises(ValueError, match="fixture asset SHA-256 mismatch"):
        model._verify_asset_set(tmp_path, expected, label="fixture")


def test_production_builders_have_no_framework_import_or_export_path() -> None:
    root = Path(__file__).parents[4]
    family = root / "python/tensorrt_model_connect/families/nemotron_voicechat"
    production = (
        "checkpoint_mapper.py",
        "conformer.py",
        "graph_blocks.py",
        "graph_ops.py",
        "model.py",
        "native_codec.py",
        "native_core.py",
        "native_tts.py",
        "plugin.py",
        "streaming_perception.py",
    )
    forbidden_roots = {"nemo", "onnx", "onnxruntime", "torch"}
    for filename in production:
        path = family / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            str(node.module).split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert not (imported & forbidden_roots), (filename, imported & forbidden_roots)
        source = path.read_text(encoding="utf-8")
        assert "OnnxParser" not in source
        assert "torch.onnx" not in source


def test_voicechat_runtime_has_no_python_subprocess_or_ffi_bridge() -> None:
    root = Path(__file__).parents[4]
    runtime = root / "src/runtime/models/nemotron_voicechat"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(runtime.glob("*.[ch]pp"))
    )
    source += "\n" + "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(runtime.glob("*.h"))
    )
    for forbidden in (
        "execvp(",
        "fork(",
        "load_ffi_kernels_from_bundle",
        "load_tvm_ffi",
        "popen(",
        "Py_Initialize",
        "system(",
    ):
        assert forbidden not in source
