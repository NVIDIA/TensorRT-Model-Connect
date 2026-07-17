# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.sam3 import tracker_memory_aoti_exporter as exporter


class _FakeCuda:
    def __init__(self) -> None:
        self.selected = 0
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_capability(self, device_index: int) -> tuple[int, int]:
        assert device_index == 0
        return (8, 9)

    def set_device(self, device_index: int) -> None:
        self.selected = device_index

    def current_device(self) -> int:
        return self.selected

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeRandom:
    def __init__(self) -> None:
        self.devices: list[list[int]] = []

    @contextmanager
    def fork_rng(self, *, devices):
        self.devices.append(list(devices))
        yield


class _FakeTorch:
    __version__ = "2.12.0+cu130"
    float32 = "float32"
    version = SimpleNamespace(cuda="13.0")
    _C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.random = _FakeRandom()
        self.seeds: list[int] = []

    def device(self, value: str) -> str:
        return value

    def manual_seed(self, value: int) -> None:
        self.seeds.append(value)


class _FakeModel:
    def __init__(self) -> None:
        self.memory_encoder = object()
        self.occlusion_spatial_embedding_parameter = object()
        self.config = SimpleNamespace(
            memory_encoder_hidden_size=256,
            memory_encoder_output_channels=64,
            mask_downsampler_hidden_act="gelu",
            mask_downsampler_total_stride=16,
            memory_fuser_hidden_act="gelu",
            memory_fuser_num_layers=2,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
        )
        self.target = None

    def eval(self):
        return self

    def to(self, target):
        self.target = target
        return self

    def parameters(self):
        return ()


def _producer() -> exporter.MemoryAotiProducerAbi:
    return exporter.MemoryAotiProducerAbi(
        torch_version="2.12.0+cu130",
        transformers_version="5.2.0",
        cuda_version="13.0",
        compute_capability=(8, 9),
        host_architecture="x86_64",
        torch_cxx11_abi=True,
        torch_aoti_abi_version=147492887796383744,
    )


def _model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"sam3_video"}\n')
    (model_dir / "model.safetensors").write_bytes(b"exact-converted-tracker-weights")
    return model_dir


def _passing_case(policy: str, batch_size: int, hard_mask: bool) -> dict[str, object]:
    metrics = {
        "cosine": 1.0,
        "relative_l2": 0.0,
        "maximum_absolute_error": 0.0,
    }
    return {
        "policy": policy,
        "batch_size": batch_size,
        "hard_mask": hard_mask,
        **metrics,
        "planes": {"memory": dict(metrics), "position": dict(metrics)},
        "passed": True,
    }


def test_fixed_tensor_contract_separates_soft_and_global_hard_mask_abis() -> None:
    assert [
        (policy_abi.policy, [(value.name, value.dtype) for value in policy_abi.tensors])
        for policy_abi in exporter.TRACKER_MEMORY_INPUT_ABI
    ] == [
        (
            "soft",
            [
                ("tracker_feature_2", "float32"),
                ("final_mask", "float32"),
                ("object_score_logits", "float32"),
                ("suppress_area_shrinkage", "int32"),
            ],
        ),
        (
            "hard",
            [
                ("tracker_feature_2", "float32"),
                ("owned_tracker_mask", "float32"),
                ("object_score_logits", "float32"),
                ("suppress_area_shrinkage", "int32"),
            ],
        ),
    ]
    assert [
        (policy, batch_size, hard_mask)
        for policy, batch_size, hard_mask in exporter._PACKAGE_VARIANTS
    ] == [
        ("soft", 1, False),
        ("hard", 1, True),
        ("soft", 2, False),
        ("hard", 2, True),
    ]
    for policy, batch_size, _ in exporter._PACKAGE_VARIANTS:
        contract = exporter._variant_contract(policy, batch_size)
        assert contract["fixed_shape"] is True
        assert [tensor["dtype"] for tensor in contract["inputs"]] == [
            "float32",
            "float32",
            "float32",
            "int32",
        ]
        expected_mask = (
            [batch_size, 1, 1008, 1008] if policy == "hard" else [batch_size, 1, 288, 288]
        )
        assert contract["inputs"][1] == {
            "name": "owned_tracker_mask" if policy == "hard" else "final_mask",
            "dtype": "float32",
            "shape": expected_mask,
        }
        assert contract["outputs"] == [
            {
                "name": "packed_memory_and_position",
                "dtype": "float32",
                "shape": ([2, 5184, 1, 64] if batch_size == 1 else [2, 2, 5184, 64]),
            }
        ]


def test_mask_policy_keeps_soft_preparation_and_consumes_owned_hard_mask() -> None:
    source = inspect.getsource(exporter._prepare_memory_mask)
    hard_branch = source.split("if hard_mask:", maxsplit=1)[1].split("else:", maxsplit=1)[0]

    assert "mask = memory_mask.float()" in hard_branch
    assert "interpolate" not in hard_branch
    assert "argmax" not in source
    assert "tracker_logits" not in source
    assert "size=(_MEMORY_MASK_SIZE, _MEMORY_MASK_SIZE)" in source
    assert "suppress_area_shrinkage.reshape(batch_size, 1, 1, 1) > 0" in source
    assert "torch.clamp(resized_logits, max=-10.0)" in source
    assert "mask = torch.sigmoid(resized_logits)" in source
    assert "mask = mask * _SIGMOID_SCALE + _SIGMOID_BIAS" in source
    assert "antialias=True" in source


def test_memory_program_uses_meta_reductions_with_transformers_leaf_weights() -> None:
    layer_norm_source = inspect.getsource(exporter._meta_layer_norm_2d)
    module_source = inspect.getsource(exporter._make_memory_module)

    assert "value.mean(1, keepdim=True)" in layer_norm_source
    assert "(value - mean).pow(2).mean(1, keepdim=True)" in layer_norm_source
    assert "torch.sqrt(variance + layer_norm.eps)" in layer_norm_source
    assert "layer_norm.weight[:, None, None]" in layer_norm_source
    assert "_conv2d_from_leaf" in module_source
    assert "_linear_from_leaf" in module_source
    assert 'functional.gelu(memory, approximate="none")' in module_source
    assert "memory = layer.scale * memory" in module_source
    assert "self.memory_encoder(" not in module_source


def test_meta_layer_norm_2d_matches_literal_channels_first_program() -> None:
    torch = pytest.importorskip("torch")
    layer_norm = torch.nn.LayerNorm(4, eps=1e-6)
    with torch.no_grad():
        layer_norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        layer_norm.bias.copy_(torch.tensor([-0.25, 0.0, 0.25, 0.5]))
    value = torch.tensor(
        [
            [
                [[1.0, -2.0], [3.0, 4.0]],
                [[-1.0, 0.5], [2.0, -3.0]],
                [[2.0, 1.5], [-4.0, 0.0]],
                [[0.5, 3.0], [1.0, -1.0]],
            ]
        ]
    )

    mean = value.mean(1, keepdim=True)
    variance = (value - mean).pow(2).mean(1, keepdim=True)
    expected = (value - mean) / torch.sqrt(variance + layer_norm.eps)
    expected = layer_norm.weight[:, None, None] * expected + layer_norm.bias[:, None, None]

    actual = exporter._meta_layer_norm_2d(torch, value, layer_norm)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_bfloat16_carrier_rounding_matches_torch_cast_exactly() -> None:
    torch = pytest.importorskip("torch")
    ordinary = torch.tensor(
        [
            -1024.125,
            -3.1415927,
            -0.001234567,
            -0.0,
            0.0,
            0.001234567,
            1.0001,
            3.1415927,
            1024.125,
        ],
        dtype=torch.float32,
    )
    edge_bits = torch.tensor(
        [
            0x00000001,  # smallest FP32 subnormal
            0x007FFFFF,  # largest FP32 subnormal
            0x3F808000,  # exact tie with an even lower BF16 mantissa
            0x3F818000,  # exact tie with an odd lower BF16 mantissa
            0x7F7FFFFF,  # largest finite FP32 value; rounds to BF16 infinity
        ],
        dtype=torch.int32,
    ).view(torch.float32)
    values = torch.cat((ordinary, edge_bits))

    expected = values.to(torch.bfloat16).float()
    actual = exporter._round_bfloat16_carrier(torch, values)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_package_validation_enforces_bfloat16_recurrent_state_boundary() -> None:
    source = inspect.getsource(exporter._validate_package)
    module_source = inspect.getsource(exporter._make_memory_module)

    assert "actual == actual.to(torch.bfloat16).float()" in source
    assert "did not preserve its BF16 state boundary" in source
    assert "_round_bfloat16_carrier(torch, memory)" in module_source
    assert "_round_bfloat16_carrier(torch, position)" in module_source


def test_fixed_memory_position_is_literal_fp32_meta_encoding() -> None:
    torch = pytest.importorskip("torch")
    actual = exporter._fixed_meta_position_encoding(torch, device=torch.device("cpu"))

    position_features = exporter._MEMORY_CHANNELS // 2
    height = exporter._FEATURE_SIZE
    y_embed = torch.arange(1, height + 1, dtype=torch.float32).view(1, -1, 1).repeat(1, 1, height)
    x_embed = torch.arange(1, height + 1, dtype=torch.float32).view(1, 1, -1).repeat(1, height, 1)
    y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * (2 * exporter.math.pi)
    x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * (2 * exporter.math.pi)
    dimensions = torch.arange(position_features, dtype=torch.float32)
    denominator = 10000 ** (2 * (dimensions // 2) / position_features)
    position_x = x_embed[:, :, :, None] / denominator
    position_y = y_embed[:, :, :, None] / denominator
    position_x = torch.stack(
        (position_x[:, :, :, 0::2].sin(), position_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    position_y = torch.stack(
        (position_y[:, :, :, 0::2].sin(), position_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    expected = torch.cat((position_y, position_x), dim=3).permute(0, 3, 1, 2).contiguous()

    assert actual.shape == (1, 64, 72, 72)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_hard_module_accepts_common_suppression_input_but_uses_hard_policy_only() -> None:
    module_source = inspect.getsource(exporter._make_memory_module)
    mask_source = inspect.getsource(exporter._prepare_memory_mask)

    assert "suppress_area_shrinkage" in module_source
    hard_branch = mask_source.split("if hard_mask:", maxsplit=1)[1].split("else:", maxsplit=1)[0]
    assert "suppress_area_shrinkage" not in hard_branch
    assert "return torch.stack((memory, position), dim=0).contiguous()" in module_source


def test_exporter_uses_only_transformers_tracker_modules_and_no_onnx() -> None:
    source = Path(exporter.__file__).read_text(encoding="utf-8")

    assert "transformers.models.sam3_tracker_video.modeling_sam3_tracker_video" in source
    assert "model.memory_encoder" in source
    assert "_fixed_meta_position_encoding" in source
    assert "import onnx" not in source
    assert "from sam3." not in source
    assert "import sam3." not in source


def test_model_loading_is_strict_except_for_removed_vision_weights() -> None:
    calls = []
    model = _FakeModel()

    class ModelClass:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            calls.append((model_dir, kwargs))
            return model, {
                "missing_keys": [],
                "unexpected_keys": ["tracker_model.vision_encoder.backbone.weight"],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    dependencies = exporter._Dependencies(_FakeTorch(), ModelClass, "5.2.0")
    loaded = exporter._load_tracker_model(
        dependencies,
        Path("/models/sam3"),
        "cuda:0",
    )

    assert loaded is model
    assert model.target == "cuda:0"
    assert calls == [
        (
            Path("/models/sam3"),
            {
                "local_files_only": True,
                "remove_vision_encoder": True,
                "dtype": "float32",
                "output_loading_info": True,
            },
        )
    ]


@pytest.mark.parametrize(
    ("diagnostics", "match"),
    [
        ({"missing_keys": ["memory_encoder.projection.weight"]}, "did not load exactly"),
        ({"unexpected_keys": ["unmapped_tracker.weight"]}, "did not load exactly"),
    ],
)
def test_model_loading_fails_closed_on_non_vision_key(
    diagnostics: dict[str, list[str]],
    match: str,
) -> None:
    class ModelClass:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):  # noqa: ARG003
            return _FakeModel(), diagnostics

    dependencies = exporter._Dependencies(_FakeTorch(), ModelClass, "5.2.0")
    with pytest.raises(RuntimeError, match=match):
        exporter._load_tracker_model(dependencies, Path("/models/sam3"), "cuda:0")


@pytest.mark.parametrize(
    ("name", "incompatible"),
    [
        ("mask_downsampler_hidden_act", "relu"),
        ("mask_downsampler_total_stride", 8),
        ("memory_fuser_hidden_act", "relu"),
    ],
)
def test_model_loading_fails_closed_on_incompatible_memory_config(
    name: str,
    incompatible: object,
) -> None:
    model = _FakeModel()
    setattr(model.config, name, incompatible)

    class ModelClass:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):  # noqa: ARG003
            return model, {"missing_keys": [], "unexpected_keys": []}

    dependencies = exporter._Dependencies(_FakeTorch(), ModelClass, "5.2.0")
    with pytest.raises(RuntimeError, match="configuration is unsupported"):
        exporter._load_tracker_model(dependencies, Path("/models/sam3"), "cuda:0")


def test_export_builds_four_content_addressed_packages_and_reuses_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch()
    dependencies = exporter._Dependencies(torch, object(), "5.2.0")
    compile_calls = []
    load_calls = []

    def fake_load(deps, model_dir, device):
        load_calls.append((deps, model_dir, device))
        return object()

    def fake_compile(deps, model, staging, device):  # noqa: ARG001
        compile_calls.append((deps, device))
        staged = []
        validation = []
        for policy, batch_size, hard_mask in exporter._PACKAGE_VARIANTS:
            path = staging / f"{policy}-b{batch_size}.pt2"
            path.write_bytes(f"{policy}:b{batch_size}:package".encode())
            staged.append(exporter._StagedPackage(policy, batch_size, hard_mask, path))
            validation.append(_passing_case(policy, batch_size, hard_mask))
        return tuple(staged), tuple(validation)

    monkeypatch.setattr(exporter, "_load_dependencies", lambda: dependencies)
    monkeypatch.setattr(exporter, "_producer_abi", lambda deps, index: _producer())
    monkeypatch.setattr(exporter, "_exporter_source_digest", lambda: "a" * 64)
    monkeypatch.setattr(exporter, "_load_tracker_model", fake_load)
    monkeypatch.setattr(exporter, "_compile_fixed_packages", fake_compile)

    artifact = exporter.export_sam3_tracker_memory_aoti(
        _model_dir(tmp_path),
        cache_dir=tmp_path / "cache",
    )

    assert [
        (package.policy, package.batch_size, package.hard_mask) for package in artifact.packages
    ] == list(exporter._PACKAGE_VARIANTS)
    assert len({package.sha256 for package in artifact.packages}) == 4
    assert all(package.path.name.endswith(f"{package.sha256}.pt2") for package in artifact.packages)
    assert [name for name, _ in artifact.bundle_sections] == [
        exporter.TRACKER_MEMORY_AOTI_MANIFEST_SECTION,
        "sam3_tracker_memory_soft_b1.pt2",
        "sam3_tracker_memory_hard_b1.pt2",
        "sam3_tracker_memory_soft_b2.pt2",
        "sam3_tracker_memory_hard_b2.pt2",
    ]
    manifest = json.loads(artifact.manifest_bytes)
    assert manifest["schema_version"] == 2
    assert manifest["input_abi"][1]["tensors"][1] == {
        "name": "owned_tracker_mask",
        "dtype": "float32",
        "shape": ["B", 1, 1008, 1008],
    }
    assert manifest["implementation"]["source_import_policy"] == "transformers-only"
    assert manifest["exporter_sha256"] == "a" * 64
    assert manifest["producer"]["torch_aoti_abi_version"] == 147492887796383744
    assert len(manifest["package_validation"]["cases"]) == 4
    assert all(case["passed"] for case in manifest["package_validation"]["cases"])
    for package in artifact.packages:
        assert package.package_global == exporter._package_global(
            package.policy,
            package.batch_size,
            package.sha256,
        )
    assert artifact.package(batch_size=1, hard_mask=False).policy == "soft"
    assert artifact.package(batch_size=2, hard_mask=True).policy == "hard"
    with pytest.raises(ValueError, match="soft/hard B1/B2"):
        artifact.package(batch_size=3, hard_mask=False)
    assert len(compile_calls) == 1
    assert len(load_calls) == 1
    assert torch.cuda.selected == 0
    assert torch.seeds == [20260717]
    assert torch.random.devices == [[0]]
    assert torch.cuda.empty_cache_calls == 1
    with pytest.raises(FrozenInstanceError):
        artifact.producer_abi = _producer()

    cached = exporter.export_sam3_tracker_memory_aoti(
        tmp_path / "model",
        cache_dir=tmp_path / "cache",
    )
    assert cached.manifest_bytes == artifact.manifest_bytes
    assert len(compile_calls) == 1
    assert len(load_calls) == 1


def test_cache_validation_rejects_package_or_policy_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch()
    dependencies = exporter._Dependencies(torch, object(), "5.2.0")

    def fake_compile(deps, model, staging, device):  # noqa: ARG001
        staged = []
        validation = []
        for policy, batch_size, hard_mask in exporter._PACKAGE_VARIANTS:
            path = staging / f"{policy}-b{batch_size}.pt2"
            path.write_bytes(f"{policy}:b{batch_size}".encode())
            staged.append(exporter._StagedPackage(policy, batch_size, hard_mask, path))
            validation.append(_passing_case(policy, batch_size, hard_mask))
        return tuple(staged), tuple(validation)

    monkeypatch.setattr(exporter, "_load_dependencies", lambda: dependencies)
    monkeypatch.setattr(exporter, "_producer_abi", lambda deps, index: _producer())
    monkeypatch.setattr(exporter, "_exporter_source_digest", lambda: "b" * 64)
    monkeypatch.setattr(exporter, "_load_tracker_model", lambda *args: object())
    monkeypatch.setattr(exporter, "_compile_fixed_packages", fake_compile)
    artifact = exporter.export_sam3_tracker_memory_aoti(
        _model_dir(tmp_path),
        cache_dir=tmp_path / "cache",
    )

    soft = artifact.package(batch_size=1, hard_mask=False)
    original = soft.path.read_bytes()
    soft.path.write_bytes(original + b"tampered")
    assert exporter._validate_cached_directory(artifact.cache_directory) is None
    soft.path.write_bytes(original)

    manifest_path = artifact.cache_directory / exporter.TRACKER_MEMORY_AOTI_MANIFEST_SECTION
    manifest = json.loads(manifest_path.read_bytes())
    manifest["mask_policy"]["soft"] = "different order"
    manifest_path.write_text(json.dumps(manifest))
    assert exporter._validate_cached_directory(artifact.cache_directory) is None


def test_cache_key_commits_to_model_exporter_and_target_abi() -> None:
    producer = _producer()
    baseline = exporter._cache_key("11" * 32, "22" * 32, producer)

    assert exporter._cache_key("33" * 32, "22" * 32, producer) != baseline
    assert exporter._cache_key("11" * 32, "44" * 32, producer) != baseline
    changed_abi = exporter.MemoryAotiProducerAbi(
        **{**producer.__dict__, "torch_aoti_abi_version": producer.torch_aoti_abi_version + 1}
    )
    assert exporter._cache_key("11" * 32, "22" * 32, changed_abi) != baseline


def test_model_digest_commits_to_config_and_all_weight_shards(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    shard = model_dir / "model-00002-of-00002.safetensors"
    shard.write_bytes(b"second-shard")
    first = exporter._model_source_digest(model_dir)
    shard.write_bytes(b"changed-shard")
    second = exporter._model_source_digest(model_dir)

    assert first != second
    assert len(first) == hashlib.sha256().digest_size * 2


def test_producer_abi_records_aoti_version_and_rejects_unknown_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch()
    dependencies = exporter._Dependencies(torch, object(), "5.2.0")
    monkeypatch.setattr(exporter, "_torch_aoti_abi_version", lambda value: 1234)

    producer = exporter._producer_abi(dependencies, 0)
    assert producer.torch_aoti_abi_version == 1234
    assert producer.compute_capability == (8, 9)

    torch.cuda.get_device_capability = lambda device_index: (9, 0)
    with pytest.raises(RuntimeError, match="does not support compute capability 9.0"):
        exporter._producer_abi(dependencies, 0)


def test_compile_contract_is_fixed_shape_and_smokes_loaded_package() -> None:
    compile_source = inspect.getsource(exporter._compile_one_package)
    build_source = inspect.getsource(exporter._compile_fixed_packages)

    assert "strict=False" in compile_source
    assert "dynamic_shapes" not in compile_source
    assert "aoti_compile_and_package" in compile_source
    assert "_validate_package(" in build_source
    assert "aoti_load_package" in inspect.getsource(exporter._validate_package)
