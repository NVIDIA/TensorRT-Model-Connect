# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.sam3 import tracker_step_aoti_exporter as exporter


class _FakeCuda:
    def __init__(self) -> None:
        self.selected: int = 0
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
        for name in (
            "memory_attention",
            "mask_decoder",
            "prompt_encoder",
            "object_pointer_proj",
            "temporal_positional_encoding_projection_layer",
            "memory_temporal_positional_encoding",
            "no_object_pointer",
            "get_image_wide_positional_embeddings",
        ):
            setattr(self, name, object())
        self.target = None

    def eval(self):
        return self

    def to(self, target):
        self.target = target
        return self

    def parameters(self):
        return ()


def _producer() -> exporter.TrackerAotiProducerAbi:
    return exporter.TrackerAotiProducerAbi(
        torch_version="2.12.0+cu130",
        transformers_version="5.2.0",
        cuda_version="13.0",
        compute_capability=(8, 9),
        host_architecture="x86_64",
        torch_cxx11_abi=True,
    )


def _model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"sam3_video"}\n')
    (model_dir / "model.safetensors").write_bytes(b"tracker-weights")
    return model_dir


def _decoder_function_source(name: str) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(exporter._make_decoder_module)))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return ast.unparse(matches[0])


def test_encoder_export_contract_uses_non_singleton_seed_and_full_dynamic_bounds() -> None:
    dimensions = []

    def dimension(name: str, *, min: int, max: int):
        value = SimpleNamespace(name=name, minimum=min, maximum=max)
        dimensions.append(value)
        return value

    fake_torch = SimpleNamespace(export=SimpleNamespace(Dim=dimension))
    shapes = exporter._encoder_dynamic_shapes(fake_torch, batch_size=2)

    assert exporter._REPRESENTATIVE_MEMORY_COUNT == 4
    assert exporter._REPRESENTATIVE_POINTER_COUNT == 3
    assert [(value.minimum, value.maximum) for value in dimensions] == [(1, 10), (1, 19)]
    assert shapes[2][1] is shapes[3][1] is shapes[4][1]
    assert shapes[5][1] is shapes[6][1]
    assert shapes[0] is None and shapes[1] is None and shapes[7] is None


def test_ten_carrier_abi_keeps_offsets_and_maximum_int32() -> None:
    abi = {value.name: value for value in exporter.TRACKER_TEN_CARRIER_ABI}

    assert len(abi) == 10
    assert abi["tracker_feature_0"].dtype == "float32"
    assert abi["tracker_feature_0"].shape == (1, 32, 288, 288)
    assert abi["tracker_feature_1"].dtype == "float32"
    assert abi["tracker_feature_1"].shape == (1, 64, 144, 144)
    assert abi["tracker_feature_2"].dtype == "float32"
    assert abi["tracker_position_2"].dtype == "float32"
    assert abi["memory_features"].shape[1] == "M"
    assert abi["object_pointers"].shape[1] == "P"
    assert abi["memory_temporal_offsets"].dtype == "int32"
    assert abi["object_pointer_temporal_offsets"].dtype == "int32"
    assert abi["max_object_pointers_to_use"].dtype == "int32"


def test_decoder_consumes_preprojected_features_at_meta_bf16_boundary() -> None:
    source = inspect.getsource(exporter._make_decoder_module)

    # TRT transports the already-projected conv_s0/conv_s1 maps as FP32.  Meta
    # rounds those maps to BF16 before the recurrent decoder's residual adds.
    assert "tracker_feature_0.to(torch.bfloat16)" in source
    assert "tracker_feature_1.to(torch.bfloat16)" in source
    assert "self.mask_decoder.conv_s0(" not in source
    assert "self.mask_decoder.conv_s1(" not in source


def test_exact_decoder_attention_keeps_rank_three_token_boundary() -> None:
    source = _decoder_function_source("_attention")

    assert "batch, query_length, _ = query.shape" in source
    assert "key_length = key.shape[1]" in source
    assert "module.q_proj(query).reshape(batch, query_length, heads, head_dim)" in source
    assert "module.k_proj(key).reshape(batch, key_length, heads, head_dim)" in source
    assert "module.v_proj(value).reshape(batch, key_length, heads, head_dim)" in source
    assert "scaled_dot_product_attention(query, key, value, dropout_p=0.0)" in source
    assert "reshape(batch, query_length, heads * head_dim)" in source
    assert "return module.o_proj(attended)" in source
    assert "unsqueeze" not in source


def test_exact_decoder_uses_manual_channels_first_layer_norm() -> None:
    layer_norm_source = _decoder_function_source("_layer_norm_2d")
    decode_source = _decoder_function_source("_decode")

    assert "mean = value.mean(1, keepdim=True)" in layer_norm_source
    assert "variance = centered.pow(2).mean(1, keepdim=True)" in layer_norm_source
    assert "module.weight[:, None, None] * normalized" in layer_norm_source
    assert "module.bias[:, None, None]" in layer_norm_source
    assert "permute" not in layer_norm_source
    assert "self._layer_norm_2d(decoder.upscale_layer_norm, upscaled)" in decode_source
    assert "decoder.upscale_layer_norm(upscaled)" not in decode_source


def test_exact_decoder_preserves_meta_output_token_order() -> None:
    source = _decoder_function_source("_decode")

    object_score = source.index("decoder.obj_score_token.weight")
    iou = source.index("decoder.iou_token.weight")
    masks = source.index("decoder.mask_tokens.weight")
    assert object_score < iou < masks
    assert "iou_token = points[:, 1]" in source
    assert "mask_tokens = points[:, 2:2 + decoder.num_mask_tokens]" in source
    assert "return (masks[:, 1:], ious[:, 1:], mask_tokens[:, 1:], scores)" in source


def test_exact_decoder_does_not_call_hf_rank_four_mask_decoder() -> None:
    source = _decoder_function_source("forward")

    assert "self._decode(conditioned, high0, high1)" in source
    assert "self.mask_decoder(" not in source
    assert "image_embeddings=" not in source
    assert "image_positional_embeddings=" not in source
    assert "sparse_prompt_embeddings=" not in source
    assert "dense_prompt_embeddings=" not in source
    assert "high_resolution_features=" not in source


def test_encoder_rounds_current_frame_inputs_at_meta_bf16_boundary() -> None:
    source = inspect.getsource(exporter._make_encoder_module)

    assert "tracker_feature_2.to(torch.bfloat16)" in source
    assert "tracker_position_2.to(torch.bfloat16)" in source


def test_default_local_model_loading_allows_only_removed_vision_weights() -> None:
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

    torch = _FakeTorch()
    dependencies = exporter._Dependencies(
        torch,
        ModelClass,
        "5.2.0",
    )
    loaded = exporter._load_tracker_model(dependencies, Path("/models/sam3"), "cuda:0")

    assert loaded is model
    assert model.target == "cuda:0"
    assert calls == [
        (
            Path("/models/sam3"),
            {
                "local_files_only": True,
                "remove_vision_encoder": True,
                "attn_implementation": "sdpa",
                "dtype": "float32",
                "output_loading_info": True,
            },
        )
    ]


def test_model_loading_fails_closed_on_missing_tracker_key() -> None:
    class ModelClass:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):  # noqa: ARG003
            return _FakeModel(), {
                "missing_keys": ["memory_attention.layers.0.weight"],
                "unexpected_keys": [],
            }

    dependencies = exporter._Dependencies(
        _FakeTorch(),
        ModelClass,
        "5.2.0",
    )
    with pytest.raises(RuntimeError, match="did not load exactly"):
        exporter._load_tracker_model(dependencies, Path("/models/sam3"), "cuda:0")


def test_export_builds_four_content_addressed_packages_and_paired_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _FakeTorch()

    dependencies = exporter._Dependencies(torch, object(), "5.2.0")
    compile_calls = []
    model_load_calls = []

    def fake_load_model(deps, model_dir, device):
        model_load_calls.append((deps, model_dir, device))
        return object()

    def fake_compile(deps, model, staging, device):  # noqa: ARG001
        compile_calls.append((deps, device))
        staged = []
        for stage, batch_size in exporter._PACKAGE_ORDER:
            path = staging / f"{stage}-b{batch_size}.pt2"
            path.write_bytes(f"{stage}:b{batch_size}:package".encode())
            staged.append(exporter._StagedPackage(stage, batch_size, path))
        validation = []
        for batch_size in (1, 2):
            for memory_count, pointer_count in exporter._SMOKE_ENCODER_SHAPES:
                validation.append(
                    {
                        "stage": "encoder",
                        "batch_size": batch_size,
                        "memory_count": memory_count,
                        "pointer_count": pointer_count,
                        "cosine": 1.0,
                        "relative_l2": 0.0,
                        "maximum_absolute_error": 0.0,
                        "passed": True,
                    }
                )
            validation.append(
                {
                    "stage": "decoder",
                    "batch_size": batch_size,
                    "memory_count": None,
                    "pointer_count": None,
                    "cosine": 1.0,
                    "relative_l2": 0.0,
                    "maximum_absolute_error": 0.0,
                    "binary_mask_agreement": 1.0,
                    "passed": True,
                }
            )
        return tuple(staged), tuple(validation)

    monkeypatch.setattr(exporter, "_load_dependencies", lambda: dependencies)
    monkeypatch.setattr(exporter, "_producer_abi", lambda deps, device_index: _producer())
    monkeypatch.setattr(exporter, "_exporter_source_digest", lambda: "a" * 64)
    monkeypatch.setattr(exporter, "_load_tracker_model", fake_load_model)
    monkeypatch.setattr(exporter, "_compile_split_packages", fake_compile)

    artifact = exporter.export_sam3_tracker_split_aoti(
        _model_dir(tmp_path), cache_dir=tmp_path / "cache"
    )

    assert [(value.stage, value.batch_size) for value in artifact.packages] == list(
        exporter._PACKAGE_ORDER
    )
    assert len({value.sha256 for value in artifact.packages}) == 4
    assert all(value.path.name.endswith(f"{value.sha256}.pt2") for value in artifact.packages)
    assert [name for name, _ in artifact.bundle_sections] == [
        exporter.TRACKER_SPLIT_AOTI_MANIFEST_SECTION,
        "sam3_tracker_encoder_b1_dynamic.pt2",
        "sam3_tracker_decoder_b1_static.pt2",
        "sam3_tracker_encoder_b2_dynamic.pt2",
        "sam3_tracker_decoder_b2_static.pt2",
    ]
    manifest = json.loads(artifact.manifest_bytes)
    assert manifest["dynamic_contract"] == {
        "memory_count": [1, 10],
        "pointer_count": [1, 19],
        "representative_memory_count": 4,
        "representative_pointer_count": 3,
    }
    assert len(manifest["package_validation"]["cases"]) == 8
    assert all(case["passed"] for case in manifest["package_validation"]["cases"])
    for batch_size in (1, 2):
        packages = {
            value.stage: value for value in artifact.packages if value.batch_size == batch_size
        }
        digest = hashlib.sha256(
            exporter._PIPELINE_DIGEST_DOMAIN
            + bytes.fromhex(packages["encoder"].sha256)
            + bytes.fromhex(packages["decoder"].sha256)
        ).hexdigest()
        assert artifact.pipeline_global(batch_size) == (
            f"trtmc.sam3.tracker_step.b{batch_size}.split_aoti.{digest[:20]}"
        )
    assert len(compile_calls) == 1
    assert len(model_load_calls) == 1
    assert torch.cuda.selected == 0
    assert torch.seeds == [20260717]
    assert torch.random.devices == [[0]]
    assert torch.cuda.empty_cache_calls == 1

    cached = exporter.export_sam3_tracker_split_aoti(
        tmp_path / "model", cache_dir=tmp_path / "cache"
    )
    assert cached.manifest_bytes == artifact.manifest_bytes
    assert len(compile_calls) == 1
    assert len(model_load_calls) == 1
    with pytest.raises(FrozenInstanceError):
        artifact.pipeline_global_b1 = "changed"


def test_pipeline_digest_changes_if_either_stage_changes() -> None:
    encoder = "11" * 32
    decoder = "22" * 32

    baseline = exporter._pipeline_digest(encoder, decoder)

    assert exporter._pipeline_digest("33" * 32, decoder) != baseline
    assert exporter._pipeline_digest(encoder, "44" * 32) != baseline


def test_export_fails_closed_on_unknown_gpu() -> None:
    torch = _FakeTorch()
    torch.cuda.get_device_capability = lambda device_index: (7, 5)
    dependencies = exporter._Dependencies(
        torch,
        object(),
        "5.2.0",
    )

    with pytest.raises(RuntimeError, match="does not support compute capability 7.5"):
        exporter._producer_abi(dependencies, 0)


def test_exporter_has_no_external_graph_or_environment_configuration() -> None:
    source = Path(exporter.__file__).read_text(encoding="utf-8").lower()

    assert "onnx" not in source
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "getenv(" not in source
    assert "trtmc_sam3" not in source
    assert "_single_frame_forward" not in source
    assert "meta-sam3" not in source
    assert "sam3.pt" not in source
    assert "spec_from_file_location" not in source
    assert "runpy." not in source
    assert "apply_rotary_pos_emb_2d" not in source
    assert "view_as_complex" not in source
