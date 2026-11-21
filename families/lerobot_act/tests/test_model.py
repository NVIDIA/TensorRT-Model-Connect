# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

from ..builder import _position_embedding_2d, define_act_network
from ..checkpoint import load_checkpoint
from ..model import _validate_initial_policy, build
from ..support import describe


def _config() -> dict:
    return {
        "type": "act",
        "n_obs_steps": 1,
        "chunk_size": 100,
        "n_action_steps": 100,
        "vision_backbone": "resnet18",
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "pre_norm": False,
        "use_vae": True,
        "latent_dim": 32,
        "temporal_ensemble_coeff": None,
        "input_features": {
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [14]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    }


def test_initial_policy_contract_and_support() -> None:
    raw = _config()
    _validate_initial_policy(raw)
    support = describe(ModelMetadata(config=raw, model_index={}))
    assert support is not None
    assert support.tasks == ("image_state_to_action_chunk",)
    assert support.default_task == "image_state_to_action_chunk"
    family, resolved = resolve_family(ModelMetadata(config=raw, model_index={}))
    assert family == "lerobot_act"
    assert resolved == support


def test_build_refuses_the_retired_robot_control_task() -> None:
    request = SimpleNamespace(mtp_seq_len=None, dynamic_kv_cache=False, task="robot_control")
    with pytest.raises(ValueError, match="image_state_to_action_chunk"):
        build(request, writer=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 99),
        ("n_action_steps", 1),
        ("temporal_ensemble_coeff", 0.01),
        ("pre_norm", True),
    ],
)
def test_initial_policy_contract_rejects_semantic_drift(field: str, value: object) -> None:
    raw = _config()
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        _validate_initial_policy(raw)


def test_checkpoint_requires_the_declared_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        load_checkpoint(tmp_path)


def test_act_2d_position_embedding_is_stable_and_finite() -> None:
    positions = _position_embedding_2d(15, 20, 256)
    assert positions.shape == (300, 512)
    assert positions.dtype == np.float32
    assert np.isfinite(positions).all()
    np.testing.assert_allclose(positions[0, 0], np.sin(2 * np.pi / 15), atol=1.0e-6)
    np.testing.assert_allclose(positions[0, 256], np.sin(2 * np.pi / 20), atol=1.0e-6)


def _request(tmp_path, precision: str) -> SimpleNamespace:
    return SimpleNamespace(
        mtp_seq_len=None,
        dynamic_kv_cache=False,
        task="image_state_to_action_chunk",
        precision=precision,
        max_sequence_length=None,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=None,
        model_dir=tmp_path,
        verbose=False,
    )


def test_bf16_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="fp32 and fp16"):
        build(_request(tmp_path, "bf16"), writer=None)


def test_fp16_is_accepted(tmp_path) -> None:
    """fp16 passes the precision check and goes on to read the checkpoint config."""
    with pytest.raises(ValueError, match="config.json"):
        build(_request(tmp_path, "fp16"), writer=None)


def test_the_default_precision_is_fp32() -> None:
    support = describe(ModelMetadata(config=_config(), model_index={}))
    assert support.default_precision == "fp32"


def _random_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random tensors with every name and shape the ACT graph reads (ResNet-18 backbone)."""
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}

    def put(name: str, *shape: int) -> None:
        weights[name] = (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def norm(prefix: str, channels: int, *, batch: bool) -> None:
        weights[f"{prefix}.weight"] = np.ones(channels, np.float32)
        put(f"{prefix}.bias", channels)
        if batch:
            put(f"{prefix}.running_mean", channels)
            weights[f"{prefix}.running_var"] = np.ones(channels, np.float32)

    put("model.backbone.conv1.weight", 64, 3, 7, 7)
    norm("model.backbone.bn1", 64, batch=True)
    channels = 64
    for stage, width in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            prefix = f"model.backbone.layer{stage}.{block}"
            inputs = channels if block == 0 else width
            put(f"{prefix}.conv1.weight", width, inputs, 3, 3)
            norm(f"{prefix}.bn1", width, batch=True)
            put(f"{prefix}.conv2.weight", width, width, 3, 3)
            norm(f"{prefix}.bn2", width, batch=True)
            if stage > 1 and block == 0:
                put(f"{prefix}.downsample.0.weight", width, inputs, 1, 1)
                norm(f"{prefix}.downsample.1", width, batch=True)
        channels = width
    put("model.encoder_img_feat_input_proj.weight", 512, 512, 1, 1)
    put("model.encoder_img_feat_input_proj.bias", 512)
    for name, inputs in (("encoder_latent_input_proj", 32), ("encoder_robot_state_input_proj", 14)):
        put(f"model.{name}.weight", 512, inputs)
        put(f"model.{name}.bias", 512)
    put("model.encoder_1d_feature_pos_embed.weight", 2, 512)
    put("model.decoder_pos_embed.weight", 100, 512)

    def attention(prefix: str) -> None:
        put(f"{prefix}.in_proj_weight", 1536, 512)
        put(f"{prefix}.in_proj_bias", 1536)
        put(f"{prefix}.out_proj.weight", 512, 512)
        put(f"{prefix}.out_proj.bias", 512)

    def feed_forward(prefix: str) -> None:
        put(f"{prefix}.linear1.weight", 3200, 512)
        put(f"{prefix}.linear1.bias", 3200)
        put(f"{prefix}.linear2.weight", 512, 3200)
        put(f"{prefix}.linear2.bias", 512)

    for index in range(4):
        prefix = f"model.encoder.layers.{index}"
        attention(f"{prefix}.self_attn")
        feed_forward(prefix)
        norm(f"{prefix}.norm1", 512, batch=False)
        norm(f"{prefix}.norm2", 512, batch=False)
    prefix = "model.decoder.layers.0"
    attention(f"{prefix}.self_attn")
    attention(f"{prefix}.multihead_attn")
    feed_forward(prefix)
    for name in ("norm1", "norm2", "norm3"):
        norm(f"{prefix}.{name}", 512, batch=False)
    norm("model.decoder.norm", 512, batch=False)
    put("model.action_head.weight", 14, 512)
    put("model.action_head.bias", 14)
    weights["normalize_inputs.buffer_observation_images_top.mean"] = np.full(
        (3, 1, 1), 0.5, np.float32
    )
    weights["normalize_inputs.buffer_observation_images_top.std"] = np.full(
        (3, 1, 1), 0.25, np.float32
    )
    for name, size in (
        ("normalize_inputs.buffer_observation_state", 14),
        ("unnormalize_outputs.buffer_action", 14),
    ):
        weights[f"{name}.mean"] = np.zeros(size, np.float32)
        weights[f"{name}.std"] = np.ones(size, np.float32)
    return weights


def _network(precision: str):
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the TensorRT builder needs a GPU")
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = define_act_network(network, _random_weights(), precision)
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    return trt, builder, network, graph, layers


def _io(trt, network) -> list:
    tensors = [network.get_input(i) for i in range(network.num_inputs)]
    tensors += [network.get_output(i) for i in range(network.num_outputs)]
    return [(tensor.name, tensor.dtype, tuple(tensor.shape)) for tensor in tensors]


def test_the_fp32_network_has_no_cast() -> None:
    """With the fp32 work type every cast helper is a no-op, so the network is the one of fp32-only builds."""
    trt, _builder, network, _graph, layers = _network("fp32")
    assert not [layer.name for layer in layers if layer.type == trt.LayerType.CAST]
    assert {layer.get_output(0).dtype for layer in layers} == {trt.float32}
    assert _io(trt, network) == [
        ("observation_image", trt.float32, (1, 480, 640, 3)),
        ("observation_state", trt.float32, (1, 14)),
        ("actions", trt.float32, (1, 100, 14)),
    ]


def test_the_fp16_network_runs_fp16_except_the_cross_attention_core() -> None:
    trt, _builder, network, _graph, layers = _network("fp16")
    assert _io(trt, network) == [
        ("observation_image", trt.float32, (1, 480, 640, 3)),
        ("observation_state", trt.float32, (1, 14)),
        ("actions", trt.float32, (1, 100, 14)),
    ]
    heavy = {
        layer.name: layer.get_output(0).dtype
        for layer in layers
        if layer.type
        in (trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.CONVOLUTION, trt.LayerType.SOFTMAX)
    }
    float32 = sorted(name for name, dtype in heavy.items() if dtype == trt.float32)
    # The TensorRT workaround of builder._ActGraph.attention: scores, softmax and context of
    # the decoder cross-attention run in float32; every other matmul, convolution and softmax
    # (21 convolutions, 37 linears, the other five attentions) runs in fp16.
    assert float32 == [
        "decoder.cross_attention.context_heads",
        "decoder.cross_attention.scores",
        "decoder.cross_attention.softmax",
    ]
    assert len(heavy) == 21 + 37 + 6 * 3
    assert {dtype for name, dtype in heavy.items() if name not in float32} == {trt.float16}
