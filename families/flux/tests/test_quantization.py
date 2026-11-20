# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 NVFP4 checkpoint reader and tiny-engine checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from families.flux import quantization as q  # noqa: E402

DIM, HEADS, FFN, TEXT_DIM = 512, 4, 1536, 64


def _bf16(*shape, seed=0):
    gen = torch.Generator().manual_seed(seed + sum(shape))
    return (torch.randn(*shape, generator=gen) * 0.05).to(torch.bfloat16)


def _to_blocked(scale: torch.Tensor) -> torch.Tensor:
    """cuBLAS 128x4 tile swizzle, as the official checkpoint stores block scales."""
    rows, cols = scale.shape
    view = scale.view(torch.uint8).reshape(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    tiles = view.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return tiles.reshape(rows, cols).view(torch.float8_e4m3fn)


CANONICAL: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _quantized(tensors: dict, stem: str, out: int, inp: int, seed: int = 0) -> None:
    """Write a projection in the official layout; remember the ModelOpt-layout truth."""
    gen = torch.Generator().manual_seed(seed + out + inp)
    canonical = torch.randint(0, 256, (out, inp // 2), generator=gen, dtype=torch.uint8)
    scale = (torch.rand(out, inp // 16, generator=gen) * 0.9 + 0.1).to(torch.float8_e4m3fn)
    CANONICAL[stem] = (canonical, scale.view(torch.uint8))
    # official files pair the first element in the HIGH nibble
    tensors[f"{stem}.weight"] = ((canonical & 0x0F) << 4) | (canonical >> 4)
    tensors[f"{stem}.weight_scale"] = _to_blocked(scale)
    tensors[f"{stem}.weight_scale_2"] = torch.tensor(0.002)
    tensors[f"{stem}.input_scale"] = torch.tensor(0.02)


def write_tiny_checkpoint(path: Path, *, quantize_txt_attn: bool = False) -> None:
    t: dict = {}
    t["img_in.weight"] = _bf16(DIM, 128)
    t["txt_in.weight"] = _bf16(DIM, TEXT_DIM)
    for name in ("time_in", "guidance_in"):
        t[f"{name}.in_layer.weight"] = _bf16(DIM, 256)
        t[f"{name}.out_layer.weight"] = _bf16(DIM, DIM)
    t["double_stream_modulation_img.lin.weight"] = _bf16(6 * DIM, DIM, seed=1)
    t["double_stream_modulation_txt.lin.weight"] = _bf16(6 * DIM, DIM, seed=2)
    t["single_stream_modulation.lin.weight"] = _bf16(3 * DIM, DIM, seed=3)
    t["final_layer.adaLN_modulation.1.weight"] = _bf16(2 * DIM, DIM)
    t["final_layer.linear.weight"] = _bf16(128, DIM)
    p = "double_blocks.0"
    for side in ("img", "txt"):
        for norm in ("query_norm", "key_norm"):
            t[f"{p}.{side}_attn.norm.{norm}.scale"] = (torch.rand(128) + 0.5).to(torch.bfloat16)
    _quantized(t, f"{p}.img_attn.qkv", 3 * DIM, DIM)
    _quantized(t, f"{p}.img_attn.proj", DIM, DIM)
    for stem, out, inp in (("txt_attn.qkv", 3 * DIM, DIM), ("txt_attn.proj", DIM, DIM)):
        if quantize_txt_attn:
            _quantized(t, f"{p}.{stem}", out, inp)
        else:
            t[f"{p}.{stem}.weight"] = _bf16(out, inp)
    for side in ("img", "txt"):
        _quantized(t, f"{p}.{side}_mlp.0", 2 * FFN, DIM)
        _quantized(t, f"{p}.{side}_mlp.2", DIM, FFN)
    p = "single_blocks.0"
    _quantized(t, f"{p}.linear1", 3 * DIM + 2 * FFN, DIM)
    _quantized(t, f"{p}.linear2", DIM, DIM + FFN)
    for norm in ("query_norm", "key_norm"):
        t[f"{p}.norm.{norm}.scale"] = (torch.rand(128) + 0.5).to(torch.bfloat16)
    safetensors_torch.save_file(t, str(path))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("flux2_nvfp4") / "flux2-dev-nvfp4.safetensors"
    write_tiny_checkpoint(path)
    return path


def test_find_checkpoint_requires_exactly_one_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="FLUX.2-dev-NVFP4"):
        q.find_nvfp4_checkpoint(tmp_path)
    (tmp_path / "flux2-dev-nvfp4.safetensors").write_bytes(b"")
    assert q.find_nvfp4_checkpoint(tmp_path).name == "flux2-dev-nvfp4.safetensors"
    (tmp_path / "flux2-dev-nvfp4-mixed.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="exactly one"):
        q.find_nvfp4_checkpoint(tmp_path)


def test_reader_maps_projections_to_diffusers_names(checkpoint):
    projections, plain = q.read_nvfp4_checkpoint(checkpoint)
    assert set(projections) == {
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.ff.linear_in",
        "transformer_blocks.0.ff.linear_out",
        "transformer_blocks.0.ff_context.linear_in",
        "transformer_blocks.0.ff_context.linear_out",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
        "single_transformer_blocks.0.attn.to_out",
    }
    # Projections left in BF16 by the export stay plain tensors and are split by rows.
    for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
        assert plain[f"transformer_blocks.0.attn.{name}.weight"].shape == (DIM, DIM)
    assert plain["transformer_blocks.0.attn.to_add_out.weight"].shape == (DIM, DIM)
    assert plain["transformer_blocks.0.attn.norm_added_q.weight"].shape == (128,)
    assert plain["single_transformer_blocks.0.attn.norm_k.weight"].shape == (128,)


def test_fused_qkv_is_split_by_rows_and_shares_scalars(checkpoint):
    projections, _ = q.read_nvfp4_checkpoint(checkpoint)
    packed, scale = (t.numpy() for t in CANONICAL["double_blocks.0.img_attn.qkv"])
    for index, name in enumerate(("to_q", "to_k", "to_v")):
        layer = projections[f"transformer_blocks.0.attn.{name}"]
        rows = slice(index * DIM, (index + 1) * DIM)
        assert layer.out_features == DIM and layer.in_features == DIM
        assert layer.packed.dtype == np.uint8 and layer.packed.shape == (DIM, DIM // 2)
        assert layer.block_scale.shape == (DIM, DIM // 16)
        np.testing.assert_array_equal(layer.packed, packed[rows])
        np.testing.assert_array_equal(layer.block_scale, scale[rows])
        assert layer.global_scale == pytest.approx(0.002) and layer.input_scale == pytest.approx(
            0.02
        )


def test_loader_transposes_plain_weights_and_hides_quantized_weights(checkpoint):
    weights = q.load_flux2_dit_weights_nvfp4(checkpoint)
    assert weights["x_embedder.weight"].shape == (128, DIM)
    assert weights["context_embedder.weight"].shape == (TEXT_DIM, DIM)
    assert weights["time_text_embed.timestep_embedder.linear_1.weight"].shape == (256, DIM)
    assert weights["double_stream_modulation_img"].shape == (DIM, 6 * DIM)
    assert weights["single_stream_modulation"].shape == (DIM, 3 * DIM)
    assert weights["proj_out.weight"].shape == (DIM, 128)
    assert weights["transformer_blocks.0.attn.add_q_proj.weight"].shape == (DIM, DIM)
    assert weights["transformer_blocks.0.attn.norm_q.weight"].dtype == np.float32
    assert "transformer_blocks.0.ff.linear_in.nvfp4" in weights
    assert "transformer_blocks.0.ff.linear_in.weight" not in weights


def test_final_norm_swaps_shift_and_scale(checkpoint):
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt") as reader:
        original = reader.get_tensor("final_layer.adaLN_modulation.1.weight").float().numpy()
    weights = q.load_flux2_dit_weights_nvfp4(checkpoint)
    shift, scale = original[:DIM], original[DIM:]
    np.testing.assert_array_equal(
        weights["norm_out.linear.weight"].T, np.concatenate([scale, shift])
    )


def test_unswizzle_and_nibble_swap_invert_the_checkpoint_layout():
    gen = torch.Generator().manual_seed(1)
    scale = torch.randint(0, 255, (256, 8), generator=gen, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    restored = q.unswizzle_block_scales(
        _to_blocked(scale).view(torch.uint8).numpy().reshape(-1), 256, 8
    )
    np.testing.assert_array_equal(restored, scale.view(torch.uint8).numpy())
    np.testing.assert_array_equal(q.swap_nibbles(np.array([0x12, 0xAB], np.uint8)), [0x21, 0xBA])
    with pytest.raises(ValueError, match="rows % 128"):
        q.unswizzle_block_scales(np.zeros(100 * 8, np.uint8), 100, 8)


def test_unknown_keys_fail_loudly(tmp_path):
    path = tmp_path / "bad.safetensors"
    safetensors_torch.save_file({"double_blocks.0.surprise.weight": _bf16(4, 4)}, str(path))
    with pytest.raises(ValueError, match="unexpected FLUX.2 NVFP4 checkpoint module"):
        q.read_nvfp4_checkpoint(path)
    safetensors_torch.save_file({"mystery": _bf16(4, 4)}, str(path))
    with pytest.raises(ValueError, match="unexpected FLUX.2 NVFP4 checkpoint key"):
        q.read_nvfp4_checkpoint(path)


def test_bad_block_scale_shape_is_rejected(tmp_path):
    tensors: dict = {}
    _quantized(tensors, "single_blocks.0.linear1", 128, 64)
    tensors["single_blocks.0.linear1.weight_scale"] = tensors[
        "single_blocks.0.linear1.weight_scale"
    ][:, :1].contiguous()
    path = tmp_path / "bad.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    with pytest.raises(ValueError, match="weight_scale has shape"):
        q.read_nvfp4_checkpoint(path)


def _blackwell_trt():
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 needs a Blackwell GPU")


def _real_trt_builder():
    """Import the DiT builder against the real TensorRT.

    Other tests in this package import the builder modules while TensorRT is mocked and leave
    them in ``sys.modules``; drop them so this test gets the real bindings.
    """
    import importlib
    import sys

    package = sys.modules.get("families.flux")
    for name in ("flux2_dit_builder", "graph_ops", "graph_blocks"):
        sys.modules.pop(f"families.flux.{name}", None)
        if package is not None and hasattr(package, name):
            delattr(package, name)
    return importlib.import_module("families.flux.flux2_dit_builder")


@pytest.mark.parametrize("quantize_txt_attn", [False, True])
def test_tiny_engine_builds_with_fp4_gemms(tmp_path, quantize_txt_attn):
    _blackwell_trt()
    import json

    import tensorrt as trt
    builder = _real_trt_builder()

    path = tmp_path / "flux2-dev-nvfp4.safetensors"
    write_tiny_checkpoint(path, quantize_txt_attn=quantize_txt_attn)
    weights = q.load_flux2_dit_weights_nvfp4(path)
    plan = builder.build_flux2_dit_engine(
        weights,
        dim=DIM,
        num_heads=HEADS,
        num_layers=1,
        num_single_layers=1,
        num_img_tokens=16,
        text_seq_len=16,
        mlp_ratio=3.0,
        packed_channels=128,
        t5_dim=TEXT_DIM,
        freq_dim=256,
        cast_dtype="bf16",
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    info = json.loads(
        engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    )
    names = " ".join(
        str(layer.get("TacticName") or layer.get("Name") or "").lower()
        for layer in info["Layers"]
        if isinstance(layer, dict)
    )
    assert "fp4" in names or "e2m1" in names


def test_projection_shape_mismatch_is_rejected():
    pytest.importorskip("tensorrt")
    builder = _real_trt_builder()

    layer = q.NVFP4Projection(
        packed=np.zeros((32, 16), np.uint8),
        block_scale=np.zeros((32, 2), np.uint8),
        global_scale=1.0,
        input_scale=1.0,
        out_features=32,
        in_features=32,
    )
    with pytest.raises(ValueError, match="graph expects"):
        builder._matmul_reduced_precision(None, None, 64, 32, None, nvfp4=layer)
