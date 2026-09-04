# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT Python construction of the SAM2 image encoder plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .checkpoint_mapper import Checkpoint
from .float_math import reciprocal_sqrtf
from .graph_ops import NetworkBuildError, TrtLayers


_ATTENTION_METADATA = tuple(f"trtmc.sam2.iattention.block.{index:02d}" for index in range(16))
_TRACKER_OUTPUTS = (
    ("tracker_fpn_0", "bf16", (1, 256, 256, 256)),
    ("tracker_fpn_1", "bf16", (1, 256, 128, 128)),
    ("tracker_fpn_2", "fp32", (1, 256, 64, 64)),
)
_BBOX_OUTPUTS = (
    ("bbox_cls_stride_8", "bf16", (1, 2, 128, 128)),
    ("bbox_cls_stride_16", "bf16", (1, 2, 64, 64)),
    ("bbox_cls_stride_32", "bf16", (1, 2, 32, 32)),
    ("bbox_reg_stride_8", "bf16", (1, 4, 128, 128)),
    ("bbox_reg_stride_16", "bf16", (1, 4, 64, 64)),
    ("bbox_reg_stride_32", "bf16", (1, 4, 32, 32)),
)


@dataclass(frozen=True)
class _Block:
    input_channels: int
    output_channels: int
    heads: int
    input_height: int
    window_size: int
    query_pool: bool


_BLOCKS = (
    _Block(96, 96, 1, 256, 8, False),
    _Block(96, 192, 2, 256, 8, True),
    _Block(192, 192, 2, 128, 4, False),
    _Block(192, 384, 4, 128, 4, True),
    _Block(384, 384, 4, 64, 14, False),
    _Block(384, 384, 4, 64, 14, False),
    _Block(384, 384, 4, 64, 14, False),
    _Block(384, 384, 4, 64, 0, False),
    _Block(384, 384, 4, 64, 14, False),
    _Block(384, 384, 4, 64, 14, False),
    _Block(384, 384, 4, 64, 0, False),
    _Block(384, 384, 4, 64, 14, False),
    _Block(384, 384, 4, 64, 14, False),
    _Block(384, 384, 4, 64, 0, False),
    _Block(384, 768, 8, 64, 14, True),
    _Block(768, 768, 8, 32, 7, False),
)


class ImageNetworkBuilder:
    def __init__(self, trt: Any, network: Any, checkpoint: Checkpoint) -> None:
        self.trt = trt
        self.network = network
        self.layers = TrtLayers(trt, network, checkpoint)

    @staticmethod
    def _prefix(index: int) -> str:
        return f"image_encoder.trunk.blocks.{index}"

    @staticmethod
    def _name(index: int, suffix: str) -> str:
        return f"hiera.block.{index}.{suffix}"

    def _attention(
        self, tensor: Any, index: int, output_channels: int, heads: int, query_pool: bool
    ) -> Any:
        shape = self.layers._shape(tensor)
        batch, height, width, input_channels = shape
        head_channels = output_channels // heads
        tokens = height * width
        weights = f"{self._prefix(index)}.attn"
        name = self._name(index, "attention")

        qkv = self.layers.linear_bf16(
            tensor, f"{weights}.qkv", input_channels, 3 * output_channels, f"{name}.qkv"
        )
        grouped = self.layers.shuffle(
            qkv, (batch, tokens, 3, heads, head_channels), f"{name}.group"
        )
        sliced = [
            self.layers.slice(
                grouped,
                (0, 0, component, 0, 0),
                (batch, tokens, 1, heads, head_channels),
                (1, 1, 1, 1, 1),
                self.trt.SampleMode.STRICT_BOUNDS,
                f"{name}.{label}_slice",
            )
            for component, label in enumerate(("q", "k", "v"))
        ]
        q, k, v = (
            self.layers.shuffle(value, (batch, tokens, heads, head_channels), f"{name}.{label}")
            for value, label in zip(sliced, ("q", "k", "v"), strict=True)
        )
        query_height, query_width = height, width
        if query_pool:
            q_image = self.layers.shuffle(
                q, (batch, height, width, output_channels), f"{name}.q_image"
            )
            q_pooled = self.layers.max_pool_nhwc(q_image, 2, 2, f"{name}.q_pool")
            query_height //= 2
            query_width //= 2
            q = self.layers.shuffle(
                q_pooled,
                (batch, query_height * query_width, heads, head_channels),
                f"{name}.q_pooled_tokens",
            )

        q = self.layers.transpose(q, (0, 2, 1, 3), f"{name}.q_heads")
        k = self.layers.transpose(k, (0, 2, 1, 3), f"{name}.k_heads")
        v = self.layers.transpose(v, (0, 2, 1, 3), f"{name}.v_heads")
        scale = self.layers.scalar(
            reciprocal_sqrtf(head_channels), 4, self.trt.bfloat16, f"{name}.q_scale"
        )
        q = self.layers.elementwise(
            q, scale, self.trt.ElementWiseOperation.PROD, f"{name}.q_scaled"
        )
        attention = self.network.add_attention_v2(
            q,
            k,
            v,
            self.trt.AttentionNormalizationOp.SOFTMAX,
            self.trt.CausalMaskKind.NONE,
        )
        if attention is None:
            raise NetworkBuildError(f"TensorRT rejected IAttentionV2 at {name}")
        attention.name = name
        attention.metadata = _ATTENTION_METADATA[index]
        attention.query_form = self.trt.AttentionIOForm.PADDED_BHND
        attention.key_value_form = self.trt.AttentionIOForm.PADDED_BHND
        attention.decomposable = False
        attended = attention.get_output(0)
        token_major = self.layers.transpose(attended, (0, 2, 1, 3), f"{name}.token_major")
        image = self.layers.shuffle(
            token_major,
            (batch, query_height, query_width, output_channels),
            f"{name}.image",
        )
        return self.layers.linear_bf16(
            image,
            f"{weights}.proj",
            output_channels,
            output_channels,
            f"{name}.projection",
        )

    def _block(self, tensor: Any, index: int, contract: _Block) -> Any:
        weights = self._prefix(index)
        name = self._name(index, "block")
        normalized = self.layers.layer_norm_fp32(
            tensor, f"{weights}.norm1", contract.input_channels, 1.0e-6, f"{name}.norm1"
        )
        shortcut = tensor
        if contract.input_channels != contract.output_channels:
            projected = self.layers.linear_bf16(
                normalized,
                f"{weights}.proj",
                contract.input_channels,
                contract.output_channels,
                f"{name}.shortcut_projection",
            )
            shortcut = self.layers.max_pool_nhwc(projected, 2, 2, f"{name}.shortcut_pool")

        if contract.window_size:
            windows = self.layers.window_partition(
                normalized,
                contract.input_height,
                contract.input_height,
                contract.input_channels,
                contract.window_size,
                f"{name}.partition",
            )
            attended = self._attention(
                windows.tensor,
                index,
                contract.output_channels,
                contract.heads,
                contract.query_pool,
            )
            output_height = (
                contract.input_height // 2 if contract.query_pool else contract.input_height
            )
            output_window = (
                contract.window_size // 2 if contract.query_pool else contract.window_size
            )
            attended = self.layers.window_unpartition(
                windows,
                attended,
                output_height,
                output_height,
                contract.output_channels,
                output_window,
                f"{name}.unpartition",
            )
        else:
            attended = self._attention(
                normalized,
                index,
                contract.output_channels,
                contract.heads,
                contract.query_pool,
            )

        attended = self.layers.cast(attended, shortcut.dtype, f"{name}.attention_residual_cast")
        result = self.layers.elementwise(
            shortcut, attended, self.trt.ElementWiseOperation.SUM, f"{name}.attention_residual"
        )
        mlp_input = self.layers.layer_norm_fp32(
            result, f"{weights}.norm2", contract.output_channels, 1.0e-6, f"{name}.norm2"
        )
        mlp = self.layers.linear_bf16(
            mlp_input,
            f"{weights}.mlp.layers.0",
            contract.output_channels,
            4 * contract.output_channels,
            f"{name}.mlp.fc1",
        )
        mlp = self.layers.gelu(mlp, f"{name}.mlp.gelu")
        mlp = self.layers.linear_bf16(
            mlp,
            f"{weights}.mlp.layers.1",
            4 * contract.output_channels,
            contract.output_channels,
            f"{name}.mlp.fc2",
        )
        mlp = self.layers.cast(mlp, result.dtype, f"{name}.mlp_residual_cast")
        return self.layers.elementwise(
            result, mlp, self.trt.ElementWiseOperation.SUM, f"{name}.mlp_residual"
        )

    def _hiera(self, pixels: Any) -> list[Any]:
        tensor = self.layers.cast(pixels, self.trt.bfloat16, "hiera.input_bf16")
        tensor = self.layers.convolution(
            tensor,
            "image_encoder.trunk.patch_embed.proj.weight",
            "image_encoder.trunk.patch_embed.proj.bias",
            3,
            96,
            7,
            4,
            3,
            1,
            "hiera.patch_embed",
        )
        tensor = self.layers.transpose(tensor, (0, 2, 3, 1), "hiera.patch_to_nhwc")
        tensor = self.layers.cast(tensor, self.trt.float32, "hiera.patch_fp32")
        background = self.layers.constant(
            "image_encoder.trunk.pos_embed",
            (1, 96, 7, 7),
            (1, 96, 7, 7),
            "hiera.position.background",
        )
        background = self.layers.resize_nchw(
            background,
            256,
            256,
            self.trt.InterpolationMode.CUBIC,
            self.trt.ResizeCoordinateTransformation.HALF_PIXEL,
            "hiera.position.background_resize",
            -0.75,
        )
        window = self.layers.constant(
            "image_encoder.trunk.pos_embed_window",
            (1, 96, 8, 8),
            (1, 96, 8, 8),
            "hiera.position.window",
        )
        window = self.layers.slice(
            window,
            (0, 0, 0, 0),
            (1, 96, 256, 256),
            (1, 1, 1, 1),
            self.trt.SampleMode.WRAP,
            "hiera.position.window_tile",
        )
        position = self.layers.elementwise(
            background, window, self.trt.ElementWiseOperation.SUM, "hiera.position.sum"
        )
        position = self.layers.transpose(position, (0, 2, 3, 1), "hiera.position.to_nhwc")
        current = self.layers.elementwise(
            tensor, position, self.trt.ElementWiseOperation.SUM, "hiera.patch_plus_position"
        )
        outputs: list[Any] = []
        for index, contract in enumerate(_BLOCKS):
            current = self._block(current, index, contract)
            if index in (0, 2, 13, 15):
                outputs.append(
                    self.layers.transpose(
                        current, (0, 3, 1, 2), f"hiera.stage.{len(outputs)}.to_nchw"
                    )
                )
        return outputs

    def _fpn(self, trunk: list[Any]) -> list[Any]:
        channels = (96, 192, 384, 768)
        fpn: list[Any] = [None] * 4
        for level in range(3, -1, -1):
            module = f"image_encoder.neck.convs.{3 - level}.conv"
            name = f"fpn.lateral.{level}"
            tensor = self.layers.cast(trunk[level], self.trt.bfloat16, f"{name}.input_bf16")
            fpn[level] = self.layers.convolution(
                tensor,
                f"{module}.weight",
                f"{module}.bias",
                channels[level],
                256,
                1,
                1,
                0,
                1,
                name,
            )
        low = self.layers.cast(fpn[3], self.trt.float32, "fpn.top_down.low_fp32")
        low = self.layers.resize_nchw(
            low,
            64,
            64,
            self.trt.InterpolationMode.NEAREST,
            self.trt.ResizeCoordinateTransformation.ASYMMETRIC,
            "fpn.top_down.upsample",
        )
        lateral = self.layers.cast(fpn[2], self.trt.float32, "fpn.top_down.lateral_fp32")
        fpn[2] = self.layers.elementwise(
            lateral, low, self.trt.ElementWiseOperation.SUM, "fpn.top_down.fusion"
        )
        return fpn

    def _bbox_head(self, fpn: list[Any]) -> tuple[list[Any], list[Any]]:
        classification: list[Any] = []
        regression: list[Any] = []
        for level in range(3):
            cls = reg = fpn[level + 1]
            for stack in range(2):
                cls = self.layers.convolution_batch_norm_silu(
                    cls,
                    f"image_encoder.bbox_head.cls_convs.{level}.{stack}",
                    256,
                    256,
                    3,
                    1,
                    1,
                    1,
                    1.0e-5,
                    f"bbox.level.{level}.cls.{stack}",
                )
                reg = self.layers.convolution_batch_norm_silu(
                    reg,
                    f"image_encoder.bbox_head.reg_convs.{level}.{stack}",
                    256,
                    256,
                    3,
                    1,
                    1,
                    1,
                    1.0e-5,
                    f"bbox.level.{level}.reg.{stack}",
                )
            cls_output = f"image_encoder.bbox_head.rtm_cls.{level}"
            reg_output = f"image_encoder.bbox_head.rtm_reg.{level}"
            classification.append(
                self.layers.convolution(
                    cls,
                    f"{cls_output}.weight",
                    f"{cls_output}.bias",
                    256,
                    2,
                    1,
                    1,
                    0,
                    1,
                    f"bbox.level.{level}.classification",
                )
            )
            regression.append(
                self.layers.convolution(
                    reg,
                    f"{reg_output}.weight",
                    f"{reg_output}.bias",
                    256,
                    4,
                    1,
                    1,
                    0,
                    1,
                    f"bbox.level.{level}.regression",
                )
            )
        return classification, regression

    def _output(self, tensor: Any, contract: tuple[str, str, tuple[int, ...]]) -> None:
        name, dtype_name, shape = contract
        dtype = self.trt.bfloat16 if dtype_name == "bf16" else self.trt.float32
        if tensor.dtype != dtype or self.layers._shape(tensor) != shape:
            raise NetworkBuildError(
                f"output contract mismatch for {name}: actual shape {self.layers._shape(tensor)}"
            )
        tensor.name = name
        self.network.mark_output(tensor)

    def build(self) -> None:
        pixels = self.network.add_input("pixel_values", self.trt.float32, (1, 3, 1024, 1024))
        fpn = self._fpn(self._hiera(pixels))
        classification, regression = self._bbox_head(fpn)
        for tensor, contract in zip(fpn[:3], _TRACKER_OUTPUTS, strict=True):
            self._output(tensor, contract)
        for tensor, contract in zip(classification + regression, _BBOX_OUTPUTS, strict=True):
            self._output(tensor, contract)


def populate_image_network(trt: Any, network: Any, checkpoint: Checkpoint) -> ImageNetworkBuilder:
    builder = ImageNetworkBuilder(trt, network, checkpoint)
    builder.build()
    return builder
