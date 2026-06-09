from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "diagnose_voxcpm2_tslm_prefill.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "diagnose_voxcpm2_tslm_prefill",
        TOOL_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_record(
    root: Path,
    manifest: Path,
    *,
    step: int,
    direction: str,
    name: str,
    dtype: str,
    shape: list[int],
    raw: bytes,
) -> None:
    raw_path = root / f"tslm_prefill_{step:06d}_{direction}_{name}.raw"
    raw_path.write_bytes(raw)
    record = {
        "stage": "tslm",
        "engine_section": "hf_reference",
        "phase": "tslm_prefill",
        "step": step,
        "direction": direction,
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "nbytes": len(raw),
        "path": str(raw_path),
    }
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def test_voxcpm2_tslm_prefill_probe_loads_manifest_rows(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = tmp_path / "manifest.jsonl"

    for step, token in enumerate((101, 202)):
        _write_record(
            tmp_path,
            manifest,
            step=step,
            direction="input",
            name="text_tokens",
            dtype="int32",
            shape=[1],
            raw=struct.pack("<i", token),
        )
        _write_record(
            tmp_path,
            manifest,
            step=step,
            direction="output",
            name="semantic_lm_states",
            dtype="bfloat16",
            shape=[1, 2],
            raw=struct.pack("<HH", 0x3F80 + step, 0x4000 + step),
        )

    records = tool._load_manifest(tmp_path)
    by_key = tool._prefill_records_by_key(records)
    steps = tool._prefill_steps(records)
    tokens = tool._stack_prefill_tensor(
        by_key,
        steps,
        direction="input",
        name="text_tokens",
        torch_module=torch,
    )
    semantic = tool._stack_prefill_tensor(
        by_key,
        steps,
        direction="output",
        name="semantic_lm_states",
        torch_module=torch,
    )

    assert steps == [0, 1]
    assert tokens.tolist() == [101, 202]
    assert semantic.shape == (2, 2)
    assert semantic.dtype == torch.bfloat16


def test_voxcpm2_tslm_prefill_probe_reports_first_bf16_mismatch() -> None:
    tool = _load_tool()
    expected = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    actual = torch.tensor([1.0, 4.0], dtype=torch.bfloat16)

    mismatch = tool._first_mismatch(expected, actual)

    assert mismatch["matched"] is False
    assert mismatch["first_different_element"] == 1
    assert mismatch["different_elements"] == 1
    assert mismatch["total_elements"] == 2
    assert mismatch["max_abs_diff"] == 2.0
    assert mismatch["expected_value"] == 2.0
    assert mismatch["actual_value"] == 4.0
    assert mismatch["expected_bits"] == "0x4000"
    assert mismatch["actual_bits"] == "0x4080"
    assert mismatch["bf16_adjacent_ulp_mismatches"] == 0
    assert mismatch["bf16_bit_delta_counts"] == {"+128": 1}
    assert mismatch["bf16_mismatch_examples"] == [
        {
            "element": 1,
            "expected_value": 2.0,
            "actual_value": 4.0,
            "expected_bits": "0x4000",
            "actual_bits": "0x4080",
            "bit_delta": 128,
        }
    ]


def test_voxcpm2_tslm_prefill_probe_reports_bf16_ulp_summary() -> None:
    tool = _load_tool()
    expected = torch.tensor([0x3B25, 0x3F80, 0xBDB5], dtype=torch.uint16).view(
        torch.bfloat16
    )
    actual = torch.tensor([0x3B26, 0x3F80, 0xBDB3], dtype=torch.uint16).view(
        torch.bfloat16
    )

    mismatch = tool._first_mismatch(expected, actual)

    assert mismatch["matched"] is False
    assert mismatch["different_elements"] == 2
    assert mismatch["bf16_adjacent_ulp_mismatches"] == 1
    assert mismatch["bf16_bit_delta_counts"] == {"+1": 1, "-2": 1}
    assert mismatch["bf16_mismatch_examples"][0] == {
        "element": 0,
        "expected_value": 0.0025177001953125,
        "actual_value": 0.002532958984375,
        "expected_bits": "0x3b25",
        "actual_bits": "0x3b26",
        "bit_delta": 1,
    }


def test_voxcpm2_tslm_prefill_probe_reports_mismatch_location_summary() -> None:
    tool = _load_tool()
    expected = torch.zeros((3, 4), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 1] = 1.0
    actual[0, 3] = 1.0
    actual[2, 3] = 2.0

    mismatch = tool._first_mismatch(expected, actual)

    assert mismatch["first_different_element"] == 1
    assert mismatch["first_different_coordinate"] == [0, 1]
    assert mismatch["mismatch_rows_with_differences"] == 2
    assert mismatch["mismatch_cols_with_differences"] == 2
    assert mismatch["top_mismatch_rows"] == [
        {"row": 0, "count": 2},
        {"row": 2, "count": 1},
    ]
    assert mismatch["top_mismatch_cols"] == [
        {"column": 3, "count": 2},
        {"column": 1, "count": 1},
    ]


def test_voxcpm2_tslm_prefill_trace_summary_reports_first_stage() -> None:
    tool = _load_tool()
    expected = {
        "embedding": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "layer_00": torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
        "final_norm": torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16),
    }
    actual = {
        "embedding": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "layer_00": torch.tensor([[3.0, 8.0]], dtype=torch.bfloat16),
        "final_norm": torch.tensor([[5.0, 9.0]], dtype=torch.bfloat16),
    }

    summary = tool._trace_summary(expected, actual)

    assert summary["first_divergent_stage"] == "layer_00"
    by_stage = {entry["stage"]: entry for entry in summary["stages"]}
    assert by_stage["embedding"]["matched"] is True
    assert by_stage["layer_00"]["matched"] is False
    assert by_stage["layer_00"]["first_different_element"] == 1
    assert by_stage["layer_00"]["expected_bits"] == "0x4080"
    assert by_stage["layer_00"]["actual_bits"] == "0x4100"


class _AddModule(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.value


class _AddAttention:
    def __init__(self, value: float, *, tuple_step: bool = False) -> None:
        self.value = value
        self.tuple_step = tuple_step

    def __call__(self, *, hidden_states: torch.Tensor, **_: Any) -> Any:
        return hidden_states + self.value, ("key", "value")

    def forward_step(
        self,
        *,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        **_: Any,
    ) -> Any:
        output = hidden_states + self.value
        if not self.tuple_step:
            return output
        key_cache, value_cache = kv_cache
        return output, (
            torch.ones_like(key_cache) * 7.0,
            torch.ones_like(value_cache) * 8.0,
        )


class _DummyLayer:
    use_mup = False

    def __init__(self, *, tuple_step: bool = False) -> None:
        self.input_layernorm = _AddModule(1.0)
        self.self_attn = _AddAttention(2.0, tuple_step=tuple_step)
        self.post_attention_layernorm = _AddModule(3.0)
        self.mlp = _AddModule(4.0)


class _TraceableMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 2, bias=False)
        self.up_proj = torch.nn.Linear(2, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.eye(2))
            self.up_proj.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
            self.down_proj.weight.copy_(torch.eye(2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.gate_proj(values) * self.up_proj(values)
        return self.down_proj(hidden)


def test_voxcpm2_tslm_prefill_full_layer_subtrace_records_substeps() -> None:
    tool = _load_tool()
    traces: dict[str, Any] = {}
    layer = _DummyLayer()

    output, cache = tool._run_full_decoder_layer(
        decoder_layer=layer,
        hidden_states=torch.tensor([[[1.0, 2.0]]]),
        position_emb=None,
        is_causal=True,
        layer_index=0,
        trace_layer_index=0,
        traces=traces,
        compute_dtype=torch.float32,
    )

    assert cache == ("key", "value")
    torch.testing.assert_close(output, torch.tensor([[[17.0, 21.0]]]))
    assert list(traces) == [
        "layer_00.input",
        "layer_00.input_norm",
        "layer_00.attention",
        "layer_00.attention_residual",
        "layer_00.post_attention_norm",
        "layer_00.mlp",
        "layer_00.output",
    ]
    torch.testing.assert_close(
        traces["layer_00.attention"],
        torch.tensor([[4.0, 5.0]]),
    )
    torch.testing.assert_close(
        traces["layer_00.output"],
        torch.tensor([[17.0, 21.0]]),
    )


def test_voxcpm2_tslm_prefill_full_subtrace_records_mlp_projections() -> None:
    tool = _load_tool()
    traces: dict[str, Any] = {}
    layer = _DummyLayer()
    layer.mlp = _TraceableMlp()

    output, _cache = tool._run_full_decoder_layer(
        decoder_layer=layer,
        hidden_states=torch.tensor([[[1.0, 2.0]]]),
        position_emb=None,
        is_causal=True,
        layer_index=0,
        trace_layer_index=0,
        traces=traces,
        compute_dtype=torch.float32,
    )

    assert "layer_00.mlp.gate_proj" in traces
    assert "layer_00.mlp.up_proj" in traces
    assert "layer_00.mlp.down_proj_input" in traces
    assert "layer_00.mlp.down_proj" in traces
    torch.testing.assert_close(
        traces["layer_00.mlp.gate_proj"],
        torch.tensor([[8.0, 10.0]]),
    )
    torch.testing.assert_close(
        traces["layer_00.mlp.down_proj_input"],
        torch.tensor([[128.0, 300.0]]),
    )
    torch.testing.assert_close(output, torch.tensor([[[133.0, 307.0]]]))


def test_voxcpm2_tslm_prefill_step_layer_subtrace_records_substeps() -> None:
    tool = _load_tool()
    trace_rows: dict[str, list[Any]] = {}
    layer = _DummyLayer()

    output = tool._run_step_decoder_layer(
        decoder_layer=layer,
        hidden_states=torch.tensor([[1.0, 2.0]]),
        position_emb=None,
        position_id=torch.tensor([0]),
        layer_cache=(torch.empty(1), torch.empty(1)),
        layer_index=0,
        trace_layer_index=0,
        trace_rows=trace_rows,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(output, torch.tensor([[17.0, 21.0]]))
    assert list(trace_rows) == [
        "layer_00.input",
        "layer_00.input_norm",
        "layer_00.attention",
        "layer_00.attention_residual",
        "layer_00.post_attention_norm",
        "layer_00.mlp",
        "layer_00.output",
    ]
    torch.testing.assert_close(
        trace_rows["layer_00.attention"][0],
        torch.tensor([4.0, 5.0]),
    )
    torch.testing.assert_close(
        trace_rows["layer_00.output"][0],
        torch.tensor([17.0, 21.0]),
    )


def test_voxcpm2_tslm_prefill_step_subtrace_records_mlp_projection_rows() -> None:
    tool = _load_tool()
    trace_rows: dict[str, list[Any]] = {}
    layer = _DummyLayer()
    layer.mlp = _TraceableMlp()

    output = tool._run_step_decoder_layer(
        decoder_layer=layer,
        hidden_states=torch.tensor([[1.0, 2.0]]),
        position_emb=None,
        position_id=torch.tensor([0]),
        layer_cache=(torch.empty(1), torch.empty(1)),
        layer_index=0,
        trace_layer_index=0,
        trace_rows=trace_rows,
        compute_dtype=torch.float32,
    )

    assert "layer_00.mlp.gate_proj" in trace_rows
    assert "layer_00.mlp.up_proj" in trace_rows
    assert "layer_00.mlp.down_proj_input" in trace_rows
    assert "layer_00.mlp.down_proj" in trace_rows
    torch.testing.assert_close(
        trace_rows["layer_00.mlp.gate_proj"][0],
        torch.tensor([8.0, 10.0]),
    )
    torch.testing.assert_close(
        trace_rows["layer_00.mlp.down_proj_input"][0],
        torch.tensor([128.0, 300.0]),
    )
    torch.testing.assert_close(output, torch.tensor([[133.0, 307.0]]))


def test_voxcpm2_tslm_prefill_step_layer_subtrace_copies_tuple_cache() -> None:
    tool = _load_tool()
    trace_rows: dict[str, list[Any]] = {}
    layer = _DummyLayer(tuple_step=True)
    key_cache = torch.zeros(2)
    value_cache = torch.zeros(2)

    output = tool._run_step_decoder_layer(
        decoder_layer=layer,
        hidden_states=torch.tensor([[1.0, 2.0]]),
        position_emb=None,
        position_id=torch.tensor([0]),
        layer_cache=(key_cache, value_cache),
        layer_index=0,
        trace_layer_index=0,
        trace_rows=trace_rows,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(output, torch.tensor([[17.0, 21.0]]))
    torch.testing.assert_close(key_cache, torch.tensor([7.0, 7.0]))
    torch.testing.assert_close(value_cache, torch.tensor([8.0, 8.0]))


def test_voxcpm2_tslm_prefill_probe_schedules_step_loop_variants() -> None:
    tool = _load_tool()

    runs = tool._selected_variant_runs(
        include_upstream=True,
        include_patched=True,
        include_step_loop=True,
    )

    assert runs == [
        ("upstream_full_prefill", False, "full_prefill"),
        ("upstream_step_loop", False, "step_loop"),
        ("patched_export_full_prefill", True, "full_prefill"),
        ("patched_export_step_loop", True, "step_loop"),
    ]


def test_voxcpm2_tslm_prefill_probe_keeps_default_variants_full_prefill_only() -> None:
    tool = _load_tool()

    runs = tool._selected_variant_runs(
        include_upstream=True,
        include_patched=True,
        include_step_loop=False,
    )

    assert runs == [
        ("upstream_full_prefill", False, "full_prefill"),
        ("patched_export_full_prefill", True, "full_prefill"),
    ]


def test_voxcpm2_tslm_prefill_trt_input_padding_preserves_rows() -> None:
    tool = _load_tool()
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    padded = tool._pad_first_dim_to_shape(
        tensor,
        (3, 2),
        torch_module=torch,
        name="local_text_features",
    )

    assert padded.shape == (3, 2)
    assert padded.dtype == torch.bfloat16
    torch.testing.assert_close(padded[0].float(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(padded[1:].float(), torch.zeros((2, 2)))

    with pytest.raises(ValueError, match="engine accepts 1"):
        tool._pad_first_dim_to_shape(
            torch.zeros((2, 2), dtype=torch.bfloat16),
            (1, 2),
            torch_module=torch,
            name="local_text_features",
        )


def test_voxcpm2_tslm_prefill_trt_output_selection_handles_normalized_names() -> None:
    tool = _load_tool()

    assert tool._select_semantic_output_name(
        [
            {"name": "output0", "mode": "output", "shape": [20, 2048]},
            {"name": "output2", "mode": "output", "shape": [20, 2]},
        ],
        (20, 2048),
    ) == "output0"

    assert tool._select_semantic_output_name(
        [
            {"name": "output0", "mode": "output", "shape": [20, 2048]},
            {"name": "semantic_lm_states", "mode": "output", "shape": [1024, 2048]},
        ],
        (20, 2048),
    ) == "semantic_lm_states"


def test_voxcpm2_tslm_prefill_parses_down_proj_tactic_source_sets() -> None:
    tool = _load_tool()

    assert tool._parse_tactic_source_sets(
        ["cublas_lt", "CUBLAS, cublas_lt", "  "]
    ) == [
        ("CUBLAS_LT",),
        ("CUBLAS", "CUBLAS_LT"),
    ]


def test_voxcpm2_tslm_prefill_parses_down_proj_builder_flag_sets() -> None:
    tool = _load_tool()

    assert tool._parse_builder_flag_sets(
        ["bf16", "BF16, prefer_precision_constraints", "  "]
    ) == [
        ("BF16",),
        ("BF16", "PREFER_PRECISION_CONSTRAINTS"),
    ]


def test_voxcpm2_tslm_prefill_sets_builder_flags() -> None:
    tool = _load_tool()

    class FakeConfig:
        def __init__(self) -> None:
            self.flags: list[int] = []

        def set_flag(self, flag: int) -> None:
            self.flags.append(flag)

    fake_config = FakeConfig()
    fake_trt = SimpleNamespace(
        BuilderFlag=SimpleNamespace(BF16=7, STRICT_NANS=11)
    )

    tool._set_builder_flags(fake_config, fake_trt, ("BF16", "STRICT_NANS"))

    assert fake_config.flags == [7, 11]

    with pytest.raises(ValueError, match="Unsupported TensorRT builder flag"):
        tool._set_builder_flags(fake_config, fake_trt, ("UNKNOWN",))


def test_voxcpm2_tslm_prefill_summarizes_trt_network_layers() -> None:
    tool = _load_tool()

    class FakeTensor:
        def __init__(self, name: str, dtype: str, shape: tuple[int, ...]) -> None:
            self.name = name
            self.dtype = dtype
            self.shape = shape

    class FakeLayer:
        name = "down_proj/Gemm"
        type = "LayerType.MATRIX_MULTIPLY"
        num_inputs = 2
        num_outputs = 1

        def get_input(self, index: int) -> Any:
            return (
                FakeTensor("down_proj_input", "DataType.BF16", (20, 6144))
                if index == 0
                else FakeTensor("down_proj_weight", "DataType.BF16", (6144, 2048))
            )

        def get_output(self, _index: int) -> Any:
            return FakeTensor("down_proj_output", "DataType.BF16", (20, 2048))

    class FakeNetwork:
        num_layers = 1

        def get_layer(self, _index: int) -> Any:
            return FakeLayer()

    assert tool._trt_network_layer_summary(FakeNetwork()) == [
        {
            "index": 0,
            "name": "down_proj/Gemm",
            "type": "LayerType.MATRIX_MULTIPLY",
            "inputs": [
                {
                    "name": "down_proj_input",
                    "dtype": "DataType.BF16",
                    "shape": [20, 6144],
                },
                {
                    "name": "down_proj_weight",
                    "dtype": "DataType.BF16",
                    "shape": [6144, 2048],
                },
            ],
            "outputs": [
                {
                    "name": "down_proj_output",
                    "dtype": "DataType.BF16",
                    "shape": [20, 2048],
                }
            ],
        }
    ]


def test_voxcpm2_tslm_prefill_reads_engine_inspector_json() -> None:
    tool = _load_tool()

    class FakeInspector:
        def get_engine_information(self, fmt: str) -> str:
            assert fmt == "json"
            return json.dumps({"Layers": [{"Name": "down_proj/Gemm"}]})

    class FakeEngine:
        def create_engine_inspector(self) -> FakeInspector:
            return FakeInspector()

    fake_trt = SimpleNamespace(LayerInformationFormat=SimpleNamespace(JSON="json"))

    assert tool._trt_engine_inspector_summary(FakeEngine(), fake_trt) == {
        "Layers": [{"Name": "down_proj/Gemm"}]
    }


def test_voxcpm2_tslm_prefill_sets_detailed_profiling_when_available() -> None:
    tool = _load_tool()

    class FakeConfig:
        profiling_verbosity = None

    fake_config = FakeConfig()
    fake_trt = SimpleNamespace(
        ProfilingVerbosity=SimpleNamespace(DETAILED="detailed")
    )

    tool._set_detailed_profiling_verbosity(fake_config, fake_trt)

    assert fake_config.profiling_verbosity == "detailed"


def test_voxcpm2_tslm_prefill_down_proj_label_includes_kernel_controls() -> None:
    tool = _load_tool()

    assert tool._down_proj_label(
        layer_index=0,
        variant="linear",
        tactic_sources=(),
        builder_flags=(),
    ) == "layer_00.mlp.down_proj.trt_default"
    assert tool._down_proj_label(
        layer_index=1,
        variant="manual_matmul_bf16",
        tactic_sources=("CUBLAS", "CUBLAS_LT"),
        builder_flags=("BF16", "PREFER_PRECISION_CONSTRAINTS"),
    ) == (
        "layer_01.mlp.down_proj.manual_matmul_bf16."
        "trt_cublas+cublas_lt_bf16+prefer_precision_constraints"
    )


def test_voxcpm2_tslm_prefill_parses_down_proj_variants() -> None:
    tool = _load_tool()

    assert tool._parse_down_proj_variants(None) == ["linear"]
    assert tool._parse_down_proj_variants(
        ["manual_matmul_bf16, fp32_output", "linear", "manual_matmul_bf16"]
    ) == ["manual_matmul_bf16", "fp32_output", "linear"]
    assert tool._parse_down_proj_variants(["all"]) == [
        "linear",
        "functional_linear",
        "einsum",
        "batched_bmm",
        "manual_matmul_bf16",
        "fp32_accum_to_bf16",
        "fp32_output",
        "split_k_1024_bf16_accum",
        "split_k_1024_fp32_accum_to_bf16",
        "split_out_256_bf16",
    ]
    assert tool._parse_down_proj_variants(
        [
            "split_k_512_bf16_accum",
            "split_k_2048_fp32_accum_to_bf16",
            "split_out_128_bf16",
        ]
    ) == [
        "split_k_512_bf16_accum",
        "split_k_2048_fp32_accum_to_bf16",
        "split_out_128_bf16",
    ]

    with pytest.raises(ValueError, match="Unsupported VoxCPM2 down-proj variant"):
        tool._parse_down_proj_variants(["unknown"])

    with pytest.raises(ValueError, match="Unsupported VoxCPM2 down-proj variant"):
        tool._parse_down_proj_variants(["split_k_0_bf16_accum"])

    with pytest.raises(ValueError, match="Unsupported VoxCPM2 down-proj variant"):
        tool._parse_down_proj_variants(["split_out_0_bf16"])


def test_voxcpm2_tslm_prefill_down_proj_variant_modules() -> None:
    tool = _load_tool()

    linear = torch.nn.Linear(3, 2, bias=True).to(dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=torch.bfloat16,
            )
        )
        linear.bias.copy_(torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.bfloat16)

    assert tool._make_down_proj_variant_module(torch, linear, "linear") is linear
    functional = tool._make_down_proj_variant_module(
        torch,
        linear,
        "functional_linear",
    )
    einsum = tool._make_down_proj_variant_module(
        torch,
        linear,
        "einsum",
    )
    batched_bmm = tool._make_down_proj_variant_module(
        torch,
        linear,
        "batched_bmm",
    )
    manual = tool._make_down_proj_variant_module(
        torch,
        linear,
        "manual_matmul_bf16",
    )
    fp32_to_bf16 = tool._make_down_proj_variant_module(
        torch,
        linear,
        "fp32_accum_to_bf16",
    )
    fp32_output = tool._make_down_proj_variant_module(
        torch,
        linear,
        "fp32_output",
    )
    split_k_bf16 = tool._make_down_proj_variant_module(
        torch,
        linear,
        "split_k_2_bf16_accum",
    )
    split_k_fp32 = tool._make_down_proj_variant_module(
        torch,
        linear,
        "split_k_2_fp32_accum_to_bf16",
    )
    split_out = tool._make_down_proj_variant_module(
        torch,
        linear,
        "split_out_1_bf16",
    )

    assert functional(x).dtype == torch.bfloat16
    assert torch.equal(functional(x), linear(x))
    assert einsum(x).dtype == torch.bfloat16
    assert batched_bmm(x).dtype == torch.bfloat16
    assert manual(x).dtype == torch.bfloat16
    assert torch.equal(manual(x), linear(x))
    assert torch.equal(einsum(x), linear(x))
    assert torch.equal(batched_bmm(x), linear(x))
    assert torch.equal(batched_bmm(x.unsqueeze(0)), linear(x.unsqueeze(0)))
    assert fp32_to_bf16(x).dtype == torch.bfloat16
    assert fp32_output(x).dtype == torch.float32
    assert split_k_bf16(x).dtype == torch.bfloat16
    assert split_k_fp32(x).dtype == torch.bfloat16
    assert split_out(x).dtype == torch.bfloat16
    assert torch.equal(split_out(x), linear(x))


def test_voxcpm2_tslm_prefill_diagnose_can_run_trt_plan_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    manifest = tmp_path / "manifest.jsonl"
    for name, dtype, shape, raw in (
        (
            "local_text_features",
            "bfloat16",
            [1, 2],
            struct.pack("<HH", 0x3F80, 0x4000),
        ),
        ("text_tokens", "int32", [1], struct.pack("<i", 101)),
        ("text_mask", "bfloat16", [1], struct.pack("<H", 0x3F80)),
        ("audio_mask", "bfloat16", [1], struct.pack("<H", 0x0000)),
    ):
        _write_record(
            tmp_path,
            manifest,
            step=0,
            direction="input",
            name=name,
            dtype=dtype,
            shape=shape,
            raw=raw,
        )
    _write_record(
        tmp_path,
        manifest,
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1, 2],
        raw=struct.pack("<HH", 0x3F80, 0x4000),
    )
    calls: list[dict[str, Any]] = []

    def fake_run_trt_prefill_plan(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "label": "trt_prefill_plan",
            "matched": True,
            "first_different_element": None,
        }

    monkeypatch.setattr(tool, "_run_trt_prefill_plan", fake_run_trt_prefill_plan)
    result = tool.diagnose(
        model_dir=tmp_path,
        hf_dump_dir=tmp_path,
        device="cuda",
        include_upstream=False,
        include_patched=False,
        trt_prefill_plan=tmp_path / "tslm.plan",
    )

    assert result["results"] == [
        {
            "label": "trt_prefill_plan",
            "matched": True,
            "first_different_element": None,
        }
    ]
    assert calls[0]["plan_path"] == tmp_path / "tslm.plan"
    assert calls[0]["inputs"]["text_tokens"].tolist() == [101]


def test_voxcpm2_tslm_prefill_diagnose_can_run_down_proj_probe_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    manifest = tmp_path / "manifest.jsonl"
    for name, dtype, shape, raw in (
        ("local_text_features", "bfloat16", [1, 2], struct.pack("<HH", 0x3F80, 0x4000)),
        ("text_tokens", "int32", [1], struct.pack("<i", 101)),
        ("text_mask", "bfloat16", [1], struct.pack("<H", 0x3F80)),
        ("audio_mask", "bfloat16", [1], struct.pack("<H", 0x0000)),
    ):
        _write_record(
            tmp_path,
            manifest,
            step=0,
            direction="input",
            name=name,
            dtype=dtype,
            shape=shape,
            raw=raw,
        )
    _write_record(
        tmp_path,
        manifest,
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1, 2],
        raw=struct.pack("<HH", 0x3F80, 0x4000),
    )
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        tool,
        "_load_tslm_state",
        lambda model_dir: (
            {"lm_config": {"hidden_size": 2}},
            {"base_lm.dummy": object()},
        ),
    )

    def fake_run_down_proj_trt_probe(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [
            {
                "label": "layer_00.mlp.down_proj.trt_default",
                "matched": False,
                "first_different_element": 3,
            }
        ]

    monkeypatch.setattr(
        tool,
        "_run_down_proj_trt_probe",
        fake_run_down_proj_trt_probe,
    )

    result = tool.diagnose(
        model_dir=tmp_path,
        hf_dump_dir=tmp_path,
        device="cuda",
        include_upstream=False,
        include_patched=False,
        trt_down_proj_layer=0,
        trt_down_proj_tactic_source_sets=[("CUBLAS_LT",)],
        trt_down_proj_builder_flag_sets=[("BF16",)],
        trt_down_proj_variants=["manual_matmul_bf16", "fp32_output"],
    )

    assert result["results"] == [
        {
            "label": "layer_00.mlp.down_proj.trt_default",
            "matched": False,
            "first_different_element": 3,
        }
    ]
    assert calls[0]["layer_index"] == 0
    assert calls[0]["tactic_source_sets"] == [("CUBLAS_LT",)]
    assert calls[0]["builder_flag_sets"] == [("BF16",)]
    assert calls[0]["variants"] == ["manual_matmul_bf16", "fp32_output"]
    assert calls[0]["inputs"]["text_tokens"].tolist() == [101]
