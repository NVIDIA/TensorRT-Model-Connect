from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from typing import Any

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
    assert mismatch["expected_value"] == 2.0
    assert mismatch["actual_value"] == 4.0
    assert mismatch["expected_bits"] == "0x4000"
    assert mismatch["actual_bits"] == "0x4080"


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
