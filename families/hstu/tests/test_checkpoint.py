# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint compatibility tests; all weights here are synthetic."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file

from families.hstu.checkpoint import _load_pytorch_checkpoint, convert_state_dict, export_checkpoint, load_dynamic_table
from families.hstu.config import expected_shapes
from families.hstu.tests.fixtures import make_checkpoint, tiny_config


def _original_source(config):
    if not (os.environ.get("TRTMC_HSTU_REFERENCE_ROOT") or os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")):
        from families.hstu.tests.test_e2e import _selected_cases

        cases, enabled = _selected_cases(config)
        if not enabled or not cases:
            pytest.skip("HSTU source oracle requires an explicit reference checkout or HSTU E2E selector")
    from families.hstu.tests.environment import reference_source

    return reference_source()


@pytest.fixture
def original_source(request):
    pytest.importorskip("torch")
    return _original_source(request.config)


@pytest.mark.parametrize("models,testcases,explicit,selected", [
    ([], [], False, False),
    (["bert"], [], False, False),
    (["hstu"], [], False, True),
    ([], ["hstu-tiny-ranking-fp32"], False, True),
    ([], [], "TRTMC_HSTU_REFERENCE_ROOT", True),
    ([], [], "TRTMC_REFERENCE_SOURCE_DIR", True),
])
def test_source_oracle_opt_in_precedes_reference_preparation(
    tmp_path, monkeypatch, models, testcases, explicit, selected,
):
    from families.hstu.tests import environment

    monkeypatch.delenv("TRTMC_HSTU_REFERENCE_ROOT", raising=False)
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    if explicit:
        monkeypatch.setenv(explicit, str(tmp_path))
    calls = []

    def prepare():
        calls.append("prepare")
        return tmp_path

    monkeypatch.setattr(environment, "reference_source", prepare)
    options = {"--e2e-model": models, "--e2e-testcase": testcases}
    config = SimpleNamespace(getoption=lambda name: options.get(name))
    if selected:
        assert _original_source(config) == tmp_path
        assert calls == ["prepare"]
    else:
        with pytest.raises(pytest.skip.Exception, match="HSTU source oracle requires"):
            _original_source(config)
        assert not calls


@pytest.mark.parametrize("variable", ["TRTMC_HSTU_REFERENCE_ROOT", "TRTMC_REFERENCE_SOURCE_DIR"])
def test_selected_source_oracle_preparation_failure_is_not_skipped(monkeypatch, variable):
    from families.hstu.tests import environment

    monkeypatch.delenv("TRTMC_HSTU_REFERENCE_ROOT", raising=False)
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    monkeypatch.setenv(variable, "/explicit-reference")

    def fail():
        raise ValueError("source differs from pinned revision")

    monkeypatch.setattr(environment, "reference_source", fail)
    with pytest.raises(ValueError, match="source differs"):
        _original_source(None)


def upstream_state(config, layout):
    """Distinct entries make type/head packing, transposes, and bias observable."""
    canonical = {
        key: (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index / 100)
        for index, (key, shape) in enumerate(expected_shapes(config).items())
    }
    state = {}
    h, a = config["num_heads"], config["head_dim"]
    for key, value in canonical.items():
        fields = key.split(".")
        if fields[0] == "blocks":
            prefix = f"_hstu_block._attention_layers.{fields[1]}."
            part, suffix = fields[2:]
            if part in ("input_norm", "output_norm"):
                target = prefix + "_" + part.replace("_norm", "_layernorm") + "_" + suffix
                if part == "output_norm" and layout == "native":
                    target = prefix + "_output_ln_dropout_mul." + suffix
            else:
                target = prefix + "_linear_" + part + ("_" if layout == "fused" else ".") + suffix
                if part == "uvqk" and layout != "native":
                    value = value.reshape(h, 4, a, *value.shape[1:]).swapaxes(0, 1).reshape(value.shape)
                if layout == "fused" and suffix == "weight":
                    value = value.T
            state[target] = value
        elif fields[0] == "embeddings":
            state[f"_embedding_collection.embeddings.{fields[1]}.weight"] = value
        elif fields[0] == "head":
            state[f"_mlp._mlp.{int(fields[1]) * 2}.{fields[2]}"] = value
        elif fields[0] in ("position", "time"):
            name = "_position_embeddings_weight" if fields[0] == "position" else "_timestamp_embeddings_weight"
            state["_hstu_block._preprocessor._positional_encoder." + name] = value
    return canonical, state


@pytest.mark.parametrize("layout", ["fused", "native", "paged"])
@pytest.mark.parametrize("learnable", [True, False])
def test_upstream_layout_and_all_weight_values(layout, learnable):
    config = tiny_config(learnable_input_layernorm=learnable, learnable_output_layernorm=learnable, add_uvqk_bias=learnable, prediction_bias=learnable)
    canonical, state = upstream_state(config, layout)
    actual = convert_state_dict(state, config, source_layout=layout)
    assert actual.keys() == canonical.keys()
    for key in canonical:
        np.testing.assert_array_equal(actual[key], canonical[key], err_msg=key)


def test_unknown_dense_weights_are_rejected():
    config = tiny_config()
    _, state = upstream_state(config, "native")
    state["_hstu_block._preprocessor._item_mlp._mlp.0.weight"] = np.zeros((16, 16))
    with pytest.raises(ValueError, match="unconverted checkpoint state"):
        convert_state_dict(state, config, source_layout="native")


def test_inference_ranking_gr_module_prefixes():
    config = tiny_config()
    canonical, state = upstream_state(config, "paged")
    wrapped = {}
    for key, value in state.items():
        if key.startswith("_embedding_collection.embeddings."):
            key = "sparse_module." + key.replace("_embedding_collection.", "_static_embedding_collection.", 1)
        else:
            key = "dense_module." + key
        wrapped[key] = value
    actual = convert_state_dict(wrapped, config, source_layout="paged")
    for key in canonical:
        np.testing.assert_array_equal(actual[key], canonical[key])


def test_wrong_dense_topology_is_rejected():
    config = tiny_config()
    _, state = upstream_state(config, "native")
    state["_hstu_block._attention_layers.0._linear_uvqk.weight"] = np.zeros((32, 16))
    with pytest.raises(ValueError, match="has shape"):
        convert_state_dict(state, config, source_layout="native")


def test_static_packed_table_order_is_preserved():
    config = tiny_config()
    canonical, state = upstream_state(config, "fused")
    names = ["context", "item", "action"]
    packed = np.concatenate([state.pop(f"_embedding_collection.embeddings.{name}.weight") for name in names])
    state["_embedding_collection._data_parallel_embedding_collection.embeddings." + "/".join(names) + "_weights"] = packed.flatten()
    actual = convert_state_dict(state, config, source_layout="fused")
    for name in names:
        np.testing.assert_array_equal(actual[f"embeddings.{name}.weight"], canonical[f"embeddings.{name}.weight"])


def write_dynamic_shard(directory, keys, values, rank=0, world_size=1, dtype="float32"):
    directory.mkdir(parents=True, exist_ok=True)
    np.asarray(keys, dtype="<i8").tofile(directory / f"item_emb_keys.rank_{rank}.world_size_{world_size}")
    values = np.asarray(values, dtype=np.float32)
    encoded = ((values.view(np.uint32) >> 16).astype("<u2") if dtype == "bfloat16" else values.astype(dtype))
    encoded.tofile(directory / f"item_emb_values.rank_{rank}.world_size_{world_size}")
    (directory / "item_opt_args.json").write_text(json.dumps({"embedding_dtype": dtype, "embedding_dim": values.shape[1]}))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_dynamic_tables_preserve_sparse_int64_keys_and_rows(tmp_path, dtype):
    values = np.asarray([[3.5, -1.25], [2.5, 0.75]], dtype=np.float32)
    write_dynamic_shard(tmp_path, [2**45 + 9], values[:1], rank=0, world_size=2, dtype=dtype)
    write_dynamic_shard(tmp_path, [2**40 + 1], values[1:], rank=1, world_size=2, dtype=dtype)
    keys, result = load_dynamic_table(tmp_path, "item", 2)
    np.testing.assert_array_equal(keys, [2**40 + 1, 2**45 + 9])
    np.testing.assert_array_equal(result, values[::-1])


def test_dynamic_table_requires_every_shard(tmp_path):
    write_dynamic_shard(tmp_path, [4], [[1, 2]], world_size=2)
    with pytest.raises(ValueError, match="incomplete"):
        load_dynamic_table(tmp_path, "item", 2)


def test_dynamic_table_duplicate_keys_rejected(tmp_path):
    write_dynamic_shard(tmp_path, [4, 4], [[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="duplicate"):
        load_dynamic_table(tmp_path, "item", 2)


def test_dynamic_table_corrupt_values_rejected(tmp_path):
    write_dynamic_shard(tmp_path, [4, 5], [[1, 2]])
    with pytest.raises(ValueError, match="value count mismatch"):
        load_dynamic_table(tmp_path, "item", 2)


def test_export_real_torch_serialization_to_native_bundle(tmp_path):
    torch = pytest.importorskip("torch")
    config = tiny_config()
    canonical, state = upstream_state(config, "fused")
    source = tmp_path / "source"
    (source / "torch_module").mkdir(parents=True)
    torch.save({"model_state_dict": {key: torch.tensor(value.copy()) for key, value in state.items()}}, source / "torch_module" / "model.0.pth")
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps(config))
    output = export_checkpoint(source, topology, tmp_path / "export", source_layout="fused")
    actual = load_file(output / "model.safetensors")
    for key in canonical:
        np.testing.assert_array_equal(actual[key], canonical[key])
    assert json.loads((output / "config.json").read_text())["schema_version"] == 1
    with pytest.raises(FileExistsError, match="not empty"):
        export_checkpoint(source, topology, output, source_layout="fused")


def test_export_compacts_dynamic_capacity_and_preserves_signed_keys(tmp_path):
    torch = pytest.importorskip("torch")
    config = tiny_config()
    canonical, state = upstream_state(config, "native")
    state.pop("_embedding_collection.embeddings.item.weight")
    source = tmp_path / "source"
    (source / "torch_module").mkdir(parents=True)
    torch.save({"model_state_dict": {key: torch.tensor(value.copy()) for key, value in state.items()}}, source / "torch_module" / "model.0.pth")
    table_dir = source / "dynamicemb_module" / "model._embedding_collection._model_parallel_embedding_collection"
    values = canonical["embeddings.item.weight"][[3, 1, 5]]
    write_dynamic_shard(table_dir, [2**45, -2**40, 2**40], values)
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps(config))
    output = export_checkpoint(source, topology, tmp_path / "export", source_layout="native")
    actual = load_file(output / "model.safetensors")
    np.testing.assert_array_equal(actual["embeddings.item.keys"], [-2**40, 2**40, 2**45])
    np.testing.assert_array_equal(actual["embeddings.item.weight"], values[[1, 2, 0]])
    exported_config = json.loads((output / "config.json").read_text())
    assert exported_config["embedding_tables"][0]["num_embeddings"] == 3


def test_safe_export_loads_real_dynamicemb_sharded_tensor_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    import io
    import torch.distributed as dist
    from torch.distributed._shard.sharded_tensor import Shard, init_from_local_shards

    config = tiny_config()
    canonical, state = upstream_state(config, "fused")
    state.pop("_embedding_collection.embeddings.item.weight")
    state = {key: torch.tensor(value.copy()) for key, value in state.items()}
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=(tmp_path / "process-group").as_uri(), rank=0, world_size=1)
    try:
        dummy = init_from_local_shards([Shard.from_tensor_and_offsets(torch.zeros(1, 1), [0, 0], 0)], 1, 1)
        state["_embedding_collection._model_parallel_embedding_collection.embeddings.item.weight"] = dummy
        state["_hstu_block._attention_layers.0._linear_uvqk._extra_state"] = io.BytesIO(b"TransformerEngine metadata is not inference weights")
        source = tmp_path / "source"
        (source / "torch_module").mkdir(parents=True)
        torch.save({"model_state_dict": state, "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.001}]}}, source / "torch_module" / "model.0.pth")
    finally:
        dist.destroy_process_group()
    table_dir = source / "dynamicemb_module" / "model._embedding_collection._model_parallel_embedding_collection"
    write_dynamic_shard(table_dir, range(len(canonical["embeddings.item.weight"])), canonical["embeddings.item.weight"])
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps(config))
    output = export_checkpoint(source, topology, tmp_path / "export", source_layout="fused")
    actual = load_file(output / "model.safetensors")
    for key in canonical:
        np.testing.assert_array_equal(actual[key], canonical[key])
    assert not dist.is_initialized()
    # A distributed placeholder in an actual dense weight position must fail.
    loaded = _load_pytorch_checkpoint(source / "torch_module" / "model.0.pth")["model_state_dict"]
    loaded["_hstu_block._attention_layers.0._linear_uvqk_weight"] = loaded["_embedding_collection._model_parallel_embedding_collection.embeddings.item.weight"]
    with pytest.raises(ValueError, match="metadata cannot be used as weights"):
        convert_state_dict(loaded, config, source_layout="fused", checkpoint_dir=source)


def test_safe_loader_rejects_unrecognized_executable_globals(tmp_path):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "must-not-exist"

    class ExecutablePayload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    path = tmp_path / "unrecognized.pth"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": ExecutablePayload()}, path)
    with pytest.raises(Exception, match="Unsupported global"):
        _load_pytorch_checkpoint(path)
    assert not marker.exists()


def test_safe_loader_getattr_is_limited_to_process_group_metadata(tmp_path):
    torch = pytest.importorskip("torch")

    class AttributePayload:
        def __reduce__(self):
            return getattr, (torch.Tensor, "__class__")

    path = tmp_path / "attribute.pth"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": AttributePayload()}, path)
    with pytest.raises(ValueError, match="unsupported attribute lookup"):
        _load_pytorch_checkpoint(path)


def test_original_source_oracle_executes_multiple_blocks(tmp_path, original_source):
    pytest.importorskip("torch")
    from families.hstu.tests.reference import run_reference
    from families.hstu.tests.fixtures import sample_request

    source = original_source
    config = make_checkpoint(tmp_path)
    request = sample_request(config)
    result = run_reference(tmp_path, request, upstream_root=source)
    assert len(result["sequences"]) == 2
    first = result["sequences"][0]
    assert len(first["logits"]) == 8
    # Candidate isolation in the original NVIDIA mask survives multiple blocks.
    request["sequences"][0]["candidate_item_ids"][1] = request["sequences"][0]["candidate_item_ids"][2]
    changed = run_reference(tmp_path, request, upstream_root=source)["sequences"][0]
    assert first["logits"][:2] == changed["logits"][:2]
    assert first["logits"][2:4] != changed["logits"][2:4]


def test_original_retrieval_postprocessor_selects_last_item_before_action(original_source):
    torch = pytest.importorskip("torch")
    from families.hstu.tests.reference import _upstream

    original = _upstream(str(original_source))
    # Two context tokens followed by three item/action pairs; final action is
    # deliberately orthogonal to the final item so using it changes the score.
    values = torch.tensor([[9., 1.], [8., 2.], [1., 1.], [2., 1.], [1., 2.], [1., 3.], [3., 0.], [0., 3.]])
    jd = original["JaggedData"](values=values, seqlen=torch.tensor([8]), seqlen_offsets=torch.tensor([0, 8]), max_seqlen=8, scaling_seqlen=-1, total_candidates_seq_len=None, max_num_candidates=0, contextual_max_seqlen=2, contextual_seqlen_offsets=torch.tensor([0, 2]), has_interleaved_action=True)
    postprocessor = original["HSTUBlockPostprocessor"]()
    postprocessor._is_inference = False
    postprocessor._sequence_parallel = False
    result = postprocessor(jd)
    assert result.values.shape == (3, 2)
    torch.testing.assert_close(result.values[-1], torch.tensor([1., 0.]), rtol=0, atol=0)
    torch.testing.assert_close(result.seqlen_offsets, torch.tensor([0, 3]), rtol=0, atol=0)
