# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only oracle executing the pinned NVIDIA HSTU PyTorch source.

Only dependency imports, instrumentation decorators, and jagged tensor storage
are adapted. The upstream attention mask, SiLU attention, layer forward,
normalization/gating, and prediction MLP methods execute unchanged. Embedding
lookup and timestamp/position indexing are CPU boundary adapters for upstream
CUDA/Triton operations. No part of this module is a production runtime.
"""

from __future__ import annotations

import ast
import functools
import subprocess
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Union

REFERENCE_REVISION = "97062d97eef53115105063801e35184e36186df5"
SOURCE_FILES = (
    "examples/hstu/ops/pt_ops/pt_hstu_attention.py",
    "examples/hstu/ops/pt_ops/pt_norm_mul_dropout.py",
    "examples/hstu/modules/debug/debug_hstu_layer.py",
    "examples/hstu/modules/mlp.py",
    "examples/hstu/modules/output_postprocessors.py",
    "examples/hstu/modules/position_encoder.py",
    "examples/hstu/ops/triton_ops/triton_position.py",
    "examples/hstu/modules/hstu_processor.py",
    "examples/hstu/modules/similarity/dot_product.py",
)



def _verify_source(root: Path) -> None:
    command = ["git", "-c", f"safe.directory={root.resolve()}", "-C", str(root)]
    revision = subprocess.run(command + ["rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    if revision != REFERENCE_REVISION:
        raise ValueError(f"HSTU oracle requires upstream revision {REFERENCE_REVISION}, found {revision}")
    subprocess.run(command + ["ls-files", "--error-unmatch", "--", *SOURCE_FILES], check=True, capture_output=True)
    changed = subprocess.run(command + ["diff", "--quiet", REFERENCE_REVISION, "--", *SOURCE_FILES], capture_output=True)
    if changed.returncode:
        raise ValueError("HSTU oracle source files differ from the pinned upstream revision")


def reference_receipt(upstream_root: str | Path | None = None) -> dict:
    root = Path(upstream_root or os.environ["TRTMC_HSTU_REFERENCE_ROOT"])
    _verify_source(root)
    return {
        "repository": "https://github.com/NVIDIA/recsys-examples",
        "revision": REFERENCE_REVISION,
        "source_files": list(SOURCE_FILES),
        "verification": "Exact Git revision and tracked source files unchanged against that revision",
        "adapters": ["CPU embedding lookup", "jagged packing and unpacking", "position and timestamp indexing", "removed instrumentation decorators and optional dependency imports"],
        "weights": "deterministic synthetic fixture; not a trained checkpoint",
    }


def _definitions(path: Path, names: set[str], namespace: dict, *, class_methods: dict[str, set[str]] | None = None) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and class_methods and node.name in class_methods:
            node.bases = [ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="nn", ctx=ast.Load())]
            node.bases = [ast.Attribute(value=node.bases[0], attr="Module", ctx=ast.Load())]
            node.keywords = []
            node.decorator_list = []
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in class_methods[node.name]]
            for method in node.body:
                method.decorator_list = []
            selected.append(node)
    if len(selected) != len(names) + len(class_methods or {}):
        raise ValueError(f"upstream source definitions changed: {path}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


@functools.lru_cache(maxsize=2)
def _upstream(root: str) -> dict:
    import torch
    import torch.nn.functional as F

    root_path = Path(root)
    _verify_source(root_path)

    def dense_to_jagged(dense, offsets, length):
        offset = offsets[0].tolist()
        return (torch.cat([dense[b, : offset[b + 1] - start] for b, start in enumerate(offset[:-1])]),)

    class TorchBoundary:
        ops = SimpleNamespace(fbgemm=SimpleNamespace(dense_to_jagged=dense_to_jagged))

        def __getattr__(self, name):
            return getattr(torch, name)

    def pad_qkv(q, k, v, offsets, n):
        packed = []
        indices = offsets.tolist()
        for tensor in (q, k, v):
            dense = tensor.new_zeros((len(indices) - 1, n, tensor.shape[1], tensor.shape[2]))
            for batch, start in enumerate(indices[:-1]):
                length = indices[batch + 1] - start
                dense[batch, :length] = tensor[start : start + length]
            packed.append(dense.transpose(1, 2))
        return tuple(packed)

    def split_jagged(values, max_seqlen, *, offsets_a, offsets_b, seq_len_a=None, seq_len_b=None):
        a_offsets, b_offsets = offsets_a.tolist(), offsets_b.tolist()
        parts_a, parts_b = [], []
        for index in range(len(a_offsets) - 1):
            start = a_offsets[index] + b_offsets[index]
            width_a = a_offsets[index + 1] - a_offsets[index]
            width_b = b_offsets[index + 1] - b_offsets[index]
            parts_a.append(values[start : start + width_a])
            parts_b.append(values[start + width_a : start + width_a + width_b])
        return torch.cat(parts_a), torch.cat(parts_b)

    class JaggedData(SimpleNamespace):
        def copy_others_but_set_values(self, *, values):
            return JaggedData(**{**vars(self), "values": values})

    ns = {
        "torch": TorchBoundary(), "F": F, "Optional": Optional, "Tuple": Tuple, "Union": Union,
        "JaggedData": JaggedData, "_pad_qkv": pad_qkv, "triton_split_2D_jagged": split_jagged,
        "nvtx": SimpleNamespace(annotate=lambda *args, **kwargs: nullcontext()),
    }
    _definitions(root_path / SOURCE_FILES[0], {"_get_valid_attn_mask", "pytorch_hstu_mha"}, ns)
    _definitions(root_path / SOURCE_FILES[1], {"pytorch_norm_mul_dropout"}, ns)
    _definitions(root_path / SOURCE_FILES[2], set(), ns, class_methods={"HSTULayer": {"get_user_value_query_key_tensors", "forward"}})
    _definitions(root_path / SOURCE_FILES[3], set(), ns, class_methods={"MLP": {"forward"}})
    _definitions(root_path / SOURCE_FILES[4], set(), ns, class_methods={"L2NormEmbeddingPostprocessor": {"forward"}})
    _definitions(root_path / SOURCE_FILES[5], {"_get_high_inds"}, ns)
    _definitions(root_path / SOURCE_FILES[7], set(), ns, class_methods={"HSTUBlockPostprocessor": {"forward"}})
    _definitions(root_path / SOURCE_FILES[8], set(), ns, class_methods={"DotProductSimilarity": {"forward"}})
    return ns


def run_reference(checkpoint_dir: str | Path, request: dict, *, upstream_root: str | Path | None = None, precision: str = "fp32") -> dict:
    """Return native CLI-shaped results from original upstream layer methods."""
    import torch
    from safetensors.torch import load_file

    from ..config import load_config

    torch.set_num_threads(1)
    checkpoint_dir = Path(checkpoint_dir)
    config = load_config(checkpoint_dir / "config.json")
    weights = load_file(str(checkpoint_dir / "model.safetensors"))
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    upstream = _upstream(str(upstream_root or os.environ["TRTMC_HSTU_REFERENCE_ROOT"]))
    d, h, a = (config[key] for key in ("hidden_size", "num_heads", "head_dim"))
    table_by_role = {table["role"]: table for table in config["embedding_tables"]}

    def lookup(name, ids):
        matrix = weights[f"embeddings.{name}.weight"]
        ids = torch.tensor(ids, dtype=torch.int64)
        keys = weights.get(f"embeddings.{name}.keys")
        if keys is not None:
            rows = torch.searchsorted(keys, ids)
            if bool((rows >= len(keys)).any()) or not torch.equal(keys[rows], ids):
                raise ValueError(f"unknown sparse ID in {name}")
        else:
            rows = ids
            if bool(((rows < 0) | (rows >= len(matrix))).any()):
                raise ValueError(f"unknown embedding ID in {name}")
        return matrix[rows]

    def linear(weight_name, bias_name=None, *, layer_dtype=dtype):
        matrix = weights[weight_name].to(layer_dtype)
        layer = torch.nn.Linear(matrix.shape[1], matrix.shape[0], bias=bias_name in weights if bias_name else False, dtype=layer_dtype)
        layer.weight = torch.nn.Parameter(matrix, requires_grad=False)
        if bias_name and bias_name in weights:
            layer.bias = torch.nn.Parameter(weights[bias_name].to(layer_dtype), requires_grad=False)
        return layer

    layers = []
    for index in range(config["num_layers"]):
        layer = upstream["HSTULayer"]()
        layer._embedding_dim, layer._num_heads = d, h
        layer._linear_dim_per_head = layer._attention_dim_per_head = a
        layer._split_arg_list = [a, a, a, a]
        layer._eps = config["layer_norm_epsilon"]
        layer._dropout_ratio = 0.0
        layer._residual = config["residual"]
        layer._disable_contextual_mask = config["disable_contextual_mask"]
        layer._target_group_size = config["target_group_size"]
        for flag in ("_debug_mock_tp", "_debug_check_tp_equal", "_debug_shortcut_proj_linear", "_debug_shortcut_output_ln_mul_dropout"):
            setattr(layer, flag, False)
        prefix = f"blocks.{index}."
        for norm in ("input", "output"):
            for suffix in ("weight", "bias"):
                value = weights.get(prefix + f"{norm}_norm.{suffix}")
                setattr(layer, f"_{norm}_layernorm_{suffix}", None if value is None else value.to(dtype))
        layer._linear_uvqk = linear(prefix + "uvqk.weight", prefix + "uvqk.bias")
        layer._linear_proj = linear(prefix + "proj.weight")
        layer._norm_mul_dropout_func = upstream["pytorch_norm_mul_dropout"]

        def attention(q, k, v, offsets, *, max_seqlen, scaling_seqlen, num_candidates, num_contextuals, target_group_size):
            return upstream["pytorch_hstu_mha"](max_seq_len=max_seqlen, alpha=1.0 / a**0.5, q=q.reshape(-1, h, a), k=k.reshape(-1, h, a), v=v.reshape(-1, h, a), seq_offsets=offsets, causal=config["is_causal"], num_targets=num_candidates, num_contextuals=num_contextuals, target_group_size=target_group_size, scaling_seqlen=scaling_seqlen, training=False).reshape(-1, h * a)

        layer._attn_func = attention
        layers.append(layer.eval())
    head = upstream["MLP"]()
    head_layers = []
    for index in range(len(config["prediction_head"])):
        head_layers.append(linear(f"head.{index}.weight", f"head.{index}.bias"))
        if index < len(config["prediction_head"]) - 1:
            head_layers.append(torch.nn.ReLU() if config["prediction_activation"] == "relu" else torch.nn.GELU())
    head._mlp = torch.nn.Sequential(*head_layers)
    normalizer = upstream["L2NormEmbeddingPostprocessor"]()
    normalizer._embedding_dim, normalizer._eps = d, config["output_norm_epsilon"]

    assembled = []
    for sequence in request["sequences"]:
        item_table = table_by_role["item"]["name"]
        history = lookup(item_table, sequence.get("history_item_ids", [])).to(dtype)
        candidate_ids = sequence.get("candidate_item_ids", [])
        candidates = lookup(item_table, candidate_ids)
        features = {feature["name"]: feature["ids"] for feature in sequence.get("contextual_features", [])}
        context_parts = [lookup(table["name"], features[table["name"]]).to(dtype) for table in config["embedding_tables"] if table["role"] == "context" and table["name"] in features]
        context = torch.cat(context_parts) if context_parts else history.new_empty((0, d))
        if "action" in table_by_role:
            action = lookup(table_by_role["action"]["name"], sequence.get("history_action_ids", [])).to(dtype)
            history = torch.stack((history, action), dim=1).reshape(-1, d)
        # Retrieval scores a completed user history against candidate table rows.
        values = torch.cat((context, history, candidates.to(dtype))) if config["mode"] == "ranking" else torch.cat((context, history))
        assembled.append((sequence, values, len(context), candidates, candidate_ids))
    max_seqlen = max(len(values) for _, values, _, _, _ in assembled)
    outputs = []
    with torch.inference_mode():
        for sequence, values, context_count, candidates, candidate_ids in assembled:
            candidate_count = len(candidate_ids) if config["mode"] == "ranking" else 0
            length, history_end = len(values), len(values) - candidate_count
            if config["position_buckets"]:
                if config["time_buckets"]:
                    positions = (history_end - torch.arange(length)).clamp(min=0, max=config["position_buckets"] - 1)
                    timestamps = sequence["token_timestamps"]
                    # The Triton `where` with 1e-6 promotes elapsed seconds to
                    # FP32 before division and sqrt. Subtract Python integers
                    # first to avoid signed-64 overflow at extreme input IDs.
                    elapsed = torch.tensor([max(timestamps[-1] - timestamp, 1e-6) for timestamp in timestamps], dtype=torch.float32)
                    time_ids = torch.sqrt(elapsed / 60.0).long().clamp(0, 2048)
                    position_values = weights["position.weight"][positions].to(dtype).float()
                    time_values = weights["time.weight"][time_ids].to(dtype).float()
                    values = values * d**0.5 + (position_values + time_values).to(dtype)
                else:
                    high = upstream["_get_high_inds"](torch.tensor([length]), weights["position.weight"], torch.tensor([candidate_count]), False)
                    positions = torch.minimum(torch.arange(length), high[0])
                    values = (values.float() * d**0.5 + weights["position.weight"][positions].to(dtype).float()).to(dtype)
            jd = upstream["JaggedData"](values=values, seqlen=torch.tensor([length]), seqlen_offsets=torch.tensor([0, length]), max_seqlen=max_seqlen, scaling_seqlen=config["scaling_seqlen"], contextual_seqlen=torch.tensor([context_count]) if context_count else None, num_candidates=torch.tensor([candidate_count]) if candidate_count else None, total_candidates_seq_len=None, max_num_candidates=candidate_count, contextual_max_seqlen=context_count, contextual_seqlen_offsets=torch.tensor([0, context_count]), has_interleaved_action="action" in table_by_role)
            for layer in layers:
                jd = layer(jd)
            sequence_embeddings = normalizer(jd.values)
            if config["mode"] == "ranking":
                embeddings = sequence_embeddings[-candidate_count:] if candidate_count else jd.values.new_empty((0, d))
                logits = head(embeddings)
                output = {"candidate_item_ids": candidate_ids, "num_candidates": len(candidate_ids), "embedding_dim": d, "output_dim": config["prediction_head"][-1], "embeddings": embeddings.flatten().tolist(), "logits": logits.flatten().tolist()}
            else:
                # RetrievalGR uses the training HSTUBlock postprocessor: remove
                # contextual outputs and every action output, then predict from
                # the final ITEM row. The final action is a distinct token.
                postprocessor = upstream["HSTUBlockPostprocessor"]()
                postprocessor._is_inference = False
                postprocessor._sequence_parallel = False
                item_outputs = postprocessor(jd)
                user = item_outputs.values[-1:].float()
                items = normalizer(candidates.float())
                # Original similarity's singleton-table branch returns (scores, {}).
                similarity = upstream["DotProductSimilarity"]()
                similarity._dtype = torch.float32
                scores, _ = similarity(user, items.unsqueeze(0))
                embeddings = items
                output = {"candidate_item_ids": candidate_ids, "num_candidates": len(candidate_ids), "embedding_dim": d, "output_dim": 1, "embeddings": embeddings.flatten().tolist(), "logits": [], "scores": scores.flatten().tolist()}
            output["sequence_embeddings"] = sequence_embeddings.flatten().tolist()
            output["sequence_length"] = len(values)
            outputs.append(output)
    return {"sequences": outputs}
