# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the original MLCommons/Meta dense model, replacing only sparse transport.

The optional 1 TB sparse table is never instantiated. No mathematical module is
substituted. The original production positional Triton kernel is selected because
the upstream PyTorch positional fallback derives its bucket cap from table width.
"""

import sys
import types
import torch
import numpy as np


def load_original(source, checkpoint):
    sys.path.insert(0, str(source))
    # The dense model imports TorchRec type names but never uses sparse lookup.
    try:
        import torchrec  # noqa: F401
    except ImportError:
        names = (
            "torchrec",
            "torchrec.modules",
            "torchrec.modules.embedding_configs",
            "torchrec.modules.embedding_modules",
        )
        for name in names:
            sys.modules[name] = types.ModuleType(name)

        class SparseLookup:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("sparse lookup is outside the dense oracle")

        sys.modules["torchrec"].KeyedJaggedTensor = SparseLookup
        sys.modules["torchrec.modules.embedding_modules"].EmbeddingCollection = SparseLookup
        sys.modules["torchrec.modules.embedding_configs"].EmbeddingConfig = SparseLookup
        sys.modules["torchrec.modules.embedding_configs"].DataType = type(
            "DataType", (), {"FP16": 0, "FP32": 1}
        )

    # FBGEMM is optional here: these routines only copy packed rows to/from padding.
    def offsets(lengths):
        return torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])

    def pad(values, offsets, max_lengths, padding_value=0):
        o = offsets[0].tolist()
        result = values.new_full((len(o) - 1, max_lengths[0], *values.shape[1:]), padding_value)
        for row, (a, b) in enumerate(zip(o, o[1:])):
            result[row, : b - a] = values[a:b]
        return result

    def unpad(dense, offsets, total_L=None):
        o = offsets[0].tolist()
        return torch.cat([dense[i, : b - a] for i, (a, b) in enumerate(zip(o, o[1:]))]), offsets

    torch.ops.fbgemm.asynchronous_complete_cumsum = offsets
    torch.ops.fbgemm.jagged_to_padded_dense = pad
    torch.ops.fbgemm.dense_to_jagged = unpad
    from configs import get_hstu_configs
    from generative_recommenders.modules.dlrm_hstu import DlrmHSTU
    from generative_recommenders.common import HammerKernel

    model = DlrmHSTU(
        get_hstu_configs(),
        embedding_tables={},
        is_inference=True,
        is_dense=True,
        bf16_training=False,
    ).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["dense_dict"]
    model.load_state_dict(state, strict=True)
    model.set_hammer_kernel(HammerKernel.PYTORCH)
    model._hstu_transducer._positional_encoder.set_hammer_kernel(HammerKernel.TRITON)
    return model.cuda()


def score_original(model, config, inputs):
    c = config.serving.candidates
    x = {k: torch.from_numpy(v).cuda() for k, v in inputs.items()}
    length, count = x["history_lengths"][:, 0], x["num_candidates"][:, 0]

    def jagged(history, targets):
        return torch.cat(
            [
                torch.cat([history[i, : int(length[i])], targets[i, : int(count[i])]])
                for i in range(len(length))
            ]
        )

    item = jagged(x["history_item_embeddings"], x["candidate_item_embeddings"])
    category = jagged(x["history_category_embeddings"], x["candidate_category_embeddings"])
    times = jagged(x["history_timestamps"], x["query_timestamp"].expand(-1, c))
    actions = torch.cat([x["history_actions"][i, : int(n)] for i, n in enumerate(length)])
    payload = dict(
        item_category_id=category,
        item_weights=actions,
        user_id=x["user_embeddings"].flatten(0, 1),
        user_id_offsets=torch.arange(len(length) + 1, device="cuda"),
    )
    with torch.no_grad():
        user, _ = model._hstu_transducer(
            max_uih_len=config.serving.history_length,
            max_targets=c,
            total_uih_len=int(length.sum()),
            total_targets=int(count.sum()),
            seq_lengths=length + count,
            seq_timestamps=times,
            seq_embeddings=item,
            num_targets=count,
            seq_payloads=payload,
        )
        candidate = torch.cat(
            [
                torch.cat(
                    [
                        x["candidate_item_embeddings"][i, : int(n)],
                        x["candidate_category_embeddings"][i, : int(n)],
                    ],
                    -1,
                )
                for i, n in enumerate(count)
            ]
        )
        item = model._item_embedding_mlp(candidate)
        scores = model._multitask_module(user, item, {}, {})[0].cpu().numpy().reshape(-1)
    result = np.zeros((len(length), c), np.float32)
    offset = 0
    for row, n in enumerate(count.tolist()):
        result[row, :n] = scores[offset : offset + n]
        offset += n
    return result
