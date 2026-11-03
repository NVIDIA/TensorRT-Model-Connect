# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The exact reduced Qwen language case shared by its gate and diagnostics."""

from families.minimax_h3 import text_encoder_builder as text
from families.minimax_h3.checkpoint import numpy_state


RTOL = 0.016
ATOL = 1e-5
SEED = 491
ROW_COUNTS = (4, 7, 11)


def make_reference_language_case(monkeypatch):
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    # One extra upstream layer makes hidden_states[4] a pre-final-norm state,
    # just as the original H3 consumes hidden_states[50] from 64 layers.
    values = dict(
        HIDDEN_SIZE=16,
        NUM_LAYERS=4,
        NUM_HEADS=2,
        NUM_KV_HEADS=1,
        HEAD_DIM=8,
        INTERMEDIATE_SIZE=32,
        VOCAB_SIZE=128,
        MROPE_SECTION=(2, 1, 1),
    )
    for name, value in values.items():
        monkeypatch.setattr(text, name, value)
    torch.manual_seed(SEED)
    config = Qwen3VLTextConfig(
        hidden_size=16,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=128,
        use_cache=False,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=text.ROPE_THETA,
            mrope_section=[2, 1, 1],
            mrope_interleaved=True,
        ),
    )
    config._attn_implementation = "sdpa"
    original = Qwen3VLTextModel(config).eval()
    with torch.no_grad():
        for name, value in original.named_parameters():
            if name.endswith("norm.weight"):
                value.copy_(torch.linspace(0.6, 1.4, value.numel()).reshape_as(value))
    rotary_frequency = original.rotary_emb.inv_freq.clone()
    original.to(dtype=torch.bfloat16)
    original.rotary_emb.inv_freq = rotary_frequency
    state = original.state_dict()
    prefix = "model.language_model."
    weights = numpy_state({name: state[name[len(prefix) :]] for name in text.checkpoint_keys()})
    return original, weights


def reference_language_inputs(rows):
    import torch

    ids = torch.randint(0, 128, (rows,), dtype=torch.int32, device="cuda")
    positions = torch.arange(rows, dtype=torch.int32, device="cuda").expand(3, -1).clone()
    # Nonidentical axes expose a one-axis RoPE shortcut.
    positions[1] *= 3
    positions[2] *= 7
    mask = torch.zeros(rows, 1, device="cuda", dtype=torch.bool)
    mask[1:3] = True
    visual = torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16)
    deepstack = [
        torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16) * 0.03 for _ in range(3)
    ]
    for value in deepstack:
        value[~mask[:, 0]] = 127
    return dict(
        input_ids=ids,
        position_ids=positions,
        visual_mask=mask.int(),
        visual_features=visual,
        **{f"deepstack_features_{i}": value for i, value in enumerate(deepstack)},
    )


def reference_language_expected(original, inputs):
    import torch

    ids = inputs["input_ids"]
    mask = inputs["visual_mask"].bool()
    embeds = original.embed_tokens(ids.long()[None])
    embeds = torch.where(mask[None], inputs["visual_features"][None], embeds)
    return original(
        inputs_embeds=embeds,
        position_ids=inputs["position_ids"].long()[:, None],
        attention_mask=torch.ones(1, ids.numel(), device="cuda", dtype=torch.long),
        visual_pos_masks=mask[:, 0][None],
        deepstack_visual_embeds=[inputs[f"deepstack_features_{i}"][mask[:, 0]] for i in range(3)],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states[4][0]
