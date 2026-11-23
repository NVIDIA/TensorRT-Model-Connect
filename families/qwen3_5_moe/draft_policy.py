# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the proposal attention contract shared by this family's native and offloaded drafts."""


def block_policy(raw: dict, variant: str) -> dict:
    """Never reinterpret a causal JetSpec checkpoint as bidirectional DFlash."""
    jet = variant == "jetspec"
    if variant not in {"jetspec", "dflash", "dflash_ddtree"}:
        raise ValueError("Unsupported Qwen3.5-MoE block draft variant")
    policy = raw.get("jetspec_config") if jet else None
    if policy is None:
        policy = raw.get("dflash_config")
    if not isinstance(policy, dict):
        raise ValueError("Missing Qwen3.5-MoE draft configuration")
    causal = policy.get("causal_head", raw.get("causal_head", False))
    if causal is not jet or (not jet and "jetspec_config" in raw):
        raise ValueError("Qwen3.5-MoE draft causal_head does not match its execution variant")
    block = policy.get("block_size", raw.get("block_size"))
    mask = policy.get("mask_token_id", raw.get("mask_token_id"))
    if type(block) is not int or block != 16:
        raise ValueError("Qwen3.5-MoE draft requires the documented block16 profile")
    return {**policy, "block_size": block, "mask_token_id": mask}
