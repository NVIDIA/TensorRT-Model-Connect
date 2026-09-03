# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Boltz-2."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("boltz2", "boltz_2"),
    architectures=("Boltz2ForStructurePrediction",),
    required_files=(
        "boltz2_conf.ckpt",
        "protein_monomer.yaml",
        "protein_monomer.a3m",
        "processed/structures/protein_monomer.npz",
        "processed/records/protein_monomer.json",
        "mols.tar",
    ),
    tasks=("structure_prediction",),
    default_task="structure_prediction",
)
