# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static regressions for Nemotron-H rank-local runtime state sizing."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_runtime_sizes_hybrid_state_from_selected_rank_engine() -> None:
    source = (
        ROOT / "src/runtime/models/nemotron_h/plugin.cpp"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.split())

    assert (
        'const int32_t kv_dim = num_attention_layers > 0 '
        '? cache_row_dim_from_module(*loaded.module, "cache_k_0") '
        ': compute_kv_dim(ctx.config);'
    ) in compact
    assert (
        'const int64_t conv_elems = num_mamba_layers > 0 '
        '? positive_numel_from_module(*loaded.module, "conv_state_0") '
        ': static_cast<int64_t>(effective_conv_dim) * mamba_d_conv;'
    ) in compact
    assert (
        'const int64_t ssm_elems = num_mamba_layers > 0 '
        '? positive_numel_from_module(*loaded.module, "ssm_state_0") '
        ': static_cast<int64_t>(mamba_nheads) * std::max(mamba_head_dim, 1) '
        '* mamba_d_state;'
    ) in compact


def test_runtime_reads_only_the_selected_lazy_tp_plan() -> None:
    source = (
        ROOT / "src/runtime/models/nemotron_h/plugin.cpp"
    ).read_text(encoding="utf-8")

    start = source.index("const std::string engine_section")
    end = source.index("auto tokenizer = create_tokenizer_from_bundle", start)
    selection = source[start:end]

    eager_lookup = "find_section(ctx.bundle, engine_section)"
    lazy_read = "ReadBundleSection(ctx.bundle_path, *section_info)"
    module_load = "ctx.backend, engine_plan, engine_section.c_str(), opts"
    assert eager_lookup in selection
    assert "if (engine_plan == nullptr)" in selection
    assert "ctx.bundle.info.sections" in selection
    assert lazy_read in selection
    assert selection.count("ReadBundleSection(") == 1
    assert "return section.name == engine_section;" in selection
    assert module_load in selection
    assert selection.index(eager_lookup) < selection.index(lazy_read)
    assert selection.index(lazy_read) < selection.index(module_load)
