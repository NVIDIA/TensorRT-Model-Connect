/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "inference_state.h"
#include "kv_cache.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

enum class GptOssTriAttentionScoreAggregation {
    kMean,
    kMax,
};

enum class GptOssTriAttentionRopeStyle {
    kHalf,
    kInterleaved,
};

struct GptOssTriAttentionConfig {
    bool enabled{false};
    int32_t kv_budget{0};
    int32_t divide_length{128};
    int32_t recent_window{128};
    GptOssTriAttentionScoreAggregation score_aggregation{GptOssTriAttentionScoreAggregation::kMean};
    GptOssTriAttentionScoreAggregation per_layer_aggregation{
        GptOssTriAttentionScoreAggregation::kMean};
    bool count_prompt_tokens{true};
    bool protect_prefill{true};
    bool disable_mlr{false};
    bool disable_trig{false};
    std::string stats_section{"triattention_stats.json"};
    int32_t offset_max_length{65536};

    // Debug / profile knobs — previously read ad-hoc via std::getenv
    // (TRTMC_TRIATTN_DEBUG, TRTMC_TRIATTN_PROFILE, TRTMC_TRIATTN_DISABLE_GPU_*,
    // TRTMC_TRIATTN_DUMP_*, TRTMC_TRIATTN_ZERO_TAIL,
    // TRTMC_TRIATTN_RUNTIME_BUCKET_ROWS). Now populated once at
    // construction from the registry-supplied ConfigBundle; the runtime
    // reads these struct fields instead of hitting the environment.
    bool debug{false};
    bool profile{false};
    int32_t runtime_bucket_rows{32};
    bool disable_gpu_selection{false};
    bool disable_gpu_compaction{false};
    bool disable_gpu_state{false};
    bool zero_tail{false};
    std::string dump_keep_path{};
    int32_t dump_compaction_index{0};
    bool abort_after_dump{false};
    bool dump_score_cache{false};
    bool dump_score_values{false};
};

struct GptOssTriAttentionHeadStats {
    std::vector<float> q_mean_real;
    std::vector<float> q_mean_imag;
    std::vector<float> q_abs_mean;
    std::vector<float> freq_scale_sq;
};

struct GptOssTriAttentionStats {
    int32_t head_dim{0};
    GptOssTriAttentionRopeStyle rope_style{GptOssTriAttentionRopeStyle::kHalf};
    float rope_theta{10000.0F};
    int32_t num_attention_heads{0};
    int32_t num_key_value_heads{0};
    int32_t stats_head_count{0};
    int32_t num_layers{0};
    std::vector<float> inv_freq;
    std::vector<std::vector<int32_t>> sampled_score_heads_by_layer;
    std::vector<GptOssTriAttentionHeadStats> layer_stats;
};

struct GptOssTriAttentionCompactionProfile {
    double host_copy_ms{0.0};
    double host_convert_ms{0.0};
    double trig_prep_ms{0.0};
    double score_ms{0.0};
    double combine_ms{0.0};
    double select_ms{0.0};
    double repack_ms{0.0};
    std::size_t host_copy_bytes{0};
    std::size_t repack_bytes{0};
    int64_t repack_calls{0};
    int32_t candidate_count{0};
    int32_t reserved_count{0};
    int32_t sampled_layers{0};
    int32_t sampled_heads{0};
};

// Forward declaration — full type in trtmc/config/config_bundle.h.
namespace config {
class ConfigBundle;
}

// Build a GptOssTriAttentionConfig from the bundle's JSON plus (optionally) the
// session-resolved ConfigBundle. Legacy bundles without the generic
// `defaults:` block still parse through the JSON path; when ``runtime_config``
// has a non-default layer value for a field, that value wins. Environment
// variables (TRTMC_TRIATTN_*) are no longer read — callers supply values via
// the registry (CLI --set / --config or bundle defaults:).
GptOssTriAttentionConfig
gpt_oss_parse_triattention_bundle_config(const std::string& config_json, int32_t max_cache_length,
                                         const config::ConfigBundle* runtime_config = nullptr);

GptOssTriAttentionStats gpt_oss_parse_triattention_stats_json(const std::string& stats_json,
                                                              int32_t num_attention_heads,
                                                              int32_t num_key_value_heads,
                                                              int32_t num_layers);

class GptOssTriAttentionKvCache : public GptOssInferenceState {
  public:
    GptOssTriAttentionKvCache(int32_t num_layers, int32_t num_kv_heads, int32_t max_length,
                              int32_t kv_dim, cudaStream_t stream, GptOssTriAttentionConfig config,
                              GptOssTriAttentionStats stats, DType cache_dtype = DType::kFloat32,
                              GptOssKvCacheNames names = {});

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    void set_prompt_length(int32_t prompt_length) override;
    void mark_prefill_complete() override;
    int32_t position() const override { return absolute_position_; }
    int32_t max_length() const override { return max_length_; }
    int32_t preferred_cache_rows() const override;
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "triattention_kv_cache"; }
    bool ok() const override;

    void build_attention_mask(std::vector<float>& mask) const;

    DeviceTensor& cache_k(int32_t layer) { return cache_k_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& cache_v(int32_t layer) { return cache_v_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& present_k(int32_t layer) { return present_k_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& present_v(int32_t layer) { return present_v_[static_cast<std::size_t>(layer)]; }

    int32_t active_length() const { return cache_length_; }
    const std::vector<int32_t>& cache_positions() const { return cache_positions_; }
    const std::vector<std::vector<int32_t>>& cache_positions_per_head() const {
        return cache_positions_per_head_;
    }
    int32_t prompt_end_position() const { return prompt_end_position_; }

  private:
    void validate_shapes();
    void normalize_sampled_heads();
    void log_init_debug() const;
    void allocate_layer_tensors();
    int32_t compaction_trigger_length() const;
    int32_t compaction_keep_budget(int32_t total_tokens) const;
    int32_t count_prefix_rows() const;
    void compact_existing_cache(bool reserve_slot_for_append = false);
    void dump_cache_rows_to_file(const DeviceTensor& tensor, int32_t rows, const std::string& path,
                                 std::vector<std::string>& out_files);
    void maybe_dump_score_cache(int32_t rows, const char* k_pattern, const char* v_pattern,
                                std::vector<std::string>& score_files,
                                std::vector<std::string>& value_files);
    bool compact_layer_on_gpu(int32_t layer, const std::vector<int32_t>& keep_indices,
                              int32_t keep_count, std::size_t row_bytes, int64_t& repack_calls,
                              std::size_t& repack_bytes);
#ifdef TRTMC_HAS_CUDA_KERNELS
    bool gpu_compaction_upload_keep(int32_t keep_count, const std::vector<int32_t>& keep_indices);
    bool gpu_compact_one_layer(int32_t layer, int32_t keep_count, std::size_t row_bytes);
#endif
    void compact_layer_on_host(int32_t layer, const std::vector<int32_t>& keep_indices,
                               int32_t keep_count, std::size_t row_bytes,
                               std::size_t head_block_bytes, int32_t old_cache_length,
                               int64_t& repack_calls, std::size_t& repack_bytes);
    void finalize_repack_profile(GptOssTriAttentionCompactionProfile* profile_ptr,
                                 cudaEvent_t repack_start, cudaEvent_t repack_stop,
                                 int64_t repack_calls, std::size_t repack_bytes) const;
    std::vector<std::vector<int32_t>>
    build_new_positions_by_head(const std::vector<int32_t>& keep_indices, int32_t keep_count) const;
    void log_compact_debug(const std::vector<int32_t>& representative_positions,
                           const std::vector<int32_t>& keep_indices, int32_t keep_count) const;
    std::vector<int32_t> collect_dropped_positions(const std::vector<int32_t>& keep_indices,
                                                   int32_t keep_count) const;
    int32_t count_prefix_positions(const std::vector<int32_t>& representative_positions) const;
    void log_compact_profile(const GptOssTriAttentionCompactionProfile* profile_ptr,
                             int32_t keep_count) const;
    bool should_emit_keep_dump(const GptOssTriAttentionCompactionProfile* profile_ptr) const;
    void emit_keep_dump_json(const std::vector<int32_t>& keep_indices, int32_t keep_count,
                             const std::vector<std::vector<int32_t>>& new_positions_by_head,
                             const GptOssTriAttentionCompactionProfile* profile_ptr,
                             const std::vector<std::string>& score_cache_files,
                             const std::vector<std::string>& value_cache_files,
                             const std::vector<std::string>& post_score_cache_files,
                             const std::vector<std::string>& post_value_cache_files);
    std::vector<int32_t>
    select_keep_indices(int32_t keep_budget,
                        GptOssTriAttentionCompactionProfile* profile = nullptr);
    std::vector<char> build_reserve_mask(int32_t total_tokens, int32_t old_budget) const;
    std::vector<int32_t> broadcast_indices_per_head(std::vector<int32_t> rows,
                                                    int32_t row_count) const;
    std::vector<int32_t>
    select_keep_indices_host(int32_t keep_budget, const std::vector<int32_t>& reserved,
                             const std::vector<int32_t>& candidates,
                             GptOssTriAttentionCompactionProfile* profile = nullptr) const;
    std::vector<int32_t>
    broadcast_reserved_for_empty_budget(int32_t keep_budget,
                                        const std::vector<int32_t>& reserved) const;
    void precompute_trig_phases(std::vector<std::vector<float>>& cos_phase,
                                std::vector<std::vector<float>>& sin_phase, int32_t half_dim,
                                GptOssTriAttentionCompactionProfile* profile) const;
    bool layer_stats_shapes_valid(const GptOssTriAttentionHeadStats& layer_stats,
                                  int32_t half_dim) const;
    void extract_k_rot(const float* row_ptr, int32_t head_offset, int32_t d, int32_t half_dim,
                       float& k_rot_real, float& k_rot_imag) const;
    float reduce_trig_sums(const std::vector<float>& trig_sums) const;
    float score_one_row(const float* row_ptr, const GptOssTriAttentionHeadStats& layer_stats,
                        std::size_t stats_base, int32_t head_offset, int32_t half_dim,
                        const std::vector<std::vector<float>>& cos_phase,
                        const std::vector<std::vector<float>>& sin_phase) const;
    void score_rows_for_head(std::vector<float>& scores, const std::vector<float>& layer_cache,
                             const GptOssTriAttentionHeadStats& layer_stats, int32_t score_head,
                             int32_t cache_head, int32_t half_dim, int32_t total_tokens,
                             const std::vector<std::vector<float>>& cos_phase,
                             const std::vector<std::vector<float>>& sin_phase) const;
    void standardize_scores(std::vector<float>& scores) const;
    void accumulate_layer_fallback(const std::vector<std::vector<float>>& layer_scores,
                                   std::vector<float>& global_fallback_sum,
                                   int32_t& global_fallback_count, int32_t total_tokens) const;
    void reduce_group_into_aggregate(int32_t cache_head, int32_t total_tokens,
                                     const std::vector<std::vector<float>>& layer_scores,
                                     const std::vector<int32_t>& sampled_group,
                                     std::vector<float>& aggregate, bool first_layer_for_cache_head,
                                     float* layer_dump) const;
    void
    accumulate_layer_to_aggregate(int32_t layer, int32_t total_tokens,
                                  const std::vector<std::vector<float>>& layer_scores,
                                  const std::vector<std::vector<int32_t>>& sampled_by_cache_head,
                                  std::vector<std::vector<float>>& aggregated_scores,
                                  std::vector<int32_t>& contributing_layers_by_cache_head,
                                  std::vector<float>& layer_aggregate_dump) const;
    std::vector<float> compute_fallback_mean(const std::vector<float>& global_fallback_sum,
                                             int32_t global_fallback_count,
                                             int32_t total_tokens) const;
    void finalize_per_head_aggregate(std::vector<std::vector<float>>& aggregated_scores,
                                     const std::vector<int32_t>& contributing_layers_by_cache_head,
                                     const std::vector<float>& global_fallback_mean) const;
    void maybe_dump_score_values(const std::vector<std::vector<float>>& aggregated_scores,
                                 const std::vector<float>& layer_aggregate_dump,
                                 int32_t total_tokens) const;
    std::vector<int32_t>
    build_keep_indices_per_head(const std::vector<std::vector<float>>& aggregated_scores,
                                const std::vector<int32_t>& reserved,
                                const std::vector<int32_t>& candidates, int32_t keep_budget,
                                int32_t need) const;
    bool process_layer_for_host_selection(int32_t layer, int32_t half_dim, int32_t total_tokens,
                                          const std::vector<std::vector<float>>& cos_phase,
                                          const std::vector<std::vector<float>>& sin_phase,
                                          std::vector<std::vector<float>>& aggregated_scores,
                                          std::vector<float>& global_fallback_sum,
                                          int32_t& global_fallback_count,
                                          std::vector<int32_t>& contributing_layers_by_cache_head,
                                          std::vector<float>& layer_aggregate_dump,
                                          GptOssTriAttentionCompactionProfile* profile) const;
    std::vector<float>
    copy_cache_rows_to_host(const DeviceTensor& tensor, int32_t rows,
                            GptOssTriAttentionCompactionProfile* profile = nullptr) const;
    void sync_shared_positions_from_head0();
#ifdef TRTMC_HAS_CUDA_KERNELS
    struct LayerGpuStats {
        DeviceTensor head_offsets;
        DeviceTensor head_cache_indices;
        DeviceTensor q_mean_real;
        DeviceTensor q_mean_imag;
        DeviceTensor q_abs_mean;
        DeviceTensor freq_scale_sq;
        DeviceTensor scores;
        std::vector<int32_t> host_cache_head_indices;
        int32_t score_head_count{0};
    };

    void initialize_gpu_state();
    void allocate_core_selection_buffers(int32_t half_dim);
    void build_layer_gpu_stats(int32_t layer, int32_t half_dim);
    bool can_use_gpu_selection() const;
    bool core_selection_buffers_ready() const;
    static bool layer_gpu_stats_ready(const LayerGpuStats& layer);
    enum class GpuLayerResult { kSkipped, kContributed, kFailed };
    GpuLayerResult process_layer_for_gpu_selection(
        int32_t layer, int32_t total_tokens, int32_t num_offsets,
        std::vector<float>& aggregated_scores, std::vector<float>& global_fallback_sum,
        int32_t& global_fallback_count, std::vector<int32_t>& contributing_layers_by_cache_head,
        GptOssTriAttentionCompactionProfile* profile);
    void standardize_score_rows(float* rows, int32_t num_rows, int32_t total_tokens) const;
    void accumulate_flat_fallback(const float* rows, int32_t num_rows, int32_t total_tokens,
                                  std::vector<float>& fallback_sum, int32_t& fallback_count) const;
    void aggregate_gpu_layer_into_cache_heads(
        const std::vector<float>& host_scores, const LayerGpuStats& gpu, int32_t total_tokens,
        std::vector<float>& aggregated_scores,
        std::vector<int32_t>& contributing_layers_by_cache_head) const;
    bool upload_candidate_indices_identity(int32_t total_tokens);
    bool upload_gpu_trig_phases(int32_t num_offsets, int32_t half_dim,
                                GptOssTriAttentionCompactionProfile* profile);
    bool run_gpu_selection_over_layers(int32_t total_tokens, int32_t num_offsets,
                                       std::vector<float>& aggregated_scores,
                                       std::vector<float>& global_fallback_sum,
                                       int32_t& global_fallback_count,
                                       std::vector<int32_t>& contributing_layers_by_cache_head,
                                       int32_t& contributing_layers,
                                       GptOssTriAttentionCompactionProfile* profile);
    void
    finalize_flat_per_head_aggregate(std::vector<float>& aggregated_scores,
                                     const std::vector<int32_t>& contributing_layers_by_cache_head,
                                     const std::vector<float>& global_fallback_mean,
                                     int32_t total_tokens) const;
    std::vector<int32_t> build_keep_from_flat_aggregate(const std::vector<float>& aggregated_scores,
                                                        const std::vector<int32_t>& reserved,
                                                        const std::vector<int32_t>& candidates,
                                                        int32_t keep_budget, int32_t need,
                                                        int32_t total_tokens) const;
    std::vector<int32_t>
    select_keep_indices_gpu(int32_t keep_budget, const std::vector<int32_t>& reserved,
                            const std::vector<int32_t>& candidates,
                            GptOssTriAttentionCompactionProfile* profile = nullptr);
#endif

    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    std::vector<DeviceTensor> present_k_;
    std::vector<DeviceTensor> present_v_;
    int32_t num_layers_{0};
    int32_t num_kv_heads_{0};
    int32_t query_head_count_{0};
    int32_t query_group_size_{0};
    int32_t cache_head_count_{0};
    int32_t score_group_size_{0};
    int32_t max_length_{0};
    int32_t kv_dim_{0};
    int32_t cache_length_{0};
    int32_t absolute_position_{0};
    int32_t planned_prompt_length_{0};
    int32_t prompt_end_position_{0};
    cudaStream_t stream_{nullptr};
    std::vector<float> mask_buf_;
    int32_t pos_buf_{0};
    bool has_position_input_{false};
    bool dynamic_binding_enabled_{false};
    int32_t bound_cache_rows_{0};
    DType cache_dtype_{DType::kFloat32};
    std::size_t cache_element_size_{sizeof(float)};
    GptOssKvCacheNames names_;
    GptOssTriAttentionConfig config_;
    GptOssTriAttentionStats stats_;
    std::vector<int32_t> cache_positions_;
    std::vector<std::vector<int32_t>> cache_positions_per_head_;
    std::vector<float> offsets_;
    bool profile_enabled_{false};
    int64_t compaction_count_{0};
    TrtModule* bound_module_{nullptr};
#ifdef TRTMC_HAS_CUDA_KERNELS
    std::vector<LayerGpuStats> layer_gpu_stats_;
    DeviceTensor candidate_indices_device_;
    DeviceTensor keep_indices_device_;
    DeviceTensor positions_device_;
    DeviceTensor inv_freq_device_;
    DeviceTensor cos_phase_device_;
    DeviceTensor sin_phase_device_;
    DeviceTensor scratch_k_device_;
    DeviceTensor scratch_v_device_;
#endif
};

} // namespace trtmc
