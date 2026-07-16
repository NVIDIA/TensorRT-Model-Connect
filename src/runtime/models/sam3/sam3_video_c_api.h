/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

// Opaque SAM3-only prompted-video ABI. The generic Model Connect pipeline ABI
// intentionally remains image-only; this surface is exported by
// libtrtmc_model_sam3.so and consumed by the customer compatibility layer.
extern "C" {

const char* trtmc_sam3_video_last_error() noexcept;

void* trtmc_sam3_video_create(const char* bundle_path, const char* plugin_dir,
                              const char* backend_dir) noexcept;
void trtmc_sam3_video_destroy(void* opaque) noexcept;

int32_t trtmc_sam3_video_begin(void* opaque, int32_t expected_frames) noexcept;
int32_t trtmc_sam3_video_append_frame(void* opaque, const float* pixels, int32_t height,
                                      int32_t width) noexcept;
int32_t trtmc_sam3_video_add_prompt(void* opaque, const char* prompt) noexcept;
int32_t trtmc_sam3_video_propagate(void* opaque, int32_t* object_counts, int32_t capacity) noexcept;
int32_t trtmc_sam3_video_close_session(void* opaque) noexcept;

} // extern "C"
