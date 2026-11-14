/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif
unsigned char* lighton_ocr_load_rgb(const char* path, int* width, int* height);
void lighton_ocr_free_rgb(unsigned char* pixels);
#ifdef __cplusplus
}
#endif
