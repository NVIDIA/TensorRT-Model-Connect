/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/tests/cpp/image_io.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

unsigned char* lighton_ocr_load_rgb(const char* path, int* width, int* height) {
    int channels = 0;
    return stbi_load(path, width, height, &channels, 3);
}

void lighton_ocr_free_rgb(unsigned char* pixels) {
    stbi_image_free(pixels);
}
