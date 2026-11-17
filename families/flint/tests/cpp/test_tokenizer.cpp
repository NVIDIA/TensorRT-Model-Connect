/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdio>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs("usage: test_flint_tokenizer PYTHON [ARG...]\n", stderr);
        return 2;
    }
    // Replace the test driver so CTest receives pytest's exact result.
    execv(argv[1], argv + 1);
    std::perror("cannot execute Flint tokenizer tests");
    return 1;
}
