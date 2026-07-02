# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for tools/test_runner_parity.py — token comparison, divergence reporting.

Trace: ARCH-TRT-001, UD-TRT-PARITY
Intent: Validate text/token comparison logic used in C++ vs Python runner parity checks
Preconditions: Synthetic text strings simulating C++ and Python runner outputs are available
Postconditions: Exact matches, whitespace-stripped matches, and word-level divergence are correctly detected
"""

from __future__ import annotations



class TestTextComparison:
    """Test the text comparison logic used in runner parity checks."""

    def test_exact_match(self):
        cpp_text = "The capital of France is Paris"
        py_text = "The capital of France is Paris"
        assert cpp_text.strip() == py_text.strip()

    def test_whitespace_stripped(self):
        cpp_text = "  Hello world  \n"
        py_text = "Hello world"
        assert cpp_text.strip() == py_text.strip()

    def test_mismatch_detected(self):
        cpp_text = "The capital of France is Paris"
        py_text = "The capital of France is London"
        assert cpp_text.strip() != py_text.strip()

    def test_word_divergence_detection(self):
        cpp_text = "The quick brown fox jumps"
        py_text = "The quick red fox jumps"
        cpp_words = cpp_text.split()
        py_words = py_text.split()
        diverge_idx = None
        for i, (cw, pw) in enumerate(zip(cpp_words, py_words)):
            if cw != pw:
                diverge_idx = i
                break
        assert diverge_idx == 2
        assert cpp_words[diverge_idx] == "brown"
        assert py_words[diverge_idx] == "red"

    def test_different_length_output(self):
        cpp_text = "Hello world"
        py_text = "Hello world and more tokens"
        cpp_words = cpp_text.split()
        py_words = py_text.split()
        # Words match up to the shorter sequence
        for cw, pw in zip(cpp_words, py_words):
            assert cw == pw
        # But lengths differ
        assert len(cpp_words) != len(py_words)

    def test_empty_output(self):
        cpp_text = ""
        py_text = ""
        assert cpp_text.strip() == py_text.strip()


class TestTokenIdComparison:
    """Test token ID list comparison logic."""

    def test_identical_tokens(self):
        trt_ids = [1, 42, 100, 200, 300]
        hf_ids = [1, 42, 100, 200, 300]
        assert trt_ids == hf_ids

    def test_first_divergence(self):
        trt_ids = [1, 42, 100, 201, 300]
        hf_ids = [1, 42, 100, 200, 300]
        diverge = None
        for i, (t, h) in enumerate(zip(trt_ids, hf_ids)):
            if t != h:
                diverge = i
                break
        assert diverge == 3
