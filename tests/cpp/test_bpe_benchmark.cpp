/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-TOK-CPP-04
// Architecture:   ARCH-TOK-001
// Unit Design:    UD-TOK-01
// Intent:         BPE tokenizer performance: init time, encode/decode latency, throughput,
// round-trip Preconditions:  Built-in small vocab or external tokenizer.json via TOKENIZER_JSON env
// var Postconditions: Encode/decode execute within measured latency; round-trip verified for real
// vocabs
// =============================================================================

// BPE Tokenizer performance benchmark.
//
// Usage:
//   ./test_bpe_benchmark                          # runs with built-in small vocab
//   TOKENIZER_JSON=/path/to/tokenizer.json ./test_bpe_benchmark  # real vocab
//
// Measures: init time, encode latency, decode latency, throughput.

#include "trtmc/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
using Us = std::chrono::microseconds;
using Ms = std::chrono::milliseconds;

// Built-in minimal tokenizer for baseline benchmark (no external files needed)
const char* kBuiltinJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5,
      "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11,
      "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17,
      "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23,
      "y": 24, "z": 25,
      "0": 26, "1": 27, "2": 28, "3": 29, "4": 30, "5": 31,
      "6": 32, "7": 33, "8": 34, "9": 35,
      "\u0120": 36, "'": 37, "!": 38, ".": 39, ",": 40,
      "th": 41, "he": 42, "in": 43, "er": 44, "an": 45,
      "the": 46, "ing": 47, "tion": 48, "and": 49,
      "\u0120t": 50, "\u0120th": 51, "\u0120the": 52,
      "\u0120a": 53, "\u0120an": 54, "\u0120and": 55,
      "\u0120i": 56, "\u0120in": 57
    },
    "merges": [
      "t h", "h e", "i n", "e r", "a n",
      "th e", "in g", "t ion", "an d",
      "\u0120 t", "\u0120t h", "\u0120th e",
      "\u0120 a", "\u0120a n", "\u0120an d",
      "\u0120 i", "\u0120i n"
    ]
  },
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false
  }
})";

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return "";
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Test strings of varying length
std::vector<std::string> make_test_strings() {
    return {
        // Short (1-5 words)
        "hello",
        "the quick brown fox",
        "testing 123",
        // Medium (1-2 sentences)
        "the quick brown fox jumps over the lazy dog and the cat",
        "in the beginning there was nothing and then there was something",
        "this is a test of the emergency broadcast system!",
        // Long (paragraph, ~360 chars)
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog.",
        // Very long (~1500 chars)
        "The Transformer architecture was introduced in 2017 by Vaswani et al. "
        "in the paper Attention Is All You Need. It has since become the "
        "foundation for modern language models including GPT, encoder, and SentencePiece. "
        "These models have revolutionized natural language processing and enabled "
        "applications like machine translation, text generation, and question "
        "answering. The key innovation is the self-attention mechanism which "
        "allows the model to attend to all positions in the input sequence "
        "simultaneously, rather than processing tokens sequentially as in RNNs. "
        "This enables much more efficient training on modern GPU hardware and "
        "has led to models with billions of parameters. The architecture consists "
        "of an encoder and decoder, each made up of multiple layers of "
        "multi-head attention and feed-forward networks. Pre-training on large "
        "text corpora followed by fine-tuning on specific tasks has become the "
        "dominant paradigm in NLP. Recent advances include instruction tuning, "
        "RLHF, and chain-of-thought prompting which further improve model "
        "capabilities for real-world applications.",
        // Code-like (with indentation and special chars)
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n"
        "\n"
        "for i in range(10):\n"
        "    print(f'fib({i}) = {fibonacci(i)}')\n",
    };
}

struct BenchResult {
    std::string label;
    double mean_us;
    double min_us;
    double max_us;
    int iterations;
};

void print_results(const std::vector<BenchResult>& results) {
    std::cerr << "\n" << std::string(75, '=') << "\n";
    std::cerr << std::left << std::setw(35) << "Benchmark" << std::right << std::setw(10)
              << "Mean(us)" << std::setw(10) << "Min(us)" << std::setw(10) << "Max(us)"
              << std::setw(10) << "Iters"
              << "\n";
    std::cerr << std::string(75, '-') << "\n";

    for (const auto& r : results) {
        std::cerr << std::left << std::setw(35) << r.label << std::right << std::setw(10)
                  << std::fixed << std::setprecision(1) << r.mean_us << std::setw(10) << r.min_us
                  << std::setw(10) << r.max_us << std::setw(10) << r.iterations << "\n";
    }
    std::cerr << std::string(75, '=') << "\n";
}

} // namespace

int main() {
    std::cerr << "BPE Tokenizer Performance Benchmark\n\n";

    // Load tokenizer JSON
    std::string json_data;
    const char* json_path = std::getenv("TOKENIZER_JSON");
    if (json_path) {
        json_data = read_file(json_path);
        if (json_data.empty()) {
            std::cerr << "ERROR: cannot read " << json_path << "\n";
            return 1;
        }
        std::cerr << "Using real tokenizer: " << json_path << " (" << json_data.size() / 1024
                  << " KB)\n";
    } else {
        json_data = kBuiltinJson;
        std::cerr << "Using built-in small vocab (set TOKENIZER_JSON for real benchmark)\n";
    }

    std::vector<BenchResult> results;

    // --- 1. Initialization benchmark ---
    {
        const int N = 10;
        std::vector<double> times;
        for (int i = 0; i < N; ++i) {
            auto t0 = Clock::now();
            auto tok = trtmc::CreateBpeTokenizer(json_data.data(), json_data.size(), false);
            auto t1 = Clock::now();
            (void)tok;
            double us = std::chrono::duration_cast<Us>(t1 - t0).count();
            times.push_back(us);
        }
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / N;
        double mn = *std::min_element(times.begin(), times.end());
        double mx = *std::max_element(times.begin(), times.end());
        results.push_back({"Init (parse JSON + build tables)", mean, mn, mx, N});
    }

    // Create tokenizer for encode/decode benchmarks
    auto tok = trtmc::CreateBpeTokenizer(json_data.data(), json_data.size(), false);
    auto test_strings = make_test_strings();

    // --- 2. Encode benchmarks ---
    {
        const int WARMUP = 50;
        const int N = 1000;

        for (size_t si = 0; si < test_strings.size(); ++si) {
            const auto& text = test_strings[si];

            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                auto ids = tok->encode(text);
                (void)ids;
            }

            // Measure
            std::vector<double> times;
            int total_tokens = 0;
            for (int i = 0; i < N; ++i) {
                auto t0 = Clock::now();
                auto ids = tok->encode(text);
                auto t1 = Clock::now();
                double us = std::chrono::duration_cast<Us>(t1 - t0).count();
                times.push_back(us);
                total_tokens += static_cast<int>(ids.size());
            }

            double mean = std::accumulate(times.begin(), times.end(), 0.0) / N;
            double mn = *std::min_element(times.begin(), times.end());
            double mx = *std::max_element(times.begin(), times.end());

            std::string label = "Encode (" + std::to_string(text.size()) + " chars, " +
                                std::to_string(total_tokens / N) + " toks)";
            results.push_back({label, mean, mn, mx, N});
        }
    }

    // --- 3. Decode benchmarks ---
    {
        const int WARMUP = 50;
        const int N = 1000;

        for (size_t si = 0; si < test_strings.size(); ++si) {
            const auto& text = test_strings[si];
            auto ids = tok->encode(text);

            for (int i = 0; i < WARMUP; ++i) {
                auto s = tok->decode(ids);
                (void)s;
            }

            std::vector<double> times;
            for (int i = 0; i < N; ++i) {
                auto t0 = Clock::now();
                auto s = tok->decode(ids);
                auto t1 = Clock::now();
                double us = std::chrono::duration_cast<Us>(t1 - t0).count();
                times.push_back(us);
            }

            double mean = std::accumulate(times.begin(), times.end(), 0.0) / N;
            double mn = *std::min_element(times.begin(), times.end());
            double mx = *std::max_element(times.begin(), times.end());

            std::string label = "Decode (" + std::to_string(ids.size()) + " tokens)";
            results.push_back({label, mean, mn, mx, N});
        }
    }

    // --- 4. Throughput benchmark (batch encode) ---
    {
        // Encode all test strings repeatedly for 1 second, count tokens
        auto t0 = Clock::now();
        int64_t total_tokens = 0;
        int64_t total_calls = 0;
        while (true) {
            for (const auto& text : test_strings) {
                auto ids = tok->encode(text);
                total_tokens += static_cast<int64_t>(ids.size());
                ++total_calls;
            }
            auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
            if (elapsed >= 1000)
                break;
        }
        auto t1 = Clock::now();
        double elapsed_s = std::chrono::duration_cast<Us>(t1 - t0).count() / 1e6;
        double toks_per_sec = total_tokens / elapsed_s;

        std::cerr << "\nThroughput: " << std::fixed << std::setprecision(0) << toks_per_sec
                  << " tokens/sec"
                  << " (" << total_calls << " calls in " << std::setprecision(2) << elapsed_s
                  << "s)\n";
    }

    print_results(results);

    // --- 5. Round-trip correctness check ---
    // Note: with small built-in vocab, some strings can't round-trip (missing chars).
    // Only report as warning; use real vocab (TOKENIZER_JSON) for strict round-trip.
    {
        int pass = 0, warn = 0;
        for (const auto& text : test_strings) {
            auto ids = tok->encode(text);
            auto decoded = tok->decode(ids);
            if (decoded == text) {
                ++pass;
            } else {
                ++warn;
                if (json_path) {
                    // Real vocab: round-trip failure is an error
                    std::cerr << "ROUND-TRIP FAIL: '" << text.substr(0, 40) << "' -> '"
                              << decoded.substr(0, 40) << "'\n";
                }
            }
        }
        std::cerr << "\nRound-trip: " << pass << " pass, " << warn << " warn\n";
        if (warn > 0 && json_path)
            return 1; // Only fail with real vocab
    }

    std::cerr << "\nBenchmark complete.\n";
    return 0;
}
