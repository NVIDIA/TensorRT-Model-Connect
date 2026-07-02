/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-TOK-CPP-05
// Architecture:   ARCH-TOK-001
// Unit Design:    UD-TOK-01
// Intent:         BPE tokenizer correctness: array/string merge formats, pre-tokenizer configs,
// special tokens, round-trip Preconditions:  None (CPU-only, inline JSON vocab definitions)
// Postconditions: Encode produces correct token IDs, decode recovers original text, special tokens
// handled correctly
// =============================================================================

#include "trtmc/tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << std::endl;
        failures++;
    } else {
        std::cerr << "PASS: " << name << std::endl;
    }
}

// ─── Minimal tokenizer JSON with ARRAY merge format (NewlineAware style) ───
static const char* kArrayMergesJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3,
      "he": 4, "ll": 5, "hel": 6, "lo": 7,
      "hello": 8,
      "\u0120": 9,
      "w": 10, "r": 11, "d": 12,
      "or": 13, "ld": 14,
      "\u0120w": 15, "orld": 16,
      "\u0120world": 17,
      "hello\u0120world": 18
    },
    "merges": [
      ["h", "e"],
      ["l", "l"],
      ["l", "o"],
      ["he", "l"],
      ["hel", "lo"],
      ["o", "r"],
      ["l", "d"],
      ["or", "ld"],
      ["\u0120", "w"],
      ["\u0120w", "orld"],
      ["hello", "\u0120world"]
    ]
  },
  "added_tokens": [
    {"id": 19, "content": "<eos>", "special": true},
    {"id": 20, "content": "<pad>", "special": true}
  ]
})";

// ─── Same tokenizer but with STRING merge format (StringMerge / SentencePiece style) ───
static const char* kStringMergesJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3,
      "he": 4, "ll": 5, "hel": 6, "lo": 7,
      "hello": 8,
      "\u0120": 9,
      "w": 10, "r": 11, "d": 12,
      "or": 13, "ld": 14,
      "\u0120w": 15, "orld": 16,
      "\u0120world": 17,
      "hello\u0120world": 18
    },
    "merges": [
      "h e",
      "l l",
      "l o",
      "he l",
      "hel lo",
      "o r",
      "l d",
      "or ld",
      "\u0120 w",
      "\u0120w orld",
      "hello \u0120world"
    ]
  },
  "added_tokens": [
    {"id": 19, "content": "<eos>", "special": true},
    {"id": 20, "content": "<pad>", "special": true}
  ]
})";

// ─── CLIP-style BPE end-of-word suffix ───
static const char* kClipEndOfWordJson = R"({
  "model": {
    "type": "BPE",
    "end_of_word_suffix": "</w>",
    "vocab": {
      "e": 0, "a": 1, "r": 2,
      "ea": 3, "r</w>": 4, "ear</w>": 5,
      "<|startoftext|>": 10, "<|endoftext|>": 11
    },
    "merges": [
      "e a",
      "ea r</w>"
    ]
  },
  "added_tokens": [
    {"id": 10, "content": "<|startoftext|>", "special": true},
    {"id": 11, "content": "<|endoftext|>", "special": true}
  ],
  "post_processor": {
    "type": "RobertaProcessing",
    "sep": ["<|endoftext|>", 11],
    "cls": ["<|startoftext|>", 10],
    "trim_offsets": false,
    "add_prefix_space": false
  }
})";

// HF BPE tokenizers commonly serialize a present-but-null end_of_word_suffix.
static const char* kNullEndOfWordSuffixJson = R"({
  "model": {
    "type": "BPE",
    "end_of_word_suffix": null,
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3,
      "he": 4, "ll": 5
    },
    "merges": [
      "h e",
      "l l"
    ]
  },
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false
  }
})";

// ─── StringMerge style pre_tokenizer config (ByteLevel) ───
static const char* kStringMergeStyleJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3,
      "he": 4, "ll": 5, "hel": 6, "lo": 7,
      "hello": 8,
      "\u0120": 9,
      "w": 10, "r": 11, "d": 12,
      "or": 13, "ld": 14,
      "\u0120w": 15, "orld": 16,
      "\u0120world": 17,
      "hello\u0120world": 18
    },
    "merges": [
      "h e",
      "l l",
      "l o",
      "he l",
      "hel lo",
      "o r",
      "l d",
      "or ld",
      "\u0120 w",
      "\u0120w orld",
      "hello \u0120world"
    ]
  },
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true
  },
  "added_tokens": [
    {"id": 19, "content": "<|endoftext|>", "special": true}
  ]
})";

// ─── NewlineAware style pre_tokenizer config (Sequence + Split + ByteLevel) ───
static const char* kNewlineAwareStyleJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3,
      "he": 4, "ll": 5, "hel": 6, "lo": 7,
      "hello": 8,
      "\u0120": 9,
      "w": 10, "r": 11, "d": 12,
      "or": 13, "ld": 14,
      "\u0120w": 15, "orld": 16,
      "\u0120world": 17,
      "hello\u0120world": 18
    },
    "merges": [
      ["h", "e"],
      ["l", "l"],
      ["l", "o"],
      ["he", "l"],
      ["hel", "lo"],
      ["o", "r"],
      ["l", "d"],
      ["or", "ld"],
      ["\u0120", "w"],
      ["\u0120w", "orld"],
      ["hello", "\u0120world"]
    ]
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {"Regex": "[^\u000d\u000a]?test"},
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false
      }
    ]
  },
  "added_tokens": [
    {"id": 19, "content": "</s>", "special": true}
  ]
})";

// ─── Pre-tokenizer boundary test ───
// Vocab includes cross-boundary merges that should NOT fire due to pre-tokenization.
// Within-boundary merges (h+e, '+t, '+s, '+m) should fire normally.
// Cross-boundary merges (n+', o+SPACE, c+1, o+!) should NOT fire because
// pre-tokenization places them in separate chunks.
//
// Token IDs:
//   a=0  b=1  c=2  d=3  e=4  h=5  l=6  m=7  n=8  o=9
//   s=10 t=11 w=12 r=13
//   1=14 2=15 3=16
//   '=17 !=18 SPACE(U+0120)=19
//   he=20 (within-boundary)
//   't=21  's=22  'm=23  (within-contraction)
//   n'=24  oSPACE=25  c1=26  o!=27  (cross-boundary, should NOT fire)
static const char* kBoundaryTestJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "a": 0, "b": 1, "c": 2, "d": 3, "e": 4,
      "h": 5, "l": 6, "m": 7, "n": 8, "o": 9,
      "s": 10, "t": 11, "w": 12, "r": 13,
      "1": 14, "2": 15, "3": 16,
      "'": 17, "!": 18,
      "\u0120": 19,
      "he": 20,
      "'t": 21, "'s": 22, "'m": 23,
      "n'": 24, "o\u0120": 25, "c1": 26, "o!": 27
    },
    "merges": [
      "h e",
      "' t",
      "' s",
      "' m",
      "n '",
      "o \u0120",
      "c 1",
      "o !"
    ]
  },
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false
  }
})";

// ─── SentencePiece style: special tokens with <|end_of_text|> ───
static const char* kSpecialTokenStyleJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3,
      "he": 4, "ll": 5, "hel": 6, "lo": 7,
      "hello": 8
    },
    "merges": [
      "h e",
      "l l",
      "l o",
      "he l",
      "hel lo"
    ]
  },
  "added_tokens": [
    {"id": 9, "content": "<|begin_of_text|>", "special": true},
    {"id": 10, "content": "<|end_of_text|>", "special": true},
    {"id": 11, "content": "<|eot_id|>", "special": true}
  ]
})";

int main() {
    std::cerr << "Running BPE tokenizer tests...\n\n";
    std::string json(kArrayMergesJson);

    // === 1. Factory / JSON parsing ===
    {
        std::cerr << "=== Factory & JSON Parsing ===\n";

        auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);
        check(tok != nullptr, "create_from_valid_json");

        // Invalid JSON
        bool threw = false;
        try {
            trtmc::CreateBpeTokenizer("not json", 8, false);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "reject_invalid_json");

        // Not BPE type
        threw = false;
        std::string not_bpe = R"({"model":{"type":"WordPiece","vocab":{},"merges":[]}})";
        try {
            trtmc::CreateBpeTokenizer(not_bpe.data(), not_bpe.size(), false);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "reject_non_bpe_type");
    }

    // === 2. id_for_token / token_for_id ===
    {
        std::cerr << "\n=== Token/ID Lookup ===\n";

        auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

        check(tok->id_for_token("h") == 0, "id_for_token_h");
        check(tok->id_for_token("hello") == 8, "id_for_token_hello");
        check(tok->id_for_token("nonexistent") == -1, "id_for_token_missing");

        check(tok->token_for_id(0) == "h", "token_for_id_0");
        check(tok->token_for_id(8) == "hello", "token_for_id_8");
        check(tok->token_for_id(-1).empty(), "token_for_id_negative");
        check(tok->token_for_id(9999).empty(), "token_for_id_out_of_range");
    }

    // === 3. Added tokens (special + beyond base vocab) ===
    {
        std::cerr << "\n=== Added Tokens ===\n";

        auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

        check(tok->id_for_token("<eos>") == 19, "added_token_eos");
        check(tok->id_for_token("<pad>") == 20, "added_token_pad");
        check(tok->token_for_id(19) == "<eos>", "added_token_reverse_eos");
        check(tok->token_for_id(20) == "<pad>", "added_token_reverse_pad");
    }

    // === 4. Encode ===
    {
        std::cerr << "\n=== Encode ===\n";

        auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

        // "hello" → h,e,l,l,o → he,l,l,o → he,ll,o → [he(4), ll(5), o(3)]
        auto ids = tok->encode("hello");
        check(!ids.empty(), "encode_hello_nonempty");
        check(ids.size() == 3 && ids[0] == 4 && ids[1] == 5 && ids[2] == 3, "encode_hello_ids");

        auto empty_ids = tok->encode("");
        check(empty_ids.empty(), "encode_empty");

        auto ids2 = tok->encode("hello");
        check(ids == ids2, "encode_deterministic");
    }

    // === 5. Decode ===
    {
        std::cerr << "\n=== Decode ===\n";

        auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

        check(tok->decode({}).empty(), "decode_empty");
        check(tok->decode({8}) == "hello", "decode_hello");
        check(tok->decode({18}) == "hello world", "decode_hello_world");

        // Special tokens should be filtered
        check(tok->decode({19}) == "", "decode_filters_eos");
        check(tok->decode({8, 19}) == "hello", "decode_filters_eos_in_sequence");
        check(tok->decode({19, 20}) == "", "decode_filters_all_special");
    }

    // === 6. Round-trip ===
    {
        std::cerr << "\n=== Round-trip ===\n";

        auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

        auto rt = [&](const std::string& text, const std::string& label) {
            auto decoded = tok->decode(tok->encode(text));
            check(decoded == text, label + " ('" + text + "' -> '" + decoded + "')");
        };

        rt("hello", "roundtrip_hello");
        rt("hello world", "roundtrip_hello_world");
        rt("", "roundtrip_empty");
    }

    // === 7. add_special_tokens flag does NOT append EOS ===
    {
        std::cerr << "\n=== add_special_tokens behavior ===\n";

        auto tok_no = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);
        auto tok_yes = trtmc::CreateBpeTokenizer(json.data(), json.size(), true);

        auto ids_no = tok_no->encode("hello");
        auto ids_yes = tok_yes->encode("hello");
        check(ids_no == ids_yes, "add_special_tokens_no_effect");
    }

    // === 7b. CLIP end-of-word suffix ===
    {
        std::cerr << "\n=== CLIP End-of-Word Suffix ===\n";

        std::string clip_json(kClipEndOfWordJson);
        auto tok = trtmc::CreateBpeTokenizer(clip_json.data(), clip_json.size(), true);
        auto ids = tok->encode("ear");
        check(ids.size() == 3 && ids[0] == 10 && ids[1] == 5 && ids[2] == 11,
              "clip_end_of_word_suffix_encode");
    }

    // === 7c. Null end-of-word suffix ===
    {
        std::cerr << "\n=== Null End-of-Word Suffix ===\n";

        std::string null_suffix_json(kNullEndOfWordSuffixJson);
        auto tok =
            trtmc::CreateBpeTokenizer(null_suffix_json.data(), null_suffix_json.size(), false);
        check(tok != nullptr, "null_end_of_word_suffix_create");
        auto ids = tok->encode("hello");
        check(!ids.empty(), "null_end_of_word_suffix_encode");
    }

    // === 8. STRING merge format (StringMerge / SentencePiece style) ===
    {
        std::cerr << "\n=== String Merge Format ===\n";

        std::string sj(kStringMergesJson);
        auto tok = trtmc::CreateBpeTokenizer(sj.data(), sj.size(), false);
        check(tok != nullptr, "string_merges_create");

        // Should produce identical encoding as array format
        auto ids = tok->encode("hello");
        check(ids.size() == 3 && ids[0] == 4 && ids[1] == 5 && ids[2] == 3,
              "string_merges_encode_hello");

        // Round-trip
        auto decoded = tok->decode(tok->encode("hello world"));
        check(decoded == "hello world", "string_merges_roundtrip");
    }

    // === 9. StringMerge pre_tokenizer config (ByteLevel) ===
    {
        std::cerr << "\n=== StringMerge Pre-tokenizer Config ===\n";

        std::string gj(kStringMergeStyleJson);
        auto tok = trtmc::CreateBpeTokenizer(gj.data(), gj.size(), false);
        check(tok != nullptr, "string_merge_style_create");

        auto ids = tok->encode("hello");
        check(!ids.empty(), "string_merge_style_encode");

        // <|endoftext|> should be filtered in decode
        check(tok->id_for_token("<|endoftext|>") == 19, "string_merge_eos_token");
        check(tok->decode({19}).empty(), "string_merge_decode_filters_eos");
    }

    // === 10. NewlineAware pre_tokenizer config (Sequence) ===
    {
        std::cerr << "\n=== NewlineAware Pre-tokenizer Config ===\n";

        std::string qj(kNewlineAwareStyleJson);
        auto tok = trtmc::CreateBpeTokenizer(qj.data(), qj.size(), false);
        check(tok != nullptr, "newline_aware_style_create");

        auto ids = tok->encode("hello");
        check(!ids.empty(), "newline_aware_style_encode");

        // </s> should be recognized as special
        check(tok->id_for_token("</s>") == 19, "newline_aware_eos_token");
        check(tok->decode({19}).empty(), "newline_aware_decode_filters_eos");
    }

    // === 11. SentencePiece style special tokens ===
    {
        std::cerr << "\n=== SentencePiece Style Special Tokens ===\n";

        std::string lj(kSpecialTokenStyleJson);
        auto tok = trtmc::CreateBpeTokenizer(lj.data(), lj.size(), false);
        check(tok != nullptr, "special_token_style_create");

        check(tok->id_for_token("<|begin_of_text|>") == 9, "special_token_bos_token");
        check(tok->id_for_token("<|end_of_text|>") == 10, "special_token_eos_token");
        check(tok->id_for_token("<|eot_id|>") == 11, "special_token_eot_token");

        // All special tokens filtered in decode
        check(tok->decode({9, 10, 11}).empty(), "special_token_decode_filters_all_special");

        auto ids = tok->encode("hello");
        check(!ids.empty(), "special_token_encode_hello");
    }

    // === 12. Merge-all optimization: multiple same pairs merged in one pass ===
    {
        std::cerr << "\n=== Merge-All Optimization ===\n";

        // Construct a tokenizer where "l"+"l" merge appears twice in input "llll"
        // After one pass: "ll","ll" (both merged), then "ll"+"ll" if that merge exists
        std::string opt_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "l": 0, "ll": 1, "llll": 2
            },
            "merges": [
              "l l",
              "ll ll"
            ]
          },
          "added_tokens": []
        })";

        auto tok = trtmc::CreateBpeTokenizer(opt_json.data(), opt_json.size(), false);
        check(tok != nullptr, "merge_all_create");

        // "llll" → l,l,l,l → (merge l+l all) → ll,ll → (merge ll+ll) → llll
        auto ids = tok->encode("llll");
        check(ids.size() == 1 && ids[0] == 2, "merge_all_llll");
    }

    // === 13. Pre-tokenizer: contraction splitting ===
    //
    // Verifies that the hand-written StringMerge pre-tokenizer correctly splits
    // contractions at the apostrophe boundary, preventing cross-boundary
    // merges from firing.
    {
        std::cerr << "\n=== Pre-tokenizer: Contraction Splitting ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "don't" → pre-tokenize ["don", "'t"]
        //   "don": d,o,n → no applicable merges → [3,9,8]
        //   "'t": ',t → merge '+t (rank 1) → 't → [21]
        //   Total: [3,9,8,21]
        // If pre-tokenization FAILS: d,o,n,',t → n+' (rank 4) fires → [3,9,24,11]
        {
            auto ids = tok->encode("don't");
            check(ids.size() == 4, "contraction_dont_size");
            check(ids.size() == 4 && ids[0] == 3 && ids[1] == 9 && ids[2] == 8 && ids[3] == 21,
                  "contraction_dont_ids");
            // Verify the cross-boundary merge (n'=24) did NOT fire
            bool has_cross = false;
            for (auto id : ids)
                if (id == 24)
                    has_cross = true;
            check(!has_cross, "contraction_dont_no_cross_merge");
        }

        // "he's" → pre-tokenize ["he", "'s"]
        //   "he": h,e → merge h+e → he → [20]
        //   "'s": ',s → merge '+s → 's → [22]
        //   Total: [20,22]
        {
            auto ids = tok->encode("he's");
            check(ids.size() == 2 && ids[0] == 20 && ids[1] == 22, "contraction_hes_ids");
        }

        // "he'm" → pre-tokenize ["he", "'m"]
        //   "he": h,e → he → [20]
        //   "'m": ',m → 'm → [23]
        //   Total: [20,23]
        {
            auto ids = tok->encode("he'm");
            check(ids.size() == 2 && ids[0] == 20 && ids[1] == 23, "contraction_hem_ids");
        }
    }

    // === 14. Pre-tokenizer: word boundary (space + letter) ===
    {
        std::cerr << "\n=== Pre-tokenizer: Word Boundary ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "o b" → pre-tokenize ["o", " b"]
        //   "o": o → [9]
        //   " b": SPACE,b → [19,1]
        //   Total: [9,19,1]
        // If pre-tokenization FAILS: o,SPACE,b → o+SPACE (rank 5) fires → [25,1]
        {
            auto ids = tok->encode("o b");
            check(ids.size() == 3 && ids[0] == 9 && ids[1] == 19 && ids[2] == 1,
                  "word_boundary_o_b");
            bool has_cross = false;
            for (auto id : ids)
                if (id == 25)
                    has_cross = true;
            check(!has_cross, "word_boundary_no_cross_merge");
        }
    }

    // === 15. Pre-tokenizer: digit boundary ===
    {
        std::cerr << "\n=== Pre-tokenizer: Digit Boundary ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "c123" → pre-tokenize ["c", "123"]
        //   "c": c → [2]
        //   "123": 1,2,3 → [14,15,16]
        //   Total: [2,14,15,16]
        // If pre-tokenization FAILS: c,1,2,3 → c+1 (rank 6) fires → [26,15,16]
        {
            auto ids = tok->encode("c123");
            check(ids.size() == 4 && ids[0] == 2 && ids[1] == 14 && ids[2] == 15 && ids[3] == 16,
                  "digit_boundary_c123");
            bool has_cross = false;
            for (auto id : ids)
                if (id == 26)
                    has_cross = true;
            check(!has_cross, "digit_boundary_no_cross_merge");
        }
    }

    // === 16. Pre-tokenizer: punctuation boundary ===
    {
        std::cerr << "\n=== Pre-tokenizer: Punctuation Boundary ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "o!" → pre-tokenize ["o", "!"]
        //   "o": o → [9]
        //   "!": ! → [18]
        //   Total: [9,18]
        // If pre-tokenization FAILS: o,! → o+! (rank 7) fires → [27]
        {
            auto ids = tok->encode("o!");
            check(ids.size() == 2 && ids[0] == 9 && ids[1] == 18, "punct_boundary_o_bang");
            bool has_cross = false;
            for (auto id : ids)
                if (id == 27)
                    has_cross = true;
            check(!has_cross, "punct_boundary_no_cross_merge");
        }
    }

    // === 17. Pre-tokenizer: whitespace runs ===
    {
        std::cerr << "\n=== Pre-tokenizer: Whitespace Runs ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "a  b" → pre-tokenize ["a", " ", " b"]
        //   StringMerge regex: \s+(?!\S) leaves last space for next token
        //   "a": a → [0]
        //   " ": SPACE → [19]
        //   " b": SPACE,b → [19,1]
        //   Total: [0,19,19,1]
        {
            auto ids = tok->encode("a  b");
            check(ids.size() == 4 && ids[0] == 0 && ids[1] == 19 && ids[2] == 19 && ids[3] == 1,
                  "whitespace_double_space");
        }

        // " a" (leading space + letter) → pre-tokenize [" a"]
        //   " a": SPACE,a → [19,0]
        {
            auto ids = tok->encode(" a");
            check(ids.size() == 2 && ids[0] == 19 && ids[1] == 0, "whitespace_leading_space");
        }
    }

    // === 18. Pre-tokenizer: within-boundary merges still fire ===
    {
        std::cerr << "\n=== Pre-tokenizer: Within-Boundary Merges ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "hello" → pre-tokenize ["hello"]
        //   byte: h,e,l,l,o → merge h+e → he,l,l,o → [20,6,6,9]
        {
            auto ids = tok->encode("hello");
            check(ids.size() == 4 && ids[0] == 20 && ids[1] == 6 && ids[2] == 6 && ids[3] == 9,
                  "within_boundary_hello");
        }

        // Round-trip for contraction
        {
            auto decoded = tok->decode(tok->encode("don't"));
            check(decoded == "don't", "roundtrip_dont");
        }
    }

    // === 19. word-separator pre-tokenizer: punctuation splitting ===
    {
        std::cerr << "\n=== word-separator Pre-tokenizer ===\n";

        // word-separator splits on punctuation: . , ! ? and CJK punct
        // Vocab: letters + Ġ(space) + punctuation
        std::string word_separator_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6,
              "\u0120": 7, "!": 8, ".": 9, ",": 10,
              "he": 11, "ll": 12, "lo": 13, "hell": 19, "hel": 20,
              "\u0120w": 14, "or": 15, "ld": 16, "orld": 21,
              "hello": 17, "\u0120world": 18
            },
            "merges": [
              "h e", "l l", "l o", "he ll", "hel lo",
              "\u0120 w", "o r", "l d", "or ld",
              "\u0120w orld", "hello \u0120world"
            ]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {
                "type": "Split",
                "pattern": {"Regex": " ?[^(\\s|[.,!?])]+" },
                "behavior": "Isolated",
                "invert": false
              },
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(word_separator_json.data(), word_separator_json.size(),
                                             false);
        check(tok != nullptr, "word_separator_create");

        // "hello!" → word-separator splits: ["hello", "!"]
        // "hello" → h,e,l,l,o → he(0),l,l,o → he,ll(1),o → hell(3),o
        //   Result: [19,3] = [hell,o]
        // "!" → [8]
        // Total: [19,3,8]
        {
            auto ids = tok->encode("hello!");
            check(ids.size() == 3 && ids[0] == 19 && ids[1] == 3 && ids[2] == 8,
                  "word_separator_punct_split_bang");
        }

        // "hello.world" → word-separator splits: ["hello", ".", "world"]
        // "hello" → [19,3] = [hell,o]
        // "." → [9]
        // "world" → w,o,r,l,d → [4,3,5,2,6] (no merges apply among these)
        //   Actually: o+r at rank 6? No, the merge list doesn't have "o r" for word-separator.
        //   Wait, merges include "\u0120 w", "o r", "l d", "or ld", "\u0120w orld"
        //   So o+r → or(15), l+d → ld(16), or+ld → orld(13+?). Wait, orld is not in vocab.
        //   Actually "\u0120w orld" merge: but "orld" itself... let me check vocab:
        //   No "orld" token in the word-separator test vocab. So: w,o,r,l,d → o+r→or, l+d→ld →
        //   w,or,ld "or ld" merge (rank 8) → w,orld. But "orld" not in vocab. So result: w,or,ld →
        //   [4,15,16]
        {
            auto ids = tok->encode("hello.world");
            // hello=[19,3], .=[9], world=w,o,r,l,d→w,or,ld→w,orld=[4,21]
            check(ids.size() == 5, "word_separator_punct_split_dot_size");
            check(ids[0] == 19 && ids[1] == 3, "word_separator_punct_split_dot_hello");
            check(ids[2] == 9, "word_separator_punct_split_dot_period");
        }

        // Round-trip: "hell" + "o" → byte_decode → "hello"
        {
            auto decoded = tok->decode(tok->encode("hello"));
            check(decoded == "hello", "word_separator_roundtrip");
        }
    }

    // === 20. Metaspace pre-tokenizer (metaspace style) ===
    {
        std::cerr << "\n=== Metaspace Pre-tokenizer ===\n";

        // Metaspace: raw Unicode BPE, spaces dropped (not in vocab),
        // adjacent chars merge across word boundaries.
        // Vocab uses Ġ (U+0120) as word separator in multi-word tokens.
        std::string meta_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6,
              "\u0120": 7,
              "he": 8, "ll": 9, "lo": 10,
              "hel": 11, "hello": 12,
              "ow": 13, "or": 14, "ld": 15, "orld": 16
            },
            "merges": [
              "h e", "l l", "l o", "he l", "hel lo",
              "o w", "o r", "l d", "or ld"
            ]
          },
          "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "always",
            "split": false
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(meta_json.data(), meta_json.size(), false);
        check(tok != nullptr, "metaspace_create");

        // "hello" → chars: h,e,l,l,o
        //   rank 0: h+e→he, rank 1: l+l→ll. Then no more merges apply
        //   (l+o at rank 2 can't fire because ll is already merged)
        //   Result: [8,9,3] = [he,ll,o]
        {
            auto ids = tok->encode("hello");
            check(ids.size() == 3 && ids[0] == 8 && ids[1] == 9 && ids[2] == 3, "metaspace_hello");
        }

        // "hello world" → space dropped → h,e,l,l,o,w,o,r,l,d
        //   → he,ll,o,w,or,ld → hel,lo,w,orld → hello,w,orld
        //   But o+w merge (rank 5) fires before hel+lo (rank 4)?
        //   Rank order: h+e(0), l+l(1), l+o(2), he+l(3), hel+lo(4), o+w(5)
        //   So hel+lo (rank 4) fires before o+w (rank 5).
        //   h,e,l,l,o,w,o,r,l,d
        //   → he(0),l,l,o,w,o,r,l,d
        //   → he,ll(1),o,w,o,r,l,d
        //   → he,ll,o,w,or(6),l,d  -- wait, o+r is rank 6
        //   Actually let me re-order: each pass finds the BEST (lowest rank) merge
        //   Pass 1: h+e(0) → he,l,l,o,w,o,r,l,d
        //   Pass 2: l+l(1) → he,ll,o,w,o,r,l,d
        //   Pass 3: l+o(2) → but "ll" and "o" → ll+o not a merge. Check all pairs:
        //     (he,ll)→no, (ll,o)→no, (o,w)→rank5, (w,o)→no, (o,r)→rank6, (r,l)→no, (l,d)→rank7
        //     Best: o+w at rank 5 → he,ll,ow,o,r,l,d
        //   Pass 4: (he,ll)no, (ll,ow)no, (ow,o)no, (o,r)rank6, (r,l)no, (l,d)rank7
        //     Best: o+r at rank 6 → he,ll,ow,or,l,d
        //   Pass 5: l+d at rank 7 → he,ll,ow,or,ld
        //   Pass 6: or+ld at rank 7... wait, or+ld merge? Let me check.
        //   Merges: "o r"(5→rank6?), no. Let me recount:
        //   merges = [h+e(0), l+l(1), l+o(2), he+l(3), hel+lo(4), o+w(5), o+r(6), l+d(7), or+ld(8)]
        //   Wait, I only listed 8 merges but the merge list has 8 entries (index 0-7).
        //   "or ld" would be rank 7 (last merge).
        //   So: or+ld is rank 7.
        //   Continuing: he,ll,ow,or,ld → or+ld(7) → he,ll,ow,orld
        //   No more merges possible. Result: [8,9,13,16] = [he,ll,ow,orld]
        {
            auto ids = tok->encode("hello world");
            check(ids.size() == 4 && ids[0] == 8 && ids[1] == 9 && ids[2] == 13 && ids[3] == 16,
                  "metaspace_hello_world");
        }

        // Decode: Ġ → space, strip leading space
        {
            // Token 7 is Ġ. decode([0, 7, 4]) = "h" + " " + "w" → "h w"
            auto decoded = tok->decode({0, 7, 4});
            check(decoded == "h w", "metaspace_decode_space");
        }

        // Decode: strip leading Ġ
        {
            // decode([7, 0]) = " " + "h" → strip leading space → "h"
            auto decoded = tok->decode({7, 0});
            check(decoded == "h", "metaspace_decode_strip_leading");
        }

        // Round-trip for single word (no spaces involved)
        {
            auto decoded = tok->decode(tok->encode("hello"));
            check(decoded == "hello", "metaspace_roundtrip_hello");
        }
    }

    // === 21. Added tokens pre-splitting ===
    {
        std::cerr << "\n=== Added Tokens Pre-splitting ===\n";

        // Non-special added tokens are matched BEFORE pre-tokenization.
        // Token "  " (double space, id=20) should be matched as a single token,
        // not split by the pre-tokenizer.
        std::string added_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "h": 0, "e": 1, "l": 2, "o": 3,
              "\u0120": 4, "w": 5, "r": 6, "d": 7,
              "he": 8, "ll": 9,
              "\u0120w": 10, "or": 11, "ld": 12,
              "orld": 13
            },
            "merges": [
              "h e", "l l",
              "\u0120 w", "o r", "l d",
              "or ld", "\u0120w orld"
            ]
          },
          "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false
          },
          "added_tokens": [
            {"id": 14, "content": "<eos>", "special": true},
            {"id": 15, "content": "    ", "special": false},
            {"id": 16, "content": "  ", "special": false}
          ]
        })";

        auto tok = trtmc::CreateBpeTokenizer(added_json.data(), added_json.size(), false);
        check(tok != nullptr, "added_tokens_create");

        // "he  ll" → added token "  " (id=16) splits the text:
        //   ["he", "  "(added=16), "ll"]
        //   "he" → h,e → merge h+e → he → [8]
        //   "  " → added token → [16]
        //   "ll" → l,l → merge l+l → ll → [9]
        //   Total: [8, 16, 9]
        {
            auto ids = tok->encode("he  ll");
            check(ids.size() == 3 && ids[0] == 8 && ids[1] == 16 && ids[2] == 9,
                  "added_tokens_double_space");
        }

        // "he    ll" → added token "    " (id=15, 4 spaces, longer) matched first
        //   ["he", "    "(added=15), "ll"]
        //   Total: [8, 15, 9]
        {
            auto ids = tok->encode("he    ll");
            check(ids.size() == 3 && ids[0] == 8 && ids[1] == 15 && ids[2] == 9,
                  "added_tokens_quad_space");
        }

        // "hello" → no added tokens match, normal BPE
        {
            auto ids = tok->encode("hello");
            check(ids.size() >= 2 && ids[0] == 8 && ids[1] == 9, "added_tokens_no_match");
        }

        // Special added tokens ARE matched during encoding (matching HF behavior).
        // The text "<eos>" should map directly to token 14.
        // The special flag only controls decode filtering, not encode matching.
        {
            auto ids = tok->encode("<eos>");
            check(ids.size() == 1 && ids[0] == 14, "added_tokens_special_matched_in_encode");
        }
    }

    // === 22. Unsupported pre-tokenizer throws ===
    {
        std::cerr << "\n=== Unsupported Pre-tokenizer Throw ===\n";

        // Unknown pre_tokenizer type should throw
        std::string unknown_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {"a": 0},
            "merges": []
          },
          "pre_tokenizer": {
            "type": "WhitespaceSplit"
          }
        })";

        bool threw = false;
        try {
            trtmc::CreateBpeTokenizer(unknown_json.data(), unknown_json.size(), false);
        } catch (const std::exception& e) {
            threw = true;
            std::string msg = e.what();
            check(msg.find("Unsupported") != std::string::npos, "unsupported_pretok_error_message");
        }
        check(threw, "unsupported_pretok_throws");

        // Sequence with unknown Split regex should also throw
        std::string unknown_regex_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {"a": 0},
            "merges": []
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": "unknown_pattern"}, "behavior": "Isolated", "invert": false}
            ]
          }
        })";

        // This should default to StringMerge, not throw (unrecognized regex → fallback)
        auto tok =
            trtmc::CreateBpeTokenizer(unknown_regex_json.data(), unknown_regex_json.size(), false);
        check(tok != nullptr, "unknown_regex_falls_back_to_string_merge");
    }

    // === 23. Metaspace decode edge cases ===
    {
        std::cerr << "\n=== Metaspace Decode Edge Cases ===\n";

        std::string meta_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "a": 0, "b": 1, "\u0120": 2, "\u0120a": 3, "\u0120b": 4
            },
            "merges": ["\u0120 a", "\u0120 b"]
          },
          "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "always",
            "split": false
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(meta_json.data(), meta_json.size(), false);

        // decode empty
        check(tok->decode({}).empty(), "metaspace_decode_empty");

        // decode single char
        check(tok->decode({0}) == "a", "metaspace_decode_single");

        // decode with Ġ tokens → space
        check(tok->decode({3, 4}) == "a b", "metaspace_decode_multi_words");

        // decode with leading Ġ → strip
        check(tok->decode({2, 0}) == "a", "metaspace_decode_leading_space_strip");
    }

    // === 23b. Metaspace decode with SentencePiece marker ===
    {
        std::cerr << "\n=== Metaspace Decode SentencePiece Marker ===\n";

        std::string meta_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "\u2581Hello": 0,
              "\u2581there": 1,
              ",": 2,
              "\u2581how": 3,
              "\u2581is": 4,
              "\u2581the": 5,
              "\u2581weather": 6,
              "\u2581today": 7,
              "?": 8
            },
            "merges": []
          },
          "decoder": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "always"
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(meta_json.data(), meta_json.size(), false);
        check(tok != nullptr, "metaspace_decode_sentencepiece_create");
        check(tok->decode({0, 1, 2, 3, 4, 5, 6, 7, 8}) == "Hello there, how is the weather today?",
              "metaspace_decode_sentencepiece_transcript");
    }

    // === 24. Missing vocab/merges → throw ===
    {
        std::cerr << "\n=== Missing Vocab/Merges ===\n";

        std::string no_vocab = R"({"model":{"type":"BPE","merges":[]}})";
        bool threw = false;
        try {
            trtmc::CreateBpeTokenizer(no_vocab.data(), no_vocab.size(), false);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "missing_vocab_throws");

        std::string no_merges = R"({"model":{"type":"BPE","vocab":{"a":0}}})";
        threw = false;
        try {
            trtmc::CreateBpeTokenizer(no_merges.data(), no_merges.size(), false);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "missing_merges_throws");
    }

    // === 25. Null pre_tokenizer ===
    {
        std::cerr << "\n=== Null Pre-tokenizer ===\n";

        // pre_tokenizer: null → fallback (return text as single chunk)
        std::string null_pt_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {"h":0,"e":1,"l":2,"o":3,"he":4},
            "merges": ["h e"]
          },
          "pre_tokenizer": null
        })";

        auto tok = trtmc::CreateBpeTokenizer(null_pt_json.data(), null_pt_json.size(), false);
        check(tok != nullptr, "null_pretok_create");
        // "he" → entire text as one chunk → byte_encode → h,e → merge → he → [4]
        auto ids = tok->encode("he");
        check(ids.size() == 1 && ids[0] == 4, "null_pretok_encode");
    }

    // === 26. Contractions 're 've 'll ===
    {
        std::cerr << "\n=== Contractions re/ve/ll ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "we're" → pre-tokenize ["we", "'re"]
        //   "we" → w,e → [12,4]
        //   "'re" → ',r,e → [17,13,4] (no merge for these)
        {
            auto ids = tok->encode("we're");
            // The key test: 're is a contraction, split correctly
            check(ids.size() >= 3, "contraction_re_size");
            // First part "we" should not merge with "'re"
            check(ids[0] == 12 && ids[1] == 4, "contraction_re_we");
        }

        // "we've" → pre-tokenize ["we", "'ve"]
        {
            auto ids = tok->encode("we've");
            check(ids.size() >= 3, "contraction_ve_size");
            check(ids[0] == 12 && ids[1] == 4, "contraction_ve_we");
        }

        // "we'll" → pre-tokenize ["we", "'ll"]
        {
            auto ids = tok->encode("we'll");
            check(ids.size() >= 3, "contraction_ll_size");
            check(ids[0] == 12 && ids[1] == 4, "contraction_ll_we");
        }
    }

    // === 27. Non-contraction apostrophe ===
    {
        std::cerr << "\n=== Non-contraction Apostrophe ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "a'b" → pre-tokenize: "a" (letter), "'" (other, not contraction), "b" (letter)
        // The apostrophe is NOT followed by s/t/m/d/re/ve/ll → treated as punctuation
        {
            auto ids = tok->encode("a'b");
            // Should have at least 3 tokens: a, ', b
            check(ids.size() == 3, "non_contraction_apost_size");
            check(ids[0] == 0, "non_contraction_apost_a");      // a=0
            check(ids[1] == 17, "non_contraction_apost_quote"); // '=17
            check(ids[2] == 1, "non_contraction_apost_b");      // b=1
        }

        // "'" alone → single punct token
        {
            auto ids = tok->encode("'");
            check(ids.size() == 1 && ids[0] == 17, "non_contraction_apost_alone");
        }
    }

    // === 28. NewlineAware trailing newlines after "other" chars ===
    {
        std::cerr << "\n=== NewlineAware Trailing Newlines ===\n";

        // NewlineAware: ` ?[^\s\p{L}\p{N}]+[\r\n]*` — other chars can absorb trailing newlines
        // \n byte (0x0A) is non-direct in StringMerge byte encoding → maps to U+010A = Ċ
        std::string q_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "a":0, "!":1, "\u0120":2, "\u010a":3, "!\u010a":4
            },
            "merges": ["! \u010a"]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": "[^\u000d\u000a]?test"}, "behavior": "Removed", "invert": true},
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(q_json.data(), q_json.size(), false);
        check(tok != nullptr, "newline_aware_trailing_nl_create");

        // "!\n" with NewlineAware: pre-tokenize as one chunk "!\n" (! is other, \n absorbed)
        // Then byte-encode: "!" → "!" (direct), "\n" → Ċ (U+010A)
        // Merge "!" + "Ċ" → "!Ċ" → token 4
        // BUT: pre-tokenizer needs to detect NewlineAware variant. The regex
        // "[^\r\n]?test" contains literal CR/LF bytes which triggers NewlineAware detection.
        {
            auto ids = tok->encode("!\n");
            // NewlineAware must keep punctuation + trailing newline in one chunk,
            // then allow BPE to merge "!" + "Ċ" -> "!Ċ".
            check(ids.size() == 1 && ids[0] == 4, "newline_aware_trailing_nl_merged");
        }

        // Double newline after punctuation must also stay in the same chunk.
        std::string q_double_nl_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              ".": 0, "\u010a": 1, "\u010a\u010a": 2, ".\u010a\u010a": 3
            },
            "merges": [
              "\u010a \u010a",
              ". \u010a\u010a"
            ]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {
                "type": "Split",
                "pattern": {"Regex": "[^\u000d\u000a\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                "behavior": "Removed",
                "invert": true
              },
              {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
              }
            ]
          }
        })";
        auto tok_double =
            trtmc::CreateBpeTokenizer(q_double_nl_json.data(), q_double_nl_json.size(), false);
        check(tok_double != nullptr, "newline_aware_double_nl_create");
        {
            auto ids = tok_double->encode(".\n\n");
            check(ids.size() == 1 && ids[0] == 3, "newline_aware_double_nl_merged");
        }
    }

    // === 29. Digit run without prefix ===
    {
        std::cerr << "\n=== Digit Run Without Prefix ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "123" (no leading space) → pre-tokenize ["123"] (StringMerge: \p{N}+)
        //   byte-encode: 1,2,3 → [14,15,16]
        {
            auto ids = tok->encode("123");
            check(ids.size() == 3 && ids[0] == 14 && ids[1] == 15 && ids[2] == 16,
                  "digit_run_no_prefix_string_merge");
        }

        // NewlineAware single digit: "123" → pre-tokenize ["1", "2", "3"]
        std::string q_digit_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {"1":0, "2":1, "3":2, "12":3},
            "merges": ["1 2"]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": "[^\u000d\u000a\\p{L}\\p{N}]?\\p{L}+"}, "behavior": "Removed", "invert": true},
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          }
        })";
        auto tok_q = trtmc::CreateBpeTokenizer(q_digit_json.data(), q_digit_json.size(), false);
        {
            auto ids = tok_q->encode("123");
            // NewlineAware: each digit is a separate pre-tokenized chunk → "1","2","3"
            // Within each chunk: single char → no merge possible across chunks
            // So "1"→[0], "2"→[1], "3"→[2], total size=3
            // (The merge "1 2" can't fire because they're in separate chunks)
            check(ids.size() == 3 && ids[0] == 0 && ids[1] == 1 && ids[2] == 2,
                  "digit_run_no_prefix_newline_aware");
        }
    }

    // === 30. word-separator whitespace handling ===
    {
        std::cerr << "\n=== word-separator Whitespace ===\n";

        // Reuse word-separator JSON from test 19 (search for word_separator_json)
        std::string word_separator_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "h":0, "e":1, "l":2, "o":3, "w":4, "r":5, "d":6,
              "\u0120":7, "!":8, ".":9, ",":10,
              "he":11, "ll":12, "lo":13, "hell":19, "hel":20, "orld":21,
              "\u0120w":14, "or":15, "ld":16,
              "hello":17, "\u0120world":18
            },
            "merges": [
              "h e", "l l", "l o", "he ll", "hel lo",
              "\u0120 w", "o r", "l d", "or ld", "\u0120w orld", "hello \u0120world"
            ]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": " ?[^(\\s|[.,!?])]+"}, "behavior": "Isolated", "invert": false},
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(word_separator_json.data(), word_separator_json.size(),
                                             false);

        // "hello  world" → word-separator: ["hello", " ", " world"]
        //   (leave-last-space rule applies to word-separator whitespace too)
        //   "hello" → hell,o → [19,3]
        //   " " → byte-encode → Ġ → [7]
        //   " world" → byte-encode → Ġ,w,o,r,l,d → Ġw,or,ld → Ġw,orld → Ġworld → [18]
        {
            auto ids = tok->encode("hello  world");
            // [19,3,7,18] = [hell,o,Ġ,Ġworld]
            check(ids.size() == 4, "word_separator_whitespace_double_space_size");
            check(ids[0] == 19 && ids[1] == 3, "word_separator_whitespace_hello");
            check(ids[2] == 7, "word_separator_whitespace_space");
            check(ids[3] == 18, "word_separator_whitespace_world");
        }
    }

    // === 31. Newline handling in StringMerge and NewlineAware ===
    {
        std::cerr << "\n=== Newline Handling ===\n";

        std::string bj(kBoundaryTestJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);

        // "a\nb" → StringMerge: \n is whitespace, byte-encoded to non-ASCII char
        // The boundary test vocab doesn't have the byte-encoded newline token,
        // so the \n token gets dropped. Result: [a, b] = [0, 1]
        {
            auto ids = tok->encode("a\nb");
            check(ids[0] == 0, "string_merge_newline_a");
            // \n byte-encoded char not in vocab → dropped
            // "b" is a separate chunk → [1]
            check(ids.back() == 1, "string_merge_newline_b");
        }

        // NewlineAware variant: \n triggers \s*[\r\n]+ pattern
        std::string q_nl_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {"a":0, "b":1, "\u010a":2},
            "merges": []
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": "[^\u000d\u000a\\p{L}\\p{N}]?\\p{L}+"}, "behavior": "Removed", "invert": true},
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          }
        })";
        auto tok_q = trtmc::CreateBpeTokenizer(q_nl_json.data(), q_nl_json.size(), false);
        {
            auto ids = tok_q->encode("a\nb");
            // NewlineAware: "a" (letter), "\n" (newline sequence), "b" (letter)
            // byte-encode \n → Ċ (U+010A, id=2)
            // Result: [0, 2, 1] = [a, Ċ, b]
            check(ids.size() == 3 && ids[0] == 0 && ids[1] == 2 && ids[2] == 1,
                  "newline_aware_newline_sequence");
        }
    }

    // === Sequence decoder (SentencePiece BPE: SentencePiece, SentencePiece, SentencePiece) ===
    {
        std::cerr << "\n=== Sequence Decoder ===\n";

        // Minimal SentencePiece-style BPE with Sequence decoder
        static const char* kSeqDecoderJson = R"({
          "model": {
            "type": "BPE",
            "byte_fallback": true,
            "vocab": {
              "a": 0, "b": 1, "c": 2,
              "\u2581": 3, "\u2581a": 4, "\u2581b": 5,
              "\u2581the": 6, "cat": 7, "\u2581cat": 12,
              "<0x0A>": 8, "<0x09>": 9,
              "\u2581hello": 10, "\u2581world": 11
            },
            "merges": [
              ["\u2581", "a"],
              ["\u2581", "b"],
              ["\u2581", "the"],
              ["c", "a"],
              ["ca", "t"],
              ["\u2581", "hello"],
              ["\u2581", "world"]
            ]
          },
          "added_tokens": [
            {"id": 100, "content": "<s>", "special": true},
            {"id": 101, "content": "</s>", "special": true}
          ],
          "pre_tokenizer": null,
          "decoder": {
            "type": "Sequence",
            "decoders": [
              {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
              {"type": "ByteFallback"},
              {"type": "Fuse"},
              {"type": "Strip", "content": " ", "start": 1, "stop": 0}
            ]
          }
        })";

        std::string sj(kSeqDecoderJson);
        auto tok = trtmc::CreateBpeTokenizer(sj.data(), sj.size(), false);
        check(tok != nullptr, "seq_decoder_create");

        // Decode: ▁hello ▁world → " hello world" → strip → "hello world"
        {
            auto text = tok->decode({10, 11});
            check(text == "hello world", "seq_decoder_basic_decode");
        }

        // Decode: ▁the ▁cat → " the cat" → strip → "the cat"
        {
            auto text = tok->decode({6, 12});
            check(text == "the cat", "seq_decoder_the_cat");
        }

        // Decode without leading ▁: "thecat" (no space, continuation)
        {
            auto text = tok->decode({6, 7});
            check(text == "thecat", "seq_decoder_continuation");
        }

        // ByteFallback: ▁a <0x0A> ▁b → " a\n b" → strip → "a\n b"
        {
            auto text = tok->decode({4, 8, 5});
            check(text == "a\n b", "seq_decoder_byte_fallback_newline");
        }

        // ByteFallback: ▁a <0x09> ▁b → " a\t b" → strip → "a\t b"
        {
            auto text = tok->decode({4, 9, 5});
            check(text == "a\t b", "seq_decoder_byte_fallback_tab");
        }

        // Strip leading space
        {
            auto text = tok->decode({3, 0});
            check(text == "a", "seq_decoder_strip_leading_space");
        }

        // Special tokens filtered
        {
            auto text = tok->decode({100, 10, 11, 101});
            check(text == "hello world", "seq_decoder_filter_special");
        }

        // Empty decode
        {
            auto text = tok->decode({});
            check(text.empty(), "seq_decoder_empty");
        }
    }

    // === ByteLevel decoder (explicit decoder field) ===
    {
        std::cerr << "\n=== Explicit ByteLevel Decoder ===\n";

        // Verify that explicit decoder.type=ByteLevel is recognized
        static const char* kExplicitByteLevelJson = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "h": 0, "e": 1, "l": 2, "o": 3,
              "he": 4, "ll": 5, "lo": 6,
              "hello": 7,
              "\u0120": 8, "w": 9, "orld": 10,
              "\u0120world": 11
            },
            "merges": [
              ["h", "e"], ["l", "l"], ["l", "o"],
              ["he", "ll"], ["hell", "o"],
              ["\u0120", "w"], ["w", "orld"],
              ["\u0120w", "orld"],
              ["hello", "\u0120world"]
            ]
          },
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
          "decoder": {"type": "ByteLevel"}
        })";

        std::string bj(kExplicitByteLevelJson);
        auto tok = trtmc::CreateBpeTokenizer(bj.data(), bj.size(), false);
        check(tok != nullptr, "explicit_bytelevel_decoder_create");

        auto text = tok->decode({7, 11});
        check(text == "hello world", "explicit_bytelevel_decoder_decode");
    }

    // === 33. Sequence post_processor with nested TemplateProcessing (SentencePiece 3.1 style) ===
    {
        std::cerr << "\n=== Sequence Post-Processor ===\n";

        // SentencePiece 3.1 style: post_processor is Sequence containing ByteLevel +
        // TemplateProcessing
        std::string seq_pp_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "h": 0, "e": 1, "l": 2, "o": 3,
              "he": 4, "ll": 5, "lo": 6,
              "hel": 7, "hello": 8
            },
            "merges": ["h e", "l l", "l o", "he l", "hel lo"]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"}, "behavior": "Isolated", "invert": true},
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          },
          "added_tokens": [
            {"id": 100, "content": "<|begin_of_text|>", "special": true},
            {"id": 101, "content": "<|end_of_text|>", "special": true}
          ],
          "post_processor": {
            "type": "Sequence",
            "processors": [
              {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
              {
                "type": "TemplateProcessing",
                "single": [
                  {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                  {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "special_tokens": {
                  "<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [100], "tokens": ["<|begin_of_text|>"]}
                }
              }
            ]
          }
        })";

        // With add_special_tokens=true: BOS should be prepended
        auto tok_sp = trtmc::CreateBpeTokenizer(seq_pp_json.data(), seq_pp_json.size(), true);
        check(tok_sp != nullptr, "seq_pp_create_with_special");
        {
            auto ids = tok_sp->encode("hello");
            check(ids.size() >= 2 && ids[0] == 100, "seq_pp_bos_prepended");
        }

        // With add_special_tokens=false: no BOS
        auto tok_no_sp = trtmc::CreateBpeTokenizer(seq_pp_json.data(), seq_pp_json.size(), false);
        {
            auto ids = tok_no_sp->encode("hello");
            check(!ids.empty() && ids[0] != 100, "seq_pp_no_bos_without_special");
        }
    }

    // === 34. Multiple BOS tokens (GLM-4 style) ===
    {
        std::cerr << "\n=== Multiple BOS Tokens ===\n";

        // GLM-4 style: two SpecialTokens before Sequence, one after
        std::string multi_bos_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "h": 0, "e": 1, "l": 2, "o": 3,
              "he": 4, "ll": 5, "lo": 6,
              "hel": 7, "hello": 8
            },
            "merges": ["h e", "l l", "l o", "he l", "hel lo"]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"}, "behavior": "Isolated", "invert": true},
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          },
          "added_tokens": [
            {"id": 200, "content": "[gMASK]", "special": true},
            {"id": 201, "content": "<sop>", "special": true},
            {"id": 202, "content": "<eop>", "special": true}
          ],
          "post_processor": {
            "type": "TemplateProcessing",
            "single": [
              {"SpecialToken": {"id": "[gMASK]", "type_id": 0}},
              {"SpecialToken": {"id": "<sop>", "type_id": 0}},
              {"Sequence": {"id": "A", "type_id": 0}},
              {"SpecialToken": {"id": "<eop>", "type_id": 0}}
            ],
            "special_tokens": {
              "[gMASK]": {"id": "[gMASK]", "ids": [200], "tokens": ["[gMASK]"]},
              "<sop>": {"id": "<sop>", "ids": [201], "tokens": ["<sop>"]},
              "<eop>": {"id": "<eop>", "ids": [202], "tokens": ["<eop>"]}
            }
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(multi_bos_json.data(), multi_bos_json.size(), true);
        check(tok != nullptr, "multi_bos_create");
        {
            auto ids = tok->encode("hello");
            // Expected: [gMASK](200), <sop>(201), ...hello..., <eop>(202)
            check(ids.size() >= 4, "multi_bos_min_size");
            check(ids[0] == 200, "multi_bos_first_token");
            check(ids[1] == 201, "multi_bos_second_token");
            check(ids.back() == 202, "multi_bos_eos_appended");
        }

        // Without special tokens: no BOS/EOS
        auto tok_ns =
            trtmc::CreateBpeTokenizer(multi_bos_json.data(), multi_bos_json.size(), false);
        {
            auto ids = tok_ns->encode("hello");
            check(!ids.empty() && ids[0] != 200 && ids.back() != 202,
                  "multi_bos_none_without_special");
        }
    }

    // === 35. Special tokens matched during VL prompt encode ===
    {
        std::cerr << "\n=== VL Special Token Encode ===\n";

        // VL style: special tokens in prompt text must be encoded to their IDs
        std::string vl_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "D": 0, "e": 1, "s": 2, "c": 3, "r": 4, "i": 5, "b": 6,
              "De": 7, "sc": 8, "ri": 9, "be": 10
            },
            "merges": ["D e", "s c", "r i", "b e"]
          },
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"}, "behavior": "Isolated", "invert": true},
              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
          },
          "added_tokens": [
            {"id": 100, "content": "<|im_start|>", "special": true},
            {"id": 101, "content": "<|im_end|>", "special": true},
            {"id": 102, "content": "<|vision_start|>", "special": true},
            {"id": 103, "content": "<|vision_end|>", "special": true},
            {"id": 104, "content": "<|image_pad|>", "special": true}
          ]
        })";

        auto tok = trtmc::CreateBpeTokenizer(vl_json.data(), vl_json.size(), false);
        check(tok != nullptr, "vl_special_create");

        // Single special token should be matched directly
        {
            auto ids = tok->encode("<|im_start|>");
            check(ids.size() == 1 && ids[0] == 100, "vl_single_special_token");
        }

        // Multiple image pads
        {
            auto ids = tok->encode("<|image_pad|><|image_pad|><|image_pad|>");
            check(ids.size() == 3 && ids[0] == 104 && ids[1] == 104 && ids[2] == 104,
                  "vl_repeated_image_pad");
        }

        // Mixed special tokens and normal text (simplified VL prompt)
        {
            auto ids = tok->encode("<|im_start|>Describe<|im_end|>");
            check(ids.size() >= 3, "vl_mixed_min_size");
            check(ids[0] == 100, "vl_mixed_im_start");
            check(ids.back() == 101, "vl_mixed_im_end");
        }

        // VL prompt with vision tokens
        {
            auto ids = tok->encode("<|vision_start|><|image_pad|><|image_pad|><|vision_end|>");
            check(ids.size() == 4 && ids[0] == 102 && ids[1] == 104 && ids[2] == 104 &&
                      ids[3] == 103,
                  "vl_vision_token_sequence");
        }

        // Special tokens should be filtered during decode
        {
            auto text = tok->decode({100, 7, 101});
            // 100 (<|im_start|>) and 101 (<|im_end|>) are special → skipped
            // Only 7 (De) is decoded
            check(text == "De", "vl_decode_filters_special");
        }
    }

    // === 36. Sequence PP with ByteLevel first, TemplateProcessing second ===
    {
        std::cerr << "\n=== Sequence PP Nested Discovery ===\n";

        // Ensure TemplateProcessing is discovered even when it's not the first element
        std::string nested_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1},
            "merges": []
          },
          "added_tokens": [
            {"id": 50, "content": "<bos>", "special": true},
            {"id": 51, "content": "<eos>", "special": true}
          ],
          "post_processor": {
            "type": "Sequence",
            "processors": [
              {"type": "ByteLevel", "add_prefix_space": true},
              {
                "type": "TemplateProcessing",
                "single": [
                  {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                  {"Sequence": {"id": "A", "type_id": 0}},
                  {"SpecialToken": {"id": "<eos>", "type_id": 0}}
                ],
                "special_tokens": {
                  "<bos>": {"id": "<bos>", "ids": [50], "tokens": ["<bos>"]},
                  "<eos>": {"id": "<eos>", "ids": [51], "tokens": ["<eos>"]}
                }
              }
            ]
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(nested_json.data(), nested_json.size(), true);
        check(tok != nullptr, "nested_pp_create");
        {
            auto ids = tok->encode("a");
            // BOS(50) + a(0) + EOS(51)
            check(ids.size() == 3 && ids[0] == 50 && ids[1] == 0 && ids[2] == 51,
                  "nested_pp_bos_and_eos");
        }
    }

    // === 37. Normalizer Split pre-tokenizer with SentencePiece-BPE normalizer ===
    {
        std::cerr << "\n=== Normalizer Split SentencePiece-BPE ===\n";

        std::string normalizer_json = R"({
          "model": {
            "type": "BPE",
            "vocab": {
              "<bos>": 0,
              "\u2581": 1,
              "i": 2,
              "s": 3,
              "is": 4,
              "\u2581is": 5,
              "2": 6,
              "+": 7,
              "\u2581+": 8,
              "?": 9,
              "W": 10,
              "h": 11,
              "a": 12,
              "t": 13,
              "Wh": 14,
              "Wha": 15,
              "What": 16
            },
            "merges": [
              "W h",
              "Wh a",
              "Wha t",
              "i s",
              "\u2581 is",
              "\u2581 +"
            ]
          },
          "added_tokens": [
            {"id": 0, "content": "<bos>", "special": true}
          ],
          "normalizer": {
            "type": "Replace",
            "pattern": {"String": " "},
            "content": "\u2581"
          },
          "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": false
          },
          "decoder": {
            "type": "Sequence",
            "decoders": [
              {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
              {"type": "ByteFallback"},
              {"type": "Fuse"}
            ]
          },
          "post_processor": {
            "type": "TemplateProcessing",
            "single": [
              {"SpecialToken": {"id": "<bos>", "type_id": 0}},
              {"Sequence": {"id": "A", "type_id": 0}}
            ],
            "special_tokens": {
              "<bos>": {"id": "<bos>", "ids": [0], "tokens": ["<bos>"]}
            }
          }
        })";

        auto tok = trtmc::CreateBpeTokenizer(normalizer_json.data(), normalizer_json.size(), true);
        check(tok != nullptr, "normalizer_split_create");
        {
            auto ids = tok->encode("What is 2 + 2?");
            std::vector<int32_t> expected = {0, 16, 5, 1, 6, 8, 1, 6, 9};
            check(ids == expected, "normalizer_split_encode_no_leading_space");
        }
        {
            auto text = tok->decode({0, 16, 5, 1, 6, 8, 1, 6, 9});
            check(text == "What is 2 + 2?", "normalizer_split_decode");
        }
    }

    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) failed\n";
        return 1;
    }

    std::cerr << "\nAll tests passed!\n";
    return 0;
}
