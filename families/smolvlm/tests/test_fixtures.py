# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native unit-test fixtures equal what the reference libraries produce.

``tests/data/*.json`` are the oracles of the C++ tests ``test_smolvlm_media`` and
``test_smolvlm_tokenizer``. Each test here recomputes its fixture with the
reference implementation (Pillow, the Transformers SmolVLM processor and chat
template renderer, Hugging Face ``tokenizers``) and requires an exact match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"

# Resize cases (input height, width -> output height, width): upscale, downscale,
# Pillow's vertical-first order, single-axis passes, the same-size copy, slivers.
RESIZE_CASES = [
    (7, 9, 16, 20), (40, 30, 9, 13), (60, 20, 10, 18), (12, 30, 12, 17), (30, 12, 17, 12),
    (40, 10, 15, 25), (5, 5, 5, 5), (3, 50, 2, 7), (50, 3, 7, 2), (9, 9, 23, 4),
]
GEOMETRY_SIZES = [
    (480, 640), (640, 480), (512, 512), (300, 200), (1000, 3000), (4000, 50), (1, 1), (2047, 1),
    (1, 5000), (999, 1001), (1365, 2048), (768, 1024), (1080, 1920), (3024, 4032), (17, 4096),
]
# A reduced-scale processor (longest edge 32, tile 16) exercises the whole tiling
# chain with fixtures small enough to keep in the repository.
PIPELINE = {"height": 13, "width": 21, "longest_edge": 32, "tile": 16}


def pattern(height: int, width: int):
    """Deterministic RGB test image; test_media.cpp computes the same formula."""
    import numpy as np

    y, x, c = np.meshgrid(np.arange(height), np.arange(width), np.arange(3), indexing="ij")
    return ((y * 37 + x * 11 + c * 83 + (x * y % 13) * 7) % 256).astype(np.uint8)


def _processor(longest_edge: int, tile: int):
    from transformers.models.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor

    return SmolVLMImageProcessor(
        do_convert_rgb=True, do_resize=True, size={"longest_edge": longest_edge},
        resample=1, do_image_splitting=True, max_image_size={"longest_edge": tile},
        do_rescale=True, rescale_factor=1 / 255, do_normalize=True,
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], do_pad=True)


def expected_media() -> dict:
    import numpy as np
    from PIL import Image
    from transformers.image_utils import ChannelDimension
    from transformers.models.smolvlm.image_processing_smolvlm import get_resize_output_image_size

    resize = []
    for height, width, out_height, out_width in RESIZE_CASES:
        image = Image.fromarray(pattern(height, width)).resize((out_width, out_height), Image.Resampling.LANCZOS)
        resize.append({"height": height, "width": width, "out_height": out_height, "out_width": out_width,
                       "expected": np.asarray(image).reshape(-1).tolist()})
    geometry = []
    processor = _processor(2048, 512)
    for height, width in GEOMETRY_SIZES:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        resized = get_resize_output_image_size(blank, 2048, input_data_format=ChannelDimension.LAST)
        tiled = processor.resize_for_vision_encoder(
            np.zeros((*resized, 3), dtype=np.uint8), 512, input_data_format=ChannelDimension.LAST).shape[:2]
        _, rows, cols = processor.split_image(
            np.zeros((*tiled, 3), dtype=np.uint8), {"longest_edge": 512},
            input_data_format=ChannelDimension.LAST)
        geometry.append({"height": height, "width": width, "resized": list(resized),
                         "tiled": list(tiled), "rows": rows, "cols": cols})
    small = _processor(PIPELINE["longest_edge"], PIPELINE["tile"])
    output = small.preprocess([[pattern(PIPELINE["height"], PIPELINE["width"])]],
                              return_row_col_info=True, return_tensors="np")
    pixels = np.asarray(output["pixel_values"], dtype=np.float32)[0]
    pipeline = dict(PIPELINE, rows=output["rows"][0][0], cols=output["cols"][0][0],
                    frames=int(pixels.shape[0]), pixels=[float(v) for v in pixels.reshape(-1)])
    return {"resize": resize, "geometry": geometry, "pipeline": pipeline}


PROMPT_CASES = [
    [{"role": "user", "content": [{"type": "image", "rows": 3, "cols": 4},
                                  {"type": "text", "text": "Describe this image."}]}],
    [{"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
     {"role": "user", "content": [{"type": "text", "text": "Compare "},
                                  {"type": "image", "rows": 4, "cols": 1},
                                  {"type": "text", "text": " with "},
                                  {"type": "image", "rows": 0, "cols": 0}]}],
    [{"role": "user", "content": [{"type": "text", "text": "Hi"}]},
     {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
     {"role": "user", "content": [{"type": "image", "rows": 2, "cols": 4}]}],
]


def expected_prompts() -> list:
    pytest.importorskip("torchvision", reason="the SmolVLM processor module imports torchvision")
    from transformers.models.smolvlm.processing_smolvlm import get_image_prompt_string
    from transformers.utils.chat_template_utils import render_jinja_template

    from families.smolvlm.config import CHAT_TEMPLATE

    cases = []
    for messages in PROMPT_CASES:
        template_messages = [{"role": m["role"], "content": [
            {"type": p["type"], **({"text": p["text"]} if p["type"] == "text" else {})} for p in m["content"]]}
            for m in messages]
        rendered, _ = render_jinja_template(conversations=[template_messages], chat_template=CHAT_TEMPLATE,
                                            add_generation_prompt=True)
        text = rendered[0]
        images = [p for m in messages for p in m["content"] if p["type"] == "image"]
        pieces = text.split("<image>")
        assert len(pieces) == len(images) + 1
        expanded = pieces[0]
        for image, piece in zip(images, pieces[1:]):
            expanded += get_image_prompt_string(image["rows"], image["cols"], 64,
                                                fake_token_around_image="<fake_token_around_image>",
                                                image_token="<image>", global_image_token="<global-img>")
            expanded += piece
        cases.append({"messages": messages, "image_seq_len": 64, "expected": expanded})
    return cases


TOKENIZER_TEXTS = [
    "Hello world", "it's 2025! It'S 12+7=19.", "  multiple   spaces\n\nand\tnewlines \n",
    "naïve café — 東京 ① ² Ⅻ ١٢٣", "<|im_start|>User:<image>Describe<end_of_utterance>\nAssistant:",
    "x<x><y>y<x>z", "nbsp ideographic　space line", "emoji 🙂 and 'll 've 're 'd 'm 't",
    "", "trailing space ", "\u0085next line\u000bvertical tab",
    "a\x04b!\x04!", "\U00040000z and \x04\x06",
    # Long single pieces: repeated merges, and a lower-ranked pair at the end of a run.
    "Hello" * 600, "l" * 1001 + "d", " " * 1000 + "x",
]


def tiny_tokenizer():
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    # Like the SmolVLM vocabulary, leave some byte units out (bytes 0x04, 0x06, 0xF1).
    alphabet = set(pre_tokenizers.ByteLevel.alphabet()) - {"Ą", "Ć", "ñ"}
    vocab = {unit: index for index, unit in enumerate(sorted(alphabet))}
    merges = [("!", "!"), ("Ġ", "w"), ("o", "r"), ("Ġw", "or"), ("l", "d"), ("Ġwor", "ld"), ("H", "e"), ("l", "l"),
              ("He", "ll"), ("Hell", "o"), ("Ġ", "Ġ"), ("ĠĠ", "Ġ"), ("Ċ", "Ċ"), ("i", "t"), ("'", "s"),
              ("a", "n"), ("an", "d"), ("Ġ", "t"), ("e", "s"), ("c", "a"), ("Ã", "¯")]
    for left, right in merges:
        vocab.setdefault(left + right, len(vocab))
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in (
        "<|im_start|>", "<image>", "<end_of_utterance>", "<x>", "<x><y>")])
    return tokenizer


def expected_tokenizer() -> dict:
    tokenizer = tiny_tokenizer()
    cases = []
    for text in TOKENIZER_TEXTS:
        ids = tokenizer.encode(text).ids
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=True)})
    emoji = tokenizer.encode("🙂").ids
    decode_only = [{"ids": emoji[:1], "decoded": tokenizer.decode(emoji[:1], skip_special_tokens=True)},
                   {"ids": emoji[:2] + tokenizer.encode("ok").ids,
                    "decoded": tokenizer.decode(emoji[:2] + tokenizer.encode("ok").ids, skip_special_tokens=True)}]
    return {"tokenizer": tokenizer.to_str(), "cases": cases, "decode_only": decode_only}


def _fixture(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_media_fixture_matches_pillow_and_the_reference_processor():
    pil = pytest.importorskip("PIL")
    pytest.importorskip("transformers")
    # Pillow 13 runs the vertical pass first when the height shrinks more than twice the
    # width; earlier releases always start horizontally. The native port follows Pillow 13.
    if int(pil.__version__.split(".")[0]) != 13:
        pytest.skip(f"the fixtures and the runtime follow Pillow 13, found {pil.__version__}")
    assert _fixture("media_cases.json") == expected_media()


def test_prompt_fixture_matches_the_reference_template_and_expansion():
    assert _fixture("prompt_cases.json") == expected_prompts()


def test_tokenizer_fixture_matches_huggingface_tokenizers():
    pytest.importorskip("tokenizers")
    assert _fixture("tokenizer_cases.json") == expected_tokenizer()
