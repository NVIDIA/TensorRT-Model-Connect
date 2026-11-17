# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the CPU goldens of the layoutlm runtime from the reference libraries.

usage: python generate_goldens.py

Needs transformers (5.2.0), tokenizers, numpy and pytesseract (only its TSV
reader runs; no Tesseract binary is called). Writes, next to this script:

* ``tiny_tokenizer.json`` - a small byte-level BPE tokenizer trained here on a
  fixed corpus, with the RoBERTa special tokens and pair framing of the
  checkpoint, so the CPU test needs no checkpoint download;
* ``document_qa_golden.json`` - word ids, overflowing windows, question splits,
  Tesseract TSV word lists and decoded spans, each recorded from the code path
  transformers' DocumentQuestionAnsweringPipeline uses for LayoutLM.

The family E2E records the word and window goldens again with the checkpoint's
own tokenizer.json and reruns the same C++ checker on them.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

CORPUS = (
    "Invoice number BW-20417 is due on April 2, 2026. The total amount due is $4,530.60, "
    "including sales tax of $335.60. Bill to Northwind Traders, 1200 Harbor Road, Seattle. "
    "Harbor Lane Robotics shipped 2,340 carts and had 418 employees at year end. It's the "
    "company's best year; they'll report again in March. Net revenue was $18.4 million. "
    "Payment terms: net 30 days. Questions? E-mail billing@brightwater.example or call "
    "(503) 555-0142. Qty Unit Price Amount Subtotal Description Date PO Number naïve café "
    "Ünïcödé 日本語 ١٢٣ ²³ € — … ‘quoted’ «guillemets» 100% #7 A1b2 x'LL 'll "
)

WORDS = (
    "Invoice", "BW-20417", "$4,530.60", "2026.", "naïve", "café", "Ünïcödé", "日本語", "١٢٣",
    "²³", "Ⅻ", "x​y", " nbsp", "tab\tin", "a　b", "😀", "a😀b", "e-mail@x.org",
    "---", "...", "'", "''", "it's", "IT'S", "'ll", "x'LL", "they'll", "A1b2", "🇺🇸", "ﬁ", "Ǆ",
    "́e", "é", "<s>", "</s>", "<pad>", "<unk>", "<mask>", "a<s>b", "<s>x", "x<s>",
    "<mask>y", "y<mask>", "<s><s>", "€100", "—", "«guillemets»", "(503)", "x", "Qq",
)

QUESTIONS = (
    "What is the invoice number?",
    "  When  is\tthe　payment due? ",
    "Who is it's <s> owner",
)

SPLITS = (
    "What is the total?", "  leading and trailing  ", "tabs\tand\nnewlines\r\nmixed",
    "x\x1cy\x1dz\x1e\x1fw", "a\x85b", "a b", "a b", "a b c", "a b c",
    "a b c　d", "a᠎b", "x​y", "", "   ", "単語 と 単語", "émigré naïve",
)


def _tiny_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        min_frequency=1,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([CORPUS] * 4, trainer=trainer)
    from tokenizers import AddedToken

    tokenizer.add_special_tokens([AddedToken("<mask>", lstrip=True, normalized=False)])
    tokenizer.post_processor = processors.RobertaProcessing(
        ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>"))
    )
    path.write_text(tokenizer.to_str(pretty=True) + "\n", encoding="utf-8")


def _reference_tokenizer(tokenizer_json: Path):
    from transformers import AutoTokenizer

    directory = Path(tempfile.mkdtemp(prefix="layoutlm-golden-"))
    (directory / "tokenizer.json").write_bytes(tokenizer_json.read_bytes())
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "RobertaTokenizer",
                "add_prefix_space": True,
                "model_max_length": 512,
                "bos_token": "<s>", "eos_token": "</s>", "sep_token": "</s>",
                "cls_token": "<s>", "unk_token": "<unk>", "pad_token": "<pad>",
                "mask_token": "<mask>",
            }
        ),
        encoding="utf-8",
    )
    return AutoTokenizer.from_pretrained(str(directory))


def word_cases(tokenizer, words) -> list[dict]:
    cases = []
    for word in words:
        encoding = tokenizer([word], is_split_into_words=True, add_special_tokens=False)
        cases.append({"word": word, "ids": list(encoding["input_ids"])})
    return cases


def window_case(tokenizer, name, question, words, boxes, max_length, stride) -> dict:
    """One pipeline preprocess call: the windows the reference feeds the model."""
    case = {"name": name, "question_words": question.split(), "document_words": list(words),
            "boxes": [list(box) for box in boxes], "max_sequence_length": max_length,
            "stride": stride}
    try:
        encoding = tokenizer(text=question.split(), text_pair=list(words), is_split_into_words=True,
                             padding="do_not_pad", max_length=max_length, stride=stride,
                             return_token_type_ids=True, truncation="only_second",
                             return_overflowing_tokens=True)
    except BaseException as error:  # noqa: BLE001 - tokenizers panics (a BaseException)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        case["error"] = True  # The reference refusing the input is the expectation.
        return case
    windows = []
    for span in range(len(encoding["input_ids"])):
        ids = encoding["input_ids"][span]
        sequence = encoding.sequence_ids(span)
        word_ids = encoding.word_ids(span)
        window_boxes = []
        for input_id, sequence_id, word_id in zip(ids, sequence, word_ids):
            if sequence_id == 1:
                window_boxes.append(list(boxes[word_id]))
            elif input_id == tokenizer.sep_token_id:
                window_boxes.append([1000] * 4)
            else:
                window_boxes.append([0] * 4)
        windows.append({"input_ids": list(ids),
                        "word_ids": [-1 if value is None else value for value in word_ids],
                        "in_document": [1 if value == 1 else 0 for value in sequence],
                        "boxes": window_boxes})
    case["windows"] = windows
    return case


def window_cases(tokenizer) -> list[dict]:
    rng = random.Random(248)
    vocabulary = [word for word in CORPUS.split() if word]
    cases = []
    for length, max_length, stride, question in (
        (12, 512, 256, QUESTIONS[0]), (300, 512, 256, QUESTIONS[0]),
        (620, 512, 256, QUESTIONS[1]),
        (90, 64, 32, QUESTIONS[0]), (60, 48, 16, QUESTIONS[2]), (0, 512, 256, QUESTIONS[0]),
        (40, 32, 16, QUESTIONS[2]), (60, 24, 12, "How much is the total amount due for the order now?"),
    ):
        words = [rng.choice(vocabulary + list(WORDS)) for _ in range(length)]
        boxes = []
        for _ in words:
            left, top = rng.randrange(0, 900), rng.randrange(0, 950)
            boxes.append((left, top, left + rng.randrange(0, 100), top + rng.randrange(0, 50)))
        name = f"{length}-words-window-{max_length}-stride-{stride}"
        cases.append(window_case(tokenizer, name, question, words, boxes, max_length, stride))
    return cases


def split_cases() -> list[dict]:
    return [{"text": text, "words": text.split()} for text in SPLITS]


def tsv_cases() -> list[dict]:
    import pytesseract
    from transformers.pipelines import document_question_answering as pipeline_module

    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"

    def page(width, height):
        return f"1\t1\t0\t0\t0\t0\t0\t0\t{width}\t{height}\t-1\t"

    def word(left, top, width, height, text, conf="96.5"):
        return f"5\t1\t1\t1\t1\t1\t{left}\t{top}\t{width}\t{height}\t{conf}\t{text}"

    def line(left, top, width, height):
        return f"4\t1\t1\t1\t1\t0\t{left}\t{top}\t{width}\t{height}\t-1\t"

    documents = {
        "plain": (1240, 1754, [page(1240, 1754), line(90, 90, 400, 30), word(90, 90, 120, 30, "Invoice"),
                               word(220, 90, 140, 30, "BW-20417"), word(380, 92, 77, 28, "$4,530.60")]),
        "whitespace-word-dropped": (800, 600, [page(800, 600), word(10, 10, 50, 20, "a"),
                                               word(70, 10, 5, 20, " "), word(80, 10, 50, 20, "b")]),
        "last-row-empty-text": (500, 400, [page(500, 400), word(0, 0, 499, 399, "edge"),
                                           line(0, 0, 499, 399)]),
        "empty-page": (300, 200, [page(300, 200)]),
        "rounding": (3, 7, [page(3, 7), word(1, 1, 1, 5, "x"), word(2, 3, 1, 4, "y")]),
        "unicode": (1000, 1000, [page(1000, 1000), word(5, 5, 10, 10, "café"),
                                 word(20, 5, 10, 10, " "), word(40, 5, 10, 10, "日本")]),
        "float-cells": (640, 480, [page(640, 480), word("10.9", "20.2", "30.7", "40.1", "fl", "90")]),
    }
    cases = []
    for name, (width, height, rows) in documents.items():
        tsv = header + "\n" + "\n".join(rows) + "\n"
        data = pytesseract.pytesseract.file_to_dict(tsv, "\t", -1)

        class _Image:
            size = (width, height)

        original = pipeline_module.pytesseract.image_to_data
        pipeline_module.pytesseract.image_to_data = lambda *_args, **_kwargs: data
        try:
            words, boxes = pipeline_module.apply_tesseract(_Image(), lang=None, tesseract_config="")
            cases.append({"name": name, "tsv": tsv, "width": width, "height": height,
                          "words": words, "boxes": boxes})
        except Exception:  # noqa: BLE001 - the reference refusing the TSV is the expectation
            cases.append({"name": name, "tsv": tsv, "width": width, "height": height,
                          "error": True})
        finally:
            pipeline_module.pytesseract.image_to_data = original
    return cases


def span_cases(tokenizer) -> list[dict]:
    import numpy as np
    from transformers.pipelines.question_answering import select_starts_ends

    rng = np.random.default_rng(248)
    question = "What is the total?".split()
    document = ["Total", "Due", "$4,530.60", "<s>", "Subtotal", "$4,195.00", "Tax", "8%"] * 3
    encoding = tokenizer(text=question, text_pair=document, is_split_into_words=True,
                         return_token_type_ids=True)
    ids = list(encoding["input_ids"])
    sequence = encoding.sequence_ids(0)
    word_ids = [-1 if value is None else value for value in encoding.word_ids(0)]
    in_document = [1 if value == 1 else 0 for value in sequence]
    p_mask = [value != 1 for value in sequence]
    for index, value in enumerate(ids):
        if value == tokenizer.cls_token_id:
            p_mask[index] = 0
    count = len(ids)

    scenarios = {
        "random": (rng.normal(0, 3, count), rng.normal(0, 3, count), 15),
        "peaked": (np.where(np.arange(count) == 14, 9.0, rng.normal(0, 1, count)),
                   np.where(np.arange(count) == 17, 9.0, rng.normal(0, 1, count)), 15),
        "end-before-start": (np.where(np.arange(count) == 20, 9.0, 0.0),
                             np.where(np.arange(count) == 12, 9.0, 0.0), 15),
        "too-long": (np.where(np.arange(count) == 10, 9.0, 0.0),
                     np.where(np.arange(count) == count - 2, 9.0, 0.0), 4),
        "ties": (np.zeros(count), np.zeros(count), 15),
        "cls-favored": (np.where(np.arange(count) == 0, 30.0, 0.0),
                        np.where(np.arange(count) == 0, 30.0, 0.0), 15),
        "question-favored": (np.where(np.arange(count) == 2, 30.0, -5.0),
                             np.where(np.arange(count) == 3, 30.0, -5.0), 15),
        "one-token": (rng.normal(0, 2, count), rng.normal(0, 2, count), 1),
    }
    cases = []
    for name, (starts, ends, max_answer_length) in scenarios.items():
        starts = np.asarray(starts, dtype=np.float32)[None]
        ends = np.asarray(ends, dtype=np.float32)[None]
        found_starts, found_ends, scores, _ = select_starts_ends(
            start=starts, end=ends, p_mask=p_mask, attention_mask=np.ones((1, count), np.int64),
            top_k=1, handle_impossible_answer=False, max_answer_len=max_answer_length)
        case = {"name": name, "input_ids": ids, "in_document": in_document, "word_ids": word_ids,
                "start_logits": starts[0].tolist(), "end_logits": ends[0].tolist(),
                "cls_id": tokenizer.cls_token_id, "max_answer_length": max_answer_length,
                "found": bool(len(found_starts))}
        if len(found_starts):
            case.update(start=int(found_starts[0]), end=int(found_ends[0]), score=float(scores[0]))
        cases.append(case)
    return cases


def main() -> None:
    tokenizer_json = HERE / "tiny_tokenizer.json"
    _tiny_tokenizer(tokenizer_json)
    tokenizer = _reference_tokenizer(tokenizer_json)
    golden = {
        "tokenizer_json": tokenizer_json.name,
        "word_cases": word_cases(tokenizer, WORDS + tuple(CORPUS.split())),
        "window_cases": window_cases(tokenizer),
        "split_cases": split_cases(),
        "tsv_cases": tsv_cases(),
        "span_cases": span_cases(tokenizer),
    }
    (HERE / "document_qa_golden.json").write_text(
        json.dumps(golden, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
