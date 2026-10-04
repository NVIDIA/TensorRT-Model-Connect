# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf's MMLU, GSM8K, and MATH-500 at pinned dataset revisions, on a deterministic subset, and
LAMBADA (last-word prediction) with its first-word grader.

The subclasses keep AIPerf's lighteval prompts and graders. They pin the dataset revision (AIPerf
loads the latest one) and select the same problems for every side, configured through the
environment because AIPerf does not forward benchmark options:

  TRTMC_ACCURACY_SEED             shuffle each task's problems with this seed first (seeded stratified
                                  sampling; unset: dataset order)
  TRTMC_ACCURACY_PER_TASK         keep the first k problems of each task (MMLU subject)
  TRTMC_ACCURACY_LIMIT            keep the first n problems overall
  TRTMC_ACCURACY_MAX_NEW_TOKENS   cap each problem's generation size
  TRTMC_ACCURACY_TOKEN_LIMIT      drop problems whose prompt plus generation exceed the bundle's
                                  sequence limit, counted with TRTMC_ACCURACY_TOKENIZER (at
                                  TRTMC_ACCURACY_TOKENIZER_REVISION; TRTMC_ACCURACY_TRUST_REMOTE_CODE=1)
  TRTMC_ACCURACY_CHAT=1           count the prompt as the chat template renders it (the chat route)
  TRTMC_ACCURACY_TEMPLATE_MARGIN  tokens reserved around the prompt when it is not rendered (default 128)
  TRTMC_ACCURACY_CACHE            a directory for selections: the first load under the same settings (the
                                  harness's plan) writes the chosen problems, later ones (each side's AIPerf run)
                                  read the same file instead of loading and tokenizing the dataset again
  HF_DATASETS_CACHE               the datasets cache (shared by the harness and AIPerf's processes)
"""

from __future__ import annotations

import asyncio
import os
import random
import zlib
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from aiperf.accuracy.benchmarks import gsm8k, math_500, mmlu
from aiperf.accuracy.benchmarks._datasets_compat import load_dataset
from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import BenchmarkProblem, GradingResult

MMLU_REVISION = "31d46ab06e6934bb0d95f6918668716d1db6f921"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
LAMBADA_REVISION = "900124bf3b8235c6daf21033af9948b3f07346c4"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
BART_SENTENCES = 1000
# Tokens a greedy continuation needs to spell out one word.
LAMBADA_GENERATION_SIZE = 8
# MMLU's answer format, added to lighteval's 0-shot instruction: instruction-tuned models otherwise open with an
# explanation and give no letter within the answer budget ("To find the factorization ..."). The budget fits
# "The answer is (B)." and the like, which AIPerf's grader extracts.
MMLU_ANSWER_INSTRUCTION = " Answer with the letter of the correct option only."
MMLU_GENERATION_SIZE = 8
# Tokens a chat template adds around the conversation (role markers, generation prompt) where it cannot
# be rendered; plain completions need only a few.
TEMPLATE_MARGIN = 128
# Tokens a rendered chat prompt may still differ by on the server (for example a thinking switch).
RENDERED_MARGIN = 8


def _int(environ: Mapping[str, str], name: str) -> int | None:
    value = environ.get(name)
    return int(value) if value else None


def _token_counter(environ: Mapping[str, str]) -> Callable[[BenchmarkProblem], int] | None:
    """Tokens a problem's prompt takes on the bundle: rendered by the chat template on the chat route
    (TRTMC_ACCURACY_CHAT=1), else the plain prompt plus TRTMC_ACCURACY_TEMPLATE_MARGIN."""
    if not environ.get("TRTMC_ACCURACY_TOKENIZER"):
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(environ["TRTMC_ACCURACY_TOKENIZER"],
                                              revision=environ.get("TRTMC_ACCURACY_TOKENIZER_REVISION") or None,
                                              trust_remote_code=environ.get("TRTMC_ACCURACY_TRUST_REMOTE_CODE") == "1")
    margin = _int(environ, "TRTMC_ACCURACY_TEMPLATE_MARGIN")
    margin = TEMPLATE_MARGIN if margin is None else margin
    if environ.get("TRTMC_ACCURACY_CHAT") == "1" and getattr(tokenizer, "chat_template", None):
        def rendered(problem: BenchmarkProblem) -> int:
            messages = problem.raw_messages or [{"role": "user", "content": problem.prompt}]
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False)
            return len(tokenizer(text, add_special_tokens=False)["input_ids"]) + RENDERED_MARGIN
        return rendered
    return lambda problem: len(tokenizer(_prompt_text(problem), add_special_tokens=True)["input_ids"]) + margin


def _prompt_text(problem: BenchmarkProblem) -> str:
    """The text a problem sends: its chat messages joined, else its prompt."""
    if problem.raw_messages:
        return "\n".join(message["content"] for message in problem.raw_messages)
    return problem.prompt


def _shuffled(problems: Sequence[BenchmarkProblem], seed: int | None) -> list[BenchmarkProblem]:
    """Each task's problems in a seeded random order (tasks keep their dataset order)."""
    if seed is None:
        return list(problems)
    groups: dict[str, list[BenchmarkProblem]] = {}
    for problem in problems:
        groups.setdefault(problem.task, []).append(problem)
    for task, members in groups.items():
        random.Random(seed ^ zlib.crc32(task.encode())).shuffle(members)
    return [problem for members in groups.values() for problem in members]


def select(problems: Sequence[BenchmarkProblem], environ: Mapping[str, str] | None = None,
           count_tokens: Callable[[str], int] | None = None) -> list[BenchmarkProblem]:
    """The configured subset: each task's problems in seeded random order (or dataset order), per-task and
    overall limits, capped generation, and only problems that fit the sequence limit (the same problems
    for every side). ``count_tokens`` (tests) counts a plain prompt; the margin is added to it."""
    environ = os.environ if environ is None else environ
    per_task, limit = _int(environ, "TRTMC_ACCURACY_PER_TASK"), _int(environ, "TRTMC_ACCURACY_LIMIT")
    max_new, token_limit = _int(environ, "TRTMC_ACCURACY_MAX_NEW_TOKENS"), _int(environ, "TRTMC_ACCURACY_TOKEN_LIMIT")
    measure = None
    if token_limit and count_tokens is not None:
        margin = _int(environ, "TRTMC_ACCURACY_TEMPLATE_MARGIN")
        margin = TEMPLATE_MARGIN if margin is None else margin
        measure = lambda problem: count_tokens(_prompt_text(problem)) + margin  # noqa: E731
    elif token_limit:
        measure = _token_counter(environ)
    seen: Counter[str] = Counter()
    selected = []
    for problem in _shuffled(problems, _int(environ, "TRTMC_ACCURACY_SEED")):
        if per_task and seen[problem.task] >= per_task:
            continue
        metadata = dict(problem.metadata or {})
        if max_new:
            metadata["generation_size"] = min(int(metadata.get("generation_size", max_new)), max_new)
        if token_limit and measure is not None:
            if measure(problem) + int(metadata.get("generation_size", 0)) > token_limit:
                continue
        seen[problem.task] += 1
        selected.append(problem.model_copy(update={"metadata": metadata}))
        if limit and len(selected) >= limit:
            break
    return selected


class _Configured:
    """The selection settings: the process environment, or a mapping the harness sets."""

    environ: Mapping[str, str] | None = None

    def _cache(self) -> dict[str, str]:
        cache = (os.environ if self.environ is None else self.environ).get("HF_DATASETS_CACHE")
        return {"cache_dir": cache} if cache else {}


class PinnedMMLU(_Configured, mmlu.MMLUBenchmark):
    async def load_problems(self, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]:
        from datasets import DatasetDict

        problems: list[BenchmarkProblem] = []
        for subject in self._resolve_subjects(tasks):
            # Only the few-shot (dev) and evaluation (test) splits: the full subject config also
            # prepares a ~100k-row auxiliary_train split.
            dev, test = await asyncio.to_thread(load_dataset, mmlu.DATASET_NAME, subject, split=["dev", "test"],
                                                revision=MMLU_REVISION, **self._cache())
            problems += await asyncio.to_thread(self._build_subject_problems, DatasetDict({"dev": dev, "test": test}),
                                                subject, n_shots, enable_cot)
        return select([problem if enable_cot else instructed(problem) for problem in problems], self.environ)


def instructed(problem: BenchmarkProblem) -> BenchmarkProblem:
    """An MMLU problem with MMLU_ANSWER_INSTRUCTION after lighteval's instruction (in the prompt and the first
    chat message alike) and MMLU_GENERATION_SIZE answer tokens."""
    marker = f"about {problem.task.replace('_', ' ')}.\n\n"

    def add(text: str) -> str:
        if marker not in text:
            raise ValueError(f"MMLU prompt without lighteval's instruction for {problem.task!r}")
        return text.replace(marker, f"about {problem.task.replace('_', ' ')}.{MMLU_ANSWER_INSTRUCTION}\n\n", 1)

    messages = [dict(message) for message in problem.raw_messages or []]
    if messages:
        messages[0]["content"] = add(messages[0]["content"])
    return problem.model_copy(update={"prompt": add(problem.prompt), "raw_messages": messages or problem.raw_messages,
                                      "metadata": {**(problem.metadata or {}), "generation_size": MMLU_GENERATION_SIZE}})


class PinnedGSM8K(_Configured, gsm8k.GSM8KBenchmark):
    async def load_problems(self, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]:
        dataset = await asyncio.to_thread(load_dataset, gsm8k.DATASET_NAME, gsm8k.DATASET_CONFIG, split="test",
                                          revision=GSM8K_REVISION, **self._cache())
        return select(await asyncio.to_thread(self._build_problems, dataset), self.environ)


class PinnedMath500(_Configured, math_500.Math500Benchmark):
    async def load_problems(self, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]:
        dataset = await asyncio.to_thread(load_dataset, math_500.DATASET_NAME, split="test", revision=MATH500_REVISION,
                                          **self._cache())
        return select(await asyncio.to_thread(self._build_problems, dataset), self.environ)


class Lambada(_Configured):
    """LAMBADA (OpenAI version, test split): each passage without its last word is the prompt, the
    last word the gold answer; a greedy continuation must start with it (lm-eval's greedy check)."""

    def __init__(self, run: Any = None, **kwargs: Any) -> None:
        self.run = run

    async def load_problems(self, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]:
        dataset = await asyncio.to_thread(load_dataset, "EleutherAI/lambada_openai", "default", split="test",
                                          revision=LAMBADA_REVISION, **self._cache())
        problems = []
        for row in dataset:
            context, word = row["text"].rsplit(" ", 1)
            problems.append(BenchmarkProblem(prompt=context, ground_truth=word, task="lambada",
                                             metadata={"generation_size": LAMBADA_GENERATION_SIZE},
                                             raw_messages=[{"role": "user", "content": context}]))
        return select(problems, self.environ)


PUNCTUATION = ".,;:!?\"'”’)]}"


class TinyStories(_Configured):
    """TinyStories (validation): each story without its last word is the prompt, that word (without
    punctuation) the gold answer; scored like LAMBADA (TRTMC_ACCURACY_LIMIT keeps the first stories
    that fit the sequence limit)."""

    def __init__(self, run: Any = None, **kwargs: Any) -> None:
        self.run = run

    async def load_problems(self, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]:
        dataset = await asyncio.to_thread(load_dataset, "roneneldan/TinyStories", split="validation",
                                          revision=TINYSTORIES_REVISION, **self._cache())
        problems = []
        for row in dataset:
            text = row["text"].strip()
            if " " not in text:
                continue
            context, word = text.rsplit(" ", 1)
            word = word.rstrip(PUNCTUATION)
            if word.isalpha():
                problems.append(BenchmarkProblem(prompt=context, ground_truth=word, task="tinystories",
                                                 metadata={"generation_size": LAMBADA_GENERATION_SIZE},
                                                 raw_messages=[{"role": "user", "content": context}]))
        return select(problems, self.environ)


class BartDenoise(_Configured):
    """BART's pre-training task on WikiText-103 test sentences (10-40 words): three consecutive words in
    the middle become ``<mask>``; the gold answer is the original sentence."""

    def __init__(self, run: Any = None, **kwargs: Any) -> None:
        self.run = run

    async def load_problems(self, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]:
        dataset = await asyncio.to_thread(load_dataset, "Salesforce/wikitext", "wikitext-103-raw-v1", split="test",
                                          revision=WIKITEXT_REVISION, **self._cache())
        problems = []
        for row in dataset:
            line = row["text"].strip()
            if not line or line.startswith("="):
                continue
            for sentence in line.split(" . "):
                words = sentence.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").split()
                if 10 <= len(words) <= 40 and len(problems) < BART_SENTENCES:
                    middle = len(words) // 2 - 1
                    masked = " ".join(words[:middle] + ["<mask>"] + words[middle + 3:]) + " ."
                    original = " ".join(words) + " ."
                    problems.append(BenchmarkProblem(prompt=masked, ground_truth=original, task="bart-denoise",
                                                     metadata={"generation_size": 96},
                                                     raw_messages=[{"role": "user", "content": masked}]))
        return select(problems, self.environ)


def _sentence(text: str) -> str:
    """Whitespace collapsed, and none before punctuation (WikiText writes ``film , television``)."""
    import re

    return re.sub(r"\s+([,.;:!?'])", r"\1", " ".join((text or "").split()))


class SentenceExactGrader(BaseGrader):
    """Correct when the response equals the gold sentence up to whitespace."""

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        return _sentence(response_text)

    async def grade(self, response_text: str, ground_truth: str, **kwargs: Any) -> GradingResult:
        answer, gold = self.extract_answer(response_text), _sentence(ground_truth)
        return GradingResult(correct=bool(answer) and answer == gold, unparsed=not answer, confidence=1.0,
                             reasoning="whitespace-normalized sentence", extracted_answer=answer,
                             ground_truth=gold)


class FirstWordGrader(BaseGrader):
    """Correct when the response's first word, without trailing punctuation, equals the gold word."""

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        words = (response_text or "").split()
        return words[0].rstrip(PUNCTUATION) if words else ""

    async def grade(self, response_text: str, ground_truth: str, **kwargs: Any) -> GradingResult:
        answer, gold = self.extract_answer(response_text), (ground_truth or "").strip()
        return GradingResult(correct=bool(answer) and answer == gold, unparsed=not answer, confidence=1.0,
                             reasoning="first word of the continuation", extracted_answer=answer, ground_truth=gold)


BENCHMARKS: dict[str, type] = {"trtmc_mmlu": PinnedMMLU, "trtmc_gsm8k": PinnedGSM8K, "trtmc_math500": PinnedMath500,
                               "trtmc_lambada": Lambada, "trtmc_tinystories": TinyStories,
                               "trtmc_bart_denoise": BartDenoise}


def _tokenizer_identity(environ: Mapping[str, str]) -> str | None:
    """The tokenizer the length filter uses, immutably: a local directory's file contents, else the commit of the
    cached Hugging Face snapshot its revision resolves to; "" without a tokenizer; None when it cannot be told
    (then nothing is cached)."""
    import hashlib

    name = environ.get("TRTMC_ACCURACY_TOKENIZER")
    if not name:
        return ""
    from pathlib import Path

    if Path(name).is_dir():
        digest = hashlib.sha256()
        for path in sorted(item for item in Path(name).rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(name)).encode() + b"\0" + path.read_bytes())
        return digest.hexdigest()
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(name, "tokenizer_config.json",
                                       revision=environ.get("TRTMC_ACCURACY_TOKENIZER_REVISION") or None)
    except Exception:  # noqa: BLE001 - an unknown identity disables the cache
        return None
    return Path(found).parent.name if isinstance(found, str) and Path(found).parent.parent.name == "snapshots" else None


def _selection_key(loader: Any, tasks: list[str] | None, n_shots: int, enable_cot: bool,
                   environ: Mapping[str, str]) -> str | None:
    """Everything a selection depends on: the benchmark and its arguments, every TRTMC_ACCURACY_* setting, the
    tokenizer's immutable identity, this module's source, and the AIPerf and Transformers versions (prompt format,
    tokenization); None when the tokenizer's identity is unknown."""
    import hashlib
    import importlib.metadata
    import json

    def version(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    tokenizer = _tokenizer_identity(environ)
    if tokenizer is None:
        return None
    settings = {key: value for key, value in environ.items()
                if key.startswith("TRTMC_ACCURACY_") and key != "TRTMC_ACCURACY_CACHE"}
    text = json.dumps({"benchmark": type(loader).__qualname__, "tasks": tasks, "n_shots": n_shots, "cot": enable_cot,
                       "settings": settings, "tokenizer": tokenizer,
                       "source": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                       "aiperf": version("aiperf"), "transformers": version("transformers")}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _cached(load: Callable) -> Callable:
    """``load_problems`` through TRTMC_ACCURACY_CACHE (when set): one selection per settings, shared by every
    process that loads it, written atomically."""
    import functools
    import json
    from pathlib import Path

    @functools.wraps(load)
    async def cached(self: Any, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]:
        import tempfile

        environ = os.environ if getattr(self, "environ", None) is None else self.environ
        if not environ.get("TRTMC_ACCURACY_CACHE"):
            return await load(self, tasks, n_shots, enable_cot)
        directory = Path(environ["TRTMC_ACCURACY_CACHE"])
        key = _selection_key(self, tasks, n_shots, enable_cot, environ)
        if key and (directory / f"{key}.json").is_file():
            return [BenchmarkProblem.model_validate(item) for item in json.loads((directory / f"{key}.json").read_text())]
        chosen = await load(self, tasks, n_shots, enable_cot)
        key = _selection_key(self, tasks, n_shots, enable_cot, environ)  # the tokenizer the load resolved
        if key:
            directory.mkdir(parents=True, exist_ok=True)
            handle, partial = tempfile.mkstemp(dir=directory, prefix=f"{key}.", suffix=".partial")
            with os.fdopen(handle, "w") as stream:
                stream.write(json.dumps([problem.model_dump(mode="json") for problem in chosen]))
            os.replace(partial, directory / f"{key}.json")
        return chosen

    return cached


for _benchmark in BENCHMARKS.values():
    _benchmark.load_problems = _cached(_benchmark.load_problems)


def problems(benchmark: str, tasks: list[str] | None, n_shots: int, environ: Mapping[str, str]) -> list[Any]:
    """The problems a run of ``benchmark`` sends under ``environ`` (the harness counts and labels them)."""
    loader = BENCHMARKS[benchmark](run=None)
    loader.environ = dict(environ)
    return asyncio.run(loader.load_problems(tasks, n_shots, False))
