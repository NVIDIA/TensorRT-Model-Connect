# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from trtmc_aiperf_qual import absolute
from trtmc_aiperf_qual.config import Environment


@pytest.fixture
def json_source(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    source = tmp_path / "asset.json"
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "<s>": 1, "hello": 2, "world": 3},
                                           unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tokenizer.save(str(source))
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"tokenizer_class": "RemoteTokenizer", "auto_map": {"AutoTokenizer": "remote.Code"},
                                  "bos_token": "<s>",
                                  "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}"}))
    calls = []

    def download(repo_id, filename, revision, **kwargs):
        calls.append((repo_id, filename, revision))
        return str(config if filename == "tokenizer_config.json" else source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(constants, "HF_HOME", str(tmp_path / "cache"))
    if hasattr(absolute, "_tokenizer_json_directory"):
        absolute._tokenizer_json_directory.cache_clear()
    model = {"candidate": {"checkpoint": "unused/candidate", "revision": "c" * 40,
                            "max_sequence_length": 20,
                            "tokenizer_json": {"repo": "org/tokenizer", "revision": "a" * 40}},
             "reference": {"trust_remote_code": True}}
    return model, source, calls


def test_json_source_is_shared_by_selection_fit_truncation_and_aiperf(json_source, tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from aiperf.accuracy.models import BenchmarkProblem
    from aiperf.common.tokenizer import Tokenizer
    from trtmc_aiperf_plugins.benchmarks import _token_counter

    model, asset, calls = json_source
    name, revision = absolute.tokenizer_source(model)
    assert Path(name).is_dir()
    assert revision is None
    assert (Path(name) / "tokenizer.json").read_bytes() == asset.read_bytes()
    assert json.loads((Path(name) / "tokenizer_config.json").read_text())["tokenizer_class"] == "PreTrainedTokenizerFast"
    configuration = json.loads((Path(name) / "tokenizer_config.json").read_text())
    assert "auto_map" not in configuration
    assert configuration["chat_template"] == "{{ bos_token }}{{ messages[0]['content'] }}"
    loaded = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    assert loaded("hello world")["input_ids"] == [1, 2, 3]
    assert absolute._candidate_tokenizer(model)("hello world")["input_ids"] == [1, 2, 3]
    sample = {"request": {"prompt": "hello world", "max_new_tokens": 16}}
    assert absolute._fitted(model, [sample], 1)[0]["request"]["max_new_tokens"] == 9
    assert absolute._head_truncated(model, [sample], 1)[0]["request"]["prompt"] == "hello"
    environment = absolute.selection_environment(Environment({"hf_datasets_cache": str(tmp_path)}), model,
                                                 {"endpoint": "chat"})
    assert environment["TRTMC_ACCURACY_TOKENIZER"] == name
    assert _token_counter(environment)(BenchmarkProblem(prompt="hello world", ground_truth="x", task="test")) == 11
    arguments = absolute._arguments(model, {"endpoint": "completions", "plugin": "gsm8k"},
                                     {"url": "http://localhost"}, 1, None)
    assert arguments[arguments.index("--tokenizer") + 1] == name
    assert "--tokenizer-revision" not in arguments
    # Note (Jiaxin Deng): run_aiperf clears inherited Hub offline flags before loading this local directory.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    client = Tokenizer.from_pretrained(name)
    assert client.encode("hello world") == [2, 3]
    assert absolute.tokenizer_source(model) == (name, None)
    assert calls.count(("org/tokenizer", "tokenizer.json", "a" * 40)) == 1


def test_concurrent_planners_publish_one_complete_tokenizer(json_source, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import huggingface_hub
    from transformers import AutoTokenizer

    model, asset, _ = json_source
    barrier = Barrier(4)

    def download(repo_id, filename, revision, **kwargs):
        if filename == "tokenizer.json":
            barrier.wait(timeout=10)
            return str(asset)
        return str(asset.parent / "tokenizer_config.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    def plan(_):
        name, _ = absolute.tokenizer_source(model)
        assert AutoTokenizer.from_pretrained(name)("hello world")["input_ids"] == [1, 2, 3]
        return name

    with ThreadPoolExecutor(max_workers=4) as pool:
        names = list(pool.map(plan, range(4)))
    assert len(set(names)) == 1


def test_publication_failure_is_not_hidden_by_candidate_fallback(json_source, monkeypatch):
    import errno

    model, _, _ = json_source

    def denied(*args):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(Path, "rename", denied)
    with pytest.raises(PermissionError, match="denied"):
        absolute.tokenizer_source(model)
