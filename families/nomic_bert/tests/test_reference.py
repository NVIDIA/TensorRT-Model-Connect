# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_reference_loads_local_weights_without_external_code_cache(tmp_path):
    if any(importlib.util.find_spec(name) is None for name in ("torch", "einops")):
        pytest.skip("publisher reference requires the family PyTorch/einops dependencies")
    cache = tmp_path / "empty-cache"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env={
            **os.environ,
            "HF_HOME": str(cache),
            "HF_HUB_CACHE": str(cache / "hub"),
            "HF_MODULES_CACHE": str(cache / "modules"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not cache.exists()


def _exercise_reference(tmp_path):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer

    from families.nomic_bert.tests.reference.configuration_hf_nomic_bert import NomicBertConfig
    from families.nomic_bert.tests.reference.modeling_hf_nomic_bert import NomicBertModel
    from families.nomic_bert.tests.test_e2e import _load_reference

    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    config = NomicBertConfig(
        vocab_size=16,
        n_embd=24,
        n_layer=1,
        n_head=4,
        n_inner=32,
        activation_function="swiglu",
        rotary_emb_fraction=1.0,
        rotary_emb_base=1000,
        qkv_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=None,
        eos_token_id=None,
    )
    config.auto_map = {
        "AutoConfig": "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
        "AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel",
    }
    config.save_pretrained(model_dir)
    original = NomicBertModel(config, add_pooling_layer=False).eval()
    save_file(original.state_dict(), model_dir / "model.safetensors")
    tokenizer = Tokenizer(WordPiece({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.normalizer = BertNormalizer()
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer", "do_lower_case": True})
    )
    actual_tokenizer, reloaded = _load_reference(model_dir)
    assert actual_tokenizer("HELLO world", add_special_tokens=False)["input_ids"] == [1, 2]
    inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    with torch.inference_mode():
        expected = original(input_ids=inputs, attention_mask=mask).last_hidden_state
        actual = reloaded(input_ids=inputs, attention_mask=mask).last_hidden_state
    assert actual.shape == (2, 4, 24)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    _exercise_reference(Path(sys.argv[1]))
