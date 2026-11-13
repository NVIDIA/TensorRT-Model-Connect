# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from families.mt5.config import ModelConfig
from tensorrt_model_connect.model_support import ModelMetadata,resolve_family


def test_mt5_has_a_distinct_owner():
    family,support = resolve_family(ModelMetadata({'model_type':'mt5','architectures':['MT5ForConditionalGeneration']},{}))
    assert family == 'mt5'
    assert support.tasks == ('text_generation',)


def test_attention_width_is_independent_of_hidden_width(tmp_path):
    raw = {'model_type':'mt5','feed_forward_proj':'gated-gelu','d_model':512,'d_ff':1024,
           'd_kv':64,'num_heads':6,'num_layers':8,'vocab_size':250112,'tie_word_embeddings':False}
    (tmp_path/'config.json').write_text(json.dumps(raw))
    cfg = ModelConfig.from_dir(tmp_path)
    assert cfg.hidden_size == 512
    assert cfg.num_attention_heads*cfg.raw['d_kv'] == 384
    raw['tie_word_embeddings'] = True
    (tmp_path/'config.json').write_text(json.dumps(raw))
    with pytest.raises(NotImplementedError,match='independent'):
        ModelConfig.from_dir(tmp_path)
