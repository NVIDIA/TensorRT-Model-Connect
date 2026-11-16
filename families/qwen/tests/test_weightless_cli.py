# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen owns its stripping option; shared build requests remain unchanged."""

import json
import pytest

from families.qwen import cli
from families.qwen.build_request import BuildRequest
from tensorrt_model_connect import family_cli, build_cli


def test_qwen_help_exposes_stripping_without_loading_tensorrt(capsys):
    with pytest.raises(SystemExit) as stopped:
        family_cli.main(['qwen', 'build', '--help'])
    assert stopped.value.code == 0
    assert '--strip-weights' in capsys.readouterr().out


def test_owner_cli_passes_the_typed_flag(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, 'build', lambda **values: calls.append(values) or 0)
    assert family_cli.main(['qwen', 'build', 'Qwen/Qwen3-0.6B',
                            '-o', str(tmp_path / 'model.bundle'), '--strip-weights']) == 0
    assert calls[0]['strip_weights'] is True
    assert calls[0]['output'] == tmp_path / 'model.bundle'


def test_shared_cli_rejects_the_qwen_only_option(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        build_cli.main(['build', 'Qwen/Qwen3-0.6B', '-o', str(tmp_path / 'model.bundle'),
                        '--strip-weights'])
    assert stopped.value.code == 2


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_owned_request_requires_a_boolean(tmp_path, value):
    with pytest.raises(ValueError, match='strip_weights must be a bool'):
        BuildRequest(tmp_path, tmp_path / 'model.bundle', 'qwen', 'text_continuation',
                     'bf16', strip_weights=value)


def test_rtx_is_rejected_before_backend_import(monkeypatch, tmp_path):
    def unexpected(_):
        raise AssertionError('must reject before loading the backend')
    monkeypatch.setattr(cli, 'select_backend', unexpected)
    request = BuildRequest(tmp_path, tmp_path / 'model.bundle', 'qwen',
                           'text_continuation', 'bf16', backend='trt_rtx', strip_weights=True)
    with pytest.raises(ValueError, match='backend=trt'):
        cli.build_bundle(request)
    assert not request.output_path.exists()


@pytest.mark.parametrize('changes', [{'precision': 'fp32'}, {'quantization': 'fp8', 'precision': 'fp16'},
                                     {'tensor_parallel_size': 2}, {'task': 'embedding'},
                                     {'context_parallel_size': 2}])
def test_unsupported_strip_modes_fail_before_loading_weights(monkeypatch, tmp_path, changes):
    from families.qwen import model
    (tmp_path / 'config.json').write_text(json.dumps({
        'model_type': 'qwen3', 'vocab_size': 32, 'hidden_size': 16,
        'intermediate_size': 32, 'num_hidden_layers': 1,
        'num_attention_heads': 2, 'num_key_value_heads': 2, 'max_position_embeddings': 256,
    }))
    def unexpected(*args, **kwargs):
        raise AssertionError('must reject before reading weights')
    monkeypatch.setattr(model._QwenModel, 'load_weights', unexpected)
    options = {'model_dir': tmp_path, 'output_path': tmp_path / 'model.bundle', 'family': 'qwen',
               'task': 'text_continuation', 'precision': 'bf16', 'strip_weights': True, **changes}
    with pytest.raises((ValueError, NotImplementedError)):
        cli.build_bundle(BuildRequest(**options))
    assert not options['output_path'].exists()
