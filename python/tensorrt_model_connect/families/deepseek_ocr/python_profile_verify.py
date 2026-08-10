# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the pinned DeepSeek-OCR-2 reference on patched Transformers."""

from importlib.metadata import version
from types import SimpleNamespace

import torch
import transformers

from tensorrt_model_connect.deepseek_ocr_reference_compat import (
    assert_deepseek_ocr_eager_attention,
    install_deepseek_ocr_transformers_compat,
)


assert version("huggingface-hub") == "1.5.0"
assert version("tokenizers") == "0.22.2"
assert transformers.__version__ == "5.5.4"

compat = install_deepseek_ocr_transformers_compat()
from transformers.models.llama.modeling_llama import (  # noqa: E402
    LlamaAttention,
    LlamaFlashAttention2,
)

assert LlamaAttention is compat.eager_attention_type
assert LlamaFlashAttention2 is compat.flash_attention_type

from transformers.configuration_utils import PreTrainedConfig  # noqa: E402


class LegacyConfig(PreTrainedConfig):
    def __init__(self, inherited_default=17, **kwargs):
        self.inherited_default = inherited_default
        super().__init__(**kwargs)


class LegacyConfigChild(LegacyConfig):
    pass


legacy_config = LegacyConfigChild(pad_token_id=None, bos_token_id=0, eos_token_id=1)
assert legacy_config.inherited_default == 17
assert legacy_config.pad_token_id is None
assert legacy_config.bos_token_id == 0
assert legacy_config.eos_token_id == 1

from transformers import Qwen2Config  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model  # noqa: E402


qwen_config = Qwen2Config(num_hidden_layers=3, _attn_implementation="sdpa")
assert qwen_config.num_hidden_layers == 3
assert qwen_config.layer_types == ["full_attention"] * 3
assert qwen_config._attn_implementation == "sdpa"


class CustomQwen2ModelInner(Qwen2Model):
    def forward(self, *, inputs_embeds, token_type_ids, attention_mask=None, **kwargs):
        self._current_token_type_ids = token_type_ids
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values,
        output_attentions,
    ):
        del attention_mask, cache_position, past_key_values, output_attentions
        self._legacy_mask_called = True
        batch_size, sequence_length = input_tensor.shape[:2]
        return torch.zeros(
            (batch_size, 1, sequence_length, sequence_length),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )


vision_config = Qwen2Config(
    hidden_size=8,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    intermediate_size=16,
    max_position_embeddings=16,
    vocab_size=16,
    _attn_implementation="sdpa",
    use_cache=False,
)
vision_model = CustomQwen2ModelInner(vision_config).eval()
CustomQwen2ModelInner.__module__ = "transformers_modules.deepseek_ocr.deepencoderv2"
vision_inputs = torch.randn(1, 4, vision_config.hidden_size)
vision_token_types = torch.zeros((1, 4), dtype=torch.long)
vision_output = vision_model(
    inputs_embeds=vision_inputs,
    token_type_ids=vision_token_types,
    use_cache=False,
)
assert vision_output.last_hidden_state.shape == vision_inputs.shape
assert vision_model._legacy_mask_called

config = SimpleNamespace(
    _attn_implementation="eager",
    attention_bias=False,
    attention_dropout=0.0,
    hidden_size=8,
    max_position_embeddings=16,
    num_attention_heads=2,
    num_hidden_layers=1,
    num_key_value_heads=2,
    pretraining_tp=1,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    rope_theta=10000.0,
    use_mla=False,
)
torch.manual_seed(7)
attention = compat.eager_attention_type(config, layer_idx=0).eval().to(torch.bfloat16)
cache = compat.dynamic_cache_type.from_legacy_cache()
hidden_states = torch.randn(1, 3, config.hidden_size).to(torch.bfloat16)
position_ids = torch.arange(3).unsqueeze(0)
output, weights, returned_cache = attention(
    hidden_states,
    position_ids=position_ids,
    past_key_value=cache,
    use_cache=True,
)
assert output.shape == hidden_states.shape
expected_first_token = torch.tensor(
    [
        -0.123046875,
        0.06640625,
        0.0194091796875,
        -0.02880859375,
        0.0279541015625,
        0.058349609375,
        -0.1923828125,
        0.0439453125,
    ],
    dtype=torch.float32,
)
assert torch.equal(output[0, 0].float(), expected_first_token)
assert weights is None
assert returned_cache is cache
assert cache.seen_tokens == 3
assert cache.get_usable_length(1) == 3
assert cache.get_max_length() is None
legacy_cache = cache.to_legacy_cache()
assert len(legacy_cache) == 1
assert compat.dynamic_cache_type.from_legacy_cache(legacy_cache).seen_tokens == 3


class TinyEagerModel(torch.nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.config = config
        self.attention = attention_module


assert assert_deepseek_ocr_eager_attention(TinyEagerModel(attention), compat) == 1
flash_model = TinyEagerModel(compat.flash_attention_type(config, layer_idx=0))
try:
    assert_deepseek_ocr_eager_attention(flash_model, compat)
except RuntimeError as error:
    assert "unsupported flash-attention" in str(error)
else:
    raise AssertionError("flash-attention compatibility selection was accepted")

print("transformers=5.5.4 deepseek_ocr_eager_compat=ok")
