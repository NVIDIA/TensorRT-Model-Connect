# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 and BF16 TensorRT graphs against transformers on a seeded tiny random PaliGemma.

The decoder is driven the way the C++ runtime drives it: host-assembled
embeddings, 1-based RoPE positions, a bidirectional prompt block, one cache row
per token. In FP32, greedy tokens must match ``generate`` exactly and every
step's logits must agree to FP32 rounding. In BF16 (a BF16 checkpoint, like the
released one) the engines are compared with the transformers BF16 model to
BF16 rounding, the decoder teacher-forced on the reference's greedy tokens: the
reference rounds attention scores and the RoPE tables to BF16 where the engines
keep FP32, so its near ties may break the other way.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
transformers = pytest.importorskip("transformers")

from families.paligemma.checkpoint import CheckpointWeights  # noqa: E402
from families.paligemma.config import ModelConfig  # noqa: E402
from families.paligemma.graph import build_decoder, build_vision  # noqa: E402

pytestmark = pytest.mark.trt

CAPACITY = 24
IMAGE_TOKEN = 127
_TORCH_TYPES = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16}
# torch.testing's relative tolerance for bfloat16, applied at the output's scale.
_BF16_RTOL = 1.6e-2


@pytest.fixture(scope="module")
def tiny_model():
    if not torch.cuda.is_available():
        pytest.skip("TensorRT graph parity requires CUDA")
    config = transformers.PaliGemmaConfig(
        vision_config={"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
                       "num_attention_heads": 4, "image_size": 28, "patch_size": 14,
                       "vision_use_head": False},
        text_config={"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
                     "vocab_size": 128},
        image_token_index=IMAGE_TOKEN, projection_dim=64, hidden_size=64)
    torch.manual_seed(0)
    config._attn_implementation = "eager"
    model = transformers.PaliGemmaForConditionalGeneration(config).eval()
    with torch.no_grad():
        # Gemma norms scale by (1 + weight) and initialize to zero; make them matter.
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.normal_(0.0, 0.2)
    return model


@pytest.fixture(scope="module", params=["fp32", "bf16"])
def tiny(request, tiny_model, tmp_path_factory):
    precision = request.param
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = copy.deepcopy(tiny_model).to({"fp32": torch.float32, "bf16": torch.bfloat16}[precision])
    directory = tmp_path_factory.mktemp(f"tiny_paligemma_{precision}")
    model.save_pretrained(directory)
    model_config = ModelConfig.from_dir(directory)
    weights = CheckpointWeights(directory)
    plans = (build_vision(model_config, weights, precision=precision),
             build_decoder(model_config, weights, CAPACITY, precision=precision))
    yield precision, model.to("cuda"), model_config, weights, plans
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


class _Engine:
    def __init__(self, plan):
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()

    def __call__(self, inputs):
        for name, tensor in inputs.items():
            assert tensor.dtype == _TORCH_TYPES[self.engine.get_tensor_dtype(name)], name
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                outputs[name] = torch.empty(tuple(self.context.get_tensor_shape(name)),
                                            dtype=_TORCH_TYPES[self.engine.get_tensor_dtype(name)],
                                            device="cuda")
                self.context.set_tensor_address(name, outputs[name].data_ptr())
        stream = torch.cuda.current_stream()
        assert self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


def _pixels(seed, dtype):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, 28, 28, generator=generator).to("cuda", dtype)


def _assert_close(precision, actual, expected, fp32_tolerance):
    assert actual.dtype == expected.dtype
    if precision == "fp32":
        torch.testing.assert_close(actual, expected, rtol=fp32_tolerance, atol=fp32_tolerance)
    else:
        scale = expected.float().abs().max().item()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=_BF16_RTOL,
                                   atol=_BF16_RTOL * scale)


def test_vision_matches_projected_scaled_features(tiny):
    precision, model, _, _, (vision_plan, _) = tiny
    pixels = _pixels(1, model.dtype)
    with torch.no_grad():
        expected = model.model.get_image_features(pixels).pooler_output[0]
    actual = _Engine(vision_plan)({"pixel_values": pixels.contiguous()})["image_features"]
    _assert_close(precision, actual, expected, 1e-5)


def _rope(config, positions):
    text = config.text
    exponent = np.arange(0, text.head_dim, 2, dtype=np.float32) / np.float32(text.head_dim)
    inverse = np.float32(1.0) / np.power(np.float32(text.rope_theta), exponent, dtype=np.float32)
    phase = positions.astype(np.float32)[:, None] * inverse[None, :]
    phase = np.concatenate([phase, phase], axis=1)
    return (torch.from_numpy(np.cos(phase)).cuda(), torch.from_numpy(np.sin(phase)).cuda())


def test_greedy_generation_matches_generate(tiny):
    precision, model, config, weights, (vision_plan, decoder_plan) = tiny
    text = config.text
    prompt = [IMAGE_TOKEN] * config.num_image_tokens + [2, 17, 33, 64, 8, 108 % 128]
    pixels = _pixels(2, model.dtype)
    steps = 6
    input_ids = torch.tensor([prompt]).cuda()
    # No EOS: every step is compared, and no logits processor alters the argmax.
    generation = transformers.GenerationConfig(do_sample=False, max_new_tokens=steps,
                                               eos_token_id=None, pad_token_id=0, bos_token_id=2)
    with torch.no_grad():
        reference = model.generate(
            input_ids=input_ids, pixel_values=pixels, attention_mask=torch.ones_like(input_ids),
            token_type_ids=torch.zeros_like(input_ids), generation_config=generation,
            output_logits=True, return_dict_in_generate=True)
    expected_tokens = reference.sequences[0, len(prompt):].tolist()

    table = torch.from_numpy(weights["text.embed_tokens.weight"]).to("cuda", model.dtype)
    features = _Engine(vision_plan)({"pixel_values": pixels.contiguous()})["image_features"]
    rows = table[torch.tensor(prompt).cuda()].clone()
    rows[:config.num_image_tokens] = features
    decoder = _Engine(decoder_plan)
    caches = {f"cache_{kind}_{layer}": torch.zeros(text.num_key_value_heads, CAPACITY,
                                                   text.head_dim, dtype=model.dtype, device="cuda")
              for layer in range(text.num_hidden_layers) for kind in ("k", "v")}
    offset, tokens = 0, []
    for step in range(steps):
        count = rows.shape[0]
        cos, sin = _rope(config, np.arange(offset, offset + count) + 1)
        mask = torch.full((1, count, CAPACITY + count), torch.finfo(torch.float32).min,
                          device="cuda")
        mask[:, :, :offset] = 0
        mask[:, :, CAPACITY:] = 0
        outputs = decoder({"inputs_embeds": rows.contiguous(), "rope_cos": cos, "rope_sin": sin,
                           "attention_mask": mask, **caches})
        _assert_close(precision, outputs["logits"][0], reference.logits[step][0].float(), 1e-4)
        for name, cache in caches.items():
            cache[:, offset:offset + count] = outputs[name.replace("cache_", "present_")]
        tokens.append(int(outputs["logits"][0].argmax()))
        offset += count
        # BF16 continues with the reference's token; its logits bound the argmax up to near ties.
        rows = table[torch.tensor([tokens[-1] if precision == "fp32" else expected_tokens[step]]).cuda()]
    if precision == "fp32":
        assert tokens == expected_tokens
