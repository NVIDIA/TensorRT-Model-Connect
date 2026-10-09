# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A static ConvBERT engine preserves the native convolution boundary of short inputs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_short_input_keeps_native_convolution_boundary(tmp_path):
    from transformers import ConvBertConfig, ConvBertModel

    from families.convbert.builder import build_convbert_encoder_engine
    from families.convbert.config import ModelConfig
    from families.convbert.model import _ConvBertModel

    torch.manual_seed(7)
    hf_config = ConvBertConfig(vocab_size=32, hidden_size=16, embedding_size=16,
                               intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                               max_position_embeddings=16, conv_kernel_size=3,
                               hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    native = ConvBertModel(hf_config).eval()
    torch.save(native.state_dict(), tmp_path / "pytorch_model.bin")
    config = ModelConfig.from_json(hf_config.to_json_string())
    weights = _ConvBertModel().load_weights(str(tmp_path), config)
    plan = build_convbert_encoder_engine(config, weights, 16, precision="fp32")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    native = native.cuda()
    for length in (3, 7, 16):
        ids = torch.zeros(16, dtype=torch.int32, device="cuda")
        ids[:length] = torch.arange(1, length + 1, device="cuda")
        mask = torch.zeros_like(ids)
        mask[:length] = 1
        output = torch.empty((16, 16), dtype=torch.float32, device="cuda")
        for name, value in (("input_ids", ids), ("attention_mask", mask), ("hidden_states", output)):
            context.set_tensor_address(name, value.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        with torch.no_grad():
            expected = native(input_ids=ids[:length].long().unsqueeze(0)).last_hidden_state[0]
        torch.cuda.synchronize()
        np.testing.assert_allclose(output[:length].cpu().numpy(), expected.cpu().numpy(), atol=2e-5, rtol=2e-5)
