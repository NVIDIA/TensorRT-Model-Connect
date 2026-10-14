# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer component comparison against the revision-pinned liquid-audio oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt
import torch


class Runner:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.tensors = {}
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [name for name in self.names if name not in self.inputs]
        self.dtype = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
        }
        self.reset()

    def reset(self):
        for name in self.inputs:
            if name.startswith(("cache_", "conv_state_")) and name != "cache_write_indices":
                shape = tuple(self.engine.get_tensor_shape(name))
                self.tensors[name] = torch.zeros(
                    shape, dtype=self.dtype[self.engine.get_tensor_dtype(name)], device="cuda"
                )
        self.position = 0

    def run(self, **inputs):
        if "position_id" in self.inputs:
            inputs |= {
                "position_id": torch.tensor([self.position], dtype=torch.int32, device="cuda"),
                "cache_write_indices": torch.tensor(
                    [self.position], dtype=torch.int32, device="cuda"
                ),
                "key_value_lengths": torch.tensor(
                    [self.position + 1], dtype=torch.int32, device="cuda"
                ),
            }
        if "decode_phase" in self.inputs and "decode_phase" not in inputs:
            inputs["decode_phase"] = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
        for name, tensor in inputs.items():
            self.tensors[name] = tensor.to(
                device="cuda", dtype=self.dtype[self.engine.get_tensor_dtype(name)]
            ).contiguous()
        for name in self.inputs:
            tensor = self.tensors[name]
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        for name in self.outputs:
            if name.startswith(("present_k_", "present_v_")):
                self.tensors[name] = self.tensors[name.replace("present_", "cache_")]
            else:
                self.tensors[name] = torch.empty(
                    tuple(self.context.get_tensor_shape(name)),
                    dtype=self.dtype[self.engine.get_tensor_dtype(name)],
                    device="cuda",
                )
            self.context.set_tensor_address(name, self.tensors[name].data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT component execution failed")
        torch.cuda.synchronize()
        for name in self.outputs:
            if name.startswith("present_conv_"):
                self.tensors[name.replace("present_conv_", "conv_state_")].copy_(self.tensors[name])
        if "position_id" in self.inputs:
            self.position += 1
        return {
            name: self.tensors[name] for name in self.outputs if not name.startswith("present_")
        }


def metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    delta = a - b
    return {
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "nrmse": (delta.square().mean().sqrt() / b.square().mean().sqrt()).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


@torch.inference_mode()
def main():
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint, engines = Path(args.checkpoint), Path(args.engines)
    torch.manual_seed(0)
    model = LFM2AudioModel.from_pretrained(checkpoint).eval()
    proc = LFM2AudioProcessor.from_pretrained(checkpoint).eval()
    report = {}
    encoder = Runner(engines / "encoder.plan")
    for frames in (9, 101, 801):
        mel = torch.randn(1, 128, frames, device="cuda", dtype=torch.bfloat16)
        ref, length = model.conformer(mel, torch.tensor([frames], device="cuda"))
        adapted = model.audio_adapter(ref.transpose(1, 2))
        extra = (
            {"valid_frames": torch.tensor([(frames + 7) // 8], device="cuda", dtype=torch.int32)}
            if "valid_frames" in encoder.inputs
            else {}
        )
        out = encoder.run(mel=mel.transpose(1, 2)[:, None].float(), **extra)
        report[f"encoder_{frames}"] = metrics(out["encoder_features"], ref.transpose(1, 2))
        report[f"adapter_{frames}"] = metrics(out["audio_embeddings"], adapted)
        print(frames, report[f"encoder_{frames}"], report[f"adapter_{frames}"], flush=True)
    if "valid_frames" in encoder.inputs:
        lengths = torch.tensor([101, 801], device="cuda")
        mel = torch.randn(2, 128, 801, device="cuda", dtype=torch.bfloat16)
        mel[0, :, 101:] = 0
        reference, encoded_lengths = model.conformer(mel, lengths)
        mask = torch.arange(reference.shape[-1], device="cuda")[None] < encoded_lengths[:, None]
        adapted = model.audio_adapter(reference.transpose(1, 2)[mask])
        offset = 0
        for index, frames in enumerate((101, 801)):
            count = (frames + 7) // 8
            actual = encoder.run(
                mel=mel[index : index + 1].transpose(1, 2)[:, None].float(),
                valid_frames=torch.tensor([count], device="cuda", dtype=torch.int32),
            )
            report[f"padded_segment_{frames}"] = metrics(
                actual["audio_embeddings"][0, :count], adapted[offset : offset + count]
            )
            offset += count
        alone, _ = model.conformer(mel[:1, :, :101], lengths[:1])
        report["reference_padding_changes_segment"] = metrics(
            alone[0], reference[0, :, : alone.shape[-1]]
        )
        print(
            "padded audio",
            {
                key: value
                for key, value in report.items()
                if "padded_segment" in key or "padding_changes" in key
            },
            flush=True,
        )
    del encoder
    chat = ChatState(proc)
    chat.new_turn("system")
    chat.add_text("Perform TTS. Use the US male voice.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_text("Hello there.")
    chat.end_turn()
    chat.new_turn("assistant")
    embeddings = model._prefill(**chat)
    reference = model.lfm(inputs_embeds=embeddings, use_cache=True)
    backbone = Runner(engines / "backbone.plan")
    for row in embeddings[0]:
        out = backbone.run(input_embedding=row[None].float())
    hidden = reference.last_hidden_state[0, -1]
    cached = None
    for row in embeddings[0]:
        sequential = model.lfm(
            inputs_embeds=row[None, None], past_key_values=cached, use_cache=True
        )
        cached = sequential.past_key_values
    report["reference_full_vs_step"] = metrics(sequential.last_hidden_state[0, -1], hidden)
    report["backbone_vs_step"] = metrics(out["hidden_state"], sequential.last_hidden_state[0, -1])
    print(
        "reference sequential",
        report["reference_full_vs_step"],
        report["backbone_vs_step"],
        flush=True,
    )
    ref_logits = torch.nn.functional.linear(hidden, model.lfm.embed_tokens.weight)
    report["backbone_hidden"] = metrics(out["hidden_state"], hidden)
    report["backbone_logits"] = metrics(out["logits"], ref_logits)
    report["backbone_top1"] = [out["logits"].argmax().item(), ref_logits.argmax().item()]
    print("backbone", report["backbone_hidden"], report["backbone_top1"], flush=True)
    del backbone
    depth = Runner(engines / "depthformer.plan")
    depth_inputs = model.depth_linear(hidden).reshape(8, 1024)
    previous = torch.zeros(1024, device="cuda", dtype=torch.bfloat16)
    cache = None
    codes = []
    for i in range(8):
        ref, cache = model.depthformer.forward_cached(
            (depth_inputs[i] + previous)[None, None, :], cache
        )
        ref_logits = model.depth_embeddings[i].get_logits(ref.squeeze())
        out = depth.run(
            input_embedding=hidden[None].float(), previous_embedding=previous[None].float()
        )
        report[f"depth_{i}"] = metrics(out["logits"], ref_logits)
        report[f"depth_top1_{i}"] = [out["logits"].argmax().item(), ref_logits.argmax().item()]
        token = ref_logits.argmax(keepdim=True)
        previous = model.depth_embeddings[i](token).squeeze()
        codes.append(token.item())
    print("depth", {k: v for k, v in report.items() if k.startswith("depth")}, flush=True)
    del depth
    detok = proc.audio_detokenizer
    codes = torch.randint(0, 2048, (1, 8, 6), device="cuda")
    repeated = detok.emb(codes).repeat_interleave(6, dim=1)
    pos = torch.arange(repeated.shape[1], device="cuda")
    delta = pos - pos[:, None]
    mask = ((delta <= 0) & (delta > -30))[None, None]
    reference = detok.lfm(
        inputs_embeds=repeated, attention_mask=mask, use_cache=False
    ).last_hidden_state
    spectra = detok.lin(reference)
    runner = Runner(engines / "detokenizer.plan")
    outputs = []
    for row in repeated[0]:
        outputs.append(runner.run(input_embedding=row[None])["logits"].clone())
    actual = torch.cat(outputs)
    report["detokenizer_spectrum"] = metrics(actual, spectra[0])
    print("detokenizer", report["detokenizer_spectrum"], flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
