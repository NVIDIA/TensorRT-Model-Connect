# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run full 20-step denoising loop in Python using TRT engines.
Compare latent divergence between two prompts at each step."""
import struct
import json
import math
import numpy as np

BUNDLE = "/workspace/tensorrt-model-connect/engines/pixart_sigma_v6.trtfb"
MODEL_DIR = "/root/.cache/huggingface/hub/models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS/snapshots/e102b3591cc82e97071b8b4cb90d834d0c487207"

def load_bundle_sections(path):
    with open(path, "rb") as f:
        f.read(8)  # skip magic bytes
        json_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(json_len).decode())
        data_start = 16 + json_len
        sections = {}
        for name, info in header.get("sections", {}).items():
            if name.endswith("_plan"):
                f.seek(data_start + info["offset"])
                sections[name] = f.read(info["size"])
    return header, sections

print("Loading engines...")
import torch  # noqa: E402
from tensorrt_model_connect import trt_compat  # noqa: E402

trt = trt_compat.get_trt()

header, sections = load_bundle_sections(BUNDLE)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
t5_engine = runtime.deserialize_cuda_engine(sections["text_encoder_0_plan"])
dit_engine = runtime.deserialize_cuda_engine(sections["denoiser_plan"])

from transformers import AutoTokenizer  # noqa: E402
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")

def run_t5(input_ids_list, seq_len=120):
    ctx = t5_engine.create_execution_context()
    stream = torch.cuda.Stream()
    padded = input_ids_list[:seq_len] + [0] * max(0, seq_len - len(input_ids_list))
    mask = [1 if t != 0 else 0 for t in padded]
    ids_t = torch.tensor([padded], dtype=torch.int32, device="cuda")
    mask_t = torch.tensor([mask], dtype=torch.int32, device="cuda")
    out_t = torch.empty(1, seq_len, 4096, dtype=torch.float32, device="cuda")
    ctx.set_input_shape("input_ids", (1, seq_len))
    ctx.set_input_shape("attention_mask", (1, seq_len))
    ctx.set_tensor_address("input_ids", ids_t.data_ptr())
    ctx.set_tensor_address("attention_mask", mask_t.data_ptr())
    ctx.set_tensor_address("output0", out_t.data_ptr())
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out_t[0].cpu().numpy()

def run_dit(latent_fp32, text_emb, timestep, num_real, seq_len=120):
    ctx = dit_engine.create_execution_context()
    stream = torch.cuda.Stream()
    sample = torch.tensor(latent_fp32, dtype=torch.float16, device="cuda").reshape(1,4,128,128)
    text = torch.tensor(text_emb, dtype=torch.float16, device="cuda").reshape(1,seq_len,4096)
    ts = torch.tensor([timestep], dtype=torch.float16, device="cuda")
    mask = torch.zeros(1, seq_len, dtype=torch.float16, device="cuda")
    mask[0, :num_real] = 1.0
    out = torch.empty(1,8,128,128, dtype=torch.float16, device="cuda")
    ctx.set_input_shape("sample", (1,4,128,128))
    ctx.set_input_shape("encoder_hidden_states", (1,seq_len,4096))
    ctx.set_input_shape("timestep", (1,))
    ctx.set_input_shape("encoder_attention_mask", (1,seq_len))
    ctx.set_tensor_address("sample", sample.data_ptr())
    ctx.set_tensor_address("encoder_hidden_states", text.data_ptr())
    ctx.set_tensor_address("timestep", ts.data_ptr())
    ctx.set_tensor_address("encoder_attention_mask", mask.data_ptr())
    ctx.set_tensor_address("output0", out.data_ptr())
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out[0,:4].cpu().float().numpy().flatten()

# DPM-Solver++ scheduler
class DPMSolver:
    def __init__(self, num_steps, T=1000, beta_start=0.0001, beta_end=0.02):
        self.T = T
        cum = 1.0
        self.alpha_t = np.zeros(T)
        self.sigma_t = np.zeros(T)
        self.lambda_t = np.zeros(T)
        for i in range(T):
            beta = beta_start + i / (T-1) * (beta_end - beta_start)
            cum *= (1 - beta)
            self.alpha_t[i] = math.sqrt(cum)
            self.sigma_t[i] = math.sqrt(1 - cum)
            self.lambda_t[i] = math.log(self.alpha_t[i] / self.sigma_t[i])

        self.timesteps = np.zeros(num_steps)
        for i in range(1, num_steps+1):
            val = i / num_steps * (T - 1)
            self.timesteps[num_steps - i] = round(val)

        self.model_outputs = []
        self.lower_order_nums = 0

    def eps_to_x0(self, eps, x_s, t):
        ti = max(0, min(t, self.T-1))
        a, s = self.alpha_t[ti], self.sigma_t[ti]
        return (x_s - s * eps) / a

    def first_order_update(self, m0, sample, t_s0, t_t):
        i0 = max(0, min(t_s0, self.T-1))
        it = max(0, min(t_t, self.T-1))
        h = self.lambda_t[it] - self.lambda_t[i0]
        return (self.sigma_t[it]/self.sigma_t[i0]) * sample - self.alpha_t[it] * (np.exp(-h) - 1) * m0

    def second_order_update(self, m0, m1, sample, t_s0, t_s1, t_t):
        i0 = max(0, min(t_s0, self.T-1))
        i1 = max(0, min(t_s1, self.T-1))
        it = max(0, min(t_t, self.T-1))
        h = self.lambda_t[it] - self.lambda_t[i0]
        h_0 = self.lambda_t[i0] - self.lambda_t[i1]
        r0 = h_0 / h
        exp_neg_h = np.exp(-h)
        base = self.alpha_t[it] * (exp_neg_h - 1)
        d0 = m0
        d1 = (1/r0) * (m0 - m1)
        return (self.sigma_t[it]/self.sigma_t[i0]) * sample - base * d0 - 0.5 * base * d1

    def step(self, eps_pred, sample, step_idx, num_steps):
        si = step_idx
        t_s0 = int(round(self.timesteps[si]))
        t_t = int(round(self.timesteps[si+1])) if si+1 < len(self.timesteps) else 0

        x0 = self.eps_to_x0(eps_pred, sample, t_s0)
        self.model_outputs.append(x0)
        if len(self.model_outputs) > 2:
            self.model_outputs.pop(0)

        order = 2
        if self.lower_order_nums < 1:
            order = 1
        elif step_idx == num_steps - 1:
            order = 1

        if order == 1 or len(self.model_outputs) < 2:
            result = self.first_order_update(self.model_outputs[-1], sample, t_s0, t_t)
        else:
            t_s1 = int(round(self.timesteps[si-1]))
            result = self.second_order_update(
                self.model_outputs[-1], self.model_outputs[-2],
                sample, t_s0, t_s1, t_t)

        if self.lower_order_nums < 2:
            self.lower_order_nums += 1
        return result

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# Encode prompts
ids1 = tokenizer.encode("a photo of a dog chewing on a bone", add_special_tokens=True)
ids2 = tokenizer.encode("a red sports car on a highway", add_special_tokens=True)
ids_null = [1]  # Just EOS for null text

emb1 = run_t5(ids1)
emb2 = run_t5(ids2)
emb_null = run_t5(ids_null)

n1, n2, nn = len(ids1), len(ids2), 1

# Initialize latent
rng = np.random.default_rng(42)
latent_init = rng.standard_normal(4*128*128).astype(np.float32)

num_steps = 20
guidance = 4.5

# Run full loop for both prompts
sched1 = DPMSolver(num_steps)
lat1 = latent_init.copy()
sched2 = DPMSolver(num_steps)
lat2 = latent_init.copy()

header = "{:>4} {:>6} {:>10} {:>10} {:>10} {:>12}".format(
    "Step", "t", "cos(lat)", "cos(eps)", "cos(cfg)", "|lat1-lat2|")
print("\n" + header)
print("-" * len(header))

for step in range(num_steps):
    t = sched1.timesteps[step]

    eps1 = run_dit(lat1, emb1, t, n1)
    eps1_unc = run_dit(lat1, emb_null, t, nn)
    cfg1 = eps1_unc + guidance * (eps1 - eps1_unc)

    eps2 = run_dit(lat2, emb2, t, n2)
    eps2_unc = run_dit(lat2, emb_null, t, nn)
    cfg2 = eps2_unc + guidance * (eps2 - eps2_unc)

    cos_lat = cosine(lat1, lat2)
    cos_eps = cosine(eps1, eps2)
    cos_cfg = cosine(cfg1, cfg2)
    diff_norm = np.linalg.norm(lat1 - lat2)

    print("{:4d} {:6.0f} {:10.6f} {:10.6f} {:10.6f} {:12.4f}".format(
        step, t, cos_lat, cos_eps, cos_cfg, diff_norm))

    lat1 = sched1.step(cfg1, lat1, step, num_steps)
    lat2 = sched2.step(cfg2, lat2, step, num_steps)

print("\nFinal latent cosine: {:.6f}".format(cosine(lat1, lat2)))
print("Final latent diff norm: {:.4f}".format(np.linalg.norm(lat1 - lat2)))
