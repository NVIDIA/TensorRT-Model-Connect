# SPDX-License-Identifier: Apache-2.0
"""PyTorch (HF Diffusers MiniMaxH3Transformer3DModel) reference for the MiniMax-H3 DiT.

Replays runtime-dumped step-0 denoiser inputs (MINIMAX_H3_DUMP_DIR layout), optionally rolls a
reference Euler trajectory (same scheduler math as runtime/pipeline.cpp) and writes the inputs of
selected steps so the TRT-RTX plans can be replayed on exactly the same tensors.

  ref_dit.py traj  --ckpt C --dump D --out O --save-steps 0,24,48 [--dtype bf16]
  ref_dit.py eval  --ckpt C --steps-dir O --dtype fp32|bf16 [--vscale 256] [--devices 0,1]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

H, W_AUDIO = 5376, 32


def load_dump(d):
    f = lambda n, dt: np.fromfile(os.path.join(d, n), dtype=dt)
    adaln = f("adaln_indices.i32", np.int32)
    tsi = f("timestep_indices.i32", np.int32)
    return dict(
        video=f("video_hidden_states.f32", np.float32).reshape(-1, 96),
        audio=f("audio_hidden_states.f32", np.float32).reshape(-1, 32),
        text=f("encoder_hidden_states.f32", np.float32).reshape(-1, 5120),
        pos=f("position_ids.f32", np.float32).reshape(-1, 3),
        adaln=adaln, tsi=tsi)


def schedule(grid, shift):
    sig = []
    for i in range(grid):
        base = np.float32(1.0 - i / (grid - 1))
        s = np.float32(shift) * base / (np.float32(1.0) + (np.float32(shift) - np.float32(1.0)) * base)
        if not sig or s != sig[-1]:
            sig.append(s)
    ts = [np.float32(1.0) - sig[i] for i in range(len(sig) - 1)]
    return sig, ts


def euler(x, v, t, sig, sig_next):
    sigma_t = 1.0 - t
    ratio = sig_next / sig
    den = x + sigma_t * v
    return ratio * x + (1.0 - ratio) * den


def build_model(ckpt, dtype, vscale, devices):
    from diffusers import MiniMaxH3Transformer3DModel
    kw = {}
    if len(devices) > 1:
        kw = dict(device_map="auto", max_memory={i: "90GiB" for i in devices})
    m = MiniMaxH3Transformer3DModel.from_pretrained(os.path.join(ckpt, "transformer"), torch_dtype=dtype, **kw)
    if len(devices) == 1:
        m = m.to(f"cuda:{devices[0]}")
    m.eval()
    if vscale != 1.0:  # exact power-of-two V fold, as the TRT builder does
        with torch.no_grad():
            for b in m.transformer_blocks:
                b.attn.to_v.weight.mul_(1.0 / vscale)
                b.attn.to_out[0].weight.mul_(vscale)
    return m


def forward(m, dev, inp, tv, ta):
    tags = (inp["adaln"] - 3 * inp["tsi"])
    t = lambda a, dt=None: torch.as_tensor(a, device=dev).to(dt) if dt else torch.as_tensor(a, device=dev)
    vi = t(np.flatnonzero(tags == 0).astype(np.int64)); ti = t(np.flatnonzero(tags == 1).astype(np.int64))
    ai = t(np.flatnonzero(tags == 2).astype(np.int64))
    with torch.no_grad():
        v, a = m(hidden_states=t(inp["video"])[None], audio_hidden_states=t(inp["audio"])[None],
                 encoder_hidden_states=t(inp["text"])[None],
                 timestep=torch.tensor([tv, ta], dtype=torch.float32, device=dev),
                 timestep_indices=t(inp["tsi"].astype(np.int64)), token_tags=t(tags.astype(np.int64)),
                 position_ids=t(inp["pos"]), video_indices=vi, audio_indices=ai, text_indices=ti,
                 return_dict=False)
    return v[0].float().cpu().numpy(), a[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--ckpt", required=True); ap.add_argument("--dump")
    ap.add_argument("--out"); ap.add_argument("--steps-dir"); ap.add_argument("--save-steps", default="0")
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--vscale", type=float, default=1.0)
    ap.add_argument("--devices", default="0"); ap.add_argument("--tag", default=None)
    ap.add_argument("--grid", type=int, default=50)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = dict(bf16=torch.bfloat16, fp32=torch.float32)[a.dtype]
    devices = [int(x) for x in a.devices.split(",")]
    dev = f"cuda:{devices[0]}"
    vsig, vts = schedule(a.grid, 12.0); asig, ats = schedule(a.grid, 3.0)
    m = build_model(a.ckpt, dtype, a.vscale, devices)
    if a.mode == "traj":
        save = {int(s) for s in a.save_steps.split(",")}
        os.makedirs(a.out, exist_ok=True)
        inp = load_dump(a.dump)
        n = len(vts)
        for k in range(n):
            t0 = time.time()
            v, au = forward(m, dev, inp, float(vts[k]), float(ats[k]))
            print(f"step {k}/{n} tv={vts[k]:.5f} ta={ats[k]:.5f} {time.time()-t0:.1f}s |v|={np.abs(v).max():.3g} finite={np.isfinite(v).all()}", flush=True)
            if k in save or k == n - 1:
                d = os.path.join(a.out, f"step{k}"); os.makedirs(d, exist_ok=True)
                for nm, arr, ext in (("video_hidden_states", inp["video"], "f32"), ("audio_hidden_states", inp["audio"], "f32"),
                                     ("encoder_hidden_states", inp["text"], "f32"), ("position_ids", inp["pos"], "f32"),
                                     ("adaln_indices", inp["adaln"], "i32"), ("timestep_indices", inp["tsi"], "i32")):
                    arr.tofile(os.path.join(d, f"{nm}.{ext}"))
                json.dump(dict(tv=float(vts[k]), ta=float(ats[k]), step=k), open(os.path.join(d, "meta.json"), "w"))
                np.savez(os.path.join(d, f"ref_{a.tag or a.dtype}.npz"), video_velocity=v, audio_velocity=au)
            inp["video"] = euler(inp["video"], v, vts[k], vsig[k], vsig[k + 1]).astype(np.float32)
            inp["audio"] = euler(inp["audio"], au, ats[k], asig[k], asig[k + 1]).astype(np.float32)
    else:
        for sd in sorted(os.listdir(a.steps_dir), key=lambda s: int(s[4:])):
            d = os.path.join(a.steps_dir, sd)
            meta = json.load(open(os.path.join(d, "meta.json")))
            inp = load_dump(d)
            t0 = time.time()
            v, au = forward(m, dev, inp, meta["tv"], meta["ta"])
            print(f"{sd} {a.dtype} vscale={a.vscale} {time.time()-t0:.1f}s", flush=True)
            np.savez(os.path.join(d, f"ref_{a.tag or (a.dtype + ('_fold' if a.vscale != 1 else ''))}.npz"), video_velocity=v, audio_velocity=au)


if __name__ == "__main__":
    main()
