# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU Ulysses context-parallel check for the MiniMax-H3 DiT (TRT-RTX).

Builds a real-dimension, few-layer DiT with ``cp=1`` (reference) and ``cp=N``,
runs the reference on rank 0 and the CP plan on N ranks joined by one NCCL
communicator, and compares the outputs. It needs no MPI: ranks are spawned
with ``multiprocessing`` so it runs the same on Windows and Linux.

    python cp_gpu_check.py --layers 2 --world 2 [--video-rows 8192] [--real CKPT_DIR]
    python cp_gpu_check.py --component vae --world 2 --real CKPT_DIR   # tile-parallel VAE

Windows recipe (validated on 2x RTX A5500, driver 591.74, WDDM): Python 3.12 venv with
``numpy ml_dtypes pytest cuda-bindings==12.9.* nvidia-cuda-runtime-cu12`` plus the
``tensorrt_rtx_cu12_{bindings,libs}`` 1.7.1.134 win_amd64 wheels (``pip install --no-deps``); PATH must
contain ``site-packages\\nvidia\\cuda_runtime\\bin``, ``site-packages\\tensorrt_rtx_libs`` and the
directory of a Windows ``nccl.dll`` >= 2.32.5 (CUDA 12.9, SASS for the GPU, static cudart).

NCCL library: ``nccl.dll`` (Windows) / ``libnccl.so.2`` (Linux), taken from
``--nccl`` or ``$MINIMAX_H3_NCCL``, otherwise searched on PATH / the loader path.
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import replace

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(x):
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _replay_input(directory, name, dtype, trt):
    """Load a runtime-dumped input (MINIMAX_H3_DUMP_DIR) shaped for the plan, or None."""
    if not directory:
        return None
    for suffix, np_dtype in ((".f32", np.float32), (".i32", np.int32), (".bf16", np.uint16)):
        path = os.path.join(directory, name + suffix)
        if os.path.exists(path):
            return np.fromfile(path, dtype=np_dtype)
    return None


def _load_nccl(path):
    if os.name == "nt":
        # nccl.dll must be in the process before TRT-RTX looks for it.
        if not path:  # Python >= 3.8 does not search PATH for DLL dependencies
            for d in os.environ.get("PATH", "").split(os.pathsep):
                if d and os.path.isfile(os.path.join(d, "nccl.dll")):
                    path = os.path.join(d, "nccl.dll")
                    break
        # An older nccl.dll links cudart64_12.dll dynamically: let its dependencies resolve via PATH.
        for d in [os.path.dirname(os.path.abspath(path))] + os.environ.get("PATH", "").split(os.pathsep):
            if d and os.path.isdir(d):
                os.add_dll_directory(d)
        return ctypes.WinDLL(path or "nccl.dll")
    return ctypes.CDLL(path or "libnccl.so.2")


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _worker(rank, world, args, barrier, uid_queue, tmpdir):
    try:
        _worker_impl(rank, world, args, barrier, uid_queue, tmpdir)
    except BaseException:
        barrier.abort()  # do not leave the other ranks waiting
        raise


def _worker_impl(rank, world, args, barrier, uid_queue, tmpdir):
    sys.path.insert(0, REPO)
    if args.workaround_wddm:
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    from cuda.bindings import runtime as cudart

    from families.minimax_h3 import _trt
    from families.minimax_h3 import dit_builder as db
    from families.minimax_h3.config import SOL_ENGINE_1344X768_124F as P
    from families.minimax_h3.tests.test_trt_builders import _weights

    trt = _trt.trt
    log = lambda *a: print(f"[r{rank}]", *a, flush=True)

    def ck(r):
        err = r[0] if isinstance(r, tuple) else r
        assert int(err) == 0, f"CUDA error {err}"
        return r[1] if isinstance(r, tuple) and len(r) > 1 else None

    ck(cudart.cudaSetDevice(rank))

    def build(cp):
        if args.component == "vae":
            from families.minimax_h3 import vae_builder as vb
            from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state

            if args.real:
                w = numpy_state(load_selected_component_state_dict(args.real + "/vae", vb.checkpoint_keys()))
            else:  # random weights; --layers sets the decoder depth (default 36 with --real)
                from families.minimax_h3.tests.test_context_parallel import _vae_weights

                vb.LAYERS = args.layers
                w = _vae_weights(vb, args.layers)
            t = time.perf_counter()
            plan = vb.build_vae_tile_decoder_engine(w, workspace_bytes=args.workspace_gb << 30, cp_size=cp)
            log(f"built VAE cp={cp}: {len(plan) / 1e9:.2f} GB in {time.perf_counter() - t:.0f}s")
            return None, plan
        if args.small:
            from families.minimax_h3.tests.test_context_parallel import _small_profile

            prof = replace(_small_profile(cp), num_layers=args.layers)
        else:
            prof = replace(P, num_layers=args.layers, num_refiner_layers=1, context_parallel_size=cp)
            if args.video_rows:
                v = args.video_rows
                prof = replace(prof, video_rows=v, padded_sequence_length=v + prof.audio_rows + prof.text_rows)
        keep = set(db.checkpoint_keys(prof))
        if args.real:
            from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state

            w = numpy_state(load_selected_component_state_dict(args.real + "/transformer", sorted(keep)))
        else:
            w = {k: v for k, v in _weights(prof).items() if k in keep}
        t = time.perf_counter()
        plan = db.build_dit_engine(w, prof, workspace_bytes=args.workspace_gb << 30, verbose=False)
        log(f"built cp={cp} L={args.layers}: {len(plan) / 1e9:.2f} GB in {time.perf_counter() - t:.0f}s")
        return prof, plan

    def make_inputs(prof, engine):
        rng = np.random.default_rng(123)
        if prof is None:  # VAE tile decoder: all inputs are dense latents
            return {
                engine.get_tensor_name(i): rng.normal(0, 1.0, tuple(engine.get_tensor_shape(engine.get_tensor_name(i)))).astype(np.float32)
                for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
            }
        rows = args.text_rows + prof.audio_rows + prof.video_rows
        out = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(engine.get_tensor_shape(name))
            shape = tuple(
                rows if (d == -1 and name in ("position_ids", "adaln_indices", "timestep_indices"))
                else args.text_rows if d == -1 else d
                for d in shape
            )
            replay = _replay_input(args.inputs, name, engine.get_tensor_dtype(name), trt)
            if replay is not None:
                out[name] = replay.reshape(shape)
                continue
            dt = engine.get_tensor_dtype(name)
            if name == "position_ids":
                a = rng.uniform(0, 30, shape).astype(np.float32)
            elif name == "adaln_indices":
                a = rng.integers(0, prof.adaln_table_rows, shape).astype(np.int32)
            elif name == "timestep_indices":
                a = rng.integers(0, prof.max_timestep_count, shape).astype(np.int32)
            elif dt == trt.bfloat16:
                a = _f32_to_bf16(rng.normal(0, 0.1, shape))
            else:
                a = rng.normal(0, 1.0, shape).astype(np.float32)
            out[name] = a
        return out

    def run(plan, prof, iters, communicator=None, sync=True):
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = rt.deserialize_cuda_engine(plan)
        assert engine is not None, "deserialize failed"
        if os.environ.get("MINIMAX_H3_RUNTIME_CONFIG"):
            ctx = engine.create_execution_context(engine.create_runtime_config())
        else:
            ctx = engine.create_execution_context()
        stream = ck(cudart.cudaStreamCreate())
        if communicator is not None:
            assert ctx.set_communicator(communicator), "set_communicator failed"
        for name, a in make_inputs(prof, engine).items():
            ptr = ck(cudart.cudaMalloc(a.nbytes))
            ck(cudart.cudaMemcpy(ptr, a.ctypes.data, a.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            assert ctx.set_input_shape(name, a.shape), f"set_input_shape {name}"
            assert ctx.set_tensor_address(name, int(ptr)), f"set_tensor_address {name}"
        outs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                dt = engine.get_tensor_dtype(name)
                esz = 2 if dt in (trt.bfloat16, trt.float16) else 4
                nb = int(np.prod(shape)) * esz
                ptr = ck(cudart.cudaMalloc(nb))
                assert ctx.set_tensor_address(name, int(ptr))
                outs[name] = (ptr, shape, dt, nb, esz)
        assert ctx.all_binding_shapes_specified

        def once():
            if sync:
                barrier.wait()
            t0 = time.perf_counter()
            assert ctx.execute_async_v3(int(stream)), "execute failed"
            ck(cudart.cudaStreamSynchronize(stream))
            return time.perf_counter() - t0

        first = once()
        times = [once() for _ in range(iters)]
        res = {}
        for name, (ptr, shape, dt, nb, esz) in outs.items():
            host = np.empty(int(np.prod(shape)), dtype=np.uint16 if dt == trt.bfloat16 else
                            np.float16 if esz == 2 else np.float32)
            ck(cudart.cudaMemcpy(host.ctypes.data, ptr, nb, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            res[name] = (_bf16_to_f32(host) if dt == trt.bfloat16 else host).reshape(shape)
        log(f"first={first * 1e3:.0f}ms median={np.median(times) * 1e3:.1f}ms min={min(times) * 1e3:.1f}ms")
        return res, float(np.median(times))

    ref = None
    if rank == 0 and not args.no_reference:
        prof1, plan1 = build(1)
        ref, t1 = run(plan1, prof1, args.iters, sync=False)
        del plan1
        if args.ref_only:
            if os.environ.get("MINIMAX_H3_SAVE_OUT"):
                np.savez(os.environ["MINIMAX_H3_SAVE_OUT"], **ref)
            for name, a in ref.items():
                a = a.astype(np.float64)
                bad = ~np.isfinite(a)
                finite = a[~bad]
                log(f"REF {name}: shape={a.shape} nonfinite={int(bad.sum())} "
                    f"absmax={np.abs(finite).max() if finite.size else float('nan'):.4g}")
                if bad.any():
                    rows_axis = 2 if a.ndim == 4 else 0
                    other = tuple(i for i in range(a.ndim) if i != rows_axis)
                    bad_rows = np.flatnonzero(bad.any(axis=other))
                    log(f"    non-finite rows: {bad_rows.size} first={bad_rows[:8].tolist()} last={bad_rows[-4:].tolist()}")
            return
    barrier.wait()

    prof, plan = build(world)

    nccl = _load_nccl(args.nccl or os.environ.get("MINIMAX_H3_NCCL"))
    nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
    nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
    if rank == 0:
        uid = _UniqueId()
        assert nccl.ncclGetUniqueId(ctypes.byref(uid)) == 0
        for _ in range(world - 1):
            uid_queue.put(ctypes.string_at(ctypes.byref(uid), ctypes.sizeof(uid)))
    else:
        uid = _UniqueId.from_buffer_copy(uid_queue.get())
    handle = ctypes.c_void_p()
    rc = nccl.ncclCommInitRank(ctypes.byref(handle), world, uid, rank)
    assert rc == 0, f"ncclCommInitRank rc={rc}"
    log("nccl comm ok")
    api = ctypes.pythonapi
    api.PyCapsule_New.restype = ctypes.py_object
    api.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = api.PyCapsule_New(handle.value, None, None)

    res, med = run(plan, prof, args.iters, communicator=capsule)
    np.savez(os.path.join(tmpdir, f"out_r{rank}.npz"), **res)
    barrier.wait()
    if rank == 0 and ref is not None:
        log(f"SPEEDUP cp={world} vs 1 GPU: {t1 * 1e3:.1f}ms -> {med * 1e3:.1f}ms = {t1 / med:.2f}x")
        for name in res:
            a, b = ref[name].astype(np.float64), res[name].astype(np.float64)
            cos = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
            log(f"{name}: shape={a.shape} cosine={cos:.6f} relL2={rel:.4g} max_abs_diff={np.abs(a - b).max():.3g}")
    if rank != 0:
        r0 = np.load(os.path.join(tmpdir, "out_r0.npz"))
        for name in res:
            log(f"{name} rank{rank} vs rank0 bit_identical={bool((r0[name] == res[name]).all())}")
    barrier.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=2, help="DiT layers (or VAE decoder layers with --component vae)")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--text-rows", type=int, default=128)
    ap.add_argument("--video-rows", type=int, default=0, help="override video token rows (smaller = less memory)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--real", help="Diffusers checkpoint dir (uses its transformer/ weights); default random")
    ap.add_argument("--component", choices=("dit", "vae"), default="dit",
                    help="vae: tile-parallel VAE decoder instead of the DiT")
    ap.add_argument("--inputs", help="directory of MINIMAX_H3_DUMP_DIR runtime inputs to replay")
    ap.add_argument("--ref-only", action="store_true", help="run only the cp=1 reference and print output statistics")
    ap.add_argument("--small", action="store_true", help="tiny profile (hidden 256)")
    ap.add_argument("--nccl", help="path to nccl.dll / libnccl.so.2")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--workaround-wddm", action=argparse.BooleanOptionalAction, default=False,
                    help="set NCCL_CUMEM_ENABLE=0 (not needed with NCCL >= 2.32.5; needed by the old 2.29.7 build)")
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.world)
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmpdir:
        procs = [ctx.Process(target=_worker, args=(r, args.world, args, barrier, q, tmpdir)) for r in range(args.world)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        bad = [p.exitcode for p in procs if p.exitcode]
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
