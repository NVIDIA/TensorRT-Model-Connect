# SPDX-License-Identifier: Apache-2.0
"""Replay saved denoiser inputs (ref_dit.py steps dirs) through the TRT-RTX MiniMax-H3 plans.

AdaLN tables are produced by the real TRT AdaLN-precompute plan from the step's timesteps, then the
full 50-layer DiT plan (cp=1 on GPU 0, and/or cp=2 over 2 GPUs) runs each step. Outputs are saved as
<stepdir>/trt_cp{1,2}[_tag].npz so compare_velocity.py can score them against the PyTorch reference.

  python trt_replay.py --real CKPT --steps-dir DIR [--world 1|2] [--only-world2] [--tag T]
"""
import argparse, ctypes, json, math, multiprocessing as mp, os, sys, tempfile, time
from dataclasses import replace
import numpy as np

REPO = os.environ.get("TMC_REPO") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def timestep_features(t):
    i = np.arange(128, dtype=np.float64)
    freq = np.exp(-math.log(10000.0) * i / 128.0)
    ph = float(t) * freq
    return np.concatenate([np.cos(ph), np.sin(ph)]).astype(np.float32)


def read(d, name, dt):
    return np.fromfile(os.path.join(d, name), dtype=dt)


def worker(rank, world, args, barrier, uidq, tmp):
    try:
        impl(rank, world, args, barrier, uidq, tmp)
    except BaseException:
        barrier.abort(); raise


def impl(rank, world, args, barrier, uidq, tmp):
    sys.path.insert(0, REPO)
    from cuda.bindings import runtime as cudart
    from families.minimax_h3 import _trt, dit_builder as db, adaln_builder as ab
    from families.minimax_h3.config import SOL_ENGINE_1344X768_124F as P
    from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state
    trt = _trt.trt
    log = lambda *a: print(f"[r{rank}]", *a, flush=True)

    def ck(r):
        err = r[0] if isinstance(r, tuple) else r
        assert int(err) == 0, f"CUDA error {err}"
        return r[1] if isinstance(r, tuple) and len(r) > 1 else None

    ck(cudart.cudaSetDevice(rank + args.gpu))
    steps = sorted([s for s in os.listdir(args.steps_dir) if s.startswith("step")], key=lambda s: int(s[4:]))
    dirs = [os.path.join(args.steps_dir, s) for s in steps]
    metas = [json.load(open(os.path.join(d, "meta.json"))) for d in dirs]

    def alloc_run(plan, feeds_per_step, comm=None, iters=1):
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        eng = rt.deserialize_cuda_engine(plan)
        ctx = eng.create_execution_context()
        stream = ck(cudart.cudaStreamCreate())
        if comm is not None:
            assert ctx.set_communicator(comm), "set_communicator failed"
        ptrs, outs = {}, {}
        first = feeds_per_step[0]
        for name, a in first.items():
            ptrs[name] = ck(cudart.cudaMalloc(a.nbytes))
            assert ctx.set_input_shape(name, a.shape)
            assert ctx.set_tensor_address(name, int(ptrs[name]))
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i)
            if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                shp = tuple(ctx.get_tensor_shape(n)); dt = eng.get_tensor_dtype(n)
                esz = 2 if dt in (trt.bfloat16, trt.float16) else 4
                p = ck(cudart.cudaMalloc(int(np.prod(shp)) * esz)); assert ctx.set_tensor_address(n, int(p))
                outs[n] = (p, shp, dt, esz)
        results, times = [], []
        for feeds in feeds_per_step:
            for name, a in feeds.items():
                ck(cudart.cudaMemcpy(ptrs[name], a.ctypes.data, a.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            ts = []
            for _ in range(iters):
                if comm is not None:
                    barrier.wait()
                t0 = time.perf_counter()
                assert ctx.execute_async_v3(int(stream)), "execute failed"
                ck(cudart.cudaStreamSynchronize(stream)); ts.append(time.perf_counter() - t0)
            times.append(float(np.median(ts)))
            res = {}
            for n, (p, shp, dt, esz) in outs.items():
                host = np.empty(int(np.prod(shp)), dtype=np.uint16 if dt == trt.bfloat16 else np.float16 if esz == 2 else np.float32)
                ck(cudart.cudaMemcpy(host.ctypes.data, p, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
                res[n] = (bf16_to_f32(host) if dt == trt.bfloat16 else host).reshape(shp)
            results.append(res)
        for p in list(ptrs.values()) + [o[0] for o in outs.values()]:
            cudart.cudaFree(p)
        return results, times

    # ---- 1. AdaLN tables (rank 0, real TRT precompute plan), cached on disk
    tabdir = os.path.join(args.steps_dir, "_mod")
    if rank == 0 and not all(os.path.exists(os.path.join(tabdir, f"{s}.npz")) for s in steps):
        os.makedirs(tabdir, exist_ok=True)
        prof = replace(P)
        w = numpy_state(load_selected_component_state_dict(args.real + "/transformer", sorted(ab.checkpoint_keys(prof))))
        t = time.time(); plan = ab.build_adaln_precompute_engine(w, prof, workspace_bytes=8 << 30); log(f"adaln built {time.time()-t:.0f}s")
        feeds = []
        for m in metas:
            f = np.zeros((4, 256), np.float32); f[0] = timestep_features(m["tv"]); f[1] = timestep_features(m["ta"])
            feeds.append({"timestep_features": f})
        # adaln outputs are bf16: fetch raw uint16 so tables are bit-exact
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING)); eng = rt.deserialize_cuda_engine(plan); ctx = eng.create_execution_context()
        stream = ck(cudart.cudaStreamCreate())
        pin = ck(cudart.cudaMalloc(4 * 256 * 4)); ctx.set_input_shape("timestep_features", (4, 256)); ctx.set_tensor_address("timestep_features", int(pin))
        outs = {}
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i)
            if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                shp = tuple(ctx.get_tensor_shape(n)); assert eng.get_tensor_dtype(n) == trt.bfloat16, (n, eng.get_tensor_dtype(n))
                p = ck(cudart.cudaMalloc(int(np.prod(shp)) * 2)); ctx.set_tensor_address(n, int(p)); outs[n] = (p, shp)
        for s, fd in zip(steps, feeds):
            a = fd["timestep_features"]
            ck(cudart.cudaMemcpy(pin, a.ctypes.data, a.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            assert ctx.execute_async_v3(int(stream)); ck(cudart.cudaStreamSynchronize(stream))
            tab = {}
            for n, (p, shp) in outs.items():
                host = np.empty(int(np.prod(shp)), np.uint16)
                ck(cudart.cudaMemcpy(host.ctypes.data, p, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)); tab[n] = host.reshape(shp)
            np.savez(os.path.join(tabdir, f"{s}.npz"), **tab)
        # validate vs the runtime dump if available (step 0)
        del plan, w
    barrier.wait()

    def feeds_for(d, s):
        tab = np.load(os.path.join(tabdir, f"{s}.npz"))
        f = {"video_hidden_states": read(d, "video_hidden_states.f32", np.float32).reshape(-1, 96),
             "audio_hidden_states": read(d, "audio_hidden_states.f32", np.float32).reshape(-1, 32),
             "encoder_hidden_states": read(d, "encoder_hidden_states.f32", np.float32).reshape(-1, 5120),
             "position_ids": read(d, "position_ids.f32", np.float32).reshape(-1, 3),
             "adaln_indices": read(d, "adaln_indices.i32", np.int32),
             "timestep_indices": read(d, "timestep_indices.i32", np.int32)}
        for k in tab.files:
            f[k] = np.ascontiguousarray(tab[k])
        return f

    def build(cp):
        prof = replace(P, context_parallel_size=cp)
        keep = set(db.checkpoint_keys(prof))
        w = numpy_state(load_selected_component_state_dict(args.real + "/transformer", sorted(keep)))
        t = time.time(); plan = db.build_dit_engine(w, prof, workspace_bytes=8 << 30); log(f"built cp={cp} {len(plan)/1e9:.1f}GB {time.time()-t:.0f}s")
        return plan

    def save(res, times, tag):
        for d, s, r, t in zip(dirs, steps, res, times):
            np.savez(os.path.join(d, f"trt_{tag}.npz"), **r, time_s=t)

    if world == 1:
        plan = build(1)
        res, times = alloc_run(plan, [feeds_for(d, s) for d, s in zip(dirs, steps)], iters=2)
        log("cp1 times", times); save(res, times, "cp1" + (("_" + args.tag) if args.tag else ""))
        return
    plan = build(2)
    class UID(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_char * 128)]
    nccl = ctypes.CDLL("libnccl.so.2")
    nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(UID)]
    nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UID, ctypes.c_int]
    if rank == 0:
        uid = UID(); assert nccl.ncclGetUniqueId(ctypes.byref(uid)) == 0
        raw = ctypes.string_at(ctypes.byref(uid), ctypes.sizeof(uid))
        for _ in range(world - 1): uidq.put(raw)
    else:
        uid = UID.from_buffer_copy(uidq.get())
    h = ctypes.c_void_p(); assert nccl.ncclCommInitRank(ctypes.byref(h), world, uid, rank) == 0
    api = ctypes.pythonapi; api.PyCapsule_New.restype = ctypes.py_object
    api.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    cap = api.PyCapsule_New(h.value, None, None)
    res, times = alloc_run(plan, [feeds_for(d, s) for d, s in zip(dirs, steps)], comm=cap, iters=2)
    log("cp2 times", times)
    if rank == 0:
        save(res, times, "cp2" + (("_" + args.tag) if args.tag else ""))
    barrier.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True); ap.add_argument("--steps-dir", required=True)
    ap.add_argument("--world", type=int, default=1); ap.add_argument("--tag", default=""); ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(args.world); q = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        ps = [ctx.Process(target=worker, args=(r, args.world, args, barrier, q, tmp)) for r in range(args.world)]
        [p.start() for p in ps]; [p.join() for p in ps]
        sys.exit(1 if any(p.exitcode for p in ps) else 0)


if __name__ == "__main__":
    main()
