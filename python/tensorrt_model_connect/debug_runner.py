"""Shared TensorRT debug infrastructure.

Concrete runners, bundle readers, and bundle-to-runner factories live in model
family modules. This module keeps only CUDA/TRT handles and distributed NCCL
setup used by model-owned debug runners and E2E subprocesses.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
import time

import numpy as np
from tensorrt_model_connect import trt_compat



trt = trt_compat.get_trt() if trt_compat.is_available() else None

# cuda-python >= 13 uses cuda.bindings.runtime; older versions use cuda.cudart.
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - exercised in TRT-free test envs
        cudart = None  # type: ignore[assignment]


def _check_cuda(status):
    """Raise on CUDA error."""
    if cudart is None:
        raise RuntimeError("cuda-python is required for debug_runner execution")
    if hasattr(cudart, "cudaError_t"):
        success = cudart.cudaError_t.cudaSuccess
    else:
        success = 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")


def _trt_nptype_safe(dtype: trt.DataType):
    """Resolve TRT dtype to a NumPy dtype, including BF16 fallback."""
    try:
        return trt.nptype(dtype)
    except TypeError:
        if dtype == trt.bfloat16:
            return np.uint16
        raise


def _trt_itemsize(dtype: trt.DataType) -> int:
    return np.dtype(_trt_nptype_safe(dtype)).itemsize


def _require_trt_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for debug_runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for debug_runner execution")


_NCCL_UNIQUE_ID_BYTES = 128
_NCCL_SUCCESS = 0


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * _NCCL_UNIQUE_ID_BYTES)]


def _env_int(names: tuple[str, ...], default: int | None = None) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _mpi_rank_info_from_env() -> tuple[int, int]:
    rank = _env_int(("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "RANK"), 0)
    world_size = _env_int(
        ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "WORLD_SIZE"), 1)
    return int(rank or 0), int(world_size or 1)


def _default_nccl_rendezvous_path() -> str:
    path = os.environ.get("TRTMC_NCCL_RENDEZVOUS")
    if path:
        return path
    job_id = (
        os.environ.get("OMPI_COMM_WORLD_JOBID")
        or os.environ.get("PMIX_NAMESPACE")
        or os.environ.get("SLURM_JOB_ID")
        or f"pid{os.getppid()}"
    )
    safe_job_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in job_id)
    return os.path.join(tempfile.gettempdir(), f"trtmc_nccl_{safe_job_id}.bin")


def _load_nccl_library() -> ctypes.CDLL:
    errors: list[str] = []
    for name in ("libnccl.so.2", "libnccl.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    else:
        raise RuntimeError("Unable to load NCCL library: " + "; ".join(errors))

    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
    lib.ncclGetUniqueId.restype = ctypes.c_int
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        _NcclUniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommInitRank.restype = ctypes.c_int
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommDestroy.restype = ctypes.c_int
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    return lib


def _nccl_error_string(lib: ctypes.CDLL, status: int) -> str:
    try:
        msg = lib.ncclGetErrorString(status)
    except Exception:
        msg = None
    if msg:
        return msg.decode("utf-8", errors="replace")
    return f"NCCL error {status}"


def _capsule_from_pointer(ptr: int):
    pycapsule_new = ctypes.pythonapi.PyCapsule_New
    pycapsule_new.restype = ctypes.py_object
    pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return pycapsule_new(ctypes.c_void_p(ptr), None, None)


class TensorParallelNcclGroup:
    """Small NCCL group helper for TensorRT distributed debug execution.

    The caller must destroy TRT execution contexts before closing this group.
    """

    def __init__(
        self,
        world_size: int | None = None,
        rendezvous_path: str | None = None,
        timeout_s: float = 60.0,
        set_device: bool = True,
    ):
        _require_trt_runtime()
        self.rank, detected_world_size = _mpi_rank_info_from_env()
        self.world_size = int(world_size or detected_world_size)
        if self.world_size <= 1:
            raise RuntimeError("TensorParallelNcclGroup requires world_size > 1")
        if detected_world_size != self.world_size:
            raise RuntimeError(
                f"MPI world size {detected_world_size} does not match "
                f"requested tensor parallel size {self.world_size}"
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise RuntimeError(
                f"MPI rank {self.rank} is outside world size {self.world_size}")

        if set_device:
            status = cudart.cudaSetDevice(self.rank)
            _check_cuda(status[0] if isinstance(status, tuple) else status)

        self.rendezvous_path = rendezvous_path or _default_nccl_rendezvous_path()
        self._lib = _load_nccl_library()
        self._comm = ctypes.c_void_p()
        unique_id = self._exchange_unique_id(timeout_s=timeout_s)
        self._check(
            self._lib.ncclCommInitRank(
                ctypes.byref(self._comm),
                self.world_size,
                unique_id,
                self.rank,
            ),
            "ncclCommInitRank",
        )
        if not self._comm.value:
            raise RuntimeError("NCCL returned a null communicator")
        self._communicator_capsule = _capsule_from_pointer(int(self._comm.value))
        self._closed = False

    @property
    def communicator(self):
        """PyCapsule wrapping the ncclComm_t pointer for TensorRT Python."""
        return self._communicator_capsule

    def _check(self, status: int, op: str) -> None:
        if int(status) != _NCCL_SUCCESS:
            raise RuntimeError(f"{op} failed: {_nccl_error_string(self._lib, status)}")

    def _exchange_unique_id(self, timeout_s: float) -> _NcclUniqueId:
        path = self.rendezvous_path
        if self.rank == 0:
            unique_id = _NcclUniqueId()
            self._check(self._lib.ncclGetUniqueId(ctypes.byref(unique_id)), "ncclGetUniqueId")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp_path = f"{path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                f.write(ctypes.string_at(ctypes.byref(unique_id), _NCCL_UNIQUE_ID_BYTES))
            os.replace(tmp_path, path)
            return unique_id

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if len(data) == _NCCL_UNIQUE_ID_BYTES:
                    unique_id = _NcclUniqueId()
                    ctypes.memmove(ctypes.byref(unique_id), data, _NCCL_UNIQUE_ID_BYTES)
                    return unique_id
            except FileNotFoundError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for NCCL rendezvous file {path!r}")
            time.sleep(0.05)

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if self._comm.value:
            self._check(self._lib.ncclCommDestroy(self._comm), "ncclCommDestroy")
            self._comm = ctypes.c_void_p()

    def __enter__(self) -> "TensorParallelNcclGroup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
